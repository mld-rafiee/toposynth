# toposynth/models/toposynth.py
# Full TopoSynth model: GAT encoder + Diffusion-TS backbone + training function.
# No torch_geometric dependency — GAT implemented manually.

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from toposynth.data.topology import (
    TOPOLOGY_CONFIG, get_upstream_neighbours, adjacency_matrix,
)

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────

GAT_D_H      = 64      # hidden dim per attention head
GAT_K        = 4       # number of attention heads
GAT_D_Z      = 128     # output topology embedding dimension
GAT_SLOPE    = 0.2     # LeakyReLU negative slope

DIFF_STEPS   = 200     # T_diff: total diffusion timesteps
DIFF_D_MODEL = 128     # Transformer hidden dim
DIFF_N_HEADS = 8       # Transformer attention heads
DIFF_N_LAYERS = 4      # Transformer encoder layers
DIFF_D_FF    = 256     # feedforward dim
DIFF_TOP_K   = 3       # top-K Fourier components kept in seasonal block
DIFF_LAMBDA1 = 1.0     # time-domain loss weight
DIFF_LAMBDA2 = 1.0     # frequency-domain loss weight
LAMBDA_TOPO  = 0.1     # topology auxiliary loss weight


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def _dist_ready():
    return dist.is_available() and dist.is_initialized()

def _rank0():
    return (not _dist_ready()) or dist.get_rank() == 0


# ─── EMA ─────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n], p.data = p.data.clone(), self.shadow[n].clone()

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()


class average_parameters:
    """Context manager: temporarily apply EMA weights, then restore."""
    def __init__(self, ema, model):
        self.ema, self.model = ema, model

    def __enter__(self):
        self.ema.apply_shadow(self.model)

    def __exit__(self, *_):
        self.ema.restore(self.model)


# ─── NOISE SCHEDULE ──────────────────────────────────────────────────────────

class NoiseSchedule(nn.Module):
    """
    Cosine noise schedule (Nichol & Dhariwal 2021).
    Registered as nn.Module so tensors auto-move with .to(device).
    """
    def __init__(self, n_steps=DIFF_STEPS):
        super().__init__()
        t  = torch.arange(n_steps + 1, dtype=torch.float32)
        f  = torch.cos((t / n_steps + 0.008) / 1.008 * math.pi / 2) ** 2
        ac = f / f[0]                           # alphas_cumprod
        betas  = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
        alphas = 1.0 - betas

        self.register_buffer('alphas_cumprod',      ac[1:])
        self.register_buffer('alphas_cumprod_prev', ac[:-1])
        self.register_buffer('sqrt_ac',             ac[1:].sqrt())
        self.register_buffer('sqrt_one_minus_ac',   (1 - ac[1:]).sqrt())
        self.register_buffer('betas',               betas)
        self.register_buffer('alphas',              alphas)
        self.T = n_steps

    def q_sample(self, x0, t, noise=None):
        """Forward: q(x_t|x_0) = N(√ᾱ_t x_0, (1−ᾱ_t)I)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_ac[t].view(-1, 1, 1, 1)
        sb = self.sqrt_one_minus_ac[t].view(-1, 1, 1, 1)
        return sa * x0 + sb * noise, noise

    def p_mean_variance(self, x0_pred, xt, t):
        """Posterior q(x_{t-1}|x_t, x̂_0) mean and variance."""
        a_t   = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        a_tm1 = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        b_t   = self.betas[t].view(-1, 1, 1, 1)
        al_t  = self.alphas[t].view(-1, 1, 1, 1)
        mean  = (a_tm1.sqrt() * b_t) / (1 - a_t) * x0_pred + \
                (al_t.sqrt() * (1 - a_tm1)) / (1 - a_t) * xt
        var   = b_t * (1 - a_tm1) / (1 - a_t)
        return mean, var


# ─── POSITIONAL ENCODING ─────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ─── TIME EMBEDDING ───────────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP (standard DDPM style)."""
    def __init__(self, d_model):
        super().__init__()
        self.d   = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, t):
        """t : [B] int → [B, d_model]"""
        half = self.d // 2
        freq = torch.exp(-math.log(10000) *
                         torch.arange(half, device=t.device).float() / (half - 1))
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)   # [B, half]
        emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)    # [B, d_model]
        return self.mlp(emb)


# ─── TREND BLOCK ─────────────────────────────────────────────────────────────

class TrendBlock(nn.Module):
    """
    Polynomial trend synthesis (Diffusion-TS, §3.2).
    V_tr = Σ_k (C · Linear(w_tr^k) + X_tr^k)  where C = [1, c, c², ..., c^p]
    """
    def __init__(self, seq_len, d_model, n_features, poly_order=3):
        super().__init__()
        self.n_features = n_features
        self.poly_order = poly_order
        t     = torch.linspace(0, 1, seq_len).unsqueeze(1)         # [W, 1]
        basis = torch.cat([t ** k for k in range(poly_order + 1)], dim=1)
        self.register_buffer('basis', basis)                        # [W, p+1]
        self.coef_proj = nn.Linear(d_model, (poly_order + 1) * n_features)

    def forward(self, h_last):
        """h_last : [B*N, d_model] → trend : [B*N, W, F]"""
        BN   = h_last.size(0)
        coef = self.coef_proj(h_last).view(BN, self.poly_order + 1, self.n_features)
        return torch.einsum('bkf,wk->bwf', coef, self.basis)  # [BN, W, F]


# ─── SEASONAL BLOCK ──────────────────────────────────────────────────────────

class SeasonalBlock(nn.Module):
    """
    Fourier seasonal synthesis — keeps top-K frequency amplitudes per feature.
    """
    def __init__(self, top_k=DIFF_TOP_K):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        """x : [B*N, W, F] → seasonal : [B*N, W, F]"""
        xf  = torch.fft.rfft(x, dim=1)                         # [BN, W//2+1, F]
        amp = xf.abs()                                          # [BN, W//2+1, F]
        _, idx = torch.topk(amp.mean(0), self.top_k, dim=0)    # [k, F]
        mask = torch.zeros_like(xf)
        for fi in range(x.size(-1)):
            mask[:, idx[:, fi], fi] = 1.0
        return torch.fft.irfft(xf * mask, n=x.size(1), dim=1) # [BN, W, F]


# ─── DIFFUSION TRANSFORMER (Denoising Backbone) ───────────────────────────────

class DiffusionTransformer(nn.Module):
    """
    Interpretable denoising backbone from Diffusion-TS (Chen & Qiao, ICLR 2024).

    Input  : x_t [B*N, W, F] + timestep t [B*N]
    Output : x̂_0 [B*N, W, F]  decomposed as trend + seasonal + residual

    Operates on each VNFC's time series independently (B*N batch axis).
    """
    def __init__(self, seq_len, n_features,
                 d_model=DIFF_D_MODEL, n_heads=DIFF_N_HEADS,
                 n_layers=DIFF_N_LAYERS, d_ff=DIFF_D_FF, top_k=DIFF_TOP_K):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features

        self.input_proj  = nn.Linear(n_features, d_model)
        self.pos_enc     = PositionalEncoding(d_model, max_len=seq_len + 1)
        self.time_emb    = TimeEmbedding(d_model)
        self.time_proj   = nn.Linear(d_model, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.trend_block    = TrendBlock(seq_len, d_model, n_features)
        self.seasonal_block = SeasonalBlock(top_k)
        self.residual_proj  = nn.Linear(d_model, n_features)

    def forward(self, xt, t):
        """
        xt : [B*N, W, F]
        t  : [B*N] int
        Returns x0_pred : [B*N, W, F]
        """
        h = self.pos_enc(self.input_proj(xt))                          # [BN, W, d_model]
        h = h + self.time_proj(self.time_emb(t)).unsqueeze(1)         # broadcast over W
        h = self.transformer(h)                                        # [BN, W, d_model]

        trend    = self.trend_block(h[:, -1, :])                       # [BN, W, F]
        residual = self.residual_proj(h)                               # [BN, W, F]
        seasonal = self.seasonal_block(residual)                       # [BN, W, F]

        return trend + seasonal + residual                             # [BN, W, F]


# ─── GAT ENCODER ─────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Single directed GAT layer — manual implementation, no torch_geometric.
    Follows Veličković et al. (ICLR 2018), Eqs. 1-6.

    Supports both:
      forward(h)          : h [N, in_dim]    → h' [N, out_dim]
      forward_batched(h)  : h [B, N, in_dim] → h' [B, N, out_dim]

    Directed edges: attention α_{ij} computed when j→i exists in SFC.
    Self-loop: every node always attends to itself first, so entry nodes
    (bono, ellis — no upstream neighbours) still produce a meaningful
    embedding rather than needing a separate bypass projection.
    """
    def __init__(self, in_dim, out_dim, n_heads=GAT_K,
                 slope=GAT_SLOPE, concat=True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat  = concat

        # W ∈ R^{K, in_dim, out_dim}
        self.W = nn.Parameter(torch.empty(n_heads, in_dim, out_dim))
        # a ∈ R^{K, 2*out_dim}
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.W.view(n_heads * in_dim, out_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(-1))

        self.leaky_relu = nn.LeakyReLU(slope)
        self.neighbours = get_upstream_neighbours()
        self.vnf_names  = TOPOLOGY_CONFIG['vnf_names']
        self.vnf_to_idx = TOPOLOGY_CONFIG['vnf_to_idx']

    def forward(self, h):
        """
        h : [N, in_dim] → h' : [N, K*out_dim] if concat else [N, out_dim]
        Used for single-sample inference (TCS, target embedding).

        Self-loop: every node always attends to itself plus its upstream
        neighbours.  This ensures entry nodes (bono, ellis — no incoming
        edges) still produce a meaningful embedding via self-attention
        rather than being replaced by a separate projection.
        """
        N     = len(self.vnf_names)
        out_d = self.n_heads * self.out_dim if self.concat else self.out_dim
        out   = torch.zeros(N, out_d, device=h.device)

        # Wh ∈ [K, N, out_dim]
        Wh = torch.einsum('kio,ni->kno', self.W, h)

        for i, vnf_i in enumerate(self.vnf_names):
            # Self-loop first, then upstream neighbours
            nb_names = [vnf_i] + list(self.neighbours[vnf_i])
            nb_idx   = [self.vnf_to_idx[nb] for nb in nb_names]

            head_outs = []
            for k in range(self.n_heads):
                Whi  = Wh[k, i]                       # [out_dim]
                Whjs = Wh[k, nb_idx]                  # [|N_i|+1, out_dim]
                cat_ij = torch.cat(
                    [Whi.unsqueeze(0).expand(len(nb_idx), -1), Whjs], dim=-1
                )                                     # [|N_i|+1, 2*out_dim]
                e     = self.leaky_relu((cat_ij * self.a[k]).sum(-1))  # [|N_i|+1]
                alpha = Fn.softmax(e, dim=0)           # [|N_i|+1]
                h_agg = (alpha.unsqueeze(-1) * Whjs).sum(0)             # [out_dim]
                head_outs.append(h_agg)

            if self.concat:
                out[i] = Fn.elu(torch.cat(head_outs, dim=-1))
            else:
                out[i] = Fn.elu(torch.stack(head_outs).mean(0))

        return out

    def forward_batched(self, h):
        """
        h : [B, N, in_dim] → h' : [B, N, K*out_dim] if concat else [B, N, out_dim]

        Processes the entire batch in one pass, replacing the per-sample Python
        loop that made guidance ~B× slower than necessary. The node-level loop
        (over 6 VNFs) is unavoidable with a custom GAT, but the batch dimension
        is fully vectorised via tensor ops, giving ~B× wall-clock speedup for
        guidance computation.

        Self-loop: same as forward() — every node attends to itself first.
        """
        B, N, _ = h.shape
        out_d   = self.n_heads * self.out_dim if self.concat else self.out_dim
        out     = torch.zeros(B, N, out_d, device=h.device)

        # Wh : [B, K, N, out_dim]
        Wh = torch.einsum('kio,bni->bkno', self.W, h)

        for i, vnf_i in enumerate(self.vnf_names):
            # Self-loop first, then upstream neighbours
            nb_names = [vnf_i] + list(self.neighbours[vnf_i])
            nb_idx   = [self.vnf_to_idx[nb] for nb in nb_names]
            M        = len(nb_idx)

            # Wh_i : [B, K, out_dim]   Wh_j : [B, K, M, out_dim]
            Wh_i = Wh[:, :, i, :]
            Wh_j = Wh[:, :, nb_idx, :]

            # cat_ij : [B, K, M, 2*out_dim]
            Wh_i_exp = Wh_i.unsqueeze(2).expand(-1, -1, M, -1)
            cat_ij   = torch.cat([Wh_i_exp, Wh_j], dim=-1)

            # e : [B, K, M]  (attention logits)
            e     = self.leaky_relu((cat_ij * self.a.view(1, self.n_heads, 1, -1)).sum(-1))
            alpha = Fn.softmax(e, dim=2)            # [B, K, M]

            # h_agg : [B, K, out_dim]
            h_agg = (alpha.unsqueeze(-1) * Wh_j).sum(2)

            if self.concat:
                out[:, i, :] = Fn.elu(h_agg.reshape(B, -1))
            else:
                out[:, i, :] = Fn.elu(h_agg.mean(1))

        return out                                  # [B, N, out_d]


class GATEncoder(nn.Module):
    """
    Two-layer GAT encoder producing topology embeddings z ∈ R^{N, d_z}.

    Layer 1 : in_dim  → K * d_h   (concatenation across heads)
    Layer 2 : K * d_h → d_z       (average across heads)

    Supports both forward (single [N, F]) and forward_batched ([B, N, F]).
    Self-loops in GATLayer ensure every node — including entry/isolated
    nodes (bono, ellis) — produces a meaningful embedding via self-attention.
    No separate entry_proj bypass is needed; all nodes flow through GAT.
    Includes correlation decoder for self-supervised topology loss.
    """
    def __init__(self, in_dim, d_h=GAT_D_H, d_z=GAT_D_Z, n_heads=GAT_K):
        super().__init__()
        self.vnf_names  = TOPOLOGY_CONFIG['vnf_names']
        self.neighbours = get_upstream_neighbours()

        self.gat1 = GATLayer(in_dim,         d_h,   n_heads=n_heads, concat=True)
        self.gat2 = GATLayer(n_heads * d_h,  d_z,   n_heads=n_heads, concat=False)
        self.norm1 = nn.LayerNorm(n_heads * d_h)
        self.norm2 = nn.LayerNorm(d_z)

        # Correlation decoder: predict Pearson C[i,j] from z_i, z_j
        self.corr_decoder = nn.Bilinear(d_z, d_z, 1)

    def forward(self, h_bar):
        """
        h_bar : [N, F]  → z : [N, d_z]
        Used for single-sample inference (TCS, target embedding).

        All nodes (including entry nodes bono and ellis) pass through the
        same two-layer GAT path — self-loops in GATLayer ensure no node
        is skipped.
        """
        h1 = self.gat1(h_bar)          # [N, K*d_h]; self-loops inside GATLayer
        h1 = self.norm1(h1)
        h2 = self.gat2(h1)             # [N, d_z]
        h2 = self.norm2(h2)
        return h2                       # [N, d_z]

    def forward_batched(self, h_bar):
        """
        h_bar : [B, N, F]  → z : [B, N, d_z]

        Batched version used inside generate() for guidance gradient computation.
        All nodes handled uniformly by GATLayer self-loops.
        """
        h1 = self.gat1.forward_batched(h_bar)  # [B, N, K*d_h]
        h1 = self.norm1(h1)
        h2 = self.gat2.forward_batched(h1)     # [B, N, d_z]
        h2 = self.norm2(h2)
        return h2                               # [B, N, d_z]

    def decode_correlations(self, z):
        """z : [N, d_z] → C_pred : [N, N] predicted cross-VNF correlations"""
        N  = z.size(0)
        zi = z.unsqueeze(1).expand(N, N, -1).reshape(N * N, -1)
        zj = z.unsqueeze(0).expand(N, N, -1).reshape(N * N, -1)
        return torch.tanh(self.corr_decoder(zi, zj).view(N, N))


# ─── TOPOSYNTH (top-level model) ─────────────────────────────────────────────

class TopoSynth(nn.Module):
    """
    Full TopoSynth model.
    DiffusionTransformer  : unconditional DDPM backbone per VNFC
    GATEncoder            : topology-aware embedding for gradient guidance
    NoiseSchedule         : cosine DDPM schedule (registered buffers → auto device)
    """
    def __init__(self, seq_len, n_features, n_vnf, n_diff_steps=DIFF_STEPS):
        super().__init__()
        self.seq_len      = seq_len
        self.n_features   = n_features
        self.n_vnf        = n_vnf
        self.n_diff_steps = n_diff_steps

        self.denoiser = DiffusionTransformer(seq_len=seq_len, n_features=n_features)
        self.gat      = GATEncoder(in_dim=n_features)
        self.schedule = NoiseSchedule(n_diff_steps)

    def forward(self, x0, t, h_bar):
        """
        Training forward pass.

        x0    : [B, W, N, F]   real data window
        t     : [B] int        diffusion timestep per sample
        h_bar : [N, F]         per-VNF training-set mean (single, not batched)

        Returns:
            x0_pred : [B, W, N, F]  denoised estimate x̂_0
            xt      : [B, W, N, F]  noisy input
            z       : [N, d_z]      topology embeddings
            C_pred  : [N, N]        predicted cross-VNF correlations
        """
        B, W, N, n_feat = x0.shape

        # ── Forward diffusion ─────────────────────────────────────────────────
        xt, _ = self.schedule.q_sample(x0, t)       # [B, W, N, F]

        # ── Denoising backbone (operates per VNFC independently) ──────────────
        xt_flat  = xt.permute(0, 2, 1, 3).reshape(B * N, W, n_feat)
        t_expand = t.unsqueeze(1).expand(B, N).reshape(B * N)

        x0_pred_flat = self.denoiser(xt_flat, t_expand)                     # [B*N, W, F]
        x0_pred = x0_pred_flat.view(B, N, W, n_feat).permute(0, 2, 1, 3)  # [B, W, N, F]

        # ── Topology encoding ─────────────────────────────────────────────────
        z      = self.gat(h_bar)                    # [N, d_z]
        C_pred = self.gat.decode_correlations(z)    # [N, N]

        return x0_pred, xt, z, C_pred

    @torch.no_grad()
    def generate(self, n_samples, h_bar, eta=0.1, gamma=0.01, device='cuda',
                 n_infer_steps=None, guidance_every=1):
        """
        Inference: reverse diffusion with topology gradient guidance.

        Parameters
        ----------
        n_samples      : number of windows to generate
        h_bar          : [N, F]  per-VNF training-set mean
        eta            : guidance strength η
        gamma          : log-likelihood weight γ
        device         : 'cuda' or 'cpu'
        n_infer_steps  : reverse-diffusion steps at inference (default: full T=200).
                         E.g. 50 gives 4× speedup with negligible quality loss.
        guidance_every : compute GAT gradient only every K steps (default: 1 = every step).
                         E.g. 5 gives ~5× speedup in guidance with negligible quality loss.

        Returns
        -------
        synthetic : [n_samples, W, N, F]  float32, unnormalised output
                    (will be near [0,1] for MinMax-scaled data but not clamped)
        """
        W, N, F  = self.seq_len, self.n_vnf, self.n_features
        h_bar    = h_bar.to(device)
        z_target = self.gat(h_bar).detach()          # [N, d_z] — fixed target

        xt = torch.randn(n_samples, W, N, F, device=device)

        # ── Build inference timestep sequence ─────────────────────────────────
        T = self.schedule.T                          # e.g. 200
        if n_infer_steps is not None and n_infer_steps < T:
            stride    = T // n_infer_steps
            timesteps = list(range(0, T, stride))[::-1]   # [T-1, T-1-stride, ..., 0]
        else:
            timesteps = list(range(T - 1, -1, -1))        # original: every step

        for step_idx, step in enumerate(timesteps):
            t_batch = torch.full((n_samples,), step, dtype=torch.long, device=device)

            xt = xt.detach()

            apply_guidance = (step_idx % guidance_every == 0)

            # ── Step 1: denoise (no grad needed for the denoiser itself) ─────────
            with torch.no_grad():
                xt_flat      = xt.permute(0, 2, 1, 3).reshape(n_samples * N, W, F)
                t_exp        = t_batch.unsqueeze(1).expand(n_samples, N).reshape(n_samples * N)
                x0_pred_flat = self.denoiser(xt_flat, t_exp)
                x0_pred      = x0_pred_flat.view(n_samples, N, W, F).permute(0, 2, 1, 3)

            # ── Step 2: guidance gradient directly w.r.t. x0_pred ─────────────
            # Correct formulation: ∂L/∂x̂₀, not ∂L/∂xₜ.
            # The denoiser Jacobian ∂x̂₀/∂xₜ at high noise levels is ill-scaled
            # (can be O(σₜ/σ_data)), so computing grad via xt and applying it to
            # x̂₀ gives a guidance correction of the wrong magnitude and direction.
            # Treating x̂₀ as a leaf and differentiating only through GAT gives a
            # clean, scale-consistent gradient.
            with torch.no_grad():
                if apply_guidance:
                    # Treat x0_pred as leaf; GAT is differentiable, denoiser is not
                    x0_leaf  = x0_pred.detach().requires_grad_(True)
                    with torch.enable_grad():
                        x0_mean  = x0_leaf.mean(dim=1)                  # [B, N, F]
                        z_pred   = self.gat.forward_batched(x0_mean)     # [B, N, d_z]
                        z_tgt_ex = z_target.unsqueeze(0).expand_as(z_pred)
                        guide_loss = Fn.mse_loss(z_pred, z_tgt_ex)
                        grad = torch.autograd.grad(guide_loss, x0_leaf)[0]  # [B, W, N, F]
                    # Clamp x0_guided to [0,1]: x0 must lie in the training-data range.
                    # Without clamping, the guidance correction (−η·∇) can overshoot
                    # the valid range, and the posterior then samples from an invalid
                    # x0 for all remaining steps — errors compound over 50 steps.
                    # MinMax-normalized real data has values at exactly 0.0 and 1.0
                    # (training-set min/max), so hard clamping introduces no spurious
                    # boundary values that the discriminator could detect.
                    x0_guided = (x0_pred - eta * grad.detach()).clamp(0.0, 1.0)
                else:
                    x0_guided = x0_pred

                # Posterior step: sample x_{t-1}
                if step > 0:
                    mean, var = self.schedule.p_mean_variance(
                        x0_guided, xt.detach(), t_batch
                    )
                    xt = mean + var.sqrt() * torch.randn_like(mean)
                else:
                    xt = x0_guided

        # Final clamp to [0,1]: belt-and-suspenders.
        # x0_guided is already clamped inside the loop, but the posterior
        # sampling (mean + σ·ε) can nudge the last step slightly outside range.
        return xt.clamp(0.0, 1.0).detach()           # [n_samples, W, N, F]


# ─── LOSS FUNCTIONS ──────────────────────────────────────────────────────────

def diffusion_loss(x0, x0_pred, lam1=DIFF_LAMBDA1, lam2=DIFF_LAMBDA2):
    """
    Diffusion-TS training loss (Chen & Qiao, Eq. 11).
    L = λ1·||x0 − x̂0||² + λ2·||FFT(x0) − FFT(x̂0)||²

    norm='ortho' is essential: without it, PyTorch's rfft is unnormalized,
    so the DC bin has magnitude ~W·mean (≈24 for W=48, mean=0.5 on [0,1] data).
    The FFT-domain MSE then operates on squared values of order W²·mean² ≈ 576,
    while the time-domain MSE operates on [0,1].  The frequency term dominates
    by ~two orders of magnitude, gradient descent trains almost exclusively on
    mean-matching (DC), and the denoiser never learns temporal dynamics.
    With norm='ortho' (Parseval's theorem), both terms have equal expected
    magnitude and the model learns both trend and fluctuation structure.
    """
    l_time = Fn.mse_loss(x0_pred, x0)
    l_freq = Fn.mse_loss(
        torch.fft.rfft(x0_pred, dim=1, norm='ortho').abs(),
        torch.fft.rfft(x0,      dim=1, norm='ortho').abs()
    )
    return lam1 * l_time + lam2 * l_freq


def topology_loss(C_pred, adj):
    """
    Structure-aware topology loss (fixed target).

    Trains the GAT decoder to output +1 for directly connected VNF pairs and
    -1 for unconnected pairs.  The adjacency matrix is topology-derived and
    never changes, so gradients are stable across every batch.  This replaces
    the previous formulation that matched per-batch empirical Pearson
    correlations: those targets swung wildly whenever a crash event shifted
    the within-batch VNF correlation structure, producing oscillating topo
    loss and an undertrained GAT.

    C_pred : [N, N]  tanh-activated predictions from GAT decoder, ∈ [-1, 1]
    adj    : [N, N]  symmetric binary adjacency (float, on same device as C_pred)
                     adj[i,j] = 1 if edge i→j or j→i exists in the topology
    """
    target = adj * 2.0 - 1.0        # {0, 1} → {-1, +1}
    return Fn.mse_loss(C_pred, target)


# ─── TRAINING FUNCTION ────────────────────────────────────────────────────────

def train_toposynth(train_loader, val_loader, train_sampler, checkpoint_dir,
                    seq_len, n_features, n_vnf,
                    n_epochs=100, lr=1e-4, weight_decay=1e-6,
                    lambda_topo=LAMBDA_TOPO, early_stop_patience=20,
                    resume=True, n_diff_steps=DIFF_STEPS):

    start_time = time.time()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device     = torch.device(f'cuda:{local_rank}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Fixed symmetric adjacency matrix for topology_loss — built once, never
    # changes.  directed=False makes adj[i,j]=1 whenever edge i→j or j→i
    # exists, giving the GAT a symmetric target (symmetric edges, directed
    # data-flow in the SFC).
    adj = torch.tensor(
        adjacency_matrix(directed=False), dtype=torch.float32, device=device
    )

    model     = TopoSynth(seq_len, n_features, n_vnf, n_diff_steps).to(device)
    model     = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    ema       = EMA(model.module)

    res_path  = os.path.join(checkpoint_dir, 'toposynth_resume.pth')
    best_path = os.path.join(checkpoint_dir, 'toposynth_best.pth')
    best_val, start_epoch, no_imp = float('inf'), 0, 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume and os.path.exists(res_path):
        ck = torch.load(res_path, map_location=device)
        model.module.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_epoch, best_val = ck['epoch'] + 1, ck['best_val']
        if 'ema_state_dict' in ck:
            ema.load_state_dict(ck['ema_state_dict'])
        if _rank0():
            print(f'[TopoSynth] Resumed epoch {start_epoch}  best_val={best_val:.5f}')

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, n_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        tot_diff = tot_topo = 0.0

        for batch in train_loader:
            x0    = batch['x'].to(device)           # [B, W, N, F]
            h_bar = batch['h_bar'][0].to(device)    # [N, F]
            B     = x0.size(0)
            t     = torch.randint(0, n_diff_steps, (B,), device=device)

            optimizer.zero_grad()
            x0_pred, xt, z, C_pred = model(x0, t, h_bar)

            l_diff = diffusion_loss(x0, x0_pred)
            l_topo = topology_loss(C_pred, adj)
            loss   = l_diff + lambda_topo * l_topo

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model.module)

            tot_diff += l_diff.item()
            tot_topo += l_topo.item()

        # ── Validation (rank 0 only) ──────────────────────────────────────────
        if _rank0():
            model.eval()
            val_diff = val_topo = 0.0
            with average_parameters(ema, model.module), torch.no_grad():
                for batch in val_loader:
                    x0    = batch['x'].to(device)
                    h_bar = batch['h_bar'][0].to(device)
                    B     = x0.size(0)
                    t     = torch.randint(0, n_diff_steps, (B,), device=device)
                    x0_pred, _, _, C_pred_val = model.module(x0, t, h_bar)
                    val_diff += diffusion_loss(x0, x0_pred).item()
                    val_topo += topology_loss(C_pred_val, adj).item()
            n_val = max(len(val_loader), 1)
            val_diff /= n_val
            val_topo /= n_val
            # Combined val loss: same weighting as training loss so early-stop
            # criterion reflects topology quality, not only diffusion quality.
            val_loss = val_diff + lambda_topo * val_topo

            n_tr = max(len(train_loader), 1)
            print(f'Epoch {epoch+1:>4} | '
                  f'diff={tot_diff/n_tr:.5f}  '
                  f'topo={tot_topo/n_tr:.5f}  '
                  f'val_diff={val_diff:.5f}  '
                  f'val_topo={val_topo:.5f}  '
                  f'val={val_loss:.5f}  '
                  f'no_imp={no_imp}', flush=True)

            if val_loss < best_val:
                best_val, no_imp = val_loss, 0
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'ema_state_dict':   ema.state_dict(),
                    'best_val':         best_val,
                    'epoch':            epoch,
                }, best_path)
            else:
                no_imp += 1

            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict':       ema.state_dict(),
                'best_val':             best_val,
            }, res_path)

        if _dist_ready():
            dist.barrier()

        scheduler.step()

        _stop = torch.zeros(1, device=device)
        if _rank0() and no_imp >= early_stop_patience:
            _stop[0] = 1.0
        if _dist_ready():
            dist.all_reduce(_stop, op=dist.ReduceOp.MAX)
        if _stop[0] > 0:
            if _rank0():
                print(f'[TopoSynth] Early stop at epoch {epoch+1}')
            break

    return {
        'model':      model,
        'train_time': time.time() - start_time,
        'best_path':  best_path,
    }


# ─── LOAD BEST MODEL ─────────────────────────────────────────────────────────

def load_best_model_for_generation(checkpoint_dir, device,
                                   seq_len, n_features, n_vnf,
                                   n_diff_steps=DIFF_STEPS):
    best_path = os.path.join(checkpoint_dir, 'toposynth_best.pth')
    if not os.path.exists(best_path):
        raise FileNotFoundError(f'Best checkpoint not found: {best_path}')

    ck    = torch.load(best_path, map_location=device)
    model = TopoSynth(seq_len, n_features, n_vnf, n_diff_steps).to(device)
    model.load_state_dict(ck['model_state_dict'])

    if 'ema_state_dict' in ck:
        for name, param in model.named_parameters():
            if param.requires_grad and name in ck['ema_state_dict']:
                param.data = ck['ema_state_dict'][name].clone()
        print(f'[TopoSynth] EMA weights applied (epoch {ck.get("epoch","?")})')
    else:
        print('[TopoSynth] WARNING: no EMA weights. Using raw weights.')

    model.eval()
    return model