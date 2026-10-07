# toposynth/models/toposynth.py
# Full TopoSynth model: GAT encoder + Diffusion-TS backbone + training function.
# No torch_geometric dependency — GAT implemented manually.
#
# FIX (2026-10): topology is now a constructor parameter throughout the stack.
# Previously GATLayer/GATEncoder read the module-level TOPOLOGY_CONFIG singleton
# (always Clearwater) even when training on SFC.  Now every class that needs
# the graph accepts topology_config=<dict> and falls back to the Clearwater
# default only when None is passed.  The training function saves topology_config
# into the checkpoint; load_best_model_for_generation reads it back, so a model
# is always loaded with the topology it was trained on.

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


# ─── TOPOLOGY HELPERS (config-aware) ─────────────────────────────────────────

def _resolve_topo(topology_config):
    """Return topology_config if provided, else fall back to Clearwater default."""
    return topology_config if topology_config is not None else TOPOLOGY_CONFIG


def _upstream_neighbours(topo: dict) -> dict:
    """Build upstream-neighbour dict from a topology_config dict."""
    vnf_names = topo['vnf_names']
    edges     = topo['edges']
    nb: dict  = {v: [] for v in vnf_names}
    for src, dst in edges:
        if dst in nb:
            nb[dst].append(src)
    return nb


def _adj_matrix(topo: dict, directed: bool = True):
    """Build adjacency matrix (list-of-lists) from a topology_config dict."""
    n   = len(topo['vnf_names'])
    idx = topo['vnf_to_idx']
    A   = [[0.0] * n for _ in range(n)]
    for src, dst in topo['edges']:
        i, j = idx[src], idx[dst]
        A[i][j] = 1.0
        if not directed:
            A[j][i] = 1.0
    return A


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
    def __init__(self, n_steps=DIFF_STEPS):
        super().__init__()
        t  = torch.arange(n_steps + 1, dtype=torch.float32)
        f  = torch.cos((t / n_steps + 0.008) / 1.008 * math.pi / 2) ** 2
        ac = f / f[0]
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
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_ac[t].view(-1, 1, 1, 1)
        sb = self.sqrt_one_minus_ac[t].view(-1, 1, 1, 1)
        return sa * x0 + sb * noise, noise

    def p_mean_variance(self, x0_pred, xt, t):
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
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ─── TIME EMBEDDING ───────────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d   = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, t):
        half = self.d // 2
        freq = torch.exp(-math.log(10000) *
                         torch.arange(half, device=t.device).float() / (half - 1))
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


# ─── TREND BLOCK ─────────────────────────────────────────────────────────────

class TrendBlock(nn.Module):
    def __init__(self, seq_len, d_model, n_features, poly_order=3):
        super().__init__()
        self.n_features = n_features
        self.poly_order = poly_order
        t     = torch.linspace(0, 1, seq_len).unsqueeze(1)
        basis = torch.cat([t ** k for k in range(poly_order + 1)], dim=1)
        self.register_buffer('basis', basis)
        self.coef_proj = nn.Linear(d_model, (poly_order + 1) * n_features)

    def forward(self, h_last):
        BN   = h_last.size(0)
        coef = self.coef_proj(h_last).view(BN, self.poly_order + 1, self.n_features)
        return torch.einsum('bkf,wk->bwf', coef, self.basis)


# ─── SEASONAL BLOCK ──────────────────────────────────────────────────────────

class SeasonalBlock(nn.Module):
    def __init__(self, top_k=DIFF_TOP_K):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf  = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        _, idx = torch.topk(amp.mean(0), self.top_k, dim=0)
        mask = torch.zeros_like(xf)
        for fi in range(x.size(-1)):
            mask[:, idx[:, fi], fi] = 1.0
        return torch.fft.irfft(xf * mask, n=x.size(1), dim=1)


# ─── DIFFUSION TRANSFORMER (Denoising Backbone) ───────────────────────────────

class DiffusionTransformer(nn.Module):
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
        h = self.pos_enc(self.input_proj(xt))
        h = h + self.time_proj(self.time_emb(t)).unsqueeze(1)
        h = self.transformer(h)

        trend    = self.trend_block(h[:, -1, :])
        residual = self.residual_proj(h)
        seasonal = self.seasonal_block(residual)

        return trend + seasonal + residual


# ─── GAT ENCODER ─────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Single directed GAT layer.

    topology_config : dict with keys vnf_names, vnf_to_idx, edges.
                      If None, falls back to the Clearwater TOPOLOGY_CONFIG.
    """
    def __init__(self, in_dim, out_dim, n_heads=GAT_K,
                 slope=GAT_SLOPE, concat=True, topology_config=None):
        super().__init__()
        topo            = _resolve_topo(topology_config)
        self.n_heads    = n_heads
        self.out_dim    = out_dim
        self.concat     = concat
        self.neighbours = _upstream_neighbours(topo)
        self.vnf_names  = topo['vnf_names']
        self.vnf_to_idx = topo['vnf_to_idx']

        self.W = nn.Parameter(torch.empty(n_heads, in_dim, out_dim))
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.W.view(n_heads * in_dim, out_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(-1))

        self.leaky_relu = nn.LeakyReLU(slope)

    def forward(self, h):
        """h : [N, in_dim] → h' : [N, K*out_dim] if concat else [N, out_dim]"""
        N     = len(self.vnf_names)
        out_d = self.n_heads * self.out_dim if self.concat else self.out_dim
        out   = torch.zeros(N, out_d, device=h.device)
        Wh    = torch.einsum('kio,ni->kno', self.W, h)

        for i, vnf_i in enumerate(self.vnf_names):
            nb_names = [vnf_i] + list(self.neighbours[vnf_i])
            nb_idx   = [self.vnf_to_idx[nb] for nb in nb_names]
            head_outs = []
            for k in range(self.n_heads):
                Whi    = Wh[k, i]
                Whjs   = Wh[k, nb_idx]
                cat_ij = torch.cat(
                    [Whi.unsqueeze(0).expand(len(nb_idx), -1), Whjs], dim=-1)
                e      = self.leaky_relu((cat_ij * self.a[k]).sum(-1))
                alpha  = Fn.softmax(e, dim=0)
                h_agg  = (alpha.unsqueeze(-1) * Whjs).sum(0)
                head_outs.append(h_agg)
            if self.concat:
                out[i] = Fn.elu(torch.cat(head_outs, dim=-1))
            else:
                out[i] = Fn.elu(torch.stack(head_outs).mean(0))
        return out

    def forward_batched(self, h):
        """h : [B, N, in_dim] → h' : [B, N, K*out_dim] if concat else [B, N, out_dim]"""
        B, N, _ = h.shape
        out_d   = self.n_heads * self.out_dim if self.concat else self.out_dim
        out     = torch.zeros(B, N, out_d, device=h.device)
        Wh      = torch.einsum('kio,bni->bkno', self.W, h)

        for i, vnf_i in enumerate(self.vnf_names):
            nb_names = [vnf_i] + list(self.neighbours[vnf_i])
            nb_idx   = [self.vnf_to_idx[nb] for nb in nb_names]
            M        = len(nb_idx)
            Wh_i     = Wh[:, :, i, :]
            Wh_j     = Wh[:, :, nb_idx, :]
            Wh_i_exp = Wh_i.unsqueeze(2).expand(-1, -1, M, -1)
            cat_ij   = torch.cat([Wh_i_exp, Wh_j], dim=-1)
            e        = self.leaky_relu((cat_ij * self.a.view(1, self.n_heads, 1, -1)).sum(-1))
            alpha    = Fn.softmax(e, dim=2)
            h_agg    = (alpha.unsqueeze(-1) * Wh_j).sum(2)
            if self.concat:
                out[:, i, :] = Fn.elu(h_agg.reshape(B, -1))
            else:
                out[:, i, :] = Fn.elu(h_agg.mean(1))
        return out


class GATEncoder(nn.Module):
    """
    Two-layer GAT encoder.

    topology_config : dict with keys vnf_names, vnf_to_idx, edges.
                      If None, falls back to the Clearwater TOPOLOGY_CONFIG.
    """
    def __init__(self, in_dim, d_h=GAT_D_H, d_z=GAT_D_Z, n_heads=GAT_K,
                 topology_config=None):
        super().__init__()
        topo            = _resolve_topo(topology_config)
        self.vnf_names  = topo['vnf_names']
        self.neighbours = _upstream_neighbours(topo)

        self.gat1 = GATLayer(in_dim,        d_h, n_heads=n_heads, concat=True,
                             topology_config=topo)
        self.gat2 = GATLayer(n_heads * d_h, d_z, n_heads=n_heads, concat=False,
                             topology_config=topo)
        self.norm1 = nn.LayerNorm(n_heads * d_h)
        self.norm2 = nn.LayerNorm(d_z)

        self.corr_decoder = nn.Bilinear(d_z, d_z, 1)

    def forward(self, h_bar):
        h1 = self.norm1(self.gat1(h_bar))
        h2 = self.norm2(self.gat2(h1))
        return h2

    def forward_batched(self, h_bar):
        h1 = self.norm1(self.gat1.forward_batched(h_bar))
        h2 = self.norm2(self.gat2.forward_batched(h1))
        return h2

    def decode_correlations(self, z):
        N  = z.size(0)
        zi = z.unsqueeze(1).expand(N, N, -1).reshape(N * N, -1)
        zj = z.unsqueeze(0).expand(N, N, -1).reshape(N * N, -1)
        return torch.tanh(self.corr_decoder(zi, zj).view(N, N))


# ─── TOPOSYNTH (top-level model) ─────────────────────────────────────────────

class TopoSynth(nn.Module):
    """
    Full TopoSynth model.

    topology_config : dict with keys vnf_names, vnf_to_idx, edges.
                      If None, uses Clearwater topology (backward-compatible).
    """
    def __init__(self, seq_len, n_features, n_vnf, n_diff_steps=DIFF_STEPS,
                 topology_config=None):
        super().__init__()
        self.seq_len         = seq_len
        self.n_features      = n_features
        self.n_vnf           = n_vnf
        self.n_diff_steps    = n_diff_steps
        self.topology_config = _resolve_topo(topology_config)

        self.denoiser = DiffusionTransformer(seq_len=seq_len, n_features=n_features)
        self.gat      = GATEncoder(in_dim=n_features,
                                   topology_config=self.topology_config)
        self.schedule = NoiseSchedule(n_diff_steps)

    def forward(self, x0, t, h_bar):
        B, W, N, n_feat = x0.shape
        xt, _ = self.schedule.q_sample(x0, t)

        xt_flat  = xt.permute(0, 2, 1, 3).reshape(B * N, W, n_feat)
        t_expand = t.unsqueeze(1).expand(B, N).reshape(B * N)

        x0_pred_flat = self.denoiser(xt_flat, t_expand)
        x0_pred = x0_pred_flat.view(B, N, W, n_feat).permute(0, 2, 1, 3)

        z      = self.gat(h_bar)
        C_pred = self.gat.decode_correlations(z)

        return x0_pred, xt, z, C_pred

    @torch.no_grad()
    def generate(self, n_samples, h_bar, eta=0.1, gamma=0.01, device='cuda',
                 n_infer_steps=None, guidance_every=1):
        W, N, F  = self.seq_len, self.n_vnf, self.n_features
        h_bar    = h_bar.to(device)
        z_target = self.gat(h_bar).detach()

        xt = torch.randn(n_samples, W, N, F, device=device)

        T = self.schedule.T
        if n_infer_steps is not None and n_infer_steps < T:
            stride    = T // n_infer_steps
            timesteps = list(range(0, T, stride))[::-1]
        else:
            timesteps = list(range(T - 1, -1, -1))

        for step_idx, step in enumerate(timesteps):
            t_batch = torch.full((n_samples,), step, dtype=torch.long, device=device)
            xt      = xt.detach()
            apply_guidance = (step_idx % guidance_every == 0)

            with torch.no_grad():
                xt_flat      = xt.permute(0, 2, 1, 3).reshape(n_samples * N, W, F)
                t_exp        = t_batch.unsqueeze(1).expand(n_samples, N).reshape(n_samples * N)
                x0_pred_flat = self.denoiser(xt_flat, t_exp)
                x0_pred      = x0_pred_flat.view(n_samples, N, W, F).permute(0, 2, 1, 3)

            with torch.no_grad():
                if apply_guidance:
                    x0_leaf = x0_pred.detach().requires_grad_(True)
                    with torch.enable_grad():
                        x0_mean  = x0_leaf.mean(dim=1)
                        z_pred   = self.gat.forward_batched(x0_mean)
                        z_tgt_ex = z_target.unsqueeze(0).expand_as(z_pred)
                        guide_loss = Fn.mse_loss(z_pred, z_tgt_ex)
                        grad = torch.autograd.grad(guide_loss, x0_leaf)[0]
                    x0_guided = (x0_pred - eta * grad.detach()).clamp(0.0, 1.0)
                else:
                    x0_guided = x0_pred

                if step > 0:
                    mean, var = self.schedule.p_mean_variance(
                        x0_guided, xt.detach(), t_batch)
                    xt = mean + var.sqrt() * torch.randn_like(mean)
                else:
                    xt = x0_guided

        return xt.clamp(0.0, 1.0).detach()


# ─── LOSS FUNCTIONS ──────────────────────────────────────────────────────────

def diffusion_loss(x0, x0_pred, lam1=DIFF_LAMBDA1, lam2=DIFF_LAMBDA2):
    l_time = Fn.mse_loss(x0_pred, x0)
    l_freq = Fn.mse_loss(
        torch.fft.rfft(x0_pred, dim=1, norm='ortho').abs(),
        torch.fft.rfft(x0,      dim=1, norm='ortho').abs()
    )
    return lam1 * l_time + lam2 * l_freq


def topology_loss(C_pred, adj):
    target = adj * 2.0 - 1.0
    return Fn.mse_loss(C_pred, target)


# ─── TRAINING FUNCTION ────────────────────────────────────────────────────────

def train_toposynth(train_loader, val_loader, train_sampler, checkpoint_dir,
                    seq_len, n_features, n_vnf,
                    n_epochs=100, lr=1e-4, weight_decay=1e-6,
                    lambda_topo=LAMBDA_TOPO, early_stop_patience=20,
                    resume=True, n_diff_steps=DIFF_STEPS,
                    topology_config=None):
    """
    topology_config : dict with vnf_names / vnf_to_idx / edges for this dataset.
                      Pass None to use the Clearwater default (backward-compat).
    """
    start_time = time.time()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device     = torch.device(f'cuda:{local_rank}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    topo = _resolve_topo(topology_config)

    # Adjacency matrix built from the actual topology (not the global default)
    adj = torch.tensor(
        _adj_matrix(topo, directed=False), dtype=torch.float32, device=device
    )

    model     = TopoSynth(seq_len, n_features, n_vnf, n_diff_steps,
                          topology_config=topo).to(device)
    model     = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    ema       = EMA(model.module)

    res_path  = os.path.join(checkpoint_dir, 'toposynth_resume.pth')
    best_path = os.path.join(checkpoint_dir, 'toposynth_best.pth')
    best_val, start_epoch, no_imp = float('inf'), 0, 0

    if resume and os.path.exists(res_path):
        ck = torch.load(res_path, map_location=device)
        model.module.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_epoch, best_val = ck['epoch'] + 1, ck['best_val']
        if 'ema_state_dict' in ck:
            ema.load_state_dict(ck['ema_state_dict'])
        if _rank0():
            print(f'[TopoSynth] Resumed epoch {start_epoch}  best_val={best_val:.5f}')

    for epoch in range(start_epoch, n_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        tot_diff = tot_topo = 0.0

        for batch in train_loader:
            x0    = batch['x'].to(device)
            h_bar = batch['h_bar'][0].to(device)
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
                    'topology_config':  topo,       # ← saved so eval loads correctly
                }, best_path)
            else:
                no_imp += 1

            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict':       ema.state_dict(),
                'best_val':             best_val,
                'topology_config':      topo,       # ← saved in resume checkpoint too
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
                                   n_diff_steps=DIFF_STEPS,
                                   topology_config=None):
    """
    Load the best checkpoint.  topology_config priority:
      1. Value stored in the checkpoint  (preferred — always matches training)
      2. topology_config argument        (fallback for old checkpoints)
      3. TOPOLOGY_CONFIG global          (Clearwater default — last resort)
    """
    best_path = os.path.join(checkpoint_dir, 'toposynth_best.pth')
    if not os.path.exists(best_path):
        raise FileNotFoundError(f'Best checkpoint not found: {best_path}')

    ck = torch.load(best_path, map_location=device)

    # Always prefer the topology saved inside the checkpoint
    topo = ck.get('topology_config', None) or topology_config or TOPOLOGY_CONFIG
    if 'topology_config' not in ck:
        print('[TopoSynth] WARNING: checkpoint has no topology_config — '
              'using provided/default topology. Old SFC checkpoints are invalid; retrain.')

    model = TopoSynth(seq_len, n_features, n_vnf, n_diff_steps,
                      topology_config=topo).to(device)
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