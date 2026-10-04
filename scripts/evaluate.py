"""
scripts/evaluate.py — Evaluation pipeline for TopoSynth.

Metrics implemented (matching Diffusion-TS ICLR-2024 + TopoSynth extensions):
  1. Discriminative Score  (DS)    ↓  |AUROC − 0.5|,  0 = indistinguishable
  2. Predictive Score/TSTR (PS)    ↓  MAE; train on synthetic, test on real
  3. Context-FID           (CFID)  ↓  FID in ts2vec embedding space (BiGRU fallback)
  4. Correlational Score   (CS)    ↓  MAE of cross-correlation matrices
  5. Topology Consistency  (TCS)   ↓  GAT-embedding MSE / ↑ cosine (TopoSynth-specific)

All metrics repeated `--repeats` times (default 5) to reduce variance;
DS and PS report the mean over repeats.

Usage (single GPU, no DDP):
  python scripts/evaluate.py \\
      [--config  configs/default.yaml] \\
      [--n_syn   5000] \\
      [--repeats 5] \\
      [--device  cuda] \\
      [--out_dir experiments]

References
----------
- Diffusion-TS (ICLR 2024): Context-FID, Correlational Score, DS, PS
- TSGBench (VLDB 2023): DS/PS LSTM architecture, evaluation protocol
- PaD-TS (AAAI 2025): additional discriminative/predictive framing
"""

import os
import sys
import json
import argparse
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

# ── package path (same as train.py) ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toposynth.data.preprocessing import load_processed
from toposynth.models.toposynth import load_best_model_for_generation

# ── optional dependencies — graceful fallbacks ────────────────────────────────
try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[evaluate] WARNING: sklearn not found — Discriminative Score will be skipped.")

try:
    from scipy.linalg import sqrtm as mat_sqrtm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[evaluate] WARNING: scipy not found — Context-FID will be skipped.")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sliding_windows(data: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """
    data   : [T, N, F]
    returns: [B, W, N, F]  where B = floor((T - W) / stride) + 1
    """
    T = data.shape[0]
    starts = range(0, T - window + 1, stride)
    return np.stack([data[i : i + window] for i in starts], axis=0)


def flatten_windows(x: np.ndarray) -> np.ndarray:
    """[B, W, N, F]  →  [B, W, N*F]"""
    B, W, N, F = x.shape
    return x.reshape(B, W, N * F)


def build_adj_matrix(vnfcs: list, edges: list, device: torch.device) -> torch.Tensor:
    """
    Build undirected binary adjacency matrix [N, N] from config edge list.
    edges: list of [src_name, dst_name]
    """
    n   = len(vnfcs)
    idx = {v: i for i, v in enumerate(vnfcs)}
    adj = torch.zeros(n, n, device=device)
    for src, dst in edges:
        s, d = idx[src], idx[dst]
        adj[s, d] = 1.0
        adj[d, s] = 1.0   # undirected
    return adj


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM modules (TSGBench / Diffusion-TS architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMClassifier(nn.Module):
    """2-layer LSTM → binary real/fake classifier."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):          # x: [B, W, D]
        out, _ = self.lstm(x)      # [B, W, H]
        return self.head(out[:, -1])  # logit [B, 1]


class LSTMPredictor(nn.Module):
    """2-layer LSTM → next-step predictor (for TSTR)."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):          # x: [B, W-1, D]
        out, _ = self.lstm(x)
        return self.head(out[:, -1])  # [B, D]


# ═══════════════════════════════════════════════════════════════════════════════
# Contextual encoder for Context-FID (ts2vec or BiGRU fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class BiGRUEncoder(nn.Module):
    """
    Lightweight bidirectional GRU that maps a window to a fixed-size embedding.
    Trained with a next-step reconstruction objective on real windows.
    Used as fallback when ts2vec is not installed.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, embed_dim: int = 128):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x):           # x: [B, W, D]
        _, h = self.gru(x)          # h: [4, B, H]  (2 layers × 2 dirs)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)  # [B, 2H]  last layer fwd+bwd
        return self.proj(h_cat)     # [B, embed_dim]


def _compute_fid(mu1: np.ndarray, sig1: np.ndarray,
                 mu2: np.ndarray, sig2: np.ndarray) -> float:
    """Fréchet distance between N(mu1,sig1) and N(mu2,sig2)."""
    diff     = mu1 - mu2
    cov_mean = mat_sqrtm(sig1 @ sig2)   # disp kwarg removed (deprecated in scipy 1.18)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * cov_mean))


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 1 — Discriminative Score
# ═══════════════════════════════════════════════════════════════════════════════

def discriminative_score(
    real: np.ndarray,        # [B_r, W, D]
    fake: np.ndarray,        # [B_f, W, D]
    device: torch.device,
    n_epochs: int   = 200,
    batch_size: int = 256,
    n_repeats: int  = 5,
) -> float:
    """
    Train a 2-layer LSTM classifier (real=1, fake=0) with an 80/20 split.
    Metric: mean |AUROC − 0.5| over n_repeats.  Lower is better (0 = perfect).

    Protocol matches TSGBench §4 and Diffusion-TS Appendix A.
    """
    if not HAS_SKLEARN:
        log.warning("sklearn unavailable — Discriminative Score skipped")
        return float("nan")

    _, W, D = real.shape
    n       = min(len(real), len(fake))
    scores  = []

    for rep in range(n_repeats):
        rng = np.random.default_rng(rep)
        ri  = rng.choice(len(real), n, replace=False)
        fi  = rng.choice(len(fake), n, replace=False)

        X   = np.concatenate([real[ri], fake[fi]], axis=0).astype(np.float32)
        y   = np.array([1.0] * n + [0.0] * n, dtype=np.float32)
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]

        split = int(len(X) * 0.8)
        Xtr = torch.tensor(X[:split]);  ytr = torch.tensor(y[:split]).unsqueeze(1)
        Xte = torch.tensor(X[split:]);  yte = y[split:]

        loader = DataLoader(TensorDataset(Xtr, ytr),
                            batch_size=batch_size, shuffle=True)

        clf = LSTMClassifier(D).to(device)
        opt = optim.Adam(clf.parameters(), lr=1e-3)
        bce = nn.BCEWithLogitsLoss()

        clf.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                bce(clf(xb.to(device)), yb.to(device)).backward()
                opt.step()

        clf.eval()
        with torch.no_grad():
            logits = clf(Xte.to(device)).squeeze(1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))   # sigmoid
        auc   = roc_auc_score(yte, probs)
        ds    = abs(auc - 0.5)
        scores.append(ds)
        log.info(f"  DS repeat {rep + 1}/{n_repeats}: "
                 f"AUROC={auc:.4f}  |AUROC−0.5|={ds:.4f}")

    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 2 — Predictive Score (TSTR)
# ═══════════════════════════════════════════════════════════════════════════════

def predictive_score(
    real: np.ndarray,        # [B_r, W, D]  — real test windows
    fake: np.ndarray,        # [B_f, W, D]  — synthetic training windows
    device: torch.device,
    n_epochs:   int = 200,
    batch_size: int = 256,
    n_repeats:  int = 5,
    train_cap:  int = 2000,
    test_cap:   int = 500,
) -> float:
    """
    Train-on-Synthetic, Test-on-Real (TSTR).
    Input: steps 0..W-2, target: step W-1.
    Metric: mean MAE on real test windows.  Lower is better.
    """
    _, W, D = fake.shape
    scores  = []

    for rep in range(n_repeats):
        rng = np.random.default_rng(42 + rep)
        fi  = rng.choice(len(fake), min(len(fake), train_cap), replace=False)
        Xf  = torch.tensor(fake[fi], dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(Xf[:, :-1, :], Xf[:, -1, :]),
            batch_size=batch_size, shuffle=True,
        )

        pred_model = LSTMPredictor(D).to(device)
        opt        = optim.Adam(pred_model.parameters(), lr=1e-3)
        l1         = nn.L1Loss()

        pred_model.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                l1(pred_model(xb.to(device)), yb.to(device)).backward()
                opt.step()

        # Evaluate on real test windows
        ri  = rng.choice(len(real), min(len(real), test_cap), replace=False)
        Xr  = torch.tensor(real[ri], dtype=torch.float32)

        pred_model.eval()
        with torch.no_grad():
            pred = pred_model(Xr[:, :-1, :].to(device)).cpu()
        mae = l1(pred, Xr[:, -1, :]).item()
        scores.append(mae)
        log.info(f"  PS repeat {rep + 1}/{n_repeats}: MAE={mae:.6f}")

    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 3 — Context-FID
# ═══════════════════════════════════════════════════════════════════════════════

def context_fid(
    real: np.ndarray,        # [B_r, W, D]
    fake: np.ndarray,        # [B_f, W, D]
    device: torch.device,
    embed_dim:  int = 128,
    enc_epochs: int = 50,
    batch_size: int = 256,
) -> float:
    """
    Compute FID in a contextual embedding space.

    Primary:  ts2vec (as used in Diffusion-TS paper) — installed via
              `pip install ts2vec`.
    Fallback: BiGRU encoder trained on real data with next-step prediction.

    Lower is better.
    """
    if not HAS_SCIPY:
        log.warning("scipy unavailable — Context-FID skipped")
        return float("nan")

    _, W, D = real.shape
    ts2vec_ok = False

    # ── ts2vec path ───────────────────────────────────────────────────────────
    try:
        from ts2vec import TS2Vec
        log.info("  Context-FID: ts2vec encoder")
        gpu_id = device.index if device.type == "cuda" else -1
        enc = TS2Vec(
            input_dims=D, device=gpu_id,
            output_dims=embed_dim, hidden_dims=64,
            depth=10, lr=0.001, batch_size=batch_size,
            max_train_length=W,
        )
        enc.fit(real, n_epochs=enc_epochs, verbose=False)
        emb_real = enc.encode(real, encoding_window="full_series")   # [B, embed_dim]
        emb_fake = enc.encode(fake, encoding_window="full_series")
        ts2vec_ok = True
    except Exception as e:
        log.info(f"  ts2vec unavailable ({e!r}) — using BiGRU fallback")

    # ── BiGRU fallback ────────────────────────────────────────────────────────
    if not ts2vec_ok:
        encoder = BiGRUEncoder(D, hidden_dim=64, embed_dim=embed_dim).to(device)
        head    = nn.Linear(embed_dim, D).to(device)     # next-step decoder
        opt     = optim.Adam(
            list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
        mse     = nn.MSELoss()

        Xtr    = torch.tensor(real, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xtr), batch_size=batch_size, shuffle=True)

        encoder.train(); head.train()
        for ep in range(enc_epochs):
            for (xb,) in loader:
                xb = xb.to(device)
                z  = encoder(xb[:, :-1, :])       # encode all-but-last step
                loss = mse(head(z), xb[:, -1, :]) # predict last step
                opt.zero_grad(); loss.backward(); opt.step()
        log.info(f"  BiGRU encoder trained ({enc_epochs} epochs)")

        def encode_batched(X_np: np.ndarray) -> np.ndarray:
            encoder.eval()
            parts = []
            with torch.no_grad():
                for i in range(0, len(X_np), batch_size):
                    xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
                    parts.append(encoder(xb).cpu().numpy())
            return np.concatenate(parts, axis=0)

        emb_real = encode_batched(real)
        emb_fake = encode_batched(fake)

    # ── FID ───────────────────────────────────────────────────────────────────
    eps  = np.eye(embed_dim) * 1e-6
    mu_r = emb_real.mean(0);  sig_r = np.cov(emb_real, rowvar=False) + eps
    mu_f = emb_fake.mean(0);  sig_f = np.cov(emb_fake, rowvar=False) + eps
    return _compute_fid(mu_r, sig_r, mu_f, sig_f)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 4 — Correlational Score
# ═══════════════════════════════════════════════════════════════════════════════

def correlational_score(
    real: np.ndarray,   # [B_r, W, D]
    fake: np.ndarray,   # [B_f, W, D]
) -> float:
    """
    Mean absolute error between the cross-correlation matrices of real
    and synthetic windows (Diffusion-TS §3.3).
    Lower is better (0 = identical correlation structure).
    """
    def corr_matrix(X: np.ndarray) -> np.ndarray:
        B, W, D = X.shape
        flat = X.reshape(B * W, D)           # [B*W, D]
        cc   = np.corrcoef(flat.T)           # [D, D]
        # Zero-variance features (constant column) produce NaN in corrcoef.
        # Treat them as uncorrelated (0) rather than propagating NaN.
        return np.nan_to_num(cc, nan=0.0)

    cc_real = corr_matrix(real)
    cc_fake = corr_matrix(fake)
    return float(np.mean(np.abs(cc_real - cc_fake)))


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 5 — Topology Consistency Score  (TopoSynth-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def topology_consistency_score(
    real_4d: np.ndarray,   # [B_r, W, N, F]
    fake_4d: np.ndarray,   # [B_f, W, N, F]
    model,                 # loaded TopoSynth model (has .gat attribute)
    vnf_names: list,       # e.g. ["bono","sprout","homestead","homer","ralf","ellis"]
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """
    Topology Consistency Score — unique to TopoSynth.

    For each window:
      1. Compute per-VNF temporal mean → [N, F]
      2. Pass through the trained GAT → [N, d_z]
    Compare distribution of GAT embeddings between real and synthetic:
      - embedding_mse    : MSE between distribution means (↓ better)
      - embedding_cosine : mean cosine similarity per VNF (↑ better)

    Rationale: the GAT encodes the Clearwater 5G topology structure.
    A good synthetic dataset should elicit the same topology embedding
    as the real data, confirming that the learned topological conditioning
    has been preserved during generation.
    """
    gat = model.gat
    gat.eval()

    def embed_batch(X_np: np.ndarray) -> np.ndarray:
        """X_np: [B, W, N, F]  →  embeddings [B, N, d_z]"""
        all_embs = []
        for i in range(0, len(X_np), batch_size):
            batch = torch.tensor(
                X_np[i : i + batch_size], dtype=torch.float32
            ).to(device)                  # [b, W, N, F]
            h_mean = batch.mean(dim=1)    # temporal mean → [b, N, F]
            with torch.no_grad():
                embs = []
                for j in range(h_mean.shape[0]):
                    # gat([N, F]) → [N, d_z]
                    e = gat(h_mean[j])
                    embs.append(e.cpu().numpy())
            all_embs.append(np.stack(embs, axis=0))  # [b, N, d_z]
        return np.concatenate(all_embs, axis=0)       # [B, N, d_z]

    log.info("  TCS: encoding real windows via GAT …")
    emb_real = embed_batch(real_4d)   # [B_r, N, d_z]
    log.info("  TCS: encoding synthetic windows via GAT …")
    emb_fake = embed_batch(fake_4d)   # [B_f, N, d_z]

    # Mean embeddings per VNF
    mu_real = emb_real.mean(axis=0)   # [N, d_z]
    mu_fake = emb_fake.mean(axis=0)   # [N, d_z]

    # Scalar MSE between mean embeddings
    mse = float(np.mean((mu_real - mu_fake) ** 2))

    # Per-VNF cosine similarity
    per_vnf_cos = {}
    cos_vals     = []
    for n, vnf in enumerate(vnf_names):
        a, b = mu_real[n], mu_fake[n]
        cos  = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        per_vnf_cos[vnf] = cos
        cos_vals.append(cos)

    return {
        "embedding_mse":    mse,
        "embedding_cosine": float(np.mean(cos_vals)),   # mean over VNFs
        "per_vnf_cosine":   per_vnf_cos,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "configs", "default.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TopoSynth")
    parser.add_argument("--config",     default=DEFAULT_CONFIG,
                        help="YAML config path (default: configs/default.yaml)")
    parser.add_argument("--n_syn",      type=int, default=None,
                        help="Synthetic windows to generate "
                             "(default: evaluation.n_synthetic from config)")
    parser.add_argument("--repeats",    type=int, default=5,
                        help="Repetitions for DS and PS (default: 5)")
    parser.add_argument("--device",     default="cuda",
                        help="cuda | cpu (default: cuda)")
    parser.add_argument("--out_dir",    default=None,
                        help="Results directory (default: training.experiment_dir)")
    parser.add_argument("--stride",     type=int, default=5,
                        help="Stride for extracting real windows (default: 5)")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    log.info(f"Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg        = load_config(args.config)
    data_cfg   = cfg.get("data",       {})
    model_cfg  = cfg.get("model",      {})
    train_cfg  = cfg.get("training",   {})
    eval_cfg   = cfg.get("evaluation", {})
    topo_cfg   = cfg.get("topology",   {})

    proc_data_dir  = data_cfg.get("processed_dir",  "data/processed")
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    out_dir        = args.out_dir or train_cfg.get("experiment_dir", "experiments")
    seq_len        = data_cfg.get("seq_len",       48)
    n_features     = data_cfg.get("n_features",    12)
    n_vnf          = data_cfg.get("n_vnf",          6)
    n_diff_steps   = model_cfg.get("n_diff_steps", 200)
    n_syn          = args.n_syn or eval_cfg.get("n_synthetic", 5000)
    guidance_eta   = train_cfg.get("guidance_eta",   0.1)
    guidance_gamma = train_cfg.get("guidance_gamma", 0.01)

    vnf_names = topo_cfg.get("vnfcs", [
        "bono", "sprout", "homestead", "homer", "ralf", "ellis",
    ])
    edges = topo_cfg.get("edges", [
        ["bono", "sprout"], ["sprout", "homestead"],
        ["sprout", "homer"], ["sprout", "ralf"],
    ])

    # ── Load processed data ───────────────────────────────────────────────────
    log.info(f"Loading processed data from {proc_data_dir} …")
    data, tsi, vsi, _scalers = load_processed(proc_data_dir)
    # data: [T, N, F]  train=0:tsi  val=tsi:vsi  test=vsi:T
    log.info(f"Data shape: {data.shape} | "
             f"train 0:{tsi} | val {tsi}:{vsi} | test {vsi}:{data.shape[0]}")

    train_data = data[:tsi]   # [T_tr, N, F]
    test_data  = data[vsi:]   # [T_te, N, F]

    # ── Extract sliding windows ───────────────────────────────────────────────
    log.info(f"Extracting windows (W={seq_len}, test stride={args.stride}) …")
    test_4d  = sliding_windows(test_data,  seq_len, stride=args.stride)
    log.info(f"Test windows : {test_4d.shape}")

    test_flat = flatten_windows(test_4d)   # [B_te, W, N*F]

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"Loading best model from {checkpoint_dir} …")
    model = load_best_model_for_generation(
        checkpoint_dir=checkpoint_dir,
        device=device,
        seq_len=seq_len,
        n_features=n_features,
        n_vnf=n_vnf,
        n_diff_steps=n_diff_steps,
    )
    model.eval()

    # ── Generate synthetic windows ────────────────────────────────────────────
    h_bar = torch.FloatTensor(train_data.mean(axis=0)).to(device)   # [N, F]
    log.info(f"Generating {n_syn} synthetic windows "
             f"(eta={guidance_eta}, gamma={guidance_gamma}) …")
    t0     = time.time()
    chunk  = 256   # per-chunk to avoid OOM
    chunks = []
    for start in range(0, n_syn, chunk):
        n = min(chunk, n_syn - start)
        # NOTE: generate() uses torch.enable_grad() internally for classifier
        # guidance — do NOT wrap this call in torch.no_grad()
        x = model.generate(
            n_samples=n, h_bar=h_bar,
            device=device, eta=guidance_eta, gamma=guidance_gamma,
        )
        chunks.append(x.cpu())
        if (start // chunk + 1) % 5 == 0:
            log.info(f"  generated {start + n}/{n_syn} windows …")

    fake_4d  = torch.cat(chunks, dim=0).numpy()   # [n_syn, W, N, F]
    fake_flat = flatten_windows(fake_4d)           # [n_syn, W, N*F]
    log.info(f"Generation done in {time.time() - t0:.1f}s  "
             f"shape={fake_4d.shape}  range=[{fake_4d.min():.3f}, {fake_4d.max():.3f}]")

    # ── RNG for sub-sampling ──────────────────────────────────────────────────
    rng = np.random.default_rng(0)

    # ── Metric 1: Discriminative Score ────────────────────────────────────────
    log.info("=" * 60)
    log.info("Metric 1/5: Discriminative Score (DS)")
    ds = discriminative_score(
        real=test_flat, fake=fake_flat,
        device=device,
        n_epochs=200, batch_size=256, n_repeats=args.repeats,
    )
    log.info(f"  DS = {ds:.6f}  (↓ better; 0 = indistinguishable)")

    # ── Metric 2: Predictive Score (TSTR) ─────────────────────────────────────
    log.info("=" * 60)
    log.info("Metric 2/5: Predictive Score / TSTR (PS)")
    ps = predictive_score(
        real=test_flat, fake=fake_flat,
        device=device,
        n_epochs=200, batch_size=256, n_repeats=args.repeats,
    )
    log.info(f"  PS = {ps:.6f}  (↓ better)")

    # ── Metric 3: Context-FID ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Metric 3/5: Context-FID (CFID)")
    n_fid = min(2000, len(test_flat), len(fake_flat))
    cfid  = context_fid(
        real=test_flat[rng.choice(len(test_flat), n_fid, replace=False)],
        fake=fake_flat[rng.choice(len(fake_flat), n_fid, replace=False)],
        device=device,
        embed_dim=128, enc_epochs=50, batch_size=256,
    )
    log.info(f"  Context-FID = {cfid:.4f}  (↓ better)")

    # ── Metric 4: Correlational Score ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("Metric 4/5: Correlational Score (CS)")
    n_cs = min(2000, len(test_flat), len(fake_flat))
    cs   = correlational_score(
        real=test_flat[rng.choice(len(test_flat), n_cs, replace=False)],
        fake=fake_flat[rng.choice(len(fake_flat), n_cs, replace=False)],
    )
    log.info(f"  CS = {cs:.6f}  (↓ better)")

    # ── Metric 5: Topology Consistency Score ──────────────────────────────────
    log.info("=" * 60)
    log.info("Metric 5/5: Topology Consistency Score (TCS)")
    n_tcs = min(500, len(test_4d), len(fake_4d))
    tcs   = topology_consistency_score(
        real_4d=test_4d [rng.choice(len(test_4d),  n_tcs, replace=False)],
        fake_4d=fake_4d [rng.choice(len(fake_4d),  n_tcs, replace=False)],
        model=model,
        vnf_names=vnf_names,
        device=device,
    )
    log.info(f"  TCS embedding_mse    = {tcs['embedding_mse']:.6f}  (↓ better)")
    log.info(f"  TCS embedding_cosine = {tcs['embedding_cosine']:.6f}  (↑ better)")
    for vnf, cos in tcs["per_vnf_cosine"].items():
        log.info(f"      {vnf:12s}: cosine = {cos:.4f}")

    # ── Collect & save results ────────────────────────────────────────────────
    results = {
        "discriminative_score":  ds,
        "predictive_score":      ps,
        "context_fid":           cfid,
        "correlational_score":   cs,
        "topology_consistency":  tcs,
        "meta": {
            "n_syn":          n_syn,
            "n_test_windows": len(test_4d),
            "repeats":        args.repeats,
            "device":         str(device),
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    ts_str   = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"eval_{ts_str}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Discriminative Score  (↓)  {ds:.6f}")
    log.info(f"  Predictive Score/TSTR (↓)  {ps:.6f}")
    log.info(f"  Context-FID           (↓)  {cfid:.4f}")
    log.info(f"  Correlational Score   (↓)  {cs:.6f}")
    log.info(f"  Topo Consistency MSE  (↓)  {tcs['embedding_mse']:.6f}")
    log.info(f"  Topo Consistency Cos  (↑)  {tcs['embedding_cosine']:.6f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()