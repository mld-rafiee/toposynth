"""
scripts/evaluate_sfc.py — Evaluation pipeline for TopoSynth on the SFC dataset.

This file is the SFC-specific counterpart of evaluate.py (Clearwater).
Key differences from the Clearwater evaluator:

  1. Directed adjacency matrix — no reverse edges added.  The SFC topology
     is a directed DAG:
       SFC1: firewall → dpi → enc → comp
       SFC2: firewall2 → dpi → nat
     Upstream VNFs influence downstream ones; the reverse is not true.

  2. Topology-aware GCS — only pairs with a clear topology prediction are
     tested.  Pairs are classified as:
       direct_edge  : (i→j) is a topology link             → should be G-causal
       no_path      : no directed path from i to j          → should be independent
       transitive   : j reachable via an intermediate VNF  → SKIPPED (ambiguous)
     GCS is reported separately for edge pairs and no-path pairs.

  3. Binary feature handling — the crash feature (and any other binary
     feature detected from training data) is thresholded to {0, 1} after
     generation, and excluded from the GCS scalar aggregation.

Default config: configs/sfc.yaml
Default out_dir: experiments_sfc/

Metrics (same six as evaluate.py):
  1. Discriminative Score  (DS)    ↓
  2. Predictive Score/TSTR (PS)    ↓
  3. Context-FID           (CFID)  ↓
  4. Correlational Score   (CS)    ↓
  5. Topology Consistency  (TCS)   ↓ MSE / ↑ cosine
  6. Granger Causal Score  (GCS)   ↑  (edge + no-path breakdown)

Usage (single GPU, no DDP):
  python scripts/evaluate_sfc.py \\
      [--config  configs/sfc.yaml] \\
      [--n_syn   500] \\
      [--repeats 5] \\
      [--device  cuda] \\
      [--out_dir experiments_sfc]
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

# ── package path ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toposynth.data.preprocessing import load_processed
from toposynth.models.toposynth import load_best_model_for_generation

# ── optional dependencies ─────────────────────────────────────────────────────
try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[evaluate_sfc] WARNING: sklearn not found — DS will be skipped.")

try:
    from scipy.linalg import sqrtm as mat_sqrtm
    from scipy.stats import f as f_dist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[evaluate_sfc] WARNING: scipy not found — Context-FID and GCS will be skipped.")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "configs", "sfc.yaml"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sliding_windows(data: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """[T, N, F] → [B, W, N, F]"""
    T = data.shape[0]
    starts = range(0, T - window + 1, stride)
    return np.stack([data[i : i + window] for i in starts], axis=0)


def flatten_windows(x: np.ndarray) -> np.ndarray:
    """[B, W, N, F] → [B, W, N*F]"""
    B, W, N, F = x.shape
    return x.reshape(B, W, N * F)


def build_adj_matrix(vnfcs: list, edges: list, device: torch.device) -> torch.Tensor:
    """
    Build a DIRECTED binary adjacency matrix [N, N].
    edges: list of [src_name, dst_name]
    No reverse edges — the SFC topology is a directed DAG.
    """
    n   = len(vnfcs)
    idx = {v: i for i, v in enumerate(vnfcs)}
    adj = torch.zeros(n, n, device=device)
    for src, dst in edges:
        s, d = idx[src], idx[dst]
        adj[s, d] = 1.0
        # No adj[d, s] — directed only
    return adj


def _detect_binary_features(train_data: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """
    Identify features that are binary (values ≈ 0 or ≈ 1) in training data.

    train_data : [T, N, F]
    Returns boolean mask [F] where True marks a binary feature.
    """
    _, _, F = train_data.shape
    binary_mask = np.zeros(F, dtype=bool)
    for f in range(F):
        col = train_data[:, :, f].ravel()
        near_binary = (col < tol) | (col > (1.0 - tol))
        if near_binary.mean() >= (1.0 - tol):
            binary_mask[f] = True
    return binary_mask


def threshold_binary_features(
    fake_4d: np.ndarray,
    binary_mask: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Round diffusion output to {0, 1} for binary features (e.g. crash).
    fake_4d     : [B, W, N, F]
    binary_mask : [F] boolean
    Returns a copy with binary columns thresholded.
    """
    out = fake_4d.copy()
    for f in np.where(binary_mask)[0]:
        out[:, :, :, f] = (out[:, :, :, f] >= threshold).astype(np.float32)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM modules (TSGBench / Diffusion-TS architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


# ═══════════════════════════════════════════════════════════════════════════════
# BiGRU encoder (Context-FID fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class BiGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, embed_dim: int = 128):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x):
        _, h = self.gru(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(h_cat)


def _compute_fid(mu1, sig1, mu2, sig2) -> float:
    diff     = mu1 - mu2
    cov_mean = mat_sqrtm(sig1 @ sig2)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * cov_mean))


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 1 — Discriminative Score
# ═══════════════════════════════════════════════════════════════════════════════

def discriminative_score(real, fake, device, n_epochs=200, batch_size=256, n_repeats=5):
    if not HAS_SKLEARN:
        log.warning("sklearn unavailable — DS skipped"); return float("nan")
    _, W, D = real.shape
    n = min(len(real), len(fake))
    scores = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(rep)
        ri  = rng.choice(len(real), n, replace=False)
        fi  = rng.choice(len(fake), n, replace=False)
        X   = np.concatenate([real[ri], fake[fi]], axis=0).astype(np.float32)
        y   = np.array([1.0]*n + [0.0]*n, dtype=np.float32)
        perm = rng.permutation(len(X));  X, y = X[perm], y[perm]
        split = int(len(X) * 0.8)
        Xtr = torch.tensor(X[:split]);  ytr = torch.tensor(y[:split]).unsqueeze(1)
        Xte = torch.tensor(X[split:]);  yte = y[split:]
        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
        clf = LSTMClassifier(D).to(device)
        opt = optim.Adam(clf.parameters(), lr=1e-3)
        bce = nn.BCEWithLogitsLoss()
        clf.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                opt.zero_grad(); bce(clf(xb.to(device)), yb.to(device)).backward(); opt.step()
        clf.eval()
        with torch.no_grad():
            logits = clf(Xte.to(device)).squeeze(1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        auc = roc_auc_score(yte, probs);  ds = abs(auc - 0.5)
        scores.append(ds)
        log.info(f"  DS repeat {rep+1}/{n_repeats}: AUROC={auc:.4f}  |AUROC−0.5|={ds:.4f}")
    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 2 — Predictive Score (TSTR)
# ═══════════════════════════════════════════════════════════════════════════════

def predictive_score(real, fake, device, n_epochs=200, batch_size=256,
                     n_repeats=5, train_cap=2000, test_cap=500):
    _, W, D = fake.shape
    scores = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(42 + rep)
        fi  = rng.choice(len(fake), min(len(fake), train_cap), replace=False)
        Xf  = torch.tensor(fake[fi], dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xf[:, :-1, :], Xf[:, -1, :]),
                            batch_size=batch_size, shuffle=True)
        pred_model = LSTMPredictor(D).to(device)
        opt = optim.Adam(pred_model.parameters(), lr=1e-3)
        l1  = nn.L1Loss()
        pred_model.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                opt.zero_grad(); l1(pred_model(xb.to(device)), yb.to(device)).backward(); opt.step()
        ri = rng.choice(len(real), min(len(real), test_cap), replace=False)
        Xr = torch.tensor(real[ri], dtype=torch.float32)
        pred_model.eval()
        with torch.no_grad():
            pred = pred_model(Xr[:, :-1, :].to(device)).cpu()
        mae = l1(pred, Xr[:, -1, :]).item()
        scores.append(mae)
        log.info(f"  PS repeat {rep+1}/{n_repeats}: MAE={mae:.6f}")
    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 3 — Context-FID
# ═══════════════════════════════════════════════════════════════════════════════

def context_fid(real, fake, device, embed_dim=128, enc_epochs=50, batch_size=256):
    if not HAS_SCIPY:
        log.warning("scipy unavailable — Context-FID skipped"); return float("nan")
    _, W, D = real.shape
    ts2vec_ok = False
    try:
        from ts2vec import TS2Vec
        log.info("  Context-FID: ts2vec encoder")
        gpu_id = device.index if device.type == "cuda" else -1
        enc = TS2Vec(input_dims=D, device=gpu_id, output_dims=embed_dim,
                     hidden_dims=64, depth=10, lr=0.001, batch_size=batch_size,
                     max_train_length=W)
        enc.fit(real, n_epochs=enc_epochs, verbose=False)
        emb_real = enc.encode(real, encoding_window="full_series")
        emb_fake = enc.encode(fake, encoding_window="full_series")
        ts2vec_ok = True
    except Exception as e:
        log.info(f"  ts2vec unavailable ({e!r}) — using BiGRU fallback")

    if not ts2vec_ok:
        encoder = BiGRUEncoder(D, hidden_dim=64, embed_dim=embed_dim).to(device)
        head    = nn.Linear(embed_dim, D).to(device)
        opt     = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
        mse     = nn.MSELoss()
        Xtr     = torch.tensor(real, dtype=torch.float32)
        loader  = DataLoader(TensorDataset(Xtr), batch_size=batch_size, shuffle=True)
        encoder.train(); head.train()
        for _ in range(enc_epochs):
            for (xb,) in loader:
                xb = xb.to(device)
                z = encoder(xb[:, :-1, :])
                loss = mse(head(z), xb[:, -1, :])
                opt.zero_grad(); loss.backward(); opt.step()
        log.info(f"  BiGRU encoder trained ({enc_epochs} epochs)")

        def encode_batched(X_np):
            encoder.eval(); parts = []
            with torch.no_grad():
                for i in range(0, len(X_np), batch_size):
                    xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
                    parts.append(encoder(xb).cpu().numpy())
            return np.concatenate(parts, axis=0)

        emb_real = encode_batched(real)
        emb_fake = encode_batched(fake)

    eps  = np.eye(embed_dim) * 1e-6
    mu_r = emb_real.mean(0);  sig_r = np.cov(emb_real, rowvar=False) + eps
    mu_f = emb_fake.mean(0);  sig_f = np.cov(emb_fake, rowvar=False) + eps
    return _compute_fid(mu_r, sig_r, mu_f, sig_f)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 4 — Correlational Score
# ═══════════════════════════════════════════════════════════════════════════════

def correlational_score(real, fake):
    def corr_matrix(X):
        B, W, D = X.shape
        flat = X.reshape(B * W, D)
        return np.nan_to_num(np.corrcoef(flat.T), nan=0.0)
    return float(np.mean(np.abs(corr_matrix(real) - corr_matrix(fake))))


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 5 — Topology Consistency Score
# ═══════════════════════════════════════════════════════════════════════════════

def topology_consistency_score(real_4d, fake_4d, model, vnf_names, device, batch_size=256):
    """
    GAT embedding MSE and cosine between real and synthetic windows.
    """
    gat = model.gat;  gat.eval()

    def embed_batch(X_np):
        all_embs = []
        for i in range(0, len(X_np), batch_size):
            batch  = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
            h_mean = batch.mean(dim=1)   # [b, N, F]
            with torch.no_grad():
                embs = [gat(h_mean[j]).cpu().numpy() for j in range(h_mean.shape[0])]
            all_embs.append(np.stack(embs, axis=0))
        return np.concatenate(all_embs, axis=0)   # [B, N, d_z]

    log.info("  TCS: encoding real windows via GAT …")
    emb_real = embed_batch(real_4d)
    log.info("  TCS: encoding synthetic windows via GAT …")
    emb_fake = embed_batch(fake_4d)

    mu_real = emb_real.mean(axis=0)   # [N, d_z]
    mu_fake = emb_fake.mean(axis=0)

    mse = float(np.mean((mu_real - mu_fake) ** 2))

    per_vnf_cos = {}
    cos_vals    = []
    for n, vnf in enumerate(vnf_names):
        a, b = mu_real[n], mu_fake[n]
        cos  = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        per_vnf_cos[vnf] = cos
        cos_vals.append(cos)

    return {
        "embedding_mse":    mse,
        "embedding_cosine": float(np.mean(cos_vals)),
        "per_vnf_cosine":   per_vnf_cos,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 6 — Granger Causal Score (topology-aware)
# ═══════════════════════════════════════════════════════════════════════════════

def _granger_test_ols(y, x, max_lag, alpha):
    """OLS F-test: returns True if x Granger-causes y at level alpha."""
    T = len(y);  p = max_lag;  n_obs = T - p
    if n_obs <= 2 * p + 2:
        return False
    Y = y[p:].copy()
    Xr = np.empty((n_obs, p + 1));  Xr[:, 0] = 1.0
    for k in range(1, p + 1):
        Xr[:, k] = y[p - k: T - k]
    Xu = np.empty((n_obs, 2 * p + 1));  Xu[:, :p + 1] = Xr
    for k in range(1, p + 1):
        Xu[:, p + k] = x[p - k: T - k]

    def _rss(X, Y):
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        res  = Y - X @ beta
        return float(res @ res)

    try:
        rss_r = _rss(Xr, Y);  rss_u = _rss(Xu, Y)
    except np.linalg.LinAlgError:
        return False
    if rss_u < 1e-10:
        return False
    q   = p
    df2 = n_obs - (2 * p + 1)
    F   = ((rss_r - rss_u) / q) / (rss_u / df2)
    if F <= 0:
        return False
    return float(f_dist.sf(F, q, df2)) < alpha


def _compute_reachable(vnf_names: list, edges: list) -> dict:
    """
    BFS reachability on the directed graph.
    Returns {src_idx: set_of_reachable_dst_idx} (excluding src itself).
    """
    N   = len(vnf_names)
    idx = {v: i for i, v in enumerate(vnf_names)}
    adj: dict[int, set] = {i: set() for i in range(N)}
    for src, dst in edges:
        adj[idx[src]].add(idx[dst])

    reachable: dict[int, set] = {}
    for start in range(N):
        visited: set = set()
        queue = list(adj[start])
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(adj[node] - visited)
        reachable[start] = visited
    return reachable


def granger_causal_score(
    real_4d:     np.ndarray,         # [B_r, W, N, F]
    fake_4d:     np.ndarray,         # [B_f, W, N, F]
    vnf_names:   list,               # ordered VNF names
    edges:       list,               # directed topology edges [[src, dst], ...]
    binary_mask: np.ndarray = None,  # [F] bool — features to exclude from aggregation
    max_lag:     int   = 3,
    alpha:       float = 0.05,
) -> dict:
    """
    Topology-aware Granger Causal Score for the SFC directed DAG.

    Pair classification:
      direct_edge  pairs — topology link; should be G-causal in real data.
      no_path      pairs — no directed path; should be independent.
      transitive   pairs — reachable via intermediary; skipped (ambiguous).

    Binary features (e.g. crash) are excluded from the per-window scalar
    aggregation used for the OLS F-test.

    Returns gcs_edges, gcs_non_edges, and gcs_overall.
    """
    if not HAS_SCIPY:
        log.warning("scipy unavailable — GCS skipped")
        return {"gcs_overall": float("nan")}

    N   = len(vnf_names)
    idx = {v: i for i, v in enumerate(vnf_names)}

    # ── Aggregate windows → [B, N] ────────────────────────────────────────────
    # Average over the time axis W and over continuous features only.
    if binary_mask is not None and binary_mask.any():
        cont = ~binary_mask
        real_ts = real_4d[:, :, :, cont].mean(axis=(1, 3))
        fake_ts = fake_4d[:, :, :, cont].mean(axis=(1, 3))
        log.info(f"  GCS: using {cont.sum()} continuous features "
                 f"(skipped {binary_mask.sum()} binary feature(s))")
    else:
        real_ts = real_4d.mean(axis=(1, 3))
        fake_ts = fake_4d.mean(axis=(1, 3))

    # ── Classify pairs ────────────────────────────────────────────────────────
    reachable       = _compute_reachable(vnf_names, edges)
    direct_edge_set = {(idx[s], idx[d]) for s, d in edges}

    edge_pairs:    list[tuple[int, int]] = []
    no_path_pairs: list[tuple[int, int]] = []
    trans_pairs:   list[tuple[int, int]] = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if (i, j) in direct_edge_set:
                edge_pairs.append((i, j))
            elif j not in reachable[i]:
                no_path_pairs.append((i, j))
            else:
                trans_pairs.append((i, j))

    log.info(
        f"  GCS pair breakdown — "
        f"direct edges={len(edge_pairs)}, "
        f"no-path={len(no_path_pairs)}, "
        f"transitive (skipped)={len(trans_pairs)}"
    )
    log.info(
        f"  GCS: testing {len(edge_pairs) + len(no_path_pairs)} pairs "
        f"(max_lag={max_lag}, alpha={alpha})  "
        f"real_len={len(real_ts)}  fake_len={len(fake_ts)}"
    )

    # ── Run Granger tests ─────────────────────────────────────────────────────
    decisions_real: dict[str, bool] = {}
    decisions_fake: dict[str, bool] = {}

    for i, j in edge_pairs + no_path_pairs:
        key = f"{vnf_names[i]}->{vnf_names[j]}"
        decisions_real[key] = _granger_test_ols(
            y=real_ts[:, j], x=real_ts[:, i], max_lag=max_lag, alpha=alpha)
        decisions_fake[key] = _granger_test_ols(
            y=fake_ts[:, j], x=fake_ts[:, i], max_lag=max_lag, alpha=alpha)

    # ── Aggregate per pair type ───────────────────────────────────────────────
    def _gcs_subset(pairs):
        n_agree = n_total = 0
        for i, j in pairs:
            key = f"{vnf_names[i]}->{vnf_names[j]}"
            if key in decisions_real:
                n_total += 1
                if decisions_real[key] == decisions_fake[key]:
                    n_agree += 1
        return (n_agree / n_total if n_total else float("nan")), n_agree, n_total

    gcs_e, n_agree_e, n_e = _gcs_subset(edge_pairs)
    gcs_n, n_agree_n, n_n = _gcs_subset(no_path_pairs)
    n_total_all  = n_e + n_n
    n_agreed_all = n_agree_e + n_agree_n
    gcs_overall  = n_agreed_all / n_total_all if n_total_all else float("nan")

    # ── Per-pair log ──────────────────────────────────────────────────────────
    log.info("  --- Direct topology edges ---")
    for i, j in edge_pairs:
        key  = f"{vnf_names[i]}->{vnf_names[j]}"
        r, f = decisions_real[key], decisions_fake[key]
        log.info(f"    {'✓' if r==f else '✗'} [edge]    {key:30s}  "
                 f"real={'G-causal' if r else 'indep   '}  "
                 f"fake={'G-causal' if f else 'indep   '}")

    log.info("  --- No-path pairs ---")
    for i, j in no_path_pairs:
        key  = f"{vnf_names[i]}->{vnf_names[j]}"
        r, f = decisions_real[key], decisions_fake[key]
        log.info(f"    {'✓' if r==f else '✗'} [no-path] {key:30s}  "
                 f"real={'G-causal' if r else 'indep   '}  "
                 f"fake={'G-causal' if f else 'indep   '}")

    if trans_pairs:
        log.info("  --- Transitive pairs (skipped) ---")
        for i, j in trans_pairs:
            log.info(f"      [skip]    {vnf_names[i]}->{vnf_names[j]}")

    log.info(f"  GCS edges    = {gcs_e:.4f}  ({n_agree_e}/{n_e} agree)")
    log.info(f"  GCS no-path  = {gcs_n:.4f}  ({n_agree_n}/{n_n} agree)")
    log.info(f"  GCS overall  = {gcs_overall:.4f}  ({n_agreed_all}/{n_total_all} agree)")

    pair_results = {}
    for k in decisions_real:
        r, f = decisions_real[k], decisions_fake[k]
        pair_results[k] = {"real_causal": r, "fake_causal": f, "agrees": r == f}

    return {
        "gcs_overall":           gcs_overall,
        "gcs_edges":             gcs_e,
        "gcs_non_edges":         gcs_n,
        "n_pairs_total":         n_total_all,
        "n_agreed_total":        n_agreed_all,
        "n_edge_pairs":          n_e,
        "n_no_path_pairs":       n_n,
        "n_transitive_skipped":  len(trans_pairs),
        "alpha":                 alpha,
        "max_lag":               max_lag,
        "pair_results":          pair_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TopoSynth — SFC dataset")
    parser.add_argument("--config",  default=DEFAULT_CONFIG,
                        help="YAML config (default: configs/sfc.yaml)")
    parser.add_argument("--n_syn",   type=int, default=None,
                        help="Synthetic windows to generate "
                             "(default: evaluation.n_synthetic from config)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Repetitions for DS and PS (default: 5)")
    parser.add_argument("--device",  default="cuda",
                        help="cuda | cpu (default: cuda)")
    parser.add_argument("--out_dir", default=None,
                        help="Results directory (default: training.experiment_dir)")
    parser.add_argument("--stride",  type=int, default=5,
                        help="Stride for extracting real test windows (default: 5)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg       = load_config(args.config)
    data_cfg  = cfg.get("data",       {})
    model_cfg = cfg.get("model",      {})
    train_cfg = cfg.get("training",   {})
    eval_cfg  = cfg.get("evaluation", {})
    topo_cfg  = cfg.get("topology",   {})

    proc_data_dir  = data_cfg.get("processed_dir",  "data/processed_sfc")
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints_sfc")
    out_dir        = args.out_dir or train_cfg.get("experiment_dir", "experiments_sfc")
    seq_len        = data_cfg.get("seq_len",       48)
    n_features     = data_cfg.get("n_features",     5)
    n_vnf          = data_cfg.get("n_vnf",           6)
    n_diff_steps   = model_cfg.get("n_diff_steps", 200)
    n_syn          = args.n_syn or eval_cfg.get("n_synthetic", 500)
    guidance_eta   = train_cfg.get("guidance_eta",   0.1)
    guidance_gamma = train_cfg.get("guidance_gamma", 0.01)
    n_infer_steps  = eval_cfg.get("n_infer_steps",   50)
    guidance_every = eval_cfg.get("guidance_every",   5)
    granger_lag    = eval_cfg.get("granger_lag",      3)

    # SFC topology — must be present in the config
    vnf_names = topo_cfg.get("vnfcs")
    edges     = topo_cfg.get("edges")
    if vnf_names is None or edges is None:
        raise ValueError(
            "configs/sfc.yaml must define topology.vnfcs and topology.edges."
        )

    log.info("SFC topology (directed):")
    for s, d in edges:
        log.info(f"  {s} → {d}")

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info(f"Loading processed data from {proc_data_dir} …")
    data, tsi, vsi, _scalers = load_processed(proc_data_dir)
    log.info(f"Data shape: {data.shape} | "
             f"train 0:{tsi} | val {tsi}:{vsi} | test {vsi}:{data.shape[0]}")

    train_data = data[:tsi]
    test_data  = data[vsi:]

    # ── Detect binary features ────────────────────────────────────────────────
    binary_mask = _detect_binary_features(train_data)
    if binary_mask.any():
        log.info(f"Binary features (indices {np.where(binary_mask)[0].tolist()}): "
                 f"will be thresholded after generation and excluded from GCS.")
    else:
        log.info("No binary features detected.")

    # ── Extract test windows ──────────────────────────────────────────────────
    log.info(f"Extracting test windows (W={seq_len}, stride={args.stride}) …")
    test_4d   = sliding_windows(test_data, seq_len, stride=args.stride)
    test_flat = flatten_windows(test_4d)
    log.info(f"Test windows: {test_4d.shape}")

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
    log.info(f"Generating {n_syn} synthetic windows …")
    t0 = time.time();  chunks = []
    for start in range(0, n_syn, 256):
        n = min(256, n_syn - start)
        x = model.generate(
            n_samples=n, h_bar=h_bar, device=device,
            eta=guidance_eta, gamma=guidance_gamma,
            n_infer_steps=n_infer_steps, guidance_every=guidance_every,
        )
        chunks.append(x.cpu())
        if (start // 256 + 1) % 5 == 0:
            log.info(f"  generated {start + n}/{n_syn} …")

    fake_4d = torch.cat(chunks, dim=0).numpy()
    log.info(f"Generation done in {time.time()-t0:.1f}s  shape={fake_4d.shape}  "
             f"range=[{fake_4d.min():.3f}, {fake_4d.max():.3f}]")

    # ── Threshold binary features ─────────────────────────────────────────────
    if binary_mask.any():
        fake_4d = threshold_binary_features(fake_4d, binary_mask)
        log.info(f"Binary features thresholded.  "
                 f"New range: [{fake_4d.min():.3f}, {fake_4d.max():.3f}]")

    fake_flat = flatten_windows(fake_4d)
    rng = np.random.default_rng(0)

    # ── Metric 1: DS ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Metric 1/6: Discriminative Score (DS)")
    ds = discriminative_score(test_flat, fake_flat, device,
                              n_epochs=200, batch_size=256, n_repeats=args.repeats)
    log.info(f"  DS = {ds:.6f}  (↓ better; 0 = indistinguishable)")

    # ── Metric 2: PS (TSTR) ───────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Metric 2/6: Predictive Score / TSTR (PS)")
    ps = predictive_score(test_flat, fake_flat, device,
                          n_epochs=200, batch_size=256, n_repeats=args.repeats)
    log.info(f"  PS = {ps:.6f}  (↓ better)")

    # ── Metric 3: Context-FID ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Metric 3/6: Context-FID (CFID)")
    n_fid = min(2000, len(test_flat), len(fake_flat))
    cfid  = context_fid(
        test_flat[rng.choice(len(test_flat), n_fid, replace=False)],
        fake_flat[rng.choice(len(fake_flat), n_fid, replace=False)],
        device, embed_dim=128, enc_epochs=50, batch_size=256)
    log.info(f"  Context-FID = {cfid:.4f}  (↓ better)")

    # ── Metric 4: CS ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Metric 4/6: Correlational Score (CS)")
    n_cs = min(2000, len(test_flat), len(fake_flat))
    cs   = correlational_score(
        test_flat[rng.choice(len(test_flat), n_cs, replace=False)],
        fake_flat[rng.choice(len(fake_flat), n_cs, replace=False)])
    log.info(f"  CS = {cs:.6f}  (↓ better)")

    # ── Metric 5: TCS ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Metric 5/6: Topology Consistency Score (TCS)")
    n_tcs = min(500, len(test_4d), len(fake_4d))
    tcs   = topology_consistency_score(
        test_4d [rng.choice(len(test_4d),  n_tcs, replace=False)],
        fake_4d [rng.choice(len(fake_4d),  n_tcs, replace=False)],
        model, vnf_names, device)
    log.info(f"  TCS embedding_mse    = {tcs['embedding_mse']:.6f}  (↓ better)")
    log.info(f"  TCS embedding_cosine = {tcs['embedding_cosine']:.6f}  (↑ better)")
    for vnf, cos in tcs["per_vnf_cosine"].items():
        log.info(f"      {vnf:12s}: cosine = {cos:.4f}")

    # ── Metric 6: GCS (topology-aware) ───────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Metric 6/6: Granger Causal Score (GCS)  [lag={granger_lag}, α=0.05]")
    gcs_result = granger_causal_score(
        real_4d=test_4d, fake_4d=fake_4d,
        vnf_names=vnf_names, edges=edges,
        binary_mask=binary_mask,
        max_lag=granger_lag, alpha=0.05,
    )
    log.info(f"  GCS overall  = {gcs_result['gcs_overall']:.4f}  "
             f"({gcs_result['n_agreed_total']}/{gcs_result['n_pairs_total']} pairs)  "
             f"(↑ better; 1.0 = perfect)")
    log.info(f"  GCS edges    = {gcs_result['gcs_edges']:.4f}  "
             f"[{gcs_result['n_edge_pairs']} direct-edge pairs]")
    log.info(f"  GCS no-path  = {gcs_result['gcs_non_edges']:.4f}  "
             f"[{gcs_result['n_no_path_pairs']} no-path pairs]")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "discriminative_score": ds,
        "predictive_score":     ps,
        "context_fid":          cfid,
        "correlational_score":  cs,
        "topology_consistency": tcs,
        "granger_causal_score": gcs_result,
        "meta": {
            "n_syn":               n_syn,
            "n_test_windows":      len(test_4d),
            "repeats":             args.repeats,
            "granger_lag":         granger_lag,
            "binary_feature_ids":  np.where(binary_mask)[0].tolist(),
            "device":              str(device),
            "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset":             "sfc",
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    ts_str   = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"eval_{ts_str}.json")
    with open(out_path, "w") as fout:
        json.dump(results, fout, indent=2)
    log.info(f"\nResults saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY  (SFC dataset)")
    log.info("=" * 60)
    log.info(f"  Discriminative Score  (↓)  {ds:.6f}")
    log.info(f"  Predictive Score/TSTR (↓)  {ps:.6f}")
    log.info(f"  Context-FID           (↓)  {cfid:.4f}")
    log.info(f"  Correlational Score   (↓)  {cs:.6f}")
    log.info(f"  Topo Consistency MSE  (↓)  {tcs['embedding_mse']:.6f}")
    log.info(f"  Topo Consistency Cos  (↑)  {tcs['embedding_cosine']:.6f}")
    log.info(f"  GCS overall           (↑)  {gcs_result['gcs_overall']:.4f}  [lag={granger_lag}]")
    log.info(f"  GCS edges             (↑)  {gcs_result['gcs_edges']:.4f}")
    log.info(f"  GCS no-path           (↑)  {gcs_result['gcs_non_edges']:.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()