"""
preprocess_sfc.py — Convert SFC SQLite database to TopoSynth-ready format.

Produces the same 4 files that preprocessing.py produces for Clearwater,
so train.py / evaluate.py work without any code changes:

  data/processed_sfc/
    data_scaled.npy     [T, N, F]  full dataset, MinMax-scaled to [0,1]
    split_indices.npy   [2]        [train_end, val_end] integer indices
    scaler_min.npy      [N, F]     per-VNF scaler min_ values
    scaler_scale.npy    [N, F]     per-VNF scaler scale_ values

VNFs (node order matches topology config):
  0: firewall   1: dpi   2: enc   3: comp   4: firewall2   5: nat

Features per VNF (5):
  cpu_usage, memory_usage, processing_delay, traffic_rate, crash
"""

import argparse
import os
import re
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def parse_cpu(val):
    if val is None:
        return float("nan")
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip()
    if val.endswith("m"):
        return float(val[:-1]) / 1000.0
    try:
        return float(val)
    except ValueError:
        return float("nan")

def parse_memory(val):
    if val is None:
        return float("nan")
    if isinstance(val, (int, float)):
        return float(val) / (1024 ** 2)
    val = str(val).strip()
    m = re.match(r"^([0-9.]+)(Ki|Mi|Gi|Ti|K|M|G|T)?$", val)
    if not m:
        return float("nan")
    num = float(m.group(1))
    unit = m.group(2) or ""
    factors = {"Ki": 1/1024, "Mi": 1.0, "Gi": 1024.0, "Ti": 1024**2,
               "K":  1/1024, "M":  1.0, "G":  1024.0, "T":  1024**2}
    return num * factors.get(unit, 1 / (1024 ** 2))

def main(args):
    db_path = args.db
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    vnf_suffixes = ["firewall", "dpi", "enc", "comp", "firewall2", "nat"]
    feature_cols = ["cpu_usage", "memory_usage",
                    "processing_delay", "traffic_rate", "crash"]

    N = len(vnf_suffixes)
    F = len(feature_cols)

    print(f"Reading {db_path} ...")
    con = sqlite3.connect(db_path)
    df  = pd.read_sql("SELECT * FROM VNF_KPI_database ORDER BY id", con)
    con.close()
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    for vnf in vnf_suffixes:
        cpu_col = f"cpu_usage_{vnf}"
        mem_col = f"memory_usage_{vnf}"
        if cpu_col in df.columns:
            df[cpu_col] = df[cpu_col].apply(parse_cpu)
        if mem_col in df.columns:
            df[mem_col] = df[mem_col].apply(parse_memory)

    T = len(df)
    X = np.zeros((T, N, F), dtype=np.float32)
    for n, vnf in enumerate(vnf_suffixes):
        for f, feat in enumerate(feature_cols):
            col = f"{feat}_{vnf}"
            if col not in df.columns:
                print(f"  WARNING: {col} not found — filling with 0")
                continue
            X[:, n, f] = df[col].values.astype(np.float32)

    X_flat = pd.DataFrame(X.reshape(T, N * F)).ffill().fillna(0.0).values
    X = X_flat.reshape(T, N, F).astype(np.float32)
    print(f"  Raw array: {X.shape}  |  NaNs: {np.isnan(X).sum()}")

    train_end = int(T * args.train_ratio)
    val_end   = int(T * (args.train_ratio + args.val_ratio))
    split_indices = np.array([train_end, val_end], dtype=np.int64)
    print(f"  Split: train 0:{train_end} | val {train_end}:{val_end} | test {val_end}:{T}")

    scaler_min   = np.zeros((N, F), dtype=np.float64)
    scaler_scale = np.zeros((N, F), dtype=np.float64)

    X_scaled = X.copy()
    for n in range(N):
        scaler = MinMaxScaler()
        scaler.fit(X[:train_end, n, :])
        X_scaled[:, n, :] = scaler.transform(X[:, n, :])
        scaler_min[n]   = scaler.min_
        scaler_scale[n] = scaler.scale_

    X_scaled = X_scaled.astype(np.float32)
    print(f"  Scaled range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")

    np.save(os.path.join(out_dir, "data_scaled.npy"),   X_scaled)
    np.save(os.path.join(out_dir, "split_indices.npy"), split_indices)
    np.save(os.path.join(out_dir, "scaler_min.npy"),    scaler_min)
    np.save(os.path.join(out_dir, "scaler_scale.npy"),  scaler_scale)

    for fname in ["train.npy", "val.npy", "test.npy", "scaler.pkl"]:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"  Removed old file: {fname}")

    print(f"  Saved 4 files to {out_dir}/")
    print(f"    data_scaled.npy   {X_scaled.shape}")
    print(f"    split_indices.npy {split_indices}")
    print(f"    scaler_min.npy    {scaler_min.shape}")
    print(f"    scaler_scale.npy  {scaler_scale.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",
        default="data/raw/VNF_KPI_database_crash_top2_v2_statefull_7May.db")
    parser.add_argument("--out-dir",     default="data/processed_sfc")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio",   type=float, default=0.10)
    main(parser.parse_args())
