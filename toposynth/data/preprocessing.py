# toposynth/data/preprocessing.py
# Loads X_126.csv, selects 72 features, applies MinMax scaling,
# and saves processed tensors to data/processed/.

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from toposynth.data.topology import (
    TOPOLOGY_CONFIG, SELECTED_METRICS, get_selected_columns
)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

RAW_PATH       = 'data/raw/X_126.csv'
PROCESSED_DIR  = 'data/processed'
TRAIN_RATIO    = 0.80
VAL_RATIO      = 0.10
# TEST_RATIO   = 0.10  (remainder)

SMOOTH_WINDOW  = 3     # rolling mean window (same as your MCAE pipeline)


# ─── MAIN LOADER ─────────────────────────────────────────────────────────────

class ClearwaterLoader:
    """
    Loads X_126.csv, selects the 72 features (12 per VNFC × 6 VNFCs),
    applies optional smoothing, fits MinMaxScaler on train split,
    and returns data dict + scalers.
    """

    def __init__(self, raw_path=RAW_PATH, smooth=True, smooth_window=SMOOTH_WINDOW):
        self.raw_path     = raw_path
        self.smooth       = smooth
        self.smooth_window = smooth_window
        self.vnf_names    = TOPOLOGY_CONFIG['vnf_names']
        self.n_vnf        = TOPOLOGY_CONFIG['n_vnf']
        self.n_features   = TOPOLOGY_CONFIG['n_features']
        self.selected_cols = get_selected_columns()

    def load(self):
        """
        Returns:
            data   : dict  vnf_name -> np.ndarray [T, F]  (raw, unscaled)
            df_raw : pd.DataFrame of the full 72-column slice (for inspection)
        """
        print(f'[Preprocessing] Loading {self.raw_path} ...', flush=True)
        df = pd.read_csv(self.raw_path, sep=';', index_col=0)
        df.columns = [c.replace('.csv', '') for c in df.columns]

        # Verify all expected columns are present
        missing = [c for c in self.selected_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f'[Preprocessing] {len(missing)} expected columns not found in CSV.\n'
                f'First 5 missing: {missing[:5]}\n'
                f'Check that column names match format: {{vnfc}}-{{metric}}'
            )

        df_sel = df[self.selected_cols].copy()

        # Fill any NaN with forward-fill then zero
        df_sel = df_sel.fillna(method='ffill').fillna(0.0)

        print(f'[Preprocessing] Shape after feature selection: {df_sel.shape}')  # (177000, 72)

        # Optional smoothing (rolling mean, same as your MCAE pipeline)
        if self.smooth:
            df_sel = df_sel.rolling(self.smooth_window, min_periods=1).mean()
            print(f'[Preprocessing] Applied rolling mean (window={self.smooth_window})')

        # Split into per-VNF arrays: each vnf_name -> [T, F]
        data = {}
        for vnf in self.vnf_names:
            cols = [f'{vnf}-{m}' for m in SELECTED_METRICS]
            data[vnf] = df_sel[cols].values.astype(np.float32)  # [T, F]

        print(f'[Preprocessing] Loaded {len(data["bono"])} timesteps, '
              f'{self.n_vnf} VNFs × {self.n_features} features')
        return data, df_sel

    @staticmethod
    def compute_split_indices(n_timesteps):
        tsi = int(n_timesteps * TRAIN_RATIO)           # train end
        vsi = int(n_timesteps * (TRAIN_RATIO + VAL_RATIO))  # val end
        return tsi, vsi

    @staticmethod
    def build_scalers(data, vnf_names, tsi):
        """
        Fits one MinMaxScaler per VNF on the training split only.
        Identical pattern to your MCAE build_scalers().
        """
        scalers = {}
        for v in vnf_names:
            sc = MinMaxScaler()
            sc.fit(data[v][:tsi])        # fit on train only
            scalers[v] = sc
        return scalers

    @staticmethod
    def apply_scalers(data, scalers, vnf_names):
        """
        Transforms all timesteps using the fitted scalers.
        Returns dict vnf_name -> np.ndarray [T, F] in [0, 1].
        """
        return {v: scalers[v].transform(data[v]).astype(np.float32)
                for v in vnf_names}


# ─── SAVE / LOAD PROCESSED ───────────────────────────────────────────────────

def preprocess_and_save(raw_path=RAW_PATH, out_dir=PROCESSED_DIR, smooth=True):
    """
    Full pipeline: load → smooth → scale → save .npy files + scaler params.
    Run once before training. Output files:

        data/processed/
            data_scaled.npy      shape [T, N, F]  float32  (all splits, scaled)
            split_indices.npy    shape [2]         int      [tsi, vsi]
            scaler_min.npy       shape [N, F]      float32  (MinMaxScaler min_)
            scaler_scale.npy     shape [N, F]      float32  (MinMaxScaler scale_)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    loader = ClearwaterLoader(raw_path=raw_path, smooth=smooth)
    data, _ = loader.load()

    vnf_names = TOPOLOGY_CONFIG['vnf_names']
    T = len(data[vnf_names[0]])
    tsi, vsi = ClearwaterLoader.compute_split_indices(T)
    print(f'[Preprocessing] Split: train=0:{tsi}  val={tsi}:{vsi}  test={vsi}:{T}')

    scalers = ClearwaterLoader.build_scalers(data, vnf_names, tsi)
    scaled  = ClearwaterLoader.apply_scalers(data, scalers, vnf_names)

    # Stack into [T, N, F]
    data_scaled = np.stack([scaled[v] for v in vnf_names], axis=1)  # [T, N, F]
    print(f'[Preprocessing] data_scaled shape: {data_scaled.shape}')

    # Save arrays
    np.save(os.path.join(out_dir, 'data_scaled.npy'),    data_scaled)
    np.save(os.path.join(out_dir, 'split_indices.npy'),  np.array([tsi, vsi]))

    # Save scaler params (one row per VNF, one col per feature)
    scaler_min   = np.stack([scalers[v].min_        for v in vnf_names])  # [N, F]
    scaler_scale = np.stack([scalers[v].scale_      for v in vnf_names])  # [N, F]
    np.save(os.path.join(out_dir, 'scaler_min.npy'),   scaler_min)
    np.save(os.path.join(out_dir, 'scaler_scale.npy'), scaler_scale)

    print(f'[Preprocessing] Saved processed data to {out_dir}/')
    print(f'  data_scaled.npy   : {data_scaled.shape}')
    print(f'  split_indices.npy : [{tsi}, {vsi}]')
    print(f'  scaler_min.npy    : {scaler_min.shape}')
    print(f'  scaler_scale.npy  : {scaler_scale.shape}')
    return data_scaled, tsi, vsi, scalers


def load_processed(out_dir=PROCESSED_DIR):
    """
    Fast load of already-processed data (skips CSV parsing).
    Called at the start of every training run.

    Returns:
        data_scaled : np.ndarray [T, N, F]
        tsi         : int   train/val boundary
        vsi         : int   val/test boundary
        scalers     : dict  vnf_name -> reconstructed MinMaxScaler
    """
    data_scaled  = np.load(os.path.join(out_dir, 'data_scaled.npy'))
    split_idx    = np.load(os.path.join(out_dir, 'split_indices.npy'))
    scaler_min   = np.load(os.path.join(out_dir, 'scaler_min.npy'))
    scaler_scale = np.load(os.path.join(out_dir, 'scaler_scale.npy'))

    tsi, vsi = int(split_idx[0]), int(split_idx[1])

    # Reconstruct MinMaxScaler objects (needed for inverse_transform at eval)
    vnf_names = TOPOLOGY_CONFIG['vnf_names']
    scalers = {}
    for i, v in enumerate(vnf_names):
        sc = MinMaxScaler()
        sc.min_   = scaler_min[i]
        sc.scale_ = scaler_scale[i]
        sc.data_min_  = scaler_min[i] / -scaler_scale[i]   # approximate
        sc.data_range_ = 1.0 / scaler_scale[i]              # approximate
        sc.n_features_in_ = TOPOLOGY_CONFIG['n_features']
        scalers[v] = sc

    print(f'[Preprocessing] Loaded processed data: {data_scaled.shape}  '
          f'train=0:{tsi}  val={tsi}:{vsi}  test={vsi}:{data_scaled.shape[0]}')
    return data_scaled, tsi, vsi, scalers


def inverse_transform(data_scaled_np, scalers):
    """
    Inverse-scale a [T, N, F] or [B, T, N, F] array back to original units.
    Useful for computing evaluation metrics in physical units.
    """
    vnf_names = TOPOLOGY_CONFIG['vnf_names']
    if data_scaled_np.ndim == 3:        # [T, N, F]
        out = np.zeros_like(data_scaled_np)
        for i, v in enumerate(vnf_names):
            out[:, i, :] = scalers[v].inverse_transform(data_scaled_np[:, i, :])
    elif data_scaled_np.ndim == 4:      # [B, T, N, F]
        out = np.zeros_like(data_scaled_np)
        for i, v in enumerate(vnf_names):
            B, T, _, F = data_scaled_np.shape
            flat = data_scaled_np[:, :, i, :].reshape(-1, F)
            out[:, :, i, :] = scalers[v].inverse_transform(flat).reshape(B, T, F)
    else:
        raise ValueError(f'Expected 3D or 4D array, got {data_scaled_np.ndim}D')
    return out


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    preprocess_and_save()