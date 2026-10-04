# toposynth/data/dataset.py
# PyTorch Dataset with sliding window logic.
# Mirrors your SFCDataset pattern exactly.

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from toposynth.data.topology import TOPOLOGY_CONFIG, get_adjacency_matrix, get_upstream_neighbours

# ─── CONFIG ──────────────────────────────────────────────────────────────────

WINDOW_SIZE = 48   # T_seq: sequence length fed to Diffusion-TS


# ─── DATASET ─────────────────────────────────────────────────────────────────

class VNFDataset(Dataset):
    """
    Sliding-window dataset over the preprocessed [T, N, F] array.

    Each item returns:
        'x'     : FloatTensor [W, N, F]   — window of W timesteps (input to model)
        'adj'   : FloatTensor [N, N]      — SFC adjacency matrix (same for all items)
        'h_bar' : FloatTensor [N, F]      — per-VNF mean over training set (GAT node attr)

    mode: 'train' | 'val' | 'test'
    """

    def __init__(self, data_scaled, tsi, vsi, mode, window_size=WINDOW_SIZE):
        """
        data_scaled : np.ndarray [T, N, F]  — full scaled dataset
        tsi         : int   — train/val split index
        vsi         : int   — val/test split index
        mode        : str   — 'train' | 'val' | 'test'
        window_size : int   — W, sliding window length
        """
        assert mode in ('train', 'val', 'test'), f'Unknown mode: {mode}'

        self.W          = window_size
        self.vnf_names  = TOPOLOGY_CONFIG['vnf_names']
        self.n_vnf      = TOPOLOGY_CONFIG['n_vnf']
        self.n_features = TOPOLOGY_CONFIG['n_features']

        # Slice the correct temporal segment
        T = len(data_scaled)
        boundaries = {
            'train': (0,   tsi),
            'val':   (tsi, vsi),
            'test':  (vsi, T),
        }
        self.start_idx, self.end_idx = boundaries[mode]

        # Store only the relevant slice (avoids holding 3× data in memory)
        # Add a lookback buffer of W so val/test windows don't bleed into train
        buf_start = max(0, self.start_idx)
        self.data  = data_scaled[buf_start : self.end_idx].astype(np.float32)  # [S, N, F]
        self.offset = buf_start   # absolute index of data[0]

        self.n_windows = max(0, (self.end_idx - self.start_idx) - self.W)

        # Adjacency matrix — static, same for every sample
        self.adj = get_adjacency_matrix(add_self_loops=False)   # [N, N]

        # Per-VNF mean over full training set — used as GAT node attribute h̃_i
        # Computed from the training portion of data_scaled (always index 0:tsi)
        train_data = data_scaled[:tsi]                          # [tsi, N, F]
        self.h_bar = torch.FloatTensor(
            train_data.mean(axis=0)                             # [N, F]
        )

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        # Absolute start of this window
        abs_start = self.start_idx + idx
        abs_end   = abs_start + self.W

        # Convert to local slice index
        local_start = abs_start - self.offset
        local_end   = abs_end   - self.offset

        x = torch.FloatTensor(self.data[local_start:local_end])  # [W, N, F]

        return {
            'x':     x,            # [W, N, F]  — sequence window
            'adj':   self.adj,     # [N, N]     — SFC adjacency (FloatTensor)
            'h_bar': self.h_bar,   # [N, F]     — VNF mean statistics
        }


# ─── DATALOADER FACTORY ──────────────────────────────────────────────────────

def build_dataloaders(data_scaled, tsi, vsi, batch_size=64,
                      window_size=WINDOW_SIZE, num_workers=4, rank=0, world_size=1):
    """
    Builds train / val / test DataLoaders.

    Train loader uses DistributedSampler (required for DDP).
    Val and test loaders run on rank 0 only — no DistributedSampler needed
    because evaluation is done inside `if rank == 0:` blocks.

    Returns:
        train_loader, val_loader, test_loader, train_sampler
    """
    train_ds = VNFDataset(data_scaled, tsi, vsi, 'train', window_size)
    val_ds   = VNFDataset(data_scaled, tsi, vsi, 'val',   window_size)
    test_ds  = VNFDataset(data_scaled, tsi, vsi, 'test',  window_size)

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    # Val / test: no DDP sampler — evaluated on rank 0 only
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    if rank == 0:
        print(f'[Dataset] window_size={window_size}  '
              f'train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)} windows')
        print(f'[Dataset] batch_size={batch_size}  '
              f'train_batches={len(train_loader)}  world_size={world_size}')

    return train_loader, val_loader, test_loader, train_sampler


# ─── SANITY CHECK ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/cluster/home/miladraf/toposynth')

    from toposynth.data.preprocessing import load_processed

    data_scaled, tsi, vsi, scalers = load_processed()

    train_loader, val_loader, test_loader, sampler = build_dataloaders(
        data_scaled, tsi, vsi,
        batch_size=64, world_size=1, rank=0
    )

    batch = next(iter(train_loader))
    print(f"\nSanity check — one training batch:")
    print(f"  x     : {batch['x'].shape}      # [B, W, N, F]")
    print(f"  adj   : {batch['adj'].shape}    # [B, N, N]")
    print(f"  h_bar : {batch['h_bar'].shape}  # [B, N, F]")
    print(f"  x min/max : {batch['x'].min():.4f} / {batch['x'].max():.4f}")
    print(f"\nExpected shapes:")
    print(f"  x     : [{64}, {WINDOW_SIZE}, {TOPOLOGY_CONFIG['n_vnf']}, {TOPOLOGY_CONFIG['n_features']}]")
    print(f"  adj   : [{64}, {TOPOLOGY_CONFIG['n_vnf']}, {TOPOLOGY_CONFIG['n_vnf']}]")