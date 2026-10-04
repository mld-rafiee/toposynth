"""
scripts/train.py — Main DDP entry point for TopoSynth training.
Called via: torchrun --nnodes=1 --nproc_per_node=N scripts/train.py [--config PATH]
"""

import os
import sys
import argparse
import logging
import time

import torch
import torch.distributed as dist
import yaml

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toposynth.data.preprocessing import preprocess_and_save, load_processed
from toposynth.data.dataset import build_dataloaders
from toposynth.models.toposynth import train_toposynth, load_best_model_for_generation

# ---------------------------------------------------------------------------
# Logging — rank-aware so only rank 0 writes to console
# ---------------------------------------------------------------------------

def setup_logging(rank: int, log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    level = logging.INFO if rank == 0 else logging.WARNING
    fmt = f"[rank{rank}] %(asctime)s %(levelname)s  %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, f"train_rank{rank}.log")),
        ],
    )


# ---------------------------------------------------------------------------
# DDP initialisation  (matches user's MCAE pattern exactly)
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple:
    """Initialise process group from torchrun env vars."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, global_rank, device


def cleanup_ddp() -> None:
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "configs", "default.yaml"
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TopoSynth (DDP)")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    # ---- DDP setup --------------------------------------------------------
    local_rank, global_rank, device = setup_ddp()
    setup_logging(global_rank)
    logger = logging.getLogger(__name__)

    logger.info(
        "TopoSynth training started | "
        f"global_rank={global_rank} local_rank={local_rank} device={device} "
        f"world_size={dist.get_world_size()}"
    )

    # ---- Config -----------------------------------------------------------
    cfg = load_config(args.config)
    data_cfg   = cfg.get("data", {})
    model_cfg  = cfg.get("model", {})
    train_cfg  = cfg.get("training", {})

    raw_data_path  = data_cfg.get("raw_path",       "data/raw/X_126.csv")
    proc_data_dir  = data_cfg.get("processed_dir",  "data/processed")
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    log_dir        = train_cfg.get("log_dir",        "logs")

    seq_len      = data_cfg.get("seq_len",      48)
    n_features   = data_cfg.get("n_features",   12)
    n_vnf        = data_cfg.get("n_vnf",         6)
    n_diff_steps = model_cfg.get("n_diff_steps", 200)
    n_epochs     = train_cfg.get("n_epochs",     300)
    batch_size   = train_cfg.get("batch_size",    64)
    lr           = train_cfg.get("lr",          1e-4)
    patience     = train_cfg.get("patience",      20)

    # ---- Preprocessing (rank 0 only, then barrier) -----------------------
    proc_data_file = os.path.join(proc_data_dir, "data_scaled.npy")

    if global_rank == 0:
        if not os.path.exists(proc_data_file):
            logger.info("Processed data not found — running preprocessing …")
            t0 = time.time()
            preprocess_and_save(
                raw_path=raw_data_path,
                out_dir=proc_data_dir,
            )
            logger.info(f"Preprocessing done in {time.time()-t0:.1f}s")
        else:
            logger.info(f"Processed data found at {proc_data_dir}")

    dist.barrier()   # all ranks wait for rank-0 to finish preprocessing

    # ---- Load processed data (all ranks) ---------------------------------
    logger.info("Loading processed data …")
    data, tsi, vsi, scalers = load_processed(proc_data_dir)
    # data: [T, N, F]   tsi: train/val boundary   vsi: val/test boundary
    logger.info(
        f"Data shape: {data.shape} | "
        f"train 0:{tsi} | val {tsi}:{vsi} | test {vsi}:{data.shape[0]}"
    )

    # ---- Build data loaders ----------------------------------------------
    # build_dataloaders returns (train_loader, val_loader, test_loader, train_sampler)
    train_loader, val_loader, test_loader, train_sampler = build_dataloaders(
        data_scaled=data,
        tsi=tsi,
        vsi=vsi,
        batch_size=batch_size,
        window_size=seq_len,
        num_workers=data_cfg.get("num_workers", 4),
        rank=global_rank,
        world_size=dist.get_world_size(),
    )
    logger.info(
        f"DataLoaders ready | "
        f"train={len(train_loader)} val={len(val_loader)} test={len(test_loader)} batches"
    )

    # ---- Training --------------------------------------------------------
    # train_toposynth signature:
    #   train_toposynth(train_loader, val_loader, train_sampler, checkpoint_dir,
    #                   seq_len, n_features, n_vnf,
    #                   n_epochs, lr, weight_decay, lambda_topo,
    #                   early_stop_patience, resume, n_diff_steps)
    logger.info("Starting train_toposynth …")
    train_toposynth(
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        checkpoint_dir=checkpoint_dir,
        seq_len=seq_len,
        n_features=n_features,
        n_vnf=n_vnf,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 1e-6),
        lambda_topo=train_cfg.get("lambda_topo", 0.1),
        early_stop_patience=patience,
        resume=train_cfg.get("resume", True),
        n_diff_steps=n_diff_steps,
    )


    # ---- Cleanup (ALL ranks before generation) ---------------------------
    # Must happen before the rank-0-only block below; otherwise ranks 1 & 2
    # hang waiting for NCCL collectives that rank 0 never sends.
    cleanup_ddp()

    # ---- Post-training: quick sanity generation on rank 0 ---------------
    if global_rank == 0:
        logger.info("Training complete. Loading best model for generation …")
        # load_best_model_for_generation signature:
        #   load_best_model_for_generation(checkpoint_dir, device,
        #                                  seq_len, n_features, n_vnf,
        #                                  n_diff_steps)
        model = load_best_model_for_generation(
            checkpoint_dir=checkpoint_dir,
            device=device,
            seq_len=seq_len,
            n_features=n_features,
            n_vnf=n_vnf,
            n_diff_steps=n_diff_steps,
        )
        model.eval()

        # h_bar: per-VNF training-set mean [N, F]
        # generate() signature: generate(n_samples, h_bar, eta, gamma, device)
        import numpy as np
        h_bar = torch.FloatTensor(data[:tsi].mean(axis=0)).to(device)

        logger.info("Running quick generation (n_samples=4) …")
        with torch.no_grad():
            x_gen = model.generate(
                n_samples=4,
                h_bar=h_bar,
                device=device,
                eta=train_cfg.get("guidance_eta", 0.1),
                gamma=train_cfg.get("guidance_gamma", 0.01),
            )
        # x_gen: [4, W, N, F]
        gen_path = os.path.join(checkpoint_dir, "quick_gen.pt")
        torch.save(x_gen.cpu(), gen_path)
        logger.info(f"Generated samples shape: {x_gen.shape} — saved to {gen_path}")
        logger.info("All done. Run scripts/evaluate.py for full evaluation.")


if __name__ == "__main__":
    main()