#!/usr/bin/env bash
# run_ddp.sh — torchrun launcher for TopoSynth inside Apptainer
# Called by submit.slurm via:
#   apptainer exec --nv ... run_ddp.sh
# Environment variables injected by SLURM + torchrun are available here.

set -euo pipefail

# ---- Repo root (directory containing this script) -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

# ---- Distributed config (set by SLURM / torchrun) ----------------------
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${SLURM_NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"          # number of GPUs per node

# ---- Optional config override ------------------------------------------
CONFIG="${CONFIG:-configs/default.yaml}"

# topology.py builds TOPOLOGY_CONFIG at import time from this env var.
# Without it, every run uses configs/default.yaml (Clearwater) regardless
# of --config, so SFC training gets Clearwater's 14-edge GAT topology.
export TOPOSYNTH_CONFIG="${REPO_ROOT}/${CONFIG}"

echo "============================================================"
echo " TopoSynth DDP training"
echo "   MASTER_ADDR    = ${MASTER_ADDR}"
echo "   MASTER_PORT    = ${MASTER_PORT}"
echo "   NNODES         = ${NNODES}"
echo "   NPROC_PER_NODE = ${NPROC_PER_NODE}"
echo "   CONFIG         = ${CONFIG}"
echo "   REPO_ROOT      = ${REPO_ROOT}"
echo "============================================================"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    scripts/train.py \
    --config "${CONFIG}"