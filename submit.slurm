#!/usr/bin/env bash
#SBATCH --job-name=toposynth
#SBATCH --account=nn12085k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # one srun task — torchrun spawns the GPU workers
#SBATCH --cpus-per-task=12          # 4 CPUs per GPU  (3 GPUs × 4 = 12)
#SBATCH --gres=gpu:3                # 3 × GH200 GPUs
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# ---- Paths --------------------------------------------------------------
REPO_ROOT="${HOME}/toposynth"
CONTAINER="/cluster/projects/nn12085k/pytorch-milad-sandbox"

# ---- Environment --------------------------------------------------------
export MASTER_ADDR="$(hostname)"
export MASTER_PORT="29500"
export NPROC_PER_NODE=3
export OMP_NUM_THREADS=4

# ---- Logging directory must exist before SLURM writes stdout/stderr -----
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/checkpoints"
mkdir -p "${REPO_ROOT}/data/processed"

echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "GPUs          : ${SLURM_JOB_GPUS:-not set by SLURM, using gres}"
echo "MASTER_ADDR   : ${MASTER_ADDR}"
echo "MASTER_PORT   : ${MASTER_PORT}"
echo "Container     : ${CONTAINER}"
echo "Repo root     : ${REPO_ROOT}"
echo "Start time    : $(date)"

# ---- Launch -------------------------------------------------------------
apptainer exec \
    --nv \
    --bind /cluster/home/miladraf \
    --bind /cluster/projects/nn12085k \
    "${CONTAINER}" \
    bash -c "cd ${REPO_ROOT} && bash run_ddp.sh"

echo "End time      : $(date)"
echo "Done."