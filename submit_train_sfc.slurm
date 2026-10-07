#!/usr/bin/env bash
#SBATCH --job-name=toposynth_sfc
#SBATCH --account=nn12085k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # one srun task — torchrun spawns the GPU workers
#SBATCH --cpus-per-task=12          # 4 CPUs per GPU  (3 GPUs × 4 = 12)
#SBATCH --gres=gpu:3                # 3 × GH200 GPUs
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sfc-%j.out
#SBATCH --error=logs/sfc-%j.err

set -euo pipefail

# ---- Paths ------------------------------------------------------------------
REPO_ROOT="${HOME}/toposynth"
CONTAINER="/cluster/projects/nn12085k/pytorch-milad-sandbox"

# ---- SFC-specific config ----------------------------------------------------
export CONFIG="configs/sfc.yaml"
export MASTER_ADDR="$(hostname)"
export MASTER_PORT="29500"
export NPROC_PER_NODE=3
export OMP_NUM_THREADS=4

# ---- Directories ------------------------------------------------------------
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/checkpoints_sfc"
mkdir -p "${REPO_ROOT}/data/processed_sfc"

echo "============================================================"
echo " TopoSynth training — SFC dataset"
echo "   Job ID    : ${SLURM_JOB_ID}"
echo "   Node      : $(hostname)"
echo "   GPUs      : ${SLURM_JOB_GPUS:-gres=gpu:3}"
echo "   Container : ${CONTAINER}"
echo "   Config    : ${CONFIG}"
echo "   Start     : $(date)"
echo "============================================================"

# ---- Launch -----------------------------------------------------------------
apptainer exec \
    --nv \
    --bind /cluster/home/miladraf \
    --bind /cluster/projects/nn12085k \
    "${CONTAINER}" \
    bash -c "cd ${REPO_ROOT} && CONFIG=${CONFIG} bash run_ddp.sh"

echo "End time : $(date)"
echo "Done."