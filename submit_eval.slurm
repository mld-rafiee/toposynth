#!/usr/bin/env bash
#SBATCH --job-name=toposynth_eval
#SBATCH --account=nn12085k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                # single GPU — evaluate.py is not DDP
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

set -euo pipefail

# ---- Paths ------------------------------------------------------------------
REPO_ROOT="${HOME}/toposynth"
CONTAINER="/cluster/projects/nn12085k/pytorch-milad-sandbox"

# ---- Optional overrides (pass via --export=VAR=val to sbatch) ---------------
CONFIG="${CONFIG:-configs/default.yaml}"
N_SYN="${N_SYN:-5000}"
REPEATS="${REPEATS:-5}"
OUT_DIR="${OUT_DIR:-experiments}"

# ---- Logging directory must exist before SLURM writes stdout/stderr ---------
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/${OUT_DIR}"

echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "GPU       : ${SLURM_JOB_GPUS:-gres=gpu:1}"
echo "Container : ${CONTAINER}"
echo "Config    : ${CONFIG}"
echo "n_syn     : ${N_SYN}"
echo "repeats   : ${REPEATS}"
echo "Start     : $(date)"

# ---- Launch (single python3, no torchrun) -----------------------------------
apptainer exec \
    --nv \
    --bind /cluster/home/miladraf \
    --bind /cluster/projects/nn12085k \
    "${CONTAINER}" \
    bash -c "
        cd ${REPO_ROOT}
        python3 scripts/evaluate.py \
            --config  ${CONFIG} \
            --n_syn   ${N_SYN} \
            --repeats ${REPEATS} \
            --device  cuda \
            --out_dir ${OUT_DIR}
    "

echo "End time  : $(date)"
echo "Done."