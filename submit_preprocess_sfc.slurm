#!/bin/bash
#SBATCH --job-name=preprocess_sfc
#SBATCH --account=nn12085k
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/preprocess_sfc_%j.log

apptainer exec --nv \
  --bind /cluster/home/miladraf \
  --bind /cluster/projects/nn12085k \
  /cluster/projects/nn12085k/pytorch-milad-sandbox \
  python3 /cluster/home/miladraf/toposynth/scripts/preprocess_sfc.py \
    --db /cluster/home/miladraf/toposynth/data/raw/VNF_KPI_database_crash_top2_v2_statefull_7May.db \
    --out-dir /cluster/home/miladraf/toposynth/data/processed_sfc
