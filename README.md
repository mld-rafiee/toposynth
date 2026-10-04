# TopoSynth

**Topology-Aware Synthetic VNF Time-Series Generation via Diffusion-TS and GAT**

TopoSynth generates synthetic multivariate time-series data for Virtual Network
Functions (VNFs) in 5G Service Function Chains (SFCs). A Graph Attention Network
(GAT) encodes the SFC topology into a conditioning embedding; a Diffusion-TS
denoising model then generates per-VNF metric traces guided toward that embedding
via gradient-based classifier guidance at inference time.

Trained and evaluated on the **Clearwater IMS** VNF dataset (177,000 timesteps,
6 VNFs, 12 features each) using 3 × NVIDIA GH200 GPUs via PyTorch DDP.

---

**GAT encoder** — 4-head GAT (Veličković et al., ICLR 2018), projects per-VNF
feature vectors through the Clearwater topology graph into a 128-dim embedding.

**Diffusion-TS denoiser** — Transformer-based diffusion model (Chen & Qiao,
ICLR 2024) with cosine noise schedule, 200 diffusion steps, separate time-domain
and frequency-domain loss terms (λ₁ = λ₂ = 1.0).

**Topology-guided generation** — at inference, each reverse-diffusion step
computes ∇ₓ MSE(GAT(x̄), z̄) and subtracts η · grad from the predicted x₀,
steering generated samples toward the target topology embedding.

---

## Clearwater topology



---

## Project structure

```
toposynth/
├── configs/
│   └── default.yaml          # all hyperparameters
├── scripts/
│   ├── train.py              # DDP training entry point (torchrun)
│   └── evaluate.py           # single-GPU evaluation
├── toposynth/
│   ├── data/
│   │   ├── preprocessing.py  # MinMax scaling, train/val/test split
│   │   ├── dataset.py        # sliding-window dataset + DDP samplers
│   │   └── topology.py       # adjacency matrix construction
│   └── models/
│       ├── gat.py            # Graph Attention Network encoder
│       ├── diffusion_ts.py   # Diffusion-TS denoiser (Transformer)
│       └── toposynth.py      # combined model + train loop + generate()
├── run_ddp.sh                # torchrun launcher (called by SLURM)
├── submit.slurm              # SLURM job: 3-GPU DDP training
└── submit_eval.slurm         # SLURM job: 1-GPU evaluation
```


---

## Usage

### 1 — Configure

Edit `configs/default.yaml`. Key fields:

| Section | Key | Default | Notes |
|---|---|---|---|
| `data` | `raw_path` | `data/raw/X_126.csv` | path to Clearwater CSV |
| `training` | `n_epochs` | `100` | max epochs (early stop: patience=20) |
| `training` | `batch_size` | `64` | per-GPU batch size |
| `model` | `n_diff_steps` | `200` | diffusion steps |
| `evaluation` | `n_synthetic` | `5000` | windows to generate at eval time |

### 2 — Train (HPC / SLURM)

```bash
# Submit 3-GPU DDP training job
sbatch submit.slurm

# Watch logs
tail -f logs/slurm-<jobid>.out
```

Preprocessing runs automatically on rank 0 before training begins.
Best checkpoint (EMA weights) is saved to `checkpoints/toposynth_best.pth`.
A quick 4-sample generation sanity check writes `checkpoints/quick_gen.pt`
at the end of the job.

### 3 — Train (local / single GPU)

```bash
# Single GPU — no torchrun needed
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --config configs/default.yaml
```

> **Note:** single-GPU mode skips DDP initialisation but still preprocesses
> data and saves checkpoints identically.

### 4 — Evaluate (HPC / SLURM)

```bash
sbatch submit_eval.slurm

# Tail the evaluation log
tail -f logs/eval-<jobid>.out
```

Results are written to `experiments/eval_<timestamp>.json`.

### 5 — Evaluate (local)

```bash
python3 scripts/evaluate.py \
    --config  configs/default.yaml \
    --n_syn   5000 \
    --repeats 5 \
    --device  cuda \
    --out_dir experiments
```

---

## Evaluation metrics

All five metrics follow the protocol of Diffusion-TS (ICLR 2024) and
TSGBench (VLDB 2023). DS and PS are repeated 5 times and averaged.

| Metric | Direction | Description |
|---|---|---|
| **Discriminative Score (DS)** | ↓ (0 = perfect) | `\|AUROC − 0.5\|` of a 2-layer LSTM classifier trained to separate real from synthetic windows |
| **Predictive Score / TSTR (PS)** | ↓ | MAE of a 2-layer LSTM next-step predictor trained on synthetic, evaluated on real test windows |
| **Context-FID (CFID)** | ↓ | Fréchet distance in ts2vec embedding space (BiGRU fallback when ts2vec is absent) |
| **Correlational Score (CS)** | ↓ | Mean absolute error between the cross-correlation matrices of real and synthetic features |
| **Topology Consistency Score (TCS)** | ↓ MSE / ↑ cos | GAT-embedding MSE and per-VNF cosine similarity between real and synthetic distributions — unique to TopoSynth |

---

## Training details

| Setting | Value |
|---|---|
| GPUs | 3 × NVIDIA GH200 (ARM64) |
| Framework | PyTorch DDP, NCCL backend, torchrun |
| Optimizer | Adam, lr=1e-4, weight_decay=1e-6 |
| Grad clip | 1.0 |
| Early stopping | patience=20 epochs (val loss) |
| EMA | applied to model weights; best checkpoint uses EMA |
| Noise schedule | cosine, T=200 |
| Topology loss weight | λ_topo = 0.1 |

---

## References

- Y. Chen & Y. Qiao, "Diffusion-TS: Interpretable Diffusion for General Time Series Generation," *ICLR 2024*
- P. Veličković et al., "Graph Attention Networks," *ICLR 2018*
- Y. Dong et al., "TSGBench: Time Series Generation Benchmark," *VLDB 2024*
- Y. Li et al., "PaD-TS: Pattern and Distribution-Aware Time Series Generation," *AAAI 2025*
- R. Mijumbi et al., "Network Function Virtualization: State-of-the-Art and Research Challenges," *IEEE TNSM 2017*
