# TopoSynth

**Topology-Aware Synthetic VNF Data Generation using Diffusion-TS and GAT**

TopoSynth generates synthetic time-series data for Virtual Network Functions (VNFs)
in 5G Service Function Chains (SFCs), conditioning generation on the SFC topology
via a Graph Attention Network (GAT) encoder and gradient guidance.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Preprocess raw data
python scripts/preprocess.py --config configs/clearwater.yaml

# 2. Train
python scripts/train.py --config configs/default.yaml

# 3. Evaluate
python scripts/evaluate.py --checkpoint experiments/<run>/best.pt

# 4. Generate synthetic data
python scripts/generate.py --checkpoint experiments/<run>/best.pt --output data/synthetic.csv
```

## References
- Chen & Qiao, "Diffusion-TS", ICLR 2024
- Veličković et al., "Graph Attention Networks", ICLR 2018
- Zhang et al., IEEE Systems Journal 2023
- Mijumbi et al., IEEE TNSM 2017
