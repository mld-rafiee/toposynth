"""
toposynth/data/topology.py
==========================
Single source of truth for the Clearwater IMS topology used in TopoSynth.

Loads vnf_names and edges from configs/default.yaml at import time so that
every module that imports TOPOLOGY_CONFIG or get_upstream_neighbours()
always reflects the current config without any code changes.

Design notes
------------
* 14 directed edges (two per documented inter-VNF communication link).
  Each direction is an independent channel; traffic volume, timing, and
  packet counts are NOT symmetric.  This differs from Jalodia et al.
  (2022), who used an undirected graph, because TopoSynth explicitly
  models directional traffic dynamics.
* The topology is architecture-derived (Project Clearwater documentation),
  NOT inferred from the Kaggle dataset, which provides per-VNF node-level
  in/out telemetry rather than per-edge source/destination flow records.
* The 'homestead' node in this abstraction subsumes the runtime Homestead
  component and the Homestead-Prov API endpoint that Ellis talks to.

References
----------
[1] Project Clearwater Architecture
    https://clearwater.readthedocs.io/en/stable/Clearwater_Architecture.html
[2] Project Clearwater Geographic Redundancy
    https://clearwater.readthedocs.io/en/no_github/Geographic_redundancy.html
[3] Sauvanaud et al., ISSRE 2016 — six-component Clearwater testbed
[4] Jalodia et al., 2022 — GNN on this dataset (undirected formulation)
"""

import os
import yaml

# ── Config path (repo_root/configs/default.yaml) ──────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_HERE, '..', '..', 'configs', 'default.yaml')


# ── Feature selection ──────────────────────────────────────────────────────────
# 12 of 21 available per-VNF metrics, chosen for dynamic range,
# low inter-metric redundancy, and relevance to topology-conditioned synthesis.
#
# Rationale (all 21 evaluated):
#   CPU    : idle + system  (wait_perc dropped — correlated with io.* metrics)
#   Memory : usable_perc + usable_mb  (free_mb dropped — dominated by usable_mb)
#   Network: all 4 (in/out bytes + packets) — primary signal for directed edges
#   IO     : read/write kbytes/sec  (req/sec & latency dropped — correlated or near-static)
#   Disk   : space_used_perc  (inode_used_perc near-constant for VNF containers)
#   Load   : avg_1_min  (5/15-min collinear with 1-min and CPU metrics)
SELECTED_METRICS: list[str] = [
    'cpu.idle_perc',
    'cpu.system_perc',
    'mem.usable_perc',
    'mem.usable_mb',
    'net.in_bytes_sec',
    'net.out_bytes_sec',
    'net.in_packets_sec',
    'net.out_packets_sec',
    'io.read_kbytes_sec',
    'io.write_kbytes_sec',
    'disk.space_used_perc',
    'load.avg_1_min',
]  # len == 12  (== data.n_features in configs/default.yaml)


def _load_topology(config_path: str = _CONFIG_PATH) -> dict:
    """
    Parse the [topology] section of the YAML config and return a dict with:
      vnf_names  : list[str]          ordered VNF / node names
      vnf_to_idx : dict[str, int]     name → integer index
      edges      : list[tuple[str,str]] directed edge pairs (src, dst)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Topology config not found: {config_path}\n"
            "Make sure configs/default.yaml exists relative to the repo root."
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    topo = cfg.get('topology', {})

    vnf_names = topo.get('vnfcs', [
        'bono', 'sprout', 'homestead', 'homer', 'ralf', 'ellis',
    ])

    raw_edges = topo.get('edges', [])
    edges = [(str(e[0]), str(e[1])) for e in raw_edges]

    return {
        'vnf_names':  vnf_names,
        'vnf_to_idx': {v: i for i, v in enumerate(vnf_names)},
        'edges':      edges,
        'n_vnf':      len(vnf_names),           # convenience alias used by preprocessing
        'n_features': len(SELECTED_METRICS),    # == 12; matches data.n_features in config
    }


# Module-level singleton — built once at import time.
TOPOLOGY_CONFIG: dict = _load_topology()


def get_selected_columns() -> list[str]:
    """
    Return the flat list of the 72 column names expected in the CSV,
    ordered as [vnf0-metric0, vnf0-metric1, …, vnf5-metric11].

    The CSV uses '{vnf}-{metric}.csv' headers; the '.csv' suffix is
    stripped during preprocessing so the column keys match this format.

    Example (first three):
        'bono-cpu.idle_perc', 'bono-cpu.system_perc', 'bono-mem.usable_perc', …
    """
    vnf_names = TOPOLOGY_CONFIG['vnf_names']
    return [f'{vnf}-{metric}' for vnf in vnf_names for metric in SELECTED_METRICS]


def get_upstream_neighbours() -> dict:
    """
    Return a dict mapping each VNF name to its list of upstream neighbours.

    Upstream neighbours of node i are all nodes j such that a directed edge
    j → i exists in the topology.  The GAT uses this to decide which nodes
    participate in the attention aggregation for each target node i.

    With the 14-edge directed topology every node has ≥ 2 upstream
    neighbours, so the old 'entry node bypass' is no longer needed.

    Example (14-edge Clearwater topology)
    --------------------------------------
    bono       ← [sprout, ralf]
    sprout     ← [bono, homestead, homer, ralf]
    homestead  ← [sprout, ellis]
    homer      ← [sprout, ellis]
    ralf       ← [bono, sprout]
    ellis      ← [homestead, homer]
    """
    vnf_names = TOPOLOGY_CONFIG['vnf_names']
    edges     = TOPOLOGY_CONFIG['edges']

    neighbours: dict = {vnf: [] for vnf in vnf_names}
    for src, dst in edges:
        if dst in neighbours:
            neighbours[dst].append(src)

    return neighbours


def get_downstream_neighbours() -> dict:
    """
    Return a dict mapping each VNF name to nodes it sends to (j → i, from i).
    Not used by the GAT directly, but useful for visualisation and debugging.
    """
    vnf_names = TOPOLOGY_CONFIG['vnf_names']
    edges     = TOPOLOGY_CONFIG['edges']

    neighbours: dict = {vnf: [] for vnf in vnf_names}
    for src, dst in edges:
        if src in neighbours:
            neighbours[src].append(dst)

    return neighbours


def get_adjacency_matrix(directed: bool = True, add_self_loops: bool = False):
    """
    Return the adjacency matrix as a list-of-lists (float).
    Alias for adjacency_matrix() that also supports the add_self_loops kwarg
    used by dataset.py.

    directed=True        : A[i][j] = 1 if edge i→j exists
    directed=False       : A[i][j] = 1 if either i→j or j→i exists
    add_self_loops=True  : additionally set A[i][i] = 1 for all i
    """
    A = adjacency_matrix(directed)
    if add_self_loops:
        for i in range(len(A)):
            A[i][i] = 1.0
    return A


def adjacency_matrix(directed: bool = True):
    """
    Return the adjacency matrix as a list-of-lists (float).

    directed=True  : A[i][j] = 1 if edge i→j exists (as in TOPOLOGY_CONFIG)
    directed=False : A[i][j] = 1 if either i→j or j→i exists
    """
    n         = len(TOPOLOGY_CONFIG['vnf_names'])
    idx       = TOPOLOGY_CONFIG['vnf_to_idx']
    A         = [[0.0] * n for _ in range(n)]

    for src, dst in TOPOLOGY_CONFIG['edges']:
        i, j = idx[src], idx[dst]
        A[i][j] = 1.0
        if not directed:
            A[j][i] = 1.0

    return A