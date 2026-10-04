# toposynth/data/topology.py
# Clearwater SFC topology definition.
# All other modules import from here — single source of truth for graph structure.

import torch
import numpy as np

# ─── TOPOLOGY CONFIG ─────────────────────────────────────────────────────────

TOPOLOGY_CONFIG = {
    'vnf_names': ['bono', 'sprout', 'homestead', 'homer', 'ralf', 'ellis'],
    'vnf_to_idx': {
        'bono': 0, 'sprout': 1, 'homestead': 2,
        'homer': 3, 'ralf': 4,  'ellis': 5,
    },
    # Directed edges: (upstream, downstream) — traffic flow direction
    # Bono  → Sprout  → Homestead
    #                 → Homer
    #                 → Ralf
    # Ellis: isolated (self-provisioning portal, not on SIP call path)
    'edges': [
        ('bono',   'sprout'),
        ('sprout', 'homestead'),
        ('sprout', 'homer'),
        ('sprout', 'ralf'),
    ],
    'n_vnf':      6,
    'n_features': 12,   # selected features per VNFC (see preprocessing.py)
}

# ─── SELECTED FEATURES ───────────────────────────────────────────────────────
# 12 features per VNFC, selected from the 21 available in X_126.csv.
# Format: column prefix = '{vnfc}-{metric}'

SELECTED_METRICS = [
    'net.in_bytes_sec',    # incoming traffic — causal input from upstream VNFC
    'net.out_bytes_sec',   # outgoing traffic — causal output to downstream VNFC
    'net.in_packets_sec',  # packet-level incoming
    'net.out_packets_sec', # packet-level outgoing
    'cpu.system_perc',     # kernel/VNF processing load
    'cpu.wait_perc',       # I/O-induced CPU stall
    'mem.usable_perc',     # normalized memory utilization
    'io.read_req_sec',     # disk read request rate
    'io.write_req_sec',    # disk write request rate
    'io.read_kbytes_sec',  # disk read throughput
    'io.write_kbytes_sec', # disk write throughput
    'load.avg_1_min',      # short-term system load average
]

# Metric → column suffix mapping (matches actual CSV column names)
METRIC_TO_COL_SUFFIX = {
    'net.in_bytes_sec':    'net.in_bytes_sec',
    'net.out_bytes_sec':   'net.out_bytes_sec',
    'net.in_packets_sec':  'net.in_packets_sec',
    'net.out_packets_sec': 'net.out_packets_sec',
    'cpu.system_perc':     'cpu.system_perc',
    'cpu.wait_perc':       'cpu.wait_perc',
    'mem.usable_perc':     'mem.usable_perc',
    'io.read_req_sec':     'io.read_req_sec',
    'io.write_req_sec':    'io.write_req_sec',
    'io.read_kbytes_sec':  'io.read_kbytes_sec',
    'io.write_kbytes_sec': 'io.write_kbytes_sec',
    'load.avg_1_min':      'load.avg_1_min',
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_edge_index():
    """
    Returns edge_index as LongTensor of shape [2, num_edges].
    edge_index[0] = source (upstream) nodes
    edge_index[1] = target (downstream) nodes

    Convention: edge (src → dst) means src influences dst,
    i.e. GAT aggregates neighbours of dst from src.
    """
    src, dst = [], []
    for (u, v) in TOPOLOGY_CONFIG['edges']:
        src.append(TOPOLOGY_CONFIG['vnf_to_idx'][u])
        dst.append(TOPOLOGY_CONFIG['vnf_to_idx'][v])
    return torch.tensor([src, dst], dtype=torch.long)  # [2, E]


def get_adjacency_matrix(add_self_loops=False):
    """
    Returns dense adjacency matrix A of shape [N, N] as FloatTensor.
    A[i, j] = 1  iff there is a directed edge from VNF j → VNF i
              (i.e. j is an upstream neighbour of i, GAT convention).

    add_self_loops: if True, sets A[i,i] = 1 for all i (Ellis already
                    has no neighbours, so self-loop is its only connection).
    """
    N = TOPOLOGY_CONFIG['n_vnf']
    A = torch.zeros(N, N, dtype=torch.float)
    for (u, v) in TOPOLOGY_CONFIG['edges']:
        u_idx = TOPOLOGY_CONFIG['vnf_to_idx'][u]
        v_idx = TOPOLOGY_CONFIG['vnf_to_idx'][v]
        A[v_idx, u_idx] = 1.0  # row = dst, col = src
    if add_self_loops:
        A = A + torch.eye(N, dtype=torch.float)
    return A


def get_upstream_neighbours():
    """
    Returns dict: vnf_name -> list of upstream vnf names.
    Used by the GAT encoder to iterate over each node's neighbourhood.

    Example:
        'sprout'    -> ['bono']
        'homestead' -> ['sprout']
        'homer'     -> ['sprout']
        'ralf'      -> ['sprout']
        'bono'      -> []          # entry node
        'ellis'     -> []          # isolated node
    """
    neighbours = {v: [] for v in TOPOLOGY_CONFIG['vnf_names']}
    for (src, dst) in TOPOLOGY_CONFIG['edges']:
        neighbours[dst].append(src)
    return neighbours


def get_selected_columns():
    """
    Returns ordered list of CSV column names for the 72 selected features
    (12 metrics × 6 VNFCs), in VNF-major order.

    Column name format in X_126.csv: '{vnfc}-{metric}'
    Example: 'bono-net.in_bytes_sec', 'sprout-cpu.system_perc', ...
    """
    cols = []
    for vnf in TOPOLOGY_CONFIG['vnf_names']:
        for metric in SELECTED_METRICS:
            col = f'{vnf}-{metric}'
            cols.append(col)
    return cols


def print_topology_summary():
    """Pretty-print the SFC topology for logging."""
    print("=" * 60)
    print("Clearwater SFC Topology")
    print("=" * 60)
    neighbours = get_upstream_neighbours()
    for vnf in TOPOLOGY_CONFIG['vnf_names']:
        idx  = TOPOLOGY_CONFIG['vnf_to_idx'][vnf]
        ups  = neighbours[vnf]
        role = _vnf_role(vnf)
        arrow = f"← {', '.join(ups)}" if ups else "(entry/isolated)"
        print(f"  [{idx}] {vnf:<12} {role:<30}  {arrow}")
    print(f"\nEdges : {TOPOLOGY_CONFIG['edges']}")
    print(f"VNFs  : {TOPOLOGY_CONFIG['n_vnf']}")
    print(f"Feats : {TOPOLOGY_CONFIG['n_features']} per VNF  "
          f"({TOPOLOGY_CONFIG['n_vnf'] * TOPOLOGY_CONFIG['n_features']} total)")
    A = get_adjacency_matrix()
    print(f"\nAdjacency matrix (row=dst, col=src):\n{A.numpy().astype(int)}")
    print("=" * 60)


def _vnf_role(name):
    roles = {
        'bono':      'SIP edge proxy (P-CSCF)',
        'sprout':    'SIP registrar (S-CSCF)',
        'homestead': 'HSS cache',
        'homer':     'XDMS service config',
        'ralf':      'CDR charging trigger',
        'ellis':     'Provisioning portal (isolated)',
    }
    return roles.get(name, '')


# ─── SANITY CHECK ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print_topology_summary()

    edge_index = get_edge_index()
    print(f"\nedge_index shape : {edge_index.shape}")
    print(f"edge_index       :\n{edge_index}")

    cols = get_selected_columns()
    print(f"\nSelected columns ({len(cols)}):")
    for i, c in enumerate(cols):
        print(f"  {i:>3}  {c}")