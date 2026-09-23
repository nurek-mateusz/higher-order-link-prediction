import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from itertools import combinations
from matplotlib.pylab import solve
import numpy as np
import pandas as pd
import math
import pickle
import os
from datetime import datetime
import scipy.sparse as sp
from scipy.sparse import linalg as spla
import networkx as nx
from utils import load_trg, load_simplices


def process_triangle_worker(batch, simplices, node_to_simplices, node_to_times, node_time_to_neighbors, pair_to_times,
                            node_to_times_bins, node_time_to_neighbors_bins, G):
    try:
        results = []

        # r_cache = {}
        # nodes = list(G.nodes())
        # node_to_index = {n: i for i, n in enumerate(nodes)}
        # A = nx.adjacency_matrix(G, nodelist=nodes).astype(float).tocsr()
        # solver = build_ppr_solver(A)
        T_end = max(ts[-1] for ts in pair_to_times.values())
        T_start = min(ts[0] for ts in pair_to_times.values())
        time_span = T_end - T_start

        for triangle in batch:
            # BENSON
            swa, swg, swh = calculate_benson(triangle, node_to_simplices)
            sda, sdg, sdh = calculate_degree_benson(triangle, node_to_simplices)

            # HEURISTICS
            hcn = calculate_hcn_tunion(triangle, G)
            hpa = calculate_hpa(triangle, G)
            haa = calculate_haa(triangle, G)
            # hppr = calculate_hppr(triangle, A, node_to_index, alpha=0.85, r_cache=r_cache, solve=solver)

            # STRUCTURE
            degree_min, degree_med, degree_max = calculate_degrees(triangle, node_to_simplices)
            weight_min, weight_med, weight_max = calculate_weights(triangle, node_to_simplices)
            degree_balance = calculate_degree_balance(triangle, node_to_simplices)
            weight_balance = calculate_weight_balance(triangle, node_to_simplices)

            # TIME
            iet_min, iet_med, iet_max = calculate_interevent_time_pair(triangle, pair_to_times, time_span)
            # iet_balance = gini_coefficient([iet_min, iet_med, iet_max])
            iet_triangle = calculate_interevent_time_triangle(triangle, pair_to_times, time_span)
            burstiness_min, burstiness_med, burstiness_max = calculate_burstiness_pair(triangle, pair_to_times)
            # burstiness_balance = gini_coefficient([burstiness_min, burstiness_med, burstiness_max])
            burstiness_triangle = calculate_burstiness_triangle(triangle, pair_to_times)
            thcn = calculate_THCN(triangle, node_to_times_bins, node_time_to_neighbors_bins)
            age, recency = calculate_age_recency(triangle, pair_to_times, T_end)

            features = {
                'triangle': triangle,
                # WEIGHT-BASED
                'swa': swa, 
                'swg': swg,
                'swh': swh,
                'weight_min': weight_min,
                'weight_med': weight_med,
                'weight_max': weight_max,
                'weight_balance': weight_balance,
                # LOCAL-HEURISTICS
                'hcn': hcn,
                'hpa': hpa,
                'haa': haa,
                'thcn': thcn,
                # DEGREE-BASED
                'sda': sda,
                'sdg': sdg,
                'sdh': sdh,
                'degree_min': degree_min,
                'degree_med': degree_med,
                'degree_max': degree_max,
                'degree_balance': degree_balance,
                # TIME/DYNAMICS
                'iet_min': iet_min,
                'iet_med': iet_med,
                'iet_max': iet_max,
                'iet_triangle': iet_triangle,
                'burstiness_min': burstiness_min,
                'burstiness_med': burstiness_med,
                'burstiness_max': burstiness_max,
                'burstiness_triangle': burstiness_triangle,
                'age': age,
                'recency': recency,
            }
            results.append(features)
        return results
    except Exception as e:
        return [{'error': str(e), 'triangle': triangle}]

# feature 1: neighborhood-based indicators
# 1-1 HCN: Higher-Order Common Neighbors;
# 1-2 neighbor motifs


# BENSON FEATURES
def calculate_benson(triangle, node_to_simplices):
    node_a, node_b, node_c = triangle
    
    # Calculate weights of the three internal edges of the triangle
    weight_ab = len(node_to_simplices.get(node_a, set()) & node_to_simplices.get(node_b, set()))
    weight_ac = len(node_to_simplices.get(node_a, set()) & node_to_simplices.get(node_c, set()))
    weight_bc = len(node_to_simplices.get(node_b, set()) & node_to_simplices.get(node_c, set()))

    swa = (weight_ab + weight_ac + weight_bc) / 3.0
    swg = (weight_ab * weight_ac * weight_bc) ** (1/3)
    swh = 0.0
    if weight_ab > 0 and weight_ac > 0 and weight_bc > 0:
        swh = 3.0 / (1.0/weight_ab + 1.0/weight_ac + 1.0/weight_bc)

    return swa, swg, swh

# HEURISTICS
def calculate_hcn_tunion(triangle, G):
    """HCN (time-union version): intersection of neighbors aggregated over all timestamps."""
    a, b, c = triangle
    na = set(G.neighbors(a)) if a in G else set()
    nb = set(G.neighbors(b)) if b in G else set()
    nc = set(G.neighbors(c)) if c in G else set()
    return float(len(na & nb & nc))

def calculate_hpa(triangle, G):
    """Higher-order preferential attachment."""
    a, b, c = triangle
    deg_a = G.degree(a) if a in G else 0
    deg_b = G.degree(b) if b in G else 0
    deg_c = G.degree(c) if c in G else 0
    return float(deg_a * deg_b * deg_c)

def calculate_haa(triangle, G):
    """Higher-order Adamic-Adar (static)."""
    a, b, c = triangle
    na = set(G.neighbors(a)) if a in G else set()
    nb = set(G.neighbors(b)) if b in G else set()
    nc = set(G.neighbors(c)) if c in G else set()
    common = na & nb & nc

    score = 0.0
    for z in common:
        deg_z = G.degree(z)
        if deg_z > 1:
            score += 1.0 / np.log(deg_z)
    return float(score)

def build_ppr_solver(A, alpha=0.85):
    """Factorize (I - alpha * P^T) once per run and return the solver callable.

    r_s = (1 - alpha) * (I - alpha * P^T)^-1 * e_s,  P = D^-1 A row-normalized.
    The sparse LU factorization is the dominant cost, so it must be computed
    exactly once and reused for every source node / triangle.
    """
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(deg, dtype=float)
    nonzero = deg > 0
    inv_deg[nonzero] = 1.0 / deg[nonzero]
    P = sp.diags(inv_deg) @ A
    M = sp.eye(n, format="csr") - alpha * P.T
    return spla.factorized(M.tocsc())


def solve_ppr_for_sources(A, source_nodes, alpha=0.85, r_cache=None, solve=None):
    """Solve personalized PageRank equations for several sources.

    ``solve`` is the LU-factorized solver from ``build_ppr_solver``; it is built
    once per run and shared across all triangles. Results are cached by source
    index so that nodes appearing in many triangles are solved only once.
    """
    n = A.shape[0]
    if solve is None:
        solve = build_ppr_solver(A, alpha)
    scale = 1.0 - alpha

    r_dict = {}
    for s in source_nodes:
        if r_cache is not None and s in r_cache:
            r_dict[s] = r_cache[s]
            continue
        b = np.zeros(n, dtype=float)
        b[s] = scale
        r = solve(b)
        r_dict[s] = r
        if r_cache is not None:
            r_cache[s] = r
    return r_dict


def calculate_hppr(triangle, A, node_to_index, alpha=0.85, r_cache=None, solve=None):
    """Higher-order Personalized PageRank (exact): sum of the six directed
    PPR entries between the triangle's nodes, s(HPPR) = F_ab + F_ac + F_bc
    + F_ba + F_cb + F_ca.

    ``node_to_index`` maps original node ids to rows of ``A``; this mapping is
    required because ``nx.adjacency_matrix(G, nodelist=nodes)`` is ordered by
    ``nodes``, not by node id.
    alpha: continuation probability in PPR (common choice 0.85)
    """
    a, b, c = triangle
    ia = node_to_index.get(a)
    ib = node_to_index.get(b)
    ic = node_to_index.get(c)
    if ia is None or ib is None or ic is None:
        return 0.0

    r_dict = solve_ppr_for_sources(A, {ia, ib, ic}, alpha=alpha, r_cache=r_cache, solve=solve)
    return float(
        r_dict[ia][ib]
        + r_dict[ia][ic]
        + r_dict[ib][ia]
        + r_dict[ib][ic]
        + r_dict[ic][ia]
        + r_dict[ic][ib]
    )

# STRUCTURE
def calculate_degree_benson(triangle, node_to_simplices):
    node_a, node_b, node_c = triangle
    
    # Calculate the degree influence of nodes
    degree_a = len(node_to_simplices.get(node_a, set()))
    degree_b = len(node_to_simplices.get(node_b, set()))
    degree_c = len(node_to_simplices.get(node_c, set()))

    sda = (degree_a + degree_b + degree_c) / 3.0
    sdg = (degree_a * degree_b * degree_c) ** (1/3)
    sdh = 0.0
    if degree_a > 0 and degree_b > 0 and degree_c > 0:
        sdh = 3.0 / (1.0/degree_a + 1.0/degree_b + 1.0/degree_c)

    return sda, sdg, sdh

def calculate_degrees(triangle, node_to_simplices):
    node_a, node_b, node_c = triangle
    
    # Calculate the degree influence of nodes
    degree_a = len(node_to_simplices.get(node_a, set()))
    degree_b = len(node_to_simplices.get(node_b, set()))
    degree_c = len(node_to_simplices.get(node_c, set()))

    degree_min, degree_med, degree_max = sorted([degree_a, degree_b, degree_c])
    
    return degree_min, degree_med, degree_max

def calculate_weights(triangle, node_to_simplices):
    node_a, node_b, node_c = triangle
    
    # Calculate weights of the three internal edges of the triangle
    weight_ab = len(node_to_simplices.get(node_a, set()) & node_to_simplices.get(node_b, set()))
    weight_ac = len(node_to_simplices.get(node_a, set()) & node_to_simplices.get(node_c, set()))
    weight_bc = len(node_to_simplices.get(node_b, set()) & node_to_simplices.get(node_c, set()))

    weight_min, weight_med, weight_max = sorted([weight_ab, weight_ac, weight_bc])

    return weight_min, weight_med, weight_max

def calculate_degree_balance(triangle, node_to_simplices):
    node_a, node_b, node_c = triangle
    degrees = [
        len(node_to_simplices.get(node_a, set())),
        len(node_to_simplices.get(node_b, set())),
        len(node_to_simplices.get(node_c, set()))
    ]

    # Calculate structural imbalance (Gini coefficient)
    return gini_coefficient(degrees)

def calculate_weight_balance(triangle, node_to_simplices):
    node_a, node_b, node_c = triangle
    
    # Calculate weights of the three internal edges of the triangle
    weight_ab = len(node_to_simplices.get(node_a, set()) & node_to_simplices.get(node_b, set()))
    weight_ac = len(node_to_simplices.get(node_a, set()) & node_to_simplices.get(node_c, set()))
    weight_bc = len(node_to_simplices.get(node_b, set()) & node_to_simplices.get(node_c, set()))
    
    weights = [weight_ab, weight_ac, weight_bc]

    # Calculate structural imbalance (Gini coefficient)
    return gini_coefficient(weights)

# TIME
def mean_log_iet_pair(node1, node2, pair_to_times, time_span):
    cooccurrence_times = sorted(set(pair_to_times.get(tuple(sorted((node1, node2))), [])))

    if len(cooccurrence_times) <= 1:
        return np.log1p(time_span)

    interevent_times = np.diff(cooccurrence_times)
    log_interevent_times = np.log1p(interevent_times)
    return np.mean(log_interevent_times)
    
def calculate_interevent_time_pair(triangle, pair_to_times, time_span):
    node_a, node_b, node_c = triangle

    iet_ab = mean_log_iet_pair(node_a, node_b, pair_to_times, time_span)
    iet_ac = mean_log_iet_pair(node_a, node_c, pair_to_times, time_span)
    iet_bc = mean_log_iet_pair(node_b, node_c, pair_to_times, time_span)

    iet_min, iet_med, iet_max = sorted([iet_ab, iet_ac, iet_bc])

    return iet_min, iet_med, iet_max

def calculate_interevent_time_triangle(triangle, pair_to_times, time_span):
    node1, node2, node3 = triangle

    cooccurrence_times_1_2 = pair_to_times.get(tuple(sorted((node1, node2))), [])
    cooccurrence_times_2_3 = pair_to_times.get(tuple(sorted((node2, node3))), [])
    cooccurrence_times_1_3 = pair_to_times.get(tuple(sorted((node1, node3))), [])

    cooccurrence_times = cooccurrence_times_1_2 + cooccurrence_times_2_3 + cooccurrence_times_1_3
    cooccurrence_times = sorted(set(cooccurrence_times))

    if len(cooccurrence_times) <= 1:
        return np.log1p(time_span)
    
    interevent_times = np.diff(cooccurrence_times)
    log_interevent_times = np.log1p(interevent_times)
    return np.mean(log_interevent_times)

def burstiness(timestamps):
    if len(timestamps) < 3:
        return 0.0

    iets = np.diff(timestamps)

    mu = np.mean(iets)
    sigma = np.std(iets, ddof=1)

    if mu + sigma == 0:
        return 0.0

    # Goha-Barabási burstiness measure
    return (sigma - mu) / (sigma + mu)

def calculate_burstiness_pair(triangle, pair_to_times):
    node1, node2, node3 = triangle

    cooccurrence_times_1_2 = sorted(set(pair_to_times.get(tuple(sorted((node1, node2))), [])))
    cooccurrence_times_2_3 = sorted(set(pair_to_times.get(tuple(sorted((node2, node3))), [])))
    cooccurrence_times_1_3 = sorted(set(pair_to_times.get(tuple(sorted((node1, node3))), [])))

    burstiness_1_2 = burstiness(cooccurrence_times_1_2)
    burstiness_2_3 = burstiness(cooccurrence_times_2_3)
    burstiness_1_3 = burstiness(cooccurrence_times_1_3)

    burstiness_min, burstiness_med, burstiness_max = sorted([burstiness_1_2, burstiness_2_3, burstiness_1_3], key=lambda x: (np.isnan(x), x))

    return burstiness_min, burstiness_med, burstiness_max

def calculate_burstiness_triangle(triangle, pair_to_times):
    node1, node2, node3 = triangle

    cooccurrence_times_1_2 = sorted(set(pair_to_times.get(tuple(sorted((node1, node2))), [])))
    cooccurrence_times_2_3 = sorted(set(pair_to_times.get(tuple(sorted((node2, node3))), [])))
    cooccurrence_times_1_3 = sorted(set(pair_to_times.get(tuple(sorted((node1, node3))), [])))

    cooccurrence_times = cooccurrence_times_1_2 + cooccurrence_times_2_3 + cooccurrence_times_1_3
    cooccurrence_times = sorted(set(cooccurrence_times))

    burstiness_triangle = burstiness(cooccurrence_times)

    return burstiness_triangle

def calculate_THCN(triangle, node_to_times, node_time_to_neighbors):
    # temporal higher-order common neighbor (THCN)
    a,b,c = triangle
    ta = node_to_times.get(a, set())
    tb = node_to_times.get(b, set())
    tc = node_to_times.get(c, set())
    t_all = ta|tb|tc

    CN_set = set()
    for t in t_all:
        na_t = node_time_to_neighbors.get((a, t), set())
        nb_t = node_time_to_neighbors.get((b, t), set())
        nc_t = node_time_to_neighbors.get((c, t), set())
        CN_set.update(na_t & nb_t & nc_t)
    return len(CN_set)

def calculate_age_recency(triangle, pair_to_times, T_end):
    pairs = [tuple(sorted(p)) for p in combinations(triangle, 2)]
    times = [pair_to_times.get(p) for p in pairs]

    if any(t is None or len(t) == 0 for t in times):
        return 0.0, 0.0
    
    t_first = max(t[0] for t in times)
    t_last = max(t[-1] for t in times)

    return (float(np.log1p(max(0, T_end - t_first))),
            float(np.log1p(max(0, T_end - t_last))))

# optional: Gini coefficient calculation
def gini_coefficient(values, eps=1e-12):
    x = np.array(values, dtype=float)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    mean = x.mean()
    if mean <= 0:
        mean = eps
    diffs = np.abs(x[:, None] - x[None, :])
    return diffs.sum() / (2.0 * (x.size**2) * mean)


class DataPreparation:
    def __init__(self, use_multiprocessing=True, n_workers=None):
        self.use_multiprocessing = use_multiprocessing
        self.n_workers = n_workers or multiprocessing.cpu_count()
        
        # Data structures
        self.simplices = []
        self.node_to_simplices = defaultdict(set)
        self.simplex_to_time = {}
        self.node_labels = {}
        self.edge_times = defaultdict(list)
        self.adjacency_list = defaultdict(set)
        self.node_to_times = defaultdict(set)
        self.node_time_to_neighbors = defaultdict(set)
        self.pair_to_times = defaultdict(set)
        self.node_to_times_bins = defaultdict(set)
        self.node_time_to_neighbors_bins = defaultdict(set)

        self.training_triangles = []
        self.test_pairs = []
        self.features_computed = False
        self.G = None
    
    def build_data_structures(self, dataset, suffix):
        data = load_simplices(f'../data/processed/{dataset}/simplices_{suffix}.parquet')

        with open(f'../data/processed/{dataset}/G_{suffix}.pickle', 'rb') as file:
            self.G = pickle.load(file)

        nverts = []
        simplices = []
        times = []
        for time, simplex in data:
            nverts.append(len(simplex))
            simplices.extend(simplex)
            times.append(time)

        df = pd.read_csv('../results/summary_best_bins_THCN.csv')
        best_bin = df.loc[df["dataset"] == dataset, "best_bin"].iloc[0]

        data = {'nverts': nverts, 'simplices': simplices, 'times': times, 'best_bin': best_bin}
        
        self._build_simplex_data(data)
        self._build_node_labels_from_simplices()
        self._build_node_simplex_mapping()
        self._build_edge_and_adjacency_data()
        self._build_node_to_times()
        self._build_node_time_to_neighbors()
        self._build_pair_to_times()
        self._build_temporal_neighbor_bin(best_bin)
    
    def _build_node_labels_from_simplices(self):
        # Extract unique node IDs from all simplices
        unique_nodes = set()
        for simplex in self.simplices:
            unique_nodes.update(simplex['nodes'])
        
        # Create node label mapping (node ID -> node label)
        # Here we directly use node ID as label
        self.node_labels = {node_id: f"Node_{node_id}" for node_id in sorted(unique_nodes)}
    
    def _build_simplex_data(self, data):
        nverts = data['nverts']
        simplices_flat = data['simplices']  
        times = data['times']
        
        # Simplices data
        self.simplices = []
        current_pos = 0
        
        for i, (nvert, time) in enumerate(zip(nverts, times)):
            # Extract nodes for the current simplex
            nodes = simplices_flat[current_pos:current_pos + nvert]
            current_pos += nvert
            
            # Create simplex object
            simplex = {
                'id': i,
                'nodes': tuple(sorted(nodes)),  # Keep nodes sorted
                'time': time,
                'order': nvert - 1  # simplex order (edge=0, triangle=1, tetrahedron=2, ...)
            }
            
            self.simplices.append(simplex)
            self.simplex_to_time[i] = time
        
        # Count simplex order distribution
        order_counts = {}
        for simplex in self.simplices:
            order = simplex['order']
            order_counts[order] = order_counts.get(order, 0) + 1
    
    def _build_node_simplex_mapping(self):
        self.node_to_simplices = defaultdict(set)
        
        for simplex in self.simplices:
            simplex_id = simplex['id']
            for node in simplex['nodes']:
                self.node_to_simplices[node].add(simplex_id)
    
    def _build_edge_and_adjacency_data(self):        
        self.edge_times = defaultdict(list)
        self.adjacency_list = defaultdict(set)
        
        for simplex in self.simplices:
            nodes = simplex['nodes']
            timestamp = simplex['time']
            
            # Build adjacency list
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if i != j:
                        self.adjacency_list[node_i].add(node_j)
            
            # Build edge timelines
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    edge_key = tuple(sorted([nodes[i], nodes[j]]))
                    self.edge_times[edge_key].append(timestamp)

    def _build_node_to_times(self):
        for simplex in self.simplices:
            for node in simplex['nodes']:
                self.node_to_times[node].add(simplex['time'])

    def _build_node_time_to_neighbors(self):
        for simplex in self.simplices:
            nodes = simplex['nodes']
            time = simplex['time']
            for node in nodes:
                self.node_time_to_neighbors[(node, time)].update(nodes)
        
        # Remove self from neighbors for each entry
        for (node, time), neighbors in self.node_time_to_neighbors.items():
            neighbors.discard(node)

    def _build_pair_to_times(self):
        for simplex in self.simplices:
            nodes = simplex['nodes']
            timestamp = simplex['time']
            
            # Generate all unique unordered pairs from nodes in this simplex
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    pair = tuple(sorted((nodes[i], nodes[j])))
                    self.pair_to_times[pair].add(timestamp)
        
        # Convert sets to sorted lists for faster later operations
        for pair in self.pair_to_times:
            self.pair_to_times[pair] = sorted(self.pair_to_times[pair])

    def _build_ts_to_bin(self, bin):
        if self.simplices is None:
            return {}

        ts = sorted({simplex['time'] for simplex in self.simplices})
        if not ts:
            return {}

        min_ts = ts[0]
        max_ts = ts[-1]
        range_ts = max_ts - min_ts + 1

        if bin is None or bin <= 0:
            return {t: 0 for t in ts}  # all timestamps in one bin
        if bin >= range_ts:
            return {t: i for i, t in enumerate(ts)}  # each timestamp gets its own bin id

        # fixed-width binning
        # empty bins are allowed (timestamps may be sparse)
        ts_to_bin = {}
        for t in ts:
            b = int((t - min_ts) / float(range_ts) * bin)
            if b >= bin:  # edge case for max_ts
                b = bin - 1
            ts_to_bin[t] = b
        return ts_to_bin

    def _build_temporal_neighbor_bin(self, bin):
        # Build temporal neighbors for each node based on simplices and timestamp bins.
        # Output:
        # - node_to_times: dict mapping node to set of time bins it appears in, e.g., {node1: {0, 1}, node2: {1}}
        # - node_time_to_neighbors: dict mapping (node, time_bin) to set of neighboring nodes in the same time bin, e.g., {(node1, 0): {node2, node3}, (node1, 1): {node4}, (node2, 1): {node1}}
        node_to_times = {}
        node_time_to_neighbors = {}
        # ts_to_bin is a dict mapping timestamp to bin id, e.g., {0: 0, 1: 0, 2: 1, 3: 1} for bin=2
        ts_to_bin = self._build_ts_to_bin(bin)
        for simplex in self.simplices:
            ts = int(simplex['time'])  # timestamp is the first element of simplex
            nodes = simplex['nodes']  # the rest are nodes list in the simplex
            if len(nodes) < 2:
                continue  # skip simplices with less than 2 nodes
            t_bin = ts_to_bin.get(ts, 0) 
            for u in nodes:
                node_to_times.setdefault(u, set()).add(t_bin)
            for u,v in combinations(nodes, 2):
                node_time_to_neighbors.setdefault((u, t_bin), set()).add(v)
                node_time_to_neighbors.setdefault((v, t_bin), set()).add(u)
        
        self.node_to_times_bins = node_to_times
        self.node_time_to_neighbors_bins = node_time_to_neighbors
    
    def calculate_triangle_features(self, triangles):
        """
        Args:
            triangles: List of triangles
        
        Returns:
            list: List of feature dictionaries
        """
        print(f"Calculating features for {len(triangles)} triangles...")
        features = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            batch_size = math.ceil(len(triangles) / self.n_workers)
            batches = [triangles[i:i+batch_size] for i in range(0, len(triangles), batch_size)]
            print(f"Batch size: {batch_size}, # batches: {len(batches)}")

            futures = [executor.submit(process_triangle_worker, 
                                       batch, 
                                       self.simplices, 
                                       self.node_to_simplices,
                                       self.node_to_times,
                                       self.node_time_to_neighbors,
                                       self.pair_to_times,
                                       self.node_to_times_bins,
                                       self.node_time_to_neighbors_bins,
                                       self.G) for batch in batches]
           
            for future in futures:
                result = future.result()
                if 'error' in result[0]:
                    print(f"Error processing triangle {result[0]['triangle']}: {result[0]['error']}")
                else:
                    features.extend(result)

        
        print(f"Processed {len(features)}/{len(triangles)} successfully")
        return pd.DataFrame(features)


if __name__ == '__main__':
    dataset_list = ['coauth-MAG-Geology', 'coauth-MAG-History', 'contact-high-school', 'contact-primary-school', 'email-Enron', 'email-Eu', 
                    'NDC-classes', 'NDC-substances', 'threads-ask-ubuntu', 'tags-ask-ubuntu']

    for dataset in dataset_list:
        for suffix in ['train', 'val', 'test']:
            print(dataset, ' ', suffix, ' ', datetime.now().strftime('%d-%m-%Y %H:%M:%S'))

            dp = DataPreparation()
            dp.build_data_structures(dataset, suffix)
            
            candidates = load_trg(f'../data/processed/{dataset}/trg_open_{suffix}.parquet')

            features = dp.calculate_triangle_features(candidates)

            save_dir = '../data/features/' + dataset
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            
            features.to_parquet(save_dir + f'/features_{suffix}.parquet', compression='zstd', index=False)