import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import math
import pickle
import os
from datetime import datetime


def process_triangle_worker(batch, simplices, node_to_simplices, node_to_times, node_time_to_neighbors, pair_to_times,
                            node_to_times_bins, node_time_to_neighbors_bins):
    try:
        results = []
        for triangle in batch:
            features = {
                'triangle': triangle,
                'hcn': calculate_THCN(triangle, node_to_times_bins, node_time_to_neighbors_bins),
                # 1-2 neighbor motifs
                'degree_reinforcement': calculate_degree_reinforcement(triangle, node_to_simplices),
                'weight_reinforcement': calculate_weight_reinforcement(triangle, node_to_simplices),
                'pairwise_timescale_density': calculate_pairwise_timescale_density(triangle, pair_to_times),
                'timescale_density_balance': calculate_timescale_density_balance(triangle, pair_to_times),
                'degree_balance': calculate_degree_balance(triangle, node_to_simplices),
                'weight_balance': calculate_weight_balance(triangle, node_to_simplices),
                'lifetime_one_edge': calculate_lifetime(triangle, pair_to_times),
                'lifetime_two_edges': calculate_lifetime_2(triangle, pair_to_times),
            }
            results.append(features)
        return results
    except Exception as e:
        return [{'error': str(e), 'triangle': triangle}]

# feature 1: neighborhood-based indicators
# 1-1 HCN: Higher-Order Common Neighbors;
# 1-2 neighbor motifs

# 1-1 HCN
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

def calculate_HCN(triangle, simplices, node_to_times, node_time_to_neighbors):
    node_a, node_b, node_c = triangle
    
    # Get all timestamps for the three nodes
    timestamps_a = node_to_times.get(node_a, set())
    timestamps_b = node_to_times.get(node_b, set())
    timestamps_c = node_to_times.get(node_c, set())
    
    # All relevant timestamps
    all_timestamps = timestamps_a | timestamps_b | timestamps_c
    
    if not all_timestamps:
        return 0.0
    
    hcn_sum = 0.0
    # valid_timestamps = 0
    
    for timestamp in all_timestamps:
        # Get neighbors at time
        neighbors_a = node_time_to_neighbors.get((node_a, timestamp), set())
        neighbors_b = node_time_to_neighbors.get((node_b, timestamp), set())
        neighbors_c = node_time_to_neighbors.get((node_c, timestamp), set())
        
        # Calculate common neighbors and all neighbors
        common_neighbors = neighbors_a & neighbors_b & neighbors_c
        all_neighbors = neighbors_a | neighbors_b | neighbors_c
        
        if len(all_neighbors) > 0:
            hcn_sum += len(common_neighbors) / len(all_neighbors)
            # valid_timestamps += 1 # ignore time

    return hcn_sum
   
# feature 2: internal activeness (with time dimension)
# 2-1 degree reinforcement
# 2-2 weight reinforcement
# 2-3 pairwise timescale density

# 2-1 degree reinforcement
def calculate_degree_reinforcement(triangle, node_to_simplices):
    node_a, node_b, node_c = triangle
    
    # Calculate the degree influence of nodes
    degree_a = len(node_to_simplices.get(node_a, set()))
    degree_b = len(node_to_simplices.get(node_b, set()))
    degree_c = len(node_to_simplices.get(node_c, set()))
    
    # Use geometric mean
    if degree_a > 0 and degree_b > 0 and degree_c > 0:
        return (degree_a * degree_b * degree_c) ** (1/3)
    else:
        return 0.0

# 2-2 weight reinforcement
def calculate_weight_reinforcement(triangle, node_to_simplices):
    node_a, node_b, node_c = triangle

    # Calculate weights of the three internal edges of the triangle
    weight_ab = len(node_to_simplices.get(node_a, set()) & node_to_simplices.get(node_b, set()))
    weight_ac = len(node_to_simplices.get(node_a, set()) & node_to_simplices.get(node_c, set()))
    weight_bc = len(node_to_simplices.get(node_b, set()) & node_to_simplices.get(node_c, set()))

    # Use geometric mean to reduce the impact of a single extreme value
    if weight_ab > 0 and weight_ac > 0 and weight_bc > 0:
        return (weight_ab * weight_ac * weight_bc) ** (1/3)
    else:
        return 0.0

# 2-3 pairwise timescale density
def calculate_pairwise_timescale_density(triangle, pair_to_times):
    """计算内部密集度/Calculate internal density"""
    node_a, node_b, node_c = triangle
    nodes = [node_a, node_b, node_c]
    
    # Calculate density values
    density = calculate_pairwise_density(node_a, node_b, node_c, pair_to_times)
    
    # Density defined as the inverse of average time interval
    mean_interval = np.mean(density)
    return 1.0 / (mean_interval + 1.0)  # +1 avoid division by zero

def calculate_pairwise_density(node1, node2, node3, pair_to_times):
    cooccurrence_times_1_2 = pair_to_times.get(tuple(sorted((node1, node2))), [])
    cooccurrence_times_2_3 = pair_to_times.get(tuple(sorted((node2, node3))), [])
    cooccurrence_times_1_3 = pair_to_times.get(tuple(sorted((node1, node3))), [])

    cooccurrence_times = cooccurrence_times_1_2 + cooccurrence_times_2_3 + cooccurrence_times_1_3
    cooccurrence_times = sorted(set(cooccurrence_times))

    if len(cooccurrence_times) <= 1:
        return 0.0
    
    # Calculate intervals between consecutive co-occurrence times
    time_intervals = []
    for i in range(1, len(cooccurrence_times)):
        interval = cooccurrence_times[i] - cooccurrence_times[i-1]
        time_intervals.append(interval)
    
    if not time_intervals:
        return [0.0]
    else:
        return time_intervals

# feature 3: structural balance
# 3-1 timescale density balance
# 3-2 degree balance
# 3-3 weight balance

# 3-1 timescale density balance
def calculate_timescale_density_balance(triangle, pair_to_times):
    node_a, node_b, node_c = triangle
    
    # Calculate density values
    density = calculate_pairwise_density(node_a, node_b, node_c, pair_to_times)

    # Calculate structural imbalance (Gini coefficient)
    return gini_coefficient(density)

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

# 3-2 degree balance
def calculate_degree_balance(triangle, node_to_simplices):
    node_a, node_b, node_c = triangle
    degrees = [
        len(node_to_simplices.get(node_a, set())),
        len(node_to_simplices.get(node_b, set())),
        len(node_to_simplices.get(node_c, set()))
    ]

    # Calculate structural imbalance (Gini coefficient)
    return gini_coefficient(degrees)

# 3-3 weight balance
def calculate_weight_balance(triangle, node_to_simplices):
    node_a, node_b, node_c = triangle
    
    # Calculate weights of the three internal edges of the triangle
    weight_ab = len(node_to_simplices.get(node_a, set()) & node_to_simplices.get(node_b, set()))
    weight_ac = len(node_to_simplices.get(node_a, set()) & node_to_simplices.get(node_c, set()))
    weight_bc = len(node_to_simplices.get(node_b, set()) & node_to_simplices.get(node_c, set()))
    
    weights = [weight_ab, weight_ac, weight_bc]

    # Calculate structural imbalance (Gini coefficient)
    return gini_coefficient(weights)

# feature 4: lifetime
# 4-1 lifetime

def calculate_lifetime(triangle, pair_to_times):
    """
    - pair_to_times: mapping (u,v) -> sorted list of timestamps (preferred)
    - closure_time: if the triangle is known to be closed, caller may supply its closure time
    - observation_end_time: fallback end time for open triangles

    Returns lifetime (float >= 0) or 0.0 if undefined.
    """
    node_a, node_b, node_c = triangle
    pairs = [tuple(sorted((node_a, node_b))), tuple(sorted((node_a, node_c))), tuple(sorted((node_b, node_c)))]

    start_time = np.median([min(pair_to_times[(node_a, node_b)]),
                           min(pair_to_times[(node_b, node_c)]),
                           min(pair_to_times[(node_a, node_c)])])

    # if caller explicitly passed closure_time via keyword 'closure_time', respect it
    # (allow caller to pass closure_time as keyword arg)
    # Note: python will place any extra kwarg into tri_first_closure only if provided that way; to support explicit closure_time,
    # callers should pass tri_first_closure mapping with the triangle key, or modify call site accordingly.

    end_time = max([min(pair_to_times[(node_a, node_b)]),
                   min(pair_to_times[(node_b, node_c)]),
                   min(pair_to_times[(node_a, node_c)])])

    if end_time is None:
        return 0.0

    lifetime = float(end_time - start_time)
    return lifetime if lifetime >= 0 else 0.0

def calculate_lifetime_2(triangle, pair_to_times):
    """
    - pair_to_times: mapping (u,v) -> sorted list of timestamps (preferred)
    - closure_time: if the triangle is known to be closed, caller may supply its closure time
    - observation_end_time: fallback end time for open triangles

    Returns lifetime (float >= 0) or 0.0 if undefined.
    """
    node_a, node_b, node_c = triangle
    pairs = [tuple(sorted((node_a, node_b))), tuple(sorted((node_a, node_c))), tuple(sorted((node_b, node_c)))]

    start_time = np.min([min(pair_to_times[(node_a, node_b)]),
                           min(pair_to_times[(node_b, node_c)]),
                           min(pair_to_times[(node_a, node_c)])])

    # if caller explicitly passed closure_time via keyword 'closure_time', respect it
    # (allow caller to pass closure_time as keyword arg)
    # Note: python will place any extra kwarg into tri_first_closure only if provided that way; to support explicit closure_time,
    # callers should pass tri_first_closure mapping with the triangle key, or modify call site accordingly.

    end_time = max([min(pair_to_times[(node_a, node_b)]),
                   min(pair_to_times[(node_b, node_c)]),
                   min(pair_to_times[(node_a, node_c)])])

    if end_time is None:
        return 0.0

    lifetime = float(end_time - start_time)
    return lifetime if lifetime >= 0 else 0.0


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
    
    def build_data_structures(self, dataset, suffix):
        with open(f'../data/processed/{dataset}/simplices_{suffix}.pickle', 'rb') as file:
            data = pickle.load(file)

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
                                       self.node_time_to_neighbors_bins) for batch in batches]
           
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
            
            with open(f'../data/processed/{dataset}/trg_open_{suffix}.pickle', 'rb') as file:
                candidates = pickle.load(file)

            features = dp.calculate_triangle_features(candidates)

            save_dir = '../data/features/' + dataset
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            
            with open(save_dir + f'/features_{suffix}.pickle', 'wb') as f:
                pickle.dump(features, f)