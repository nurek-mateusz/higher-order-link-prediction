import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import itertools
import numpy as np
import gzip
from pathlib import Path
from tqdm import tqdm
import math
import pickle as pkl


def process_triangle_worker(batch, simplices, edge_times, node_to_simplices, node_to_times, node_time_to_neighbors, pair_to_times):
    try:
        results = []
        for triangle in batch:
            features = {
                'triangle': triangle,
                'intensity': calculate_intensity(triangle, simplices, node_to_times, node_time_to_neighbors),
                'lifetime': calculate_lifetime(triangle, edge_times),
                'internal_density': calculate_internal_density(triangle, pair_to_times),
                'structural_imbalance': calculate_structural_imbalance(triangle, pair_to_times),
                'reinforcement': calculate_reinforcement(triangle, node_to_simplices)
            }
            results.append(features)
        return results
    except Exception as e:
        return [{'error': str(e), 'triangle': triangle}]
        
def calculate_intensity(triangle, simplices, node_to_times, node_time_to_neighbors):
    """计算强度特征（平均intensity保证公平性）"""
    node_a, node_b, node_c = triangle
    
    # 获取三个节点的所有时间戳
    timestamps_a = node_to_times.get(node_a, set())
    timestamps_b = node_to_times.get(node_b, set())
    timestamps_c = node_to_times.get(node_c, set())
    
    # 所有相关时间戳
    all_timestamps = timestamps_a | timestamps_b | timestamps_c
    
    if not all_timestamps:
        return 0.0
    
    intensity_sum = 0.0
    valid_timestamps = 0
    
    for timestamp in all_timestamps:
        # 获取每个节点在这个时间戳的邻居
        # Get neighbors at time
        neighbors_a = node_time_to_neighbors.get((node_a, timestamp), set())
        neighbors_b = node_time_to_neighbors.get((node_b, timestamp), set())
        neighbors_c = node_time_to_neighbors.get((node_c, timestamp), set())
        
        # 计算共同邻居和所有邻居
        common_neighbors = neighbors_a & neighbors_b & neighbors_c
        all_neighbors = neighbors_a | neighbors_b | neighbors_c
        
        if len(all_neighbors) > 0:
            intensity_sum += len(common_neighbors) / len(all_neighbors)
            valid_timestamps += 1
    
    # 计算时间跨度
    if all_timestamps:
        min_timestamp = min(all_timestamps)
        max_timestamp = max(all_timestamps)
        time_span = max_timestamp - min_timestamp + 1
        return intensity_sum / time_span
    else:
        return 0.0

def calculate_lifetime(triangle, edge_times):
    """计算生命周期特征"""
    node_a, node_b, node_c = triangle
    
    # 获取所有相关时间戳
    all_times = []
    for edge in [(node_a, node_b), (node_a, node_c), (node_b, node_c)]:
        edge_key = tuple(sorted(edge))
        all_times.extend(edge_times.get(edge_key, []))
    
    if not all_times:
        return 0
    
    return max(all_times) - min(all_times)

def calculate_internal_density(triangle, pair_to_times):
    """计算内部密集度"""
    node_a, node_b, node_c = triangle
    nodes = [node_a, node_b, node_c]
    
    # 计算三个内部密集度值
    internal_densities = []
    
    for i, target_node in enumerate(nodes):
        other_nodes = [nodes[j] for j in range(3) if j != i]
        neighbor1, neighbor2 = other_nodes
        
        density1 = calculate_pairwise_density(target_node, neighbor1, pair_to_times)
        density2 = calculate_pairwise_density(target_node, neighbor2, pair_to_times)
        
        if density1 > 0 and density2 > 0:
            internal_density = np.sqrt(density1 * density2)
        else:
            internal_density = 0.0
        
        internal_densities.append(internal_density)
    
    return np.mean(internal_densities)

def calculate_structural_imbalance(triangle, pair_to_times):
    """计算结构不平衡性"""
    node_a, node_b, node_c = triangle
    nodes = [node_a, node_b, node_c]
    
    # 计算三个内部密集度值
    internal_densities = []
    
    for i, target_node in enumerate(nodes):
        other_nodes = [nodes[j] for j in range(3) if j != i]
        neighbor1, neighbor2 = other_nodes
        
        density1 = calculate_pairwise_density(target_node, neighbor1, pair_to_times)
        density2 = calculate_pairwise_density(target_node, neighbor2, pair_to_times)
        
        if density1 > 0 and density2 > 0:
            internal_density = np.sqrt(density1 * density2)
        else:
            internal_density = 0.0
        
        internal_densities.append(internal_density)
    
    # 计算结构不平衡性（变异系数）
    mean_density = np.mean(internal_densities)
    if mean_density > 0:
        std_density = np.std(internal_densities)
        return std_density / mean_density
    else:
        return 0.0

def calculate_pairwise_density(node1, node2, pair_to_times):
    """计算两个节点之间的共现密集度"""
    cooccurrence_times = pair_to_times.get(tuple(sorted((node1, node2))), [])
    
    if len(cooccurrence_times) <= 1:
        return 0.0
    
    # 计算相邻共现时间的间隔
    cooccurrence_times.sort()
    time_intervals = []
    for i in range(1, len(cooccurrence_times)):
        interval = cooccurrence_times[i] - cooccurrence_times[i-1]
        time_intervals.append(interval)
    
    if not time_intervals:
        return 0.0
    
    # 密集度定义为平均时间间隔的倒数
    mean_interval = np.mean(time_intervals)
    return 1.0 / (mean_interval + 1.0)  # +1避免除零

def calculate_reinforcement(triangle, node_to_simplices):
    """计算增强效应特征"""
    node_a, node_b, node_c = triangle
    
    # 计算节点度数的影响
    degree_a = len(node_to_simplices.get(node_a, set()))
    degree_b = len(node_to_simplices.get(node_b, set()))
    degree_c = len(node_to_simplices.get(node_c, set()))
    
    # 使用几何平均值
    if degree_a > 0 and degree_b > 0 and degree_c > 0:
        return (degree_a * degree_b * degree_c) ** (1/3)
    else:
        return 0.0

class DataPreparation:
    """
    数据准备和特征计算
    Data preparation and feature calculation

    只使用三个核心文件：nverts, simplices, times
    Only uses three core files: nverts, simplices, times
    """
    
    def __init__(self, use_multiprocessing=True, n_workers=None):
        """初始化高阶链路预测器"""
        self.use_multiprocessing = use_multiprocessing
        self.n_workers = n_workers or multiprocessing.cpu_count()
        
        # 数据结构 / Data structures
        self.simplices = []      # 所有simplices的列表
        self.node_to_simplices = defaultdict(set)  # 节点到包含它的simplices的映射
        self.simplex_to_time = {}  # simplex到时间戳的映射
        self.node_labels = {}    # 节点ID到标签的映射 (将从simplices生成)
        self.edge_times = defaultdict(list)  # 边时间线
        self.adjacency_list = defaultdict(set)  # 邻接表
        self.node_to_times = defaultdict(set)
        self.node_time_to_neighbors = defaultdict(set)
        self.pair_to_times = defaultdict(set)
        
        # 内部数据
        self.training_triangles = []
        self.test_pairs = []
        self.features_computed = False
        
        print(f"Initialized DataPreparation with {self.n_workers} workers")
    
    def load_dataset_from_files(self, dataset_path):
        """
        从标准数据文件加载数据集（简化版，只使用nverts、simplices、times）
        Load dataset from standard data files (simplified, only uses nverts, simplices, times)
        
        支持多种文件名格式：
        - 标准格式: nverts.txt, simplices.txt, times.txt
        - 压缩格式: nverts.txt.gz, simplices.txt.gz, times.txt.gz
        - 带前缀格式: coauth-DBLP-nverts.txt.gz, dataset-name-simplices.txt, etc.
        
        Args:
            dataset_path: 数据集文件夹路径，包含以上格式的文件
        
        Returns:
            dict: 包含所有数据的字典
        """
        dataset_path = Path(dataset_path)
        print(f"Loading simplified dataset from: {dataset_path}")
        
        # 检查路径是否存在 / Check if the path exists
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        dataset = {}
        
        # 定义需要查找的文件后缀 / Define file suffixes to search for
        required_suffixes = ['nverts.txt', 'simplices.txt', 'times.txt']
        
        # 加载必需文件 / Load required files
        for suffix in required_suffixes:
            found_file = None
            
            # 首先查找带前缀的压缩文件 (如 coauth-DBLP-nverts.txt.gz)
            for file_path in dataset_path.iterdir():
                if file_path.is_file() and file_path.name.endswith(suffix + '.gz'):
                    found_file = file_path
                    break
            
            # 如果没找到压缩文件，查找普通文件 (如 coauth-DBLP-nverts.txt)
            if not found_file:
                for file_path in dataset_path.iterdir():
                    if file_path.is_file() and file_path.name.endswith(suffix):
                        found_file = file_path
                        break
            
            # 如果还没找到，尝试标准文件名
            if not found_file:
                standard_file = dataset_path / suffix
                standard_gz_file = dataset_path / f"{suffix}.gz"
                
                if standard_gz_file.exists():
                    found_file = standard_gz_file
                elif standard_file.exists():
                    found_file = standard_file
            
            # 加载找到的文件
            if found_file:
                print(f"Loading file: {found_file.name}")
                try:
                    if found_file.name.endswith('.gz'):
                        with gzip.open(found_file, 'rt', encoding='utf-8') as f:
                            data = self._read_file_content(f, suffix)
                            dataset[suffix.replace('.txt', '')] = data
                    else:
                        with open(found_file, 'r', encoding='utf-8') as f:
                            data = self._read_file_content(f, suffix)
                            dataset[suffix.replace('.txt', '')] = data
                except Exception as e:
                    print(f"Error reading {found_file}: {e}")
                    raise FileNotFoundError(f"Failed to load required file with suffix: {suffix}")
            else:
                raise FileNotFoundError(f"Required file with suffix '{suffix}' not found in {dataset_path}")
        
        # 验证数据完整性 / Validate dataset integrity
        self._validate_dataset(dataset)
        
        return dataset
    
    def _read_file_content(self, file_handle, filename):
        """
        读取文件内容并根据文件类型进行解析
        Read file content and parse according to file type
        """
        content = []
        
        if filename in ['nverts.txt', 'times.txt']:
            # 数值文件：每行一个整数 / Numeric file: one integer per line
            for line_num, line in enumerate(file_handle):
                line = line.strip()
                if line:
                    try:
                        value = int(line)
                        content.append(value)
                    except ValueError as e:
                        print(f"Warning: Invalid integer at line {line_num + 1} in {filename}: {line}")
                        continue
        
        elif filename == 'simplices.txt':
            # Simplices文件：连续的节点索引列表
            for line_num, line in enumerate(file_handle):
                line = line.strip()
                if line:
                    try:
                        # 可能是空格分隔或单个数字每行 / space-separated or single number per line
                        if ' ' in line:
                            # 空格分隔的多个数字 / Space-separated multiple numbers
                            values = [int(x) for x in line.split()]
                            content.extend(values)
                        else:
                            # 单个数字 / Single number
                            value = int(line)
                            content.append(value)
                    except ValueError as e:
                        print(f"Warning: Invalid integer at line {line_num + 1} in {filename}: {line}")
                        continue
        
        print(f"Loaded {len(content)} items from {filename}")
        return content
    
    def _validate_dataset(self, dataset):
        """
        验证数据集的完整性和一致性
        Validate dataset integrity and consistency 
        """
        required_keys = ['nverts', 'simplices', 'times']
        missing_keys = [key for key in required_keys if key not in dataset]
        
        if missing_keys:
            raise ValueError(f"Missing required data: {missing_keys}")
        
        # 检查数据长度一致性 / Check data length consistency
        nverts_len = len(dataset['nverts'])
        times_len = len(dataset['times'])
        
        if nverts_len != times_len:
            print(f"Warning: Length mismatch - nverts: {nverts_len}, times: {times_len}")
        
        # 检查simplices数组长度 / Check simplices array length
        expected_simplices_len = sum(dataset['nverts'])
        actual_simplices_len = len(dataset['simplices'])
        
        if expected_simplices_len != actual_simplices_len:
            print(f"Warning: Simplices length mismatch - expected: {expected_simplices_len}, actual: {actual_simplices_len}")
        
        print("Dataset validation completed successfully!")
    
    def build_data_structures(self, dataset):
        """
        从数据集构建内部数据结构
        Build internal data structures from dataset
        """
        print("Building simplified data structures...")
        
        # 构建simplex数据结构 / Build simplex data structure
        self._build_simplex_data(dataset)
        
        # 生成节点标签 (从simplices中的唯一节点生成) / Generate node labels from unique nodes in simplices
        self._build_node_labels_from_simplices(dataset)
        
        # 构建节点到simplices的映射 / Build node to simplices mapping
        self._build_node_simplex_mapping()
        
        # 构建边时间线和邻接表 / Build edge timelines and adjacency list
        self._build_edge_and_adjacency_data()

        self._build_node_to_times()

        self._build_node_time_to_neighbors()

        self._build_pair_to_times()
        
        print(f"Built data structures:")
        print(f"- Total simplices: {len(self.simplices)}")
        print(f"- Total nodes: {len(self.node_labels)}")
        print(f"- Node-simplex mappings: {sum(len(simplices) for simplices in self.node_to_simplices.values())}")
        print(f"- Edge timelines: {len(self.edge_times)}")
    
    def _build_node_labels_from_simplices(self, dataset):
        """
        从simplices数据中生成节点标签
        Generate node labels from simplices data (simplified)
        """
        print("Generating node labels from simplices data...")
        
        # 从所有simplices中提取唯一的节点ID / Extract unique node IDs from all simplices
        unique_nodes = set()
        for simplex in self.simplices:
            unique_nodes.update(simplex['nodes'])
        
        # 创建节点标签映射 (节点ID -> 节点标签) / Create node label mapping (node ID -> node label)
        # 这里我们直接使用节点ID作为标签 / Here we directly use node ID as label
        self.node_labels = {node_id: f"Node_{node_id}" for node_id in sorted(unique_nodes)}
        
        print(f"Generated {len(self.node_labels)} node labels from simplices")
        print(f"Node ID range: {min(unique_nodes)} to {max(unique_nodes)}")
    
    def _build_simplex_data(self, dataset):
        """
        构建simplex数据结构
        Build simplex data structure
        """
        nverts = dataset['nverts']
        simplices_flat = dataset['simplices']  
        times = dataset['times']
        
        print(f"Building simplex data from {len(nverts)} simplices...")
        
        # 解析simplices数据 / simplices data
        self.simplices = []
        current_pos = 0
        
        for i, (nvert, time) in enumerate(zip(nverts, times)):
            # 提取当前simplex的节点 / Extract nodes for the current simplex
            nodes = simplices_flat[current_pos:current_pos + nvert]
            current_pos += nvert
            
            # 创建simplex对象 / Create simplex object
            simplex = {
                'id': i,
                'nodes': tuple(sorted(nodes)),  # 保持节点排序 / Keep nodes sorted
                'time': time,
                'order': nvert - 1  # simplex的阶数 (边=0, 三角形=1, 四面体=2, ...) / simplex order (edge=0, triangle=1, tetrahedron=2, ...)
            }
            
            self.simplices.append(simplex)
            self.simplex_to_time[i] = time
        
        print(f"Built {len(self.simplices)} simplices")
        
        # 统计simplex阶数分布 / Count simplex order distribution
        order_counts = {}
        for simplex in self.simplices:
            order = simplex['order']
            order_counts[order] = order_counts.get(order, 0) + 1
        
        print("Simplex order distribution:")
        for order, count in sorted(order_counts.items()):
            if order == 0:
                print(f"  Order {order} (edges): {count}")
            elif order == 1:
                print(f"  Order {order} (triangles): {count}")
            elif order == 2:
                print(f"  Order {order} (tetrahedra): {count}")
            else:
                print(f"  Order {order}: {count}")
    
    def _build_node_simplex_mapping(self):
        """
        构建节点到simplices的映射关系
        Build node to simplices mapping 
        """
        print("Building node-simplex mappings...")
        
        self.node_to_simplices = defaultdict(set)
        
        for simplex in self.simplices:
            simplex_id = simplex['id']
            for node in simplex['nodes']:
                self.node_to_simplices[node].add(simplex_id)
        
        print(f"Built mappings for {len(self.node_to_simplices)} nodes")
    
    def _build_edge_and_adjacency_data(self):
        """
        构建边时间线和邻接表
        Build edge timelines and adjacency list
        """
        print("Building edge timelines and adjacency list...")
        
        self.edge_times = defaultdict(list)
        self.adjacency_list = defaultdict(set)
        
        for simplex in self.simplices:
            nodes = simplex['nodes']
            timestamp = simplex['time']
            
            # 构建邻接表 / Build adjacency list
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if i != j:
                        self.adjacency_list[node_i].add(node_j)
            
            # 构建边时间线 / Build edge timelines
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    edge_key = tuple(sorted([nodes[i], nodes[j]]))
                    self.edge_times[edge_key].append(timestamp)
        
        print(f"Built {len(self.edge_times)} edge timelines")

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
    
    def get_data_statistics(self):
        """
        获取数据统计信息
        Get data statistics 
        """
        if not self.simplices:
            return "No data loaded"
        
        stats = {
            'total_simplices': len(self.simplices),
            'total_nodes': len(self.node_labels),
            'time_range': (
                min(s['time'] for s in self.simplices),
                max(s['time'] for s in self.simplices)
            ),
            'simplex_orders': {}
        }
        
        # 统计不同阶数的simplex数量
        for simplex in self.simplices:
            order = simplex['order']
            stats['simplex_orders'][order] = stats['simplex_orders'].get(order, 0) + 1
        
        return stats

    # ====== 五个特征计算方法 / Five Feature Calculation Methods ======
    
    def generate_candidate_triangles(self, limit=None):
        """
        生成候选三角形（开放三角形）
        Generate candidate triangles (open triangles)
        """
        print("Generating candidate triangles...")
        candidates = set()
        
        for node_a in self.adjacency_list:
            for node_b in self.adjacency_list[node_a]:
                if node_a < node_b:  # 避免重复
                    # 找到与a和b都相邻的节点c
                    common_neighbors = self.adjacency_list[node_a] & self.adjacency_list[node_b]
                    for node_c in common_neighbors:
                        if node_b < node_c:  # 保持有序
                            triangle = tuple(sorted([node_a, node_b, node_c]))
                            candidates.add(triangle)
                            
                            if limit and len(candidates) >= limit:
                                break
                    if limit and len(candidates) >= limit:
                        break
            if limit and len(candidates) >= limit:
                break
        
        print(f"Generated {len(candidates)} candidate triangles")
        return list(candidates)
    
    def calculate_triangle_features(self, triangles):
        """
        计算三角形的五个特征
        Calculate five features for triangles
        
        Args:
            triangles: 三角形列表 / List of triangles
        
        Returns:
            list: 包含特征字典的列表 / List of feature dictionaries
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
                                       self.edge_times, 
                                       self.node_to_simplices,
                                       self.node_to_times,
                                       self.node_time_to_neighbors,
                                       self.pair_to_times) for batch in batches]
           
            for future in tqdm(as_completed(futures), total=len(batches)):
                result = future.result()
                if 'error' in result[0]:
                    print(f"Error processing triangle {result[0]['triangle']}: {result[0]['error']}")
                else:
                    features.extend(result)

        
        print(f"Processed {len(features)}/{len(triangles)} successfully")
        return features
    
    

if __name__ == "__main__":
    predictor = DataPreparation(n_workers=10)
    dataset = predictor.load_dataset_from_files('data/train/email-Enron')
    predictor.build_data_structures(dataset)
    candidates = predictor.generate_candidate_triangles()
    features = predictor.calculate_triangle_features(candidates)

    with open('data/features/email-Enron.pkl', 'wb') as file:
        pkl.dump(features, file)