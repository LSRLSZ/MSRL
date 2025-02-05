import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Dict, List, Tuple

class DynamicHeteroDataset(Dataset):
    
    def __init__(self, dataset_name: str):
        super().__init__()
        self.dataset_name = dataset_name
        self.root_path = f"./data/{dataset_name}/"
        
        # Initialize data structures
        self.node_types: Dict[int, str] = {}       # {node_id: type}
        self.node_labels: Dict[int, int] = {}      # {node_id: label}
        self.edges: List[Tuple[int, int, str, float]] = []  # (src, dst, relation_type, timestamp)
        self.meta_paths: Dict[str, List[List[str]]] = {}    # Defined meta-paths
        
        # Dynamic graph state
        self.current_time = 0.0
        self.event_history: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)
        self.adjacency_matrices: Dict[float, torch.Tensor] = {}
        
        # Load dataset files
        self._load_node_data()
        self._load_edge_data()
        self._load_meta_paths()
        self._preprocess_temporal_data()
        
        # Split datasets
        self._create_splits()

    def _load_node_data(self):
        """Load node information and labels"""
        # Load node types
        with open(os.path.join(self.root_path, "node_types.json")) as f:
            raw_types = json.load(f)
            self.node_types = {int(k): v for k, v in raw_types.items()}
            
        # Load node labels (if available)
        label_path = os.path.join(self.root_path, "node_labels.json")
        if os.path.exists(label_path):
            with open(label_path) as f:
                raw_labels = json.load(f)
                self.node_labels = {int(k): v for k, v in raw_labels.items()}
                
        # Create node ID mappings
        self.node_ids = list(self.node_types.keys())
        self.num_nodes = len(self.node_ids)
        self.type_to_nodes = defaultdict(list)
        for nid, t in self.node_types.items():
            self.type_to_nodes[t].append(nid)

    def _load_edge_data(self):
        """Load temporal edge data with relation types"""
        edge_file = os.path.join(self.root_path, "edges.csv")
        
        # Expected format: src,dst,relation_type,timestamp
        with open(edge_file) as f:
            for line in f:
                src, dst, rel, t = line.strip().split(',')
                self.edges.append((
                    int(src), int(dst), 
                    rel, float(t)
                ))
                
        # Sort edges by timestamp
        self.edges.sort(key=lambda x: x[3])
        
        # Get unique relation types
        self.relation_types = list(set([e[2] for e in self.edges]))
        self.num_relations = len(self.relation_types)

    def _load_meta_paths(self):
        """Load predefined meta-paths for each dataset"""
        meta_path_file = os.path.join(self.root_path, "meta_paths.json")
        if os.path.exists(meta_path_file):
            with open(meta_path_file) as f:
                self.meta_paths = json.load(f)

    def _preprocess_temporal_data(self):
        """Preprocess temporal edge data into event history"""
        # Create time-ordered event sequence
        self.timestamps = sorted(list(set([e[3] for e in self.edges])))
        self.max_timestamp = max(self.timestamps) if self.timestamps else 0.0
        
        # Initialize adjacency matrix
        self.initial_adj = torch.zeros((self.num_nodes, self.num_nodes), 
                                      dtype=torch.float32)
        
        # Build event history
        for src, dst, rel, t in self.edges:
            key = (src, dst, rel)
            self.event_history[key].append(t)

    def _create_splits(self):
        """Create temporal train/validation/test splits"""
        # Split strategy: last 20% timestamps for test
        split_idx = int(0.8 * len(self.timestamps))
        self.train_times = self.timestamps[:split_idx]
        self.test_times = self.timestamps[split_idx:]
        
        # Create event lists for each split
        self.train_events = [e for e in self.edges if e[3] in self.train_times]
        self.test_events = [e for e in self.edges if e[3] in self.test_times]
        
        # For node classification: use all nodes with labels
        if self.node_labels:
            self.test_nodes = list(self.node_labels.keys())

    def get_adjacency_matrix(self, timestamp: float) -> torch.Tensor:
        """Get adjacency matrix up to specified timestamp
        
        Args:
            timestamp: Temporal threshold for graph state
            
        Returns:
            adj_matrix: Sparse adjacency matrix up to timestamp
        """
        # Check cached matrices
        if timestamp in self.adjacency_matrices:
            return self.adjacency_matrices[timestamp]
        
        # Create new adjacency matrix
        adj = self.initial_adj.clone()
        for src, dst, rel, t in self.edges:
            if t <= timestamp:
                adj[src, dst] = 1.0
                adj[dst, src] = 1.0  # Assuming undirected graph
                
        # Cache and return
        self.adjacency_matrices[timestamp] = adj
        return adj

    def get_neighbor_data(self, node_pairs: List[Tuple[int, int]]) -> Dict:
        """Get neighborhood information for node pairs
        
        Args:
            node_pairs: List of (src, dst) node pairs
            
        Returns:
            neighbor_data: Dictionary containing:
                - 'micro': Direct neighbors with timestamps
                - 'meso': Type-based neighbor groups
                - 'macro': Full neighborhood structures
        """
        neighbor_data = {}
        current_adj = self.get_adjacency_matrix(self.current_time)
        
        for src, dst in node_pairs:
            # Micro-level: Direct neighbors
            micro_neighbors = {
                'src': self._get_micro_neighbors(src, current_adj),
                'dst': self._get_micro_neighbors(dst, current_adj)
            }
            
            # Meso-level: Type-based groups
            meso_neighbors = {
                'src': self._get_meso_neighbors(src),
                'dst': self._get_meso_neighbors(dst)
            }
            
            # Macro-level: Full neighborhood
            macro_neighbors = {
                'src': current_adj[src].nonzero().squeeze().tolist(),
                'dst': current_adj[dst].nonzero().squeeze().tolist()
            }
            
            neighbor_data[(src, dst)] = {
                'micro': micro_neighbors,
                'meso': meso_neighbors,
                'macro': macro_neighbors
            }
            
        return neighbor_data

    def _get_micro_neighbors(self, node: int, adj_matrix: torch.Tensor) -> List[Tuple[int, float]]:
        """Get direct neighbors with temporal information"""
        neighbors = adj_matrix[node].nonzero().squeeze().tolist()
        return [(n, self._get_last_interaction_time(node, n)) 
                for n in neighbors]

    def _get_meso_neighbors(self, node: int) -> Dict[str, List[int]]:
        """Group neighbors by node types"""
        node_type = self.node_types[node]
        neighbors = defaultdict(list)
        
        # Get all possible neighbor types from meta-paths
        for mp in self.meta_paths.get(node_type, []):
            target_type = mp[-1]
            neighbors[target_type] = self._get_nodes_by_meta_path(node, mp)
            
        return neighbors

    def _get_nodes_by_meta_path(self, node: int, meta_path: List[str]) -> List[int]:
        """Find nodes reachable via specific meta-path"""
        # Implementation requires meta-path-based traversal
        # Placeholder for actual path traversal logic
        return []

    def _get_last_interaction_time(self, src: int, dst: int) -> float:
        """Get latest interaction time between nodes"""
        candidates = []
        for rel in self.relation_types:
            key = (src, dst, rel)
            if key in self.event_history:
                candidates.extend(self.event_history[key])
        return max(candidates) if candidates else 0.0

    def collate_fn(self, batch: List) -> Dict:
        """Custom collation function for dynamic graph data
        
        Args:
            batch: List of events (src, dst, rel, timestamp)
            
        Returns:
            batch_dict: Dictionary containing:
                - node_pairs: Tensor of (src, dst) pairs
                - event_times: Tensor of timestamps
                - relation_types: Tensor of relation type indices
        """
        src_nodes = torch.tensor([e[0] for e in batch], dtype=torch.long)
        dst_nodes = torch.tensor([e[1] for e in batch], dtype=torch.long)
        times = torch.tensor([e[3] for e in batch], dtype=torch.float)
        rels = torch.tensor([self.relation_types.index(e[2]) for e in batch], 
                          dtype=torch.long)
        
        return {
            'node_pairs': torch.stack([src_nodes, dst_nodes], dim=1),
            'event_times': times,
            'relation_types': rels
        }

    def __len__(self) -> int:
        return len(self.train_events)

    def __getitem__(self, idx: int) -> Tuple:
        return self.train_events[idx]

# Example usage
if __name__ == '__main__':
    dataset = DynamicHeteroDataset('imdb')
    print(f"Dataset loaded with {len(dataset)} training events")
    print(f"Node types: {set(dataset.node_types.values())}")
    print(f"Relation types: {dataset.relation_types}")