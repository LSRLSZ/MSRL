import os
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Union
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def negative_sampling(node_pairs: torch.Tensor, 
                      node_types: Dict[int, str], 
                      num_negatives: int, 
                      device: torch.device) -> torch.Tensor:
    """
    Generate negative samples following the strategy in the paper
    
    Args:
        node_pairs: Positive node pairs (shape: [batch_size, 2])
        node_types: Dictionary mapping node IDs to their types
        num_negatives: Number of negative samples per positive pair
        device: Target device for tensor operations
    
    Returns:
        neg_samples: Negative node pairs (shape: [batch_size * num_negatives, 2])
    """
    batch_size = node_pairs.size(0)
    neg_samples = []
    
    # Build type distribution
    type_dist = defaultdict(list)
    for nid, t in node_types.items():
        type_dist[t].append(nid)
    
    # Generate negatives for each positive pair
    for src, dst in node_pairs:
        src_type = node_types[src.item()]
        dst_type = node_types[dst.item()]
        
        # Get valid negative candidates based on types
        valid_dst_types = config.DATASET_META[config.DATASET_NAME]['valid_pairs'][src_type]
        candidates = []
        for t in valid_dst_types:
            candidates.extend(type_dist[t])
        
        # Randomly sample negatives
        if len(candidates) == 0:
            selected = torch.randint(0, len(node_types), (num_negatives,))
        else:
            selected = np.random.choice(candidates, size=num_negatives, replace=True)
        
        # Create negative pairs
        for neg_dst in selected:
            neg_samples.append([src.item(), neg_dst])
    
    return torch.tensor(neg_samples, dtype=torch.long, device=device)

def time_decay_kernel(delta_t: torch.Tensor, theta: float) -> torch.Tensor:
    """
    Compute temporal decay kernel values (Equation 1)
    
    Args:
        delta_t: Time differences (t - t_s)
        theta: Decay rate parameter
    
    Returns:
        decay_values: Computed decay values
    """
    return torch.exp(-theta * delta_t)

def compute_clustering_coefficient(adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate local clustering coefficients (Equation 3)
    
    Args:
        adj_matrix: Dense adjacency matrix (shape: [N, N])
    
    Returns:
        clustering_coeffs: Tensor of clustering coefficients (shape: [N])
    """
    num_nodes = adj_matrix.size(0)
    tri_counts = torch.diagonal(adj_matrix @ adj_matrix @ adj_matrix)
    degrees = adj_matrix.sum(dim=1)
    
    # Handle zero-degree nodes
    denom = degrees * (degrees - 1)
    denom[denom <= 0] = 1  # Avoid division by zero
    
    return 2 * tri_counts / denom

def compute_metrics(labels: np.ndarray, 
                    preds: np.ndarray, 
                    probs: np.ndarray = None, 
                    task: str = 'link_prediction') -> Dict[str, float]:
    """
    Compute evaluation metrics based on task type
    
    Args:
        labels: Ground truth labels
        preds: Model predictions
        probs: Prediction probabilities (for AUC)
        task: Evaluation task type
    
    Returns:
        metrics: Dictionary of computed metrics
    """
    metrics = {}
    
    if task == 'link_prediction':
        metrics.update({
            'Micro-F1': f1_score(labels, preds, average='micro'),
            'Macro-F1': f1_score(labels, preds, average='macro'),
            'Precision': precision_score(labels, preds),
            'Recall': recall_score(labels, preds)
        })
    elif task == 'node_classification' and probs is not None:
        metrics.update({
            'AUC-ROC': roc_auc_score(labels, probs, multi_class='ovr'),
            'Micro-F1': f1_score(labels, preds, average='micro'),
            'Macro-F1': f1_score(labels, preds, average='macro')
        })
    
    return metrics

def save_checkpoint(state: Dict, filename: str):
    """
    Save model checkpoint with training state
    
    Args:
        state: Dictionary containing model state
        filename: Output file path
    """
    torch.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")

def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """
    Load model checkpoint and restore state
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to saved checkpoint
        device: Target device for loading
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")

def setup_logging(log_dir: str = None):
    """
    Configure logging system with file and console handlers
    
    Args:
        log_dir: Directory to save log files
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir)/'training.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

def attention_mechanism(query: torch.Tensor, 
                        keys: torch.Tensor, 
                        values: torch.Tensor, 
                        mask: torch.Tensor = None) -> torch.Tensor:
    """
    Implement scaled dot-product attention mechanism
    
    Args:
        query: Query tensor (shape: [batch_size, dim])
        keys: Key tensor (shape: [batch_size, seq_len, dim])
        values: Value tensor (shape: [batch_size, seq_len, dim])
        mask: Optional attention mask
    
    Returns:
        context: Context vector (shape: [batch_size, dim])
        attn_weights: Attention weights (shape: [batch_size, seq_len])
    """
    scores = torch.matmul(query.unsqueeze(1), keys.transpose(-2, -1))
    scores = scores / np.sqrt(query.size(-1))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn_weights = F.softmax(scores, dim=-1)
    context = torch.matmul(attn_weights, values).squeeze(1)
    return context, attn_weights

def neighbor_aggregation(features: torch.Tensor, 
                         aggregation: str = 'mean') -> torch.Tensor:
    """
    Aggregate neighbor features using specified strategy
    
    Args:
        features: Neighbor features (shape: [batch_size, num_neighbors, dim])
        aggregation: Aggregation method (mean/sum/max)
    
    Returns:
        aggregated: Aggregated features (shape: [batch_size, dim])
    """
    if aggregation == 'mean':
        return torch.mean(features, dim=1)
    elif aggregation == 'sum':
        return torch.sum(features, dim=1)
    elif aggregation == 'max':
        return torch.max(features, dim=1)[0]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

def save_results(results: Dict, filename: str):
    """
    Save evaluation results to JSON file
    
    Args:
        results: Dictionary of results
        filename: Output file path
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {filename}")

def set_seed(seed: int = config.SEED):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")

def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    Normalize features using L2 normalization
    
    Args:
        features: Input feature tensor
    
    Returns:
        normalized: Normalized features
    """
    return F.normalize(features, p=2, dim=-1)

if __name__ == '__main__':
    # Test utility functions
    set_seed(42)
    test_adj = torch.eye(5)
    print("Clustering coefficients:", compute_clustering_coefficient(test_adj))