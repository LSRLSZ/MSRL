import os
from pathlib import Path

class Config:
    """Central configuration class for MSRL model training and evaluation"""
    
    # --------------------------
    # Dataset Configuration
    # --------------------------
    DATASET_NAME = "imdb"               # Name of dataset to use (imdb/aminer/yelp/dblp)
    DATA_ROOT = Path("./data")          # Root directory for all datasets
    FORCE_REPROCESS = False             # Force reprocessing of raw data
    
    # Dataset-specific parameters (based on Table 2 in paper)
    DATASET_META = {
        "imdb": {
            "node_types": ["movie", "director", "actor", "genre"],
            "relations": ["M-D", "M-A", "M-G"],
            "default_metapaths": [
                ["M-D-M"], 
                ["M-A-M"],
                ["M-G-M"]
            ]
        },
        "aminer": {
            "node_types": ["author", "paper", "term", "venue"],
            "relations": ["P-A", "P-T", "P-V"],
            "default_metapaths": [
                ["A-P-A"],
                ["P-T-P"],
                ["A-P-T-P-A"]
            ]
        },
        # Add configurations for other datasets similarly
    }
    
    # --------------------------
    # Model Architecture
    # --------------------------
    EMBED_DIM = 128                     # Dimension of node embeddings
    HAWKES_DECAY_THETA = 0.1            # Decay rate θ in Hawkes process (Eq.1)
    ATTENTION_DROPOUT = 0.2             # Dropout rate for attention layers
    NUM_RELATIONS = 4                   # Number of relation types (dataset-specific)
    
    # Neighbor Influence Module Parameters
    MICRO_ATTN_HIDDEN = 64              # Hidden dimension for micro attention
    MESO_AGG_HIDDEN = 64                # Hidden dimension for meso aggregation
    MACRO_AGG_STRATEGY = "mean"         # Aggregation strategy for macro level
    
    # --------------------------
    # Training Parameters
    # --------------------------
    BATCH_SIZE = 256                     # Training batch size
    NUM_EPOCHS = 100                     # Maximum training epochs
    LEARNING_RATE = 0.001                # Initial learning rate
    WEIGHT_DECAY = 1e-5                  # L2 regularization weight
    NEGATIVE_RATIO = 5                   # Negative sampling ratio (positive:negative)
    PATIENCE = 10                        # Early stopping patience
    
    # Optimization Parameters
    OPTIMIZER = "adam"                   # Optimizer (adam/sgd/rmsprop)
    LR_SCHEDULER = "plateau"             # Learning rate scheduler
    GRAD_CLIP = 5.0                      # Gradient clipping value
    
    # --------------------------
    # Evaluation Parameters
    # --------------------------
    EVAL_BATCH_SIZE = 1024               # Evaluation batch size
    TOP_K_VALUES = [1, 2]                # Top-k values for evaluation
    CLASSIFICATION_SPLIT = 0.8           # Train-test split for node classification
    
    # --------------------------
    # System Configuration
    # --------------------------
    SEED = 42                            # Random seed for reproducibility
    DEVICE = "auto"                      # auto/cuda/cpu
    NUM_WORKERS = 4                      # DataLoader workers
    LOG_INTERVAL = 50                    # Training log interval
    
    # --------------------------
    # Path Configuration
    # --------------------------
    @property
    def dataset_path(self):
        """Get path to specific dataset"""
        return self.DATA_ROOT / self.DATASET_NAME
    
    @property
    def processed_dir(self):
        """Directory for processed data"""
        return self.dataset_path / "processed"
    
    CHECKPOINT_DIR = Path("./checkpoints")  # Model checkpoint directory
    RESULT_DIR = Path("./results")          # Evaluation results directory
    LOG_DIR = Path("./logs")                # Training logs directory

    # --------------------------
    # Runtime Validation
    # --------------------------
    def __init__(self):
        self._validate_paths()
        self._validate_parameters()

    def _validate_paths(self):
        """Ensure required directories exist"""
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} not found!")

    def _validate_parameters(self):
        """Validate configuration parameters"""
        assert self.DATASET_NAME in ["imdb", "aminer", "yelp", "dblp"], \
            "Invalid dataset name"
        assert self.OPTIMIZER in ["adam", "sgd", "rmsprop"], \
            "Invalid optimizer selection"
        assert self.MACRO_AGG_STRATEGY in ["mean", "sum", "max"], \
            "Invalid macro aggregation strategy"

# Initialize global configuration
config = Config()