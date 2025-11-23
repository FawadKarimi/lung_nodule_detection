"""
Configuration Module for Lung Nodule Detection Framework
Contains all hyperparameters and settings
"""

import os
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration parameters"""
    # Dataset paths
    data_root: str = "./data"  # Default: current directory/data
    cache_dir: str = "./cache"
    
    # Data splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Random seed
    random_seed: int = 42
    
    # Preprocessing
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # mm
    patch_size: Tuple[int, int, int] = (64, 64, 64)  # voxels
    
    # Intensity normalization (thesis Section 3.3.2: -1000 to +400 HU)
    window_level: int = -300  # HU (center of -1000 to +400 range)
    window_width: int = 1400  # HU
    
    # Batch composition
    batch_size: int = 16
    positive_per_batch: int = 4
    hard_negative_per_batch: int = 8
    random_negative_per_batch: int = 4
    
    # Number of workers
    num_workers: int = 8
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: float = 15.0  # degrees
    elastic_deform: bool = True
    elastic_alpha: float = 100.0
    elastic_sigma: float = 10.0
    noise_std: float = 0.01
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0
        assert self.batch_size == (self.positive_per_batch + 
                                   self.hard_negative_per_batch + 
                                   self.random_negative_per_batch)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # ResNet3D backbone
    resnet_depth: int = 18  # Options: 18, 34, 50
    resnet_in_channels: int = 1
    resnet_base_channels: int = 64
    
    # Transformer settings
    use_transformer: bool = True
    transformer_dim: int = 512
    transformer_depth: int = 6
    transformer_heads: int = 8
    transformer_mlp_dim: int = 2048
    dropout: float = 0.1
    
    # Multi-scale processing
    use_multiscale: bool = True
    scales: Tuple[int, ...] = (1, 2, 3)  # Which ResNet stages to use
    
    # Positional encoding
    use_positional_encoding: bool = True
    
    # Fusion strategy
    fusion_type: str = "hierarchical"  # Options: "concat", "residual", "hierarchical"
    
    # Classification head
    classifier_hidden_dim: int = 256
    classifier_dropout: float = 0.5
    num_classes: int = 2


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Epochs
    num_epochs: int = 60
    warmup_epochs: int = 5
    
    # Learning rates
    lr_cnn: float = 1e-4
    lr_transformer: float = 5e-5
    min_lr: float = 1e-6
    
    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    
    # Learning rate schedule
    scheduler: str = "cosine"  # Options: "cosine", "step", "plateau"
    cosine_t0: int = 10  # Epochs for first restart
    
    # Loss weights
    loss_type: str = "weighted_ce"
    class_weights: Tuple[float, float] = (1.0, 3.0)  # non-nodule, nodule
    
    # XAI-guided training
    use_xai_loss: bool = True
    xai_loss_weight: float = 0.1
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Checkpointing
    save_every: int = 5  # epochs
    save_best_only: bool = False
    
    # Mixed precision training
    use_amp: bool = True
    
    # Hard example mining
    use_hard_mining: bool = True
    mining_start_epoch: int = 20


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Metrics to compute
    compute_roc: bool = True
    compute_froc: bool = True
    compute_confusion_matrix: bool = True
    
    # Size-stratified analysis
    size_categories: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'small': (0, 10),     # mm
        'medium': (10, 20),   # mm
        'large': (20, 100)    # mm
    })
    
    # XAI evaluation
    evaluate_explainability: bool = True
    xai_validation_samples: int = 100
    
    # Cross-validation
    num_folds: int = 5
    
    # Statistical testing
    confidence_level: float = 0.95
    num_bootstrap: int = 1000


@dataclass
class ExplainabilityConfig:
    """Explainability configuration"""
    # Methods to use
    use_gradcam: bool = True
    use_integrated_gradients: bool = True
    use_attention_viz: bool = True
    
    # Grad-CAM++ settings
    gradcam_target_layer: str = "layer4"
    
    # Integrated Gradients settings
    ig_steps: int = 50
    ig_baseline: str = "zeros"  # Options: "zeros", "black", "blur"
    
    # Attention visualization settings
    attention_aggregation: str = "mean"  # Options: "mean", "max", "rollout"
    
    # Multi-modal fusion
    fusion_weights: Tuple[float, float, float] = (0.40, 0.35, 0.25)  # GradCAM, IG, Attention
    learned_fusion: bool = True


@dataclass
class Config:
    """Main configuration class"""
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    
    # Experiment settings
    experiment_name: str = "lung_nodule_detection"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    # Device settings
    device: str = "cuda"  # Options: "cuda", "cpu"
    gpu_id: int = 0
    
    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False  # Set True for faster training if input size is fixed
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'explainability': self.explainability.__dict__,
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'log_dir': self.log_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'device': self.device,
            'gpu_id': self.gpu_id,
            'deterministic': self.deterministic,
            'benchmark': self.benchmark
        }
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        import yaml
        
        # Register tuple constructor for SafeLoader
        def tuple_constructor(loader, node):
            return tuple(loader.construct_sequence(node))
        
        yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config object
        config = cls()
        
        # Update with loaded values
        for key, value in config_dict.items():
            if hasattr(config, key):
                if isinstance(value, dict) and hasattr(getattr(config, key), '__dict__'):
                    # Update nested config
                    for k, v in value.items():
                        setattr(getattr(config, key), k, v)
                else:
                    setattr(config, key, value)
        
        return config


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


# Example usage
if __name__ == "__main__":
    config = get_default_config()
    
    print("Configuration created successfully!")
    print(f"\nBatch size: {config.data.batch_size}")
    print(f"Learning rate (CNN): {config.training.lr_cnn}")
    print(f"Learning rate (Transformer): {config.training.lr_transformer}")
    print(f"Number of epochs: {config.training.num_epochs}")
    print(f"Patch size: {config.data.patch_size}")
    
    # Save configuration
    config.save("config.yaml")
    print("\nConfiguration saved to config.yaml")
    
    # Load configuration
    loaded_config = Config.load("config.yaml")
    print("Configuration loaded successfully!")