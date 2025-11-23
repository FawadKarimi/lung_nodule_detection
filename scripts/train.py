"""
Complete Training Script for Lung Nodule Detection
Run this script to train the model
"""

import os
import sys
import torch
import numpy as np
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config, get_default_config
from models.hybrid_model import create_model
from data.dataset import create_dataloaders
from training.trainer import Trainer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_environment(config: Config):
    """Setup training environment"""
    # Set device
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{config.gpu_id}')
        print(f"[OK] Using GPU: {torch.cuda.get_device_name(config.gpu_id)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(config.gpu_id).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("[WARN] Using CPU (training will be slow)")
    
    return device


def main():
    """Main training function"""
    print("="*70)
    print("Lung Nodule Detection - Training Script")
    print("="*70)
    
    # Load configuration
    config = get_default_config()
    
    # You can also load from file:
    # config = Config.load('config.yaml')
    
    # Set random seed
    set_seed(config.data.random_seed)
    print(f"[OK] Random seed set to: {config.data.random_seed}")
    
    # Setup environment
    device = setup_environment(config)
    
    # Create data loaders
    print("\n" + "="*70)
    print("Creating Data Loaders")
    print("="*70)
    
    # Get paths from config or environment
    data_dir = os.getenv('DATA_DIR', config.data.cache_dir if hasattr(config.data, 'cache_dir') else 'processed_data')
    annotations_dir = os.getenv('ANNOTATIONS_DIR', 'annotations')
    
    train_annotation = os.path.join(annotations_dir, 'train_annotations.json')
    val_annotation = os.path.join(annotations_dir, 'val_annotations.json')
    test_annotation = os.path.join(annotations_dir, 'test_annotations.json')
    
    # Check if data exists
    if not Path(train_annotation).exists():
        print("[WARN] Annotation files not found!")
        print("   Please create annotation files first:")
        print(f"   - {train_annotation}")
        print(f"   - {val_annotation}")
        print(f"   - {test_annotation}")
        print("\n   See documentation for annotation format.")
        return
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            train_annotation=train_annotation,
            val_annotation=val_annotation,
            test_annotation=test_annotation,
            config=config,
            num_workers=config.data.num_workers
        )
        
        print(f"[OK] Data loaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"[ERROR] Error creating data loaders: {e}")
        print("\n   If data is not yet preprocessed, run preprocessing first:")
        print("   python scripts/preprocess_data.py")
        return
    
    # Create model
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)
    
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[OK] Model created:")
    print(f"   Architecture: Hybrid 3D CNN-Transformer")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Create trainer
    print("\n" + "="*70)
    print("Initializing Trainer")
    print("="*70)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=str(device),
        checkpoint_dir=config.checkpoint_dir
    )
    
    print(f"[OK] Trainer initialized:")
    print(f"   Optimizer: AdamW")
    print(f"   LR (CNN): {config.training.lr_cnn}")
    print(f"   LR (Transformer): {config.training.lr_transformer}")
    print(f"   Scheduler: {config.training.scheduler}")
    print(f"   Mixed precision: {config.training.use_amp}")
    print(f"   Gradient clipping: {config.training.max_grad_norm}")
    
    # Start training
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Warmup epochs: {config.training.warmup_epochs}")
    print(f"Early stopping patience: {config.training.patience}")
    print("="*70 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user")
        print("   Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pth')
        print("   [OK] Checkpoint saved: interrupted.pth")
        return
    except Exception as e:
        print(f"\n[ERROR] Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_checkpoint('final_model.pth')
    
    # Save configuration
    config.save(Path(config.checkpoint_dir) / 'config.yaml')
    print("[OK] Configuration saved")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print("\nTo evaluate the model, run:")
    print("  python scripts/evaluate.py --checkpoint checkpoints/best_model.pth")


if __name__ == "__main__":
    main()