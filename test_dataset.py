# test_dataset.py
import numpy as np
from data.augmentation import DataAugmentation
from config.config import get_default_config

# Test augmentation
print("Testing Data Augmentation...")
aug = DataAugmentation(
    rotation_range=15.0,
    apply_elastic=True,
    apply_noise=True,
    p=1.0
)

# Create dummy 3D image
dummy_image = np.random.rand(64, 64, 64).astype(np.float32)
print(f"Original shape: {dummy_image.shape}")

# Apply augmentation
augmented, _ = aug(dummy_image, None)
print(f"Augmented shape: {augmented.shape}")
print("[OK] Augmentation working!")

# Test configuration
config = get_default_config()
print(f"\n[OK] Batch size: {config.data.batch_size}")
print(f"[OK] Patch size: {config.data.patch_size}")
print(f"[OK] Augmentation enabled: {config.data.use_augmentation}")