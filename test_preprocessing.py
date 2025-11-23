# test_preprocessing.py
from data.preprocessing import PreprocessingPipeline
import numpy as np

# Initialize pipeline
pipeline = PreprocessingPipeline(
    target_spacing=(1.0, 1.0, 1.0),
    window_level=-600,
    window_width=1400,
    apply_lung_segmentation=True
)

print("[OK] Preprocessing pipeline initialized successfully!")
print("\nPipeline includes:")
print("  - DICOM to NIfTI conversion")
print("  - Intensity normalization")
print("  - Isotropic resampling") 
print("  - Lung segmentation")