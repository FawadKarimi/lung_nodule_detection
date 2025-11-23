# test_models.py
import torch
from models.resnet3d import ResNet3DBackbone
from models.transformer import MultiScaleTransformer

print("="*60)
print("Testing 3D ResNet Backbone")
print("="*60)

# Create ResNet
resnet = ResNet3DBackbone(depth=18, in_channels=1, base_channels=64)
dummy_input = torch.randn(2, 1, 64, 64, 64)

print(f"Input shape: {dummy_input.shape}")

# Extract multi-scale features
feat1, feat2, feat3 = resnet(dummy_input, return_multiscale=True)

print(f"[OK] ResNet output:")
print(f"  Scale 1: {feat1.shape}")  # (2, 128, 8, 8, 8)
print(f"  Scale 2: {feat2.shape}")  # (2, 256, 4, 4, 4)
print(f"  Scale 3: {feat3.shape}")  # (2, 512, 2, 2, 2)

print("\n" + "="*60)
print("Testing Multi-Scale Transformer")
print("="*60)

# Create Transformer
transformer = MultiScaleTransformer(
    in_channels_list=(128, 256, 512),
    d_model=512,
    num_layers=6,
    num_heads=8
)

# Transform features
features = (feat1, feat2, feat3)
transformed, _ = transformer(features, return_attention=False)

print(f"[OK] Transformer output:")
for i, f in enumerate(transformed):
    print(f"  Scale {i+1}: {f.shape}")

# Count total parameters
resnet_params = sum(p.numel() for p in resnet.parameters())
transformer_params = sum(p.numel() for p in transformer.parameters())

print(f"\n[OK] Model Statistics:")
print(f"  ResNet parameters: {resnet_params:,}")
print(f"  Transformer parameters: {transformer_params:,}")
print(f"  Total: {(resnet_params + transformer_params):,}")

print("\n[OK] Both modules working correctly!")