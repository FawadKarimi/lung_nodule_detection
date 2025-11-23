# test_complete_model.py
import torch
from models.hybrid_model import HybridModel
from explainability.xai_modules import MultiModalExplainer

print("="*70)
print("Testing Complete Hybrid Model with Explainability")
print("="*70)

# Create model
model = HybridModel(
    resnet_depth=18,
    resnet_base_channels=64,
    transformer_dim=512,
    transformer_depth=6,
    transformer_heads=8,
    use_transformer=True,
    use_multiscale=True,
    num_classes=2
)

print("\n[OK] Model created successfully!")

# Create dummy input
batch_size = 2
dummy_input = torch.randn(batch_size, 1, 64, 64, 64)
print(f"Input shape: {dummy_input.shape}")

# Test forward pass
model.eval()
with torch.no_grad():
    output = model(dummy_input, return_features=True, return_attention=True)

print(f"\n[OK] Forward pass successful!")
print(f"  Logits: {output['logits'].shape}")
print(f"  Predictions: {output['logits'].argmax(dim=1)}")

# Test explainability
print("\n" + "="*70)
print("Testing Explainability")
print("="*70)

explainer = MultiModalExplainer(
    model=model,
    target_layer='resnet.resnet.layer4',
    fusion_weights=(0.40, 0.35, 0.25)
)

print("\nGenerating explanations...")
explanations = explainer.explain(
    dummy_input,
    return_individual=True
)

print(f"\n[OK] Explanations generated:")
for name, expl in explanations.items():
    print(f"  {name}: {expl.shape}")

# Model statistics
total_params = sum(p.numel() for p in model.parameters())
print(f"\n[OK] Model Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

print("\n" + "="*70)
print("[OK] Complete model with explainability working!")
print("="*70)