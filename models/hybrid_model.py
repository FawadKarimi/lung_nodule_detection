"""
Hybrid CNN-Transformer Architecture for Lung Nodule Detection
Combines 3D ResNet backbone with multi-scale Transformer modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from models.resnet3d import ResNet3DBackbone
from models.transformer import MultiScaleTransformer


class FeatureFusion(nn.Module):
    """Feature fusion module for combining CNN and Transformer features"""
    
    def __init__(
        self,
        in_channels: int,
        fusion_type: str = "hierarchical"
    ):
        """
        Args:
            in_channels: Number of input channels
            fusion_type: Type of fusion ("concat", "residual", "hierarchical")
        """
        super(FeatureFusion, self).__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            # Concatenate and reduce channels
            self.fusion = nn.Sequential(
                nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == "residual":
            # Weighted residual fusion
            self.alpha = nn.Parameter(torch.tensor(0.1))
        elif fusion_type == "hierarchical":
            # No additional parameters needed
            pass
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        cnn_features: torch.Tensor,
        transformer_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse CNN and Transformer features
        
        Args:
            cnn_features: CNN features (B, C, D, H, W)
            transformer_features: Transformer features (B, C, D, H, W)
            
        Returns:
            Fused features (B, C, D, H, W)
        """
        if self.fusion_type == "concat":
            # Concatenate along channel dimension
            combined = torch.cat([cnn_features, transformer_features], dim=1)
            return self.fusion(combined)
        
        elif self.fusion_type == "residual":
            # Weighted addition
            return cnn_features + self.alpha * transformer_features
        
        elif self.fusion_type == "hierarchical":
            # Simple addition (identity fusion)
            return cnn_features + transformer_features


class ClassificationHead(nn.Module):
    """Classification head for nodule detection"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            in_channels: Number of input channels
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(ClassificationHead, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, D, H, W)
            
        Returns:
            Class logits (B, num_classes)
        """
        # Global average pooling
        x = self.global_pool(x)  # (B, C, 1, 1, 1)
        x = torch.flatten(x, 1)  # (B, C)
        
        # Classification
        x = self.classifier(x)  # (B, num_classes)
        
        return x


class HybridModel(nn.Module):
    """
    Hybrid 3D CNN-Transformer Model for Lung Nodule Detection
    Combines ResNet3D backbone with multi-scale Transformer processing
    """
    
    def __init__(
        self,
        resnet_depth: int = 18,
        resnet_base_channels: int = 64,
        transformer_dim: int = 512,
        transformer_depth: int = 6,
        transformer_heads: int = 8,
        transformer_mlp_dim: int = 2048,
        use_transformer: bool = True,
        use_multiscale: bool = True,
        fusion_type: str = "hierarchical",
        num_classes: int = 2,
        classifier_hidden_dim: int = 256,
        classifier_dropout: float = 0.5
    ):
        """
        Args:
            resnet_depth: Depth of ResNet (18, 34)
            resnet_base_channels: Base channels for ResNet
            transformer_dim: Transformer model dimension
            transformer_depth: Number of transformer layers
            transformer_heads: Number of attention heads
            transformer_mlp_dim: MLP dimension in transformer
            use_transformer: Whether to use transformer modules
            use_multiscale: Whether to use multi-scale features
            fusion_type: Type of feature fusion
            num_classes: Number of output classes
            classifier_hidden_dim: Hidden dimension in classifier
            classifier_dropout: Dropout rate in classifier
        """
        super(HybridModel, self).__init__()
        
        self.use_transformer = use_transformer
        self.use_multiscale = use_multiscale
        
        # 3D ResNet Backbone
        self.resnet = ResNet3DBackbone(
            depth=resnet_depth,
            in_channels=1,
            base_channels=resnet_base_channels
        )
        
        # Calculate channel dimensions based on ResNet architecture
        if resnet_depth in [18, 34]:
            self.scale_channels = (
                resnet_base_channels * 2,  # layer2: 128
                resnet_base_channels * 4,  # layer3: 256
                resnet_base_channels * 8   # layer4: 512
            )
        
        # Multi-scale Transformer
        if self.use_transformer:
            self.transformer = MultiScaleTransformer(
                in_channels_list=self.scale_channels,
                d_model=transformer_dim,
                num_layers=transformer_depth,
                num_heads=transformer_heads,
                d_ff=transformer_mlp_dim,
                dropout=0.1,
                spatial_sizes=((8, 8, 8), (4, 4, 4), (2, 2, 2))
            )
            
            # Feature fusion modules for each scale
            self.fusion_modules = nn.ModuleList([
                FeatureFusion(channels, fusion_type)
                for channels in self.scale_channels
            ])
        
        # Multi-scale feature aggregation
        if self.use_multiscale:
            # Upsample and combine features
            self.upsample_scale2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(self.scale_channels[1], self.scale_channels[0], kernel_size=1)
            )
            self.upsample_scale3 = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
                nn.Conv3d(self.scale_channels[2], self.scale_channels[0], kernel_size=1)
            )
            
            # Final feature dimension after aggregation
            final_channels = self.scale_channels[0]
        else:
            # Only use finest scale
            final_channels = self.scale_channels[2]
        
        # Classification head
        self.classifier = ClassificationHead(
            in_channels=final_channels,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 1, D, H, W)
            return_features: Whether to return intermediate features
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - 'logits': Classification logits (B, num_classes)
                - 'features': Multi-scale features (optional)
                - 'attention': Attention weights (optional)
        """
        # Extract multi-scale features from ResNet
        cnn_feat1, cnn_feat2, cnn_feat3 = self.resnet(
            x,
            return_multiscale=True
        )
        
        cnn_features = (cnn_feat1, cnn_feat2, cnn_feat3)
        
        # Apply Transformer if enabled
        if self.use_transformer:
            transformer_features, attention_weights = self.transformer(
                cnn_features,
                return_attention=return_attention
            )
            
            # Fuse CNN and Transformer features
            fused_features = []
            for i, (cnn_f, trans_f) in enumerate(zip(cnn_features, transformer_features)):
                fused_f = self.fusion_modules[i](cnn_f, trans_f)
                fused_features.append(fused_f)
            
            feat1, feat2, feat3 = fused_features
        else:
            feat1, feat2, feat3 = cnn_features
            attention_weights = None
        
        # Multi-scale feature aggregation
        if self.use_multiscale:
            # Upsample coarser scales to finest scale
            feat2_up = self.upsample_scale2(feat2)  # (B, 128, 8, 8, 8)
            feat3_up = self.upsample_scale3(feat3)  # (B, 128, 8, 8, 8)
            
            # Hierarchical fusion: coarse -> medium -> fine
            combined = feat3_up + feat2_up + feat1
            final_features = combined
        else:
            # Use only the finest scale
            final_features = feat3
        
        # Classification
        logits = self.classifier(final_features)
        
        # Prepare output
        output = {
            'logits': logits
        }
        
        if return_features:
            output['features'] = {
                'scale1': feat1,
                'scale2': feat2,
                'scale3': feat3,
                'final': final_features
            }
        
        if return_attention and attention_weights is not None:
            output['attention'] = attention_weights
        
        return output


def create_model(config) -> HybridModel:
    """
    Create model from configuration
    
    Args:
        config: Configuration object
        
    Returns:
        HybridModel instance
    """
    model = HybridModel(
        resnet_depth=config.model.resnet_depth,
        resnet_base_channels=config.model.resnet_base_channels,
        transformer_dim=config.model.transformer_dim,
        transformer_depth=config.model.transformer_depth,
        transformer_heads=config.model.transformer_heads,
        transformer_mlp_dim=config.model.transformer_mlp_dim,
        use_transformer=config.model.use_transformer,
        use_multiscale=config.model.use_multiscale,
        fusion_type=config.model.fusion_type,
        num_classes=config.model.num_classes,
        classifier_hidden_dim=config.model.classifier_hidden_dim,
        classifier_dropout=config.model.classifier_dropout
    )
    
    return model


# Testing and validation
if __name__ == "__main__":
    print("="*70)
    print("Testing Hybrid CNN-Transformer Model")
    print("="*70)
    
    # Create model
    model = HybridModel(
        resnet_depth=18,
        resnet_base_channels=64,
        transformer_dim=512,
        transformer_depth=6,
        transformer_heads=8,
        transformer_mlp_dim=2048,
        use_transformer=True,
        use_multiscale=True,
        fusion_type="hierarchical",
        num_classes=2
    )
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 64, 64, 64)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(
            dummy_input,
            return_features=True,
            return_attention=True
        )
    
    print(f"\n✅ Model Output:")
    
    # Test inference time
    import time
    model.eval()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    end = time.time()
    
    avg_time = (end - start) / 10 * 1000
    print(f"  Average inference time: {avg_time:.2f} ms")
    
    # Test with transformer disabled
    print(f"\n{'='*70}")
    print("Testing with Transformer Disabled")
    print("="*70)
    
    model_no_transformer = HybridModel(
        resnet_depth=18,
        use_transformer=False,
        use_multiscale=True,
        num_classes=2
    )
    
    with torch.no_grad():
        output_no_trans = model_no_transformer(dummy_input)
    
    no_trans_params = sum(p.numel() for p in model_no_transformer.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n✅ CNN-only model:")
    print(f"  Output shape: {output_no_trans['logits'].shape}")
    print(f"  Parameters: {no_trans_params:,}")
    print(f"  Reduction: {(total_params - no_trans_params):,} parameters")
    
    print("\n" + "="*70)
    print("✅ Hybrid Model Implementation Complete!")
    print("="*70)