"""
3D ResNet-18 Backbone for Lung Nodule Detection
Implements 3D residual network for volumetric feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class BasicBlock3D(nn.Module):
    """3D Basic Residual Block"""
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
            downsample: Downsample layer for residual connection
        """
        super(BasicBlock3D, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet3D(nn.Module):
    """3D ResNet Architecture"""
    
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 2
    ):
        """
        Args:
            block: Residual block type (BasicBlock3D or Bottleneck3D)
            layers: Number of blocks in each layer
            in_channels: Number of input channels (1 for CT)
            base_channels: Base number of channels
            num_classes: Number of output classes
        """
        super(ResNet3D, self).__init__()
        
        self.in_channels = base_channels
        self.base_channels = base_channels
        
        # Initial convolution block
        self.conv1 = nn.Conv3d(
            in_channels,
            base_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(
        self,
        block: nn.Module,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer"""
        downsample = None
        
        # Create downsample layer if needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels * block.expansion)
            )
        
        layers = []
        # First block (may downsample)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet"""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def extract_features(
        self,
        x: torch.Tensor,
        return_multiscale: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Extract features from multiple scales
        
        Args:
            x: Input tensor (B, C, D, H, W)
            return_multiscale: Whether to return features from multiple scales
            
        Returns:
            If return_multiscale=False: features from layer4
            If return_multiscale=True: (layer2_features, layer3_features, layer4_features)
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        
        if return_multiscale:
            # Extract features from multiple scales
            feat_scale1 = self.layer2(x)    # 128 channels, 8x8x8
            feat_scale2 = self.layer3(feat_scale1)  # 256 channels, 4x4x4
            feat_scale3 = self.layer4(feat_scale2)  # 512 channels, 2x2x2
            
            return feat_scale1, feat_scale2, feat_scale3
        else:
            # Only return final features
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            return x


class ResNet3DBackbone(nn.Module):
    """
    3D ResNet Backbone optimized for feature extraction
    Used as the CNN component in hybrid CNN-Transformer architecture
    """
    
    def __init__(
        self,
        depth: int = 18,
        in_channels: int = 1,
        base_channels: int = 64,
        pretrained: bool = False
    ):
        """
        Args:
            depth: ResNet depth (18, 34, 50)
            in_channels: Number of input channels
            base_channels: Base number of channels
            pretrained: Whether to use pretrained weights (2D -> 3D inflation)
        """
        super(ResNet3DBackbone, self).__init__()
        
        if depth == 18:
            layers = [2, 2, 2, 2]
        elif depth == 34:
            layers = [3, 4, 6, 3]
        elif depth == 50:
            raise NotImplementedError("ResNet-50 not implemented yet")
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        
        # Build ResNet
        self.resnet = ResNet3D(
            block=BasicBlock3D,
            layers=layers,
            in_channels=in_channels,
            base_channels=base_channels,
            num_classes=2  # Will be replaced in hybrid model
        )
        
        # Remove classification head (will be replaced)
        delattr(self.resnet, 'fc')
        delattr(self.resnet, 'avgpool')
        
        if pretrained:
            self._load_pretrained_2d()
    
    def _load_pretrained_2d(self):
        """
        Load pretrained 2D ResNet weights and inflate to 3D
        (Simplified version - in practice, you'd load from torchvision)
        """
        print("Note: Pretrained weight loading not implemented in this example")
        print("In practice, load 2D ResNet weights and inflate kernels along depth dimension")
        # Implementation would involve:
        # 1. Load 2D ResNet from torchvision
        # 2. Inflate 2D kernels to 3D by repeating along depth dimension
        # 3. Divide by depth to maintain magnitude
        pass
    
    def forward(self, x: torch.Tensor, return_multiscale: bool = True):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 1, D, H, W)
            return_multiscale: Whether to return multi-scale features
            
        Returns:
            Multi-scale features or single feature map
        """
        return self.resnet.extract_features(x, return_multiscale=return_multiscale)


def resnet18_3d(in_channels: int = 1, base_channels: int = 64, **kwargs) -> ResNet3D:
    """Construct a 3D ResNet-18 model"""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels, base_channels, **kwargs)


def resnet34_3d(in_channels: int = 1, base_channels: int = 64, **kwargs) -> ResNet3D:
    """Construct a 3D ResNet-34 model"""
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], in_channels, base_channels, **kwargs)


# Testing and validation
if __name__ == "__main__":
    print("Testing 3D ResNet-18...")
    
    # Create model
    model = ResNet3DBackbone(depth=18, in_channels=1, base_channels=64)
    
    # Create dummy input (batch_size=2, channels=1, depth=64, height=64, width=64)
    dummy_input = torch.randn(2, 1, 64, 64, 64)
    
    # Test multi-scale feature extraction
    print(f"Input shape: {dummy_input.shape}")
    
    feat1, feat2, feat3 = model(dummy_input, return_multiscale=True)
    
    print(f"\n[OK] Multi-scale features:")
    print(f"  Scale 1 (layer2): {feat1.shape}")  # Should be (2, 128, 8, 8, 8)
    print(f"  Scale 2 (layer3): {feat2.shape}")  # Should be (2, 256, 4, 4, 4)
    print(f"  Scale 3 (layer4): {feat3.shape}")  # Should be (2, 512, 2, 2, 2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[OK] Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test single forward pass timing
    import time
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = model(dummy_input)
        end = time.time()
    
    print(f"  Average inference time: {(end - start) / 10 * 1000:.2f} ms")
    
    print("\n[OK] 3D ResNet-18 backbone implemented successfully!")