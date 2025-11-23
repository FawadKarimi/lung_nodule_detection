"""
Multi-Scale Transformer Module for 3D Medical Imaging
Implements self-attention mechanisms with 3D positional encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding3D(nn.Module):
    """Learnable 3D positional encoding"""
    
    def __init__(self, d_model: int, max_depth: int = 8, max_height: int = 8, max_width: int = 8):
        """
        Args:
            d_model: Dimension of the model
            max_depth: Maximum depth dimension
            max_height: Maximum height dimension
            max_width: Maximum width dimension
        """
        super(PositionalEncoding3D, self).__init__()
        
        self.d_model = d_model
        
        # Calculate dimensions
        dim_per_axis = d_model // 3
        extra_dim = d_model - 3 * dim_per_axis
        
        # Learnable positional embeddings for each dimension
        self.pos_embed_d = nn.Parameter(torch.randn(max_depth, dim_per_axis))
        self.pos_embed_h = nn.Parameter(torch.randn(max_height, dim_per_axis))
        self.pos_embed_w = nn.Parameter(torch.randn(max_width, dim_per_axis))
        
        # Remaining dimension if d_model not divisible by 3
        if extra_dim > 0:
            self.pos_embed_extra = nn.Parameter(torch.randn(max_depth, max_height, max_width, extra_dim))
        else:
            self.pos_embed_extra = None
        
        # Initialize
        nn.init.normal_(self.pos_embed_d, std=0.02)
        nn.init.normal_(self.pos_embed_h, std=0.02)
        nn.init.normal_(self.pos_embed_w, std=0.02)
        nn.init.normal_(self.pos_embed_extra, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, C) where N = D*H*W
            
        Returns:
            Positional encoding (B, N, C)
        """
        B, N, C = x.shape
        
        # Determine spatial dimensions
        D = self.pos_embed_d.shape[0]
        H = self.pos_embed_h.shape[0]
        W = self.pos_embed_w.shape[0]
        
        assert D * H * W == N, f"Mismatch: {D}*{H}*{W} != {N}"
        
        # Create positional encoding
        pos_enc = []
        
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    # Concatenate positional embeddings from each dimension
                    pos_parts = [
                        self.pos_embed_d[d],
                        self.pos_embed_h[h],
                        self.pos_embed_w[w]
                    ]
                    if self.pos_embed_extra is not None:
                        pos_parts.append(self.pos_embed_extra[d, h, w])
                    pos = torch.cat(pos_parts)
                    pos_enc.append(pos)
        
        pos_enc = torch.stack(pos_enc)  # (N, C)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)  # (B, N, C)
        
        return pos_enc


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadSelfAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, N, C)
            mask: Attention mask (B, N, N) or None
            
        Returns:
            output: (B, N, C)
            attention_weights: (B, num_heads, N, N)
        """
        B, N, C = x.shape
        
        # Linear projections and reshape for multi-head attention
        Q = self.q_linear(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, N, d_k)
        K = self.k_linear(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, N, d_k)
        V = self.v_linear(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, N, d_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (B, H, N, d_k)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        
        # Final linear projection
        output = self.out_linear(context)
        
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model
            d_ff: Dimension of feed-forward network
            dropout: Dropout rate
        """
        super(FeedForwardNetwork, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, C)
            
        Returns:
            Output tensor (B, N, C)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, N, C)
            mask: Attention mask
            
        Returns:
            output: (B, N, C)
            attention_weights: (B, num_heads, N, N)
        """
        # Self-attention with residual connection and layer norm
        attn_out, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection and layer norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder blocks"""
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        """
        Args:
            num_layers: Number of transformer blocks
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            x: Input tensor (B, N, C)
            mask: Attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            output: (B, N, C)
            attention_weights: List of attention weights if return_attention=True
        """
        attention_weights = [] if return_attention else None
        
        for layer in self.layers:
            x, attn = layer(x, mask)
            if return_attention:
                attention_weights.append(attn)
        
        return x, attention_weights


class MultiScaleTransformer(nn.Module):
    """
    Multi-scale Transformer module for 3D features
    Processes features at different spatial resolutions
    """
    
    def __init__(
        self,
        in_channels_list: Tuple[int, int, int] = (128, 256, 512),
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        spatial_sizes: Tuple[Tuple[int, int, int], ...] = ((8, 8, 8), (4, 4, 4), (2, 2, 2))
    ):
        """
        Args:
            in_channels_list: Input channels for each scale
            d_model: Transformer model dimension
            num_layers: Number of transformer layers per scale
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            spatial_sizes: Spatial sizes for each scale (D, H, W)
        """
        super(MultiScaleTransformer, self).__init__()
        
        self.num_scales = len(in_channels_list)
        self.d_model = d_model
        
        # Input projections for each scale
        self.input_projections = nn.ModuleList([
            nn.Conv3d(in_channels, d_model, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Positional encodings for each scale
        self.pos_encodings = nn.ModuleList([
            PositionalEncoding3D(d_model, d, h, w)
            for d, h, w in spatial_sizes
        ])
        
        # Transformer encoders for each scale
        self.transformers = nn.ModuleList([
            TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
            for _ in range(self.num_scales)
        ])
        
        # Output projections back to original channels
        self.output_projections = nn.ModuleList([
            nn.Conv3d(d_model, in_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
    
    def forward(
        self,
        features: Tuple[torch.Tensor, ...],
        return_attention: bool = False
    ) -> Tuple[Tuple[torch.Tensor, ...], Optional[list]]:
        """
        Args:
            features: Tuple of feature tensors from different scales
                      Each tensor: (B, C, D, H, W)
            return_attention: Whether to return attention weights
            
        Returns:
            transformed_features: Tuple of transformed features
            attention_weights: List of attention weights if return_attention=True
        """
        assert len(features) == self.num_scales
        
        transformed_features = []
        all_attention_weights = [] if return_attention else None
        
        for i, feat in enumerate(features):
            B, C, D, H, W = feat.shape
            
            # Project to d_model
            x = self.input_projections[i](feat)  # (B, d_model, D, H, W)
            
            # Reshape to sequence: (B, d_model, D, H, W) -> (B, N, d_model)
            x = x.flatten(2).transpose(1, 2)  # (B, D*H*W, d_model)
            
            # Add positional encoding
            pos_enc = self.pos_encodings[i](x)
            x = x + pos_enc
            
            # Apply transformer
            x, attn_weights = self.transformers[i](x, return_attention=return_attention)
            
            if return_attention:
                all_attention_weights.append(attn_weights)
            
            # Reshape back to spatial: (B, N, d_model) -> (B, d_model, D, H, W)
            x = x.transpose(1, 2).view(B, self.d_model, D, H, W)
            
            # Project back to original channels
            x = self.output_projections[i](x)  # (B, C, D, H, W)
            
            transformed_features.append(x)
        
        return tuple(transformed_features), all_attention_weights


# Testing
if __name__ == "__main__":
    print("Testing Multi-Scale Transformer...")
    
    # Create dummy multi-scale features (from ResNet)
    feat1 = torch.randn(2, 128, 8, 8, 8)  # Scale 1: (B, 128, 8, 8, 8)
    feat2 = torch.randn(2, 256, 4, 4, 4)  # Scale 2: (B, 256, 4, 4, 4)
    feat3 = torch.randn(2, 512, 2, 2, 2)  # Scale 3: (B, 512, 2, 2, 2)
    
    features = (feat1, feat2, feat3)
    
    # Create transformer
    transformer = MultiScaleTransformer(
        in_channels_list=(128, 256, 512),
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        spatial_sizes=((8, 8, 8), (4, 4, 4), (2, 2, 2))
    )
    
    # Forward pass
    print(f"\n[OK] Input features:")
    for i, f in enumerate(features):
        print(f"  Scale {i+1}: {f.shape}")
    
    transformed, attention = transformer(features, return_attention=True)
    
    print(f"\n[OK] Transformed features:")
    for i, f in enumerate(transformed):
        print(f"  Scale {i+1}: {f.shape}")
    
    print(f"\n[OK] Attention weights collected: {len(attention)} scales")
    
    # Count parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"\n[OK] Transformer parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n[OK] Multi-scale Transformer implemented successfully!")