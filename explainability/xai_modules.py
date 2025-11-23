"""
Explainable AI Modules for Lung Nodule Detection
Implements 3D Grad-CAM++, Integrated Gradients, and Attention Visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np


class GradCAMPlusPlus3D:
    """3D Grad-CAM++ for generating visual explanations"""
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Args:
            model: The neural network model
            target_layer: Name of the target layer for CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Layer {self.target_layer} not found in model")
        
        # Register hooks
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate Grad-CAM++ explanation
        
        Args:
            input_tensor: Input tensor (B, C, D, H, W)
            target_class: Target class index (None = predicted class)
            
        Returns:
            CAM heatmap (B, D, H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        logits = output['logits']
        
        # Get target class
        if target_class is None:
            target_class = logits.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        
        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        
        # Backward
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (B, C, D, H, W)
        activations = self.activations  # (B, C, D, H, W)
        
        B, C, D, H, W = gradients.shape
        
        # Calculate Grad-CAM++ weights
        # Second derivative
        grad_2 = gradients.pow(2)
        
        # Third derivative
        grad_3 = gradients.pow(3)
        
        # Calculate alpha (Grad-CAM++ weights)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + (grad_3 * activations).sum(dim=(2, 3, 4), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        
        alpha = alpha_num / alpha_denom
        
        # Calculate weights
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        
        # Generate CAM
        cam = (weights * activations).sum(dim=1)  # (B, D, H, W)
        cam = F.relu(cam)
        
        # Normalize
        cam = self._normalize_cam(cam)
        
        return cam
    
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """Normalize CAM to [0, 1]"""
        B = cam.shape[0]
        
        for i in range(B):
            cam_i = cam[i]
            cam_min = cam_i.min()
            cam_max = cam_i.max()
            
            if cam_max - cam_min > 1e-6:
                cam[i] = (cam_i - cam_min) / (cam_max - cam_min)
            else:
                cam[i] = torch.zeros_like(cam_i)
        
        return cam


class IntegratedGradients3D:
    """3D Integrated Gradients for attribution"""
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: The neural network model
        """
        self.model = model
    
    def generate_attribution(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> torch.Tensor:
        """
        Generate Integrated Gradients attribution
        
        Args:
            input_tensor: Input tensor (B, C, D, H, W)
            target_class: Target class index
            baseline: Baseline input (default: zeros)
            steps: Number of integration steps
            
        Returns:
            Attribution map (B, C, D, H, W)
        """
        self.model.eval()
        
        # Create baseline
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Get target class if not provided
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output['logits'].argmax(dim=1)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps + 1).to(input_tensor.device)
        
        # Storage for gradients
        integrated_grads = torch.zeros_like(input_tensor)
        
        for alpha in alphas[1:]:  # Skip alpha=0
            # Interpolate
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True
            
            # Forward pass
            output = self.model(interpolated)
            logits = output['logits']
            
            # Get target score
            target_score = logits[range(len(target_class)), target_class].sum()
            
            # Backward
            self.model.zero_grad()
            target_score.backward(retain_graph=True)
            
            # Accumulate gradients
            integrated_grads += interpolated.grad / steps
        
        # Multiply by input difference
        attribution = (input_tensor - baseline) * integrated_grads
        
        return attribution
    
    def generate_explanation(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        steps: int = 50
    ) -> torch.Tensor:
        """
        Generate explanation heatmap
        
        Args:
            input_tensor: Input tensor
            target_class: Target class
            steps: Integration steps
            
        Returns:
            Heatmap (B, D, H, W)
        """
        # Get attribution
        attribution = self.generate_attribution(input_tensor, target_class, steps=steps)
        
        # Take absolute value and sum across channels
        heatmap = attribution.abs().sum(dim=1)  # (B, D, H, W)
        
        # Normalize
        heatmap = self._normalize_heatmap(heatmap)
        
        return heatmap
    
    def _normalize_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Normalize heatmap to [0, 1]"""
        B = heatmap.shape[0]
        
        for i in range(B):
            heat_i = heatmap[i]
            heat_min = heat_i.min()
            heat_max = heat_i.max()
            
            if heat_max - heat_min > 1e-6:
                heatmap[i] = (heat_i - heat_min) / (heat_max - heat_min)
            else:
                heatmap[i] = torch.zeros_like(heat_i)
        
        return heatmap


class AttentionVisualizer:
    """Visualize attention weights from Transformer"""
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: The neural network model with Transformer
        """
        self.model = model
    
    def visualize_attention(
        self,
        input_tensor: torch.Tensor,
        aggregation: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Visualize attention weights
        
        Args:
            input_tensor: Input tensor (B, C, D, H, W)
            aggregation: How to aggregate attention ('mean', 'max', 'rollout')
            
        Returns:
            Dictionary of attention maps for each scale
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass with attention
            output = self.model(input_tensor, return_attention=True)
        
        if 'attention' not in output:
            raise ValueError("Model does not return attention weights")
        
        attention_weights = output['attention']
        
        # Process attention for each scale
        attention_maps = {}
        
        for scale_idx, scale_attention in enumerate(attention_weights):
            # scale_attention is a list of attention weights from each layer
            # Each has shape (B, num_heads, N, N)
            
            if aggregation == 'mean':
                # Average across layers and heads
                scale_map = self._aggregate_mean(scale_attention)
            elif aggregation == 'max':
                # Max across layers and heads
                scale_map = self._aggregate_max(scale_attention)
            elif aggregation == 'rollout':
                # Attention rollout
                scale_map = self._aggregate_rollout(scale_attention)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            attention_maps[f'scale_{scale_idx + 1}'] = scale_map
        
        return attention_maps
    
    def _aggregate_mean(self, attention_list: List[torch.Tensor]) -> torch.Tensor:
        """Average attention across layers and heads"""
        # Stack and average
        stacked = torch.stack(attention_list)  # (num_layers, B, num_heads, N, N)
        
        # Average across layers and heads
        attention = stacked.mean(dim=(0, 2))  # (B, N, N)
        
        # Average across query dimension
        attention_map = attention.mean(dim=1)  # (B, N)
        
        return attention_map
    
    def _aggregate_max(self, attention_list: List[torch.Tensor]) -> torch.Tensor:
        """Max attention across layers and heads"""
        # Stack
        stacked = torch.stack(attention_list)  # (num_layers, B, num_heads, N, N)
        
        # Max across layers and heads
        attention, _ = stacked.max(dim=0)  # (B, num_heads, N, N)
        attention, _ = attention.max(dim=1)  # (B, N, N)
        
        # Average across query dimension
        attention_map = attention.mean(dim=1)  # (B, N)
        
        return attention_map
    
    def _aggregate_rollout(self, attention_list: List[torch.Tensor]) -> torch.Tensor:
        """Attention rollout (recursive multiplication)"""
        # Start with identity
        result = None
        
        for attention in attention_list:
            # attention: (B, num_heads, N, N)
            # Average across heads
            attn = attention.mean(dim=1)  # (B, N, N)
            
            # Add residual connection
            I = torch.eye(attn.shape[1]).to(attn.device).unsqueeze(0)
            attn = 0.5 * attn + 0.5 * I
            
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            # Multiply
            if result is None:
                result = attn
            else:
                result = torch.matmul(result, attn)
        
        # Average across query dimension
        attention_map = result.mean(dim=1)  # (B, N)
        
        return attention_map


class MultiModalExplainer:
    """Multi-modal explanation fusion"""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str = 'resnet.layer4',
        fusion_weights: Tuple[float, float, float] = (0.40, 0.35, 0.25)
    ):
        """
        Args:
            model: The neural network model
            target_layer: Target layer for Grad-CAM
            fusion_weights: Weights for (GradCAM, IG, Attention)
        """
        self.model = model
        self.fusion_weights = fusion_weights
        
        # Initialize explainability modules
        self.gradcam = GradCAMPlusPlus3D(model, target_layer)
        self.integrated_gradients = IntegratedGradients3D(model)
        self.attention_viz = AttentionVisualizer(model)
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate multi-modal explanation
        
        Args:
            input_tensor: Input tensor (B, C, D, H, W)
            target_class: Target class
            return_individual: Whether to return individual explanations
            
        Returns:
            Dictionary containing fused and individual explanations
        """
        B, C, D, H, W = input_tensor.shape
        
        # Generate Grad-CAM
        gradcam_map = self.gradcam.generate_cam(input_tensor, target_class)
        
        # Generate Integrated Gradients
        ig_map = self.integrated_gradients.generate_explanation(
            input_tensor,
            target_class,
            steps=50
        )
        
        # Generate Attention visualization
        attention_maps = self.attention_viz.visualize_attention(input_tensor)
        
        # Use finest scale attention (scale_1)
        attention_map = attention_maps['scale_1']
        
        # Reshape attention to spatial dimensions
        # attention_map is (B, N)
        N = attention_map.shape[1]
        # Calculate spatial dimensions assuming cubic volume
        # For scale_1 (8x8x8), N=512. 512^(1/3) = 8
        d = h = w = int(round(N ** (1/3)))
        
        attention_map = attention_map.view(B, d, h, w)
        
        # Upsample all to input resolution if needed
        gradcam_map = F.interpolate(
            gradcam_map.unsqueeze(1),
            size=(D, H, W),
            mode='trilinear',
            align_corners=True
        ).squeeze(1)
        
        ig_map = F.interpolate(
            ig_map.unsqueeze(1),
            size=(D, H, W),
            mode='trilinear',
            align_corners=True
        ).squeeze(1)
        
        attention_map = F.interpolate(
            attention_map.unsqueeze(1),
            size=(D, H, W),
            mode='trilinear',
            align_corners=True
        ).squeeze(1)
        
        # Normalize each explanation
        gradcam_map = self._normalize(gradcam_map)
        ig_map = self._normalize(ig_map)
        attention_map = self._normalize(attention_map)
        
        # Fuse explanations
        w1, w2, w3 = self.fusion_weights
        fused_explanation = (
            w1 * gradcam_map +
            w2 * ig_map +
            w3 * attention_map
        )
        
        # Normalize fused explanation
        fused_explanation = self._normalize(fused_explanation)
        
        result = {'fused': fused_explanation}
        
        if return_individual:
            result.update({
                'gradcam': gradcam_map,
                'integrated_gradients': ig_map,
                'attention': attention_map
            })
        
        return result
    
    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1]"""
        B = tensor.shape[0]
        
        for i in range(B):
            t_i = tensor[i]
            t_min = t_i.min()
            t_max = t_i.max()
            
            if t_max - t_min > 1e-6:
                tensor[i] = (t_i - t_min) / (t_max - t_min)
            else:
                tensor[i] = torch.zeros_like(t_i)
        
        return tensor


# Testing
if __name__ == "__main__":
    print("="*70)
    print("Testing Explainability Modules")
    print("="*70)
    
    from models.hybrid_model import HybridModel
    
    # Create model
    model = HybridModel(
        resnet_depth=18,
        use_transformer=True,
        use_multiscale=True,
        num_classes=2
    )
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(2, 1, 64, 64, 64)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Initialize multi-modal explainer
    explainer = MultiModalExplainer(
        model=model,
        target_layer='resnet.resnet.layer4',
        fusion_weights=(0.40, 0.35, 0.25)
    )
    
    # Generate explanations
    print("\nGenerating explanations...")
    explanations = explainer.explain(
        dummy_input,
        target_class=None,
        return_individual=True
    )
    
    print(f"\n[OK] Generated Explanations:")
    for name, expl in explanations.items():
        print(f"  {name}: {expl.shape}")
    
    print("\n[OK] Explainability modules implemented successfully!")
    print("="*70)