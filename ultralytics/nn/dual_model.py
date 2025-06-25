"""
Dual-stream YOLO model for RGBT (RGB-Thermal) detection.
"""

import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.dual_head import DualStreamDetect, DualStreamFusion
from ultralytics.utils import LOGGER


class DualStreamYOLO(DetectionModel):
    """
    Dual-stream YOLO model for RGBT detection.
    
    This model has two parallel backbones for processing wide and narrow camera streams,
    with feature fusion before the detection head.
    """
    
    def __init__(self, cfg="yolo11n-dual.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize dual-stream YOLO model.
        
        Args:
            cfg (str): Model configuration file
            ch (int): Number of input channels (typically 3 for RGB)
            nc (int): Number of classes
            verbose (bool): Print model info
        """
        # Initialize as a standard detection model first
        super().__init__(cfg, ch, nc, verbose)
        
        # Mark as dual-stream model
        self.is_dual_stream = True
        
        LOGGER.info("Dual-stream YOLO model initialized")

    def forward(self, x, *args, **kwargs):
        """
        Forward pass for dual-stream model.
        
        Args:
            x: Input can be:
                - Tuple of (wide_img, narrow_img) tensors for dual-stream
                - Single tensor for fallback single-stream mode
                
        Returns:
            Model predictions
        """
        if isinstance(x, tuple) and len(x) == 2:
            # Dual-stream mode
            wide_img, narrow_img = x
            
            # Process through backbone to get features
            wide_features = self._extract_features(wide_img)
            narrow_features = self._extract_features(narrow_img)
            
            # Pass dual features to head
            return self._forward_head((wide_features, narrow_features))
        
        else:
            # Single-stream fallback mode
            return super().forward(x, *args, **kwargs)
    
    def _extract_features(self, x):
        """
        Extract features using the backbone and neck.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            list: Multi-scale feature maps
        """
        y = []  # outputs
        
        # Run through model layers except the head
        for i, m in enumerate(self.model[:-1]):  # Skip the last layer (head)
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            
        return x if isinstance(x, list) else [x]
    
    def _forward_head(self, features):
        """
        Forward pass through detection head.
        
        Args:
            features: Features from backbone/neck
            
        Returns:
            Detection predictions
        """
        head = self.model[-1]  # Get the head (last layer)
        return head(features)

    def predict(self, x, *args, **kwargs):
        """
        Predict method that handles dual-stream input.
        
        Args:
            x: Input images (dual-stream tuple or single tensor)
            
        Returns:
            Predictions
        """
        return self.forward(x, *args, **kwargs)


class SimpleDualStreamYOLO(nn.Module):
    """
    Simplified dual-stream YOLO that reuses existing YOLO backbone.
    """
    
    def __init__(self, backbone_model, fusion_method='concat'):
        """
        Initialize simplified dual-stream model.
        
        Args:
            backbone_model: Pre-trained YOLO model to use as backbone
            fusion_method (str): Feature fusion method
        """
        super().__init__()
        
        # Use the same backbone for both streams
        self.backbone = backbone_model.model[:-1]  # All layers except head
        self.head_layer_idx = len(backbone_model.model) - 1
        
        # Get head configuration
        original_head = backbone_model.model[-1]
        
        # Create dual-stream head
        if hasattr(original_head, 'nc') and hasattr(original_head, 'ch'):
            self.head = DualStreamDetect(
                nc=original_head.nc,
                ch=original_head.ch,
                fusion_method=fusion_method
            )
        else:
            # Fallback to original head
            self.head = original_head
            
        # Copy other important attributes
        self.stride = getattr(backbone_model, 'stride', torch.tensor([32]))
        self.names = getattr(backbone_model, 'names', {})
        self.nc = getattr(backbone_model, 'nc', 80)
        
        # Mark as dual-stream
        self.is_dual_stream = True
        
    def forward(self, x):
        """
        Forward pass through simplified dual-stream model.
        
        Args:
            x (tuple): (wide_img, narrow_img) tensors
            
        Returns:
            Detection predictions
        """
        if isinstance(x, tuple) and len(x) == 2:
            wide_img, narrow_img = x
            
            # Extract features from both streams
            wide_features = self._extract_features(wide_img)
            narrow_features = self._extract_features(narrow_img)
            
            # Pass to dual head
            return self.head((wide_features, narrow_features))
        else:
            # Single stream fallback
            features = self._extract_features(x)
            return self.head(features)
    
    def _extract_features(self, x):
        """Extract multi-scale features using backbone."""
        y = []
        
        for i, m in enumerate(self.backbone):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in getattr(self, 'save', []) else None)
            
        # Return the multi-scale features expected by head
        if isinstance(x, list):
            return x
        else:
            return [x]  # Wrap single output in list


def create_dual_stream_model(base_model_path="yolo11n.pt", fusion_method='concat'):
    """
    Create a dual-stream model from a pre-trained YOLO model.
    
    Args:
        base_model_path (str): Path to pre-trained YOLO model
        fusion_method (str): Feature fusion method
        
    Returns:
        SimpleDualStreamYOLO: Dual-stream model
    """
    from ultralytics import YOLO
    
    # Load base model
    base_model = YOLO(base_model_path)
    
    # Create dual-stream model
    dual_model = SimpleDualStreamYOLO(base_model.model, fusion_method)
    
    LOGGER.info(f"Created dual-stream model with {fusion_method} fusion from {base_model_path}")
    
    return dual_model
