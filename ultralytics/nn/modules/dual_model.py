# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Dual-stream YOLO model wrapper for multi-camera processing.
"""

import torch
import torch.nn as nn
from ultralytics.utils import LOGGER


class DualStreamWrapper(nn.Module):
    """
    A wrapper for dual-stream YOLO models.
    
    This wrapper takes a standard YOLO model and adapts it to receive two input streams
    (e.g., wide and narrow camera inputs) and process them using the model's MultiStream 
    modules.
    
    Attributes:
        model (nn.Module): The base YOLO model containing MultiStream modules.
        requires_dual_input (bool): Whether the model requires dual-stream inputs.
        stride (torch.Tensor): Model stride from the wrapped model.
        names (dict): Class names from the wrapped model.
    """
    
    def __init__(self, model):
        """
        Initialize the DualStreamWrapper with a base model.
        
        Args:
            model (nn.Module): A YOLO model containing MultiStream modules.
        """
        super().__init__()
        self.model = model
        # Check if model contains MultiStream modules
        self.requires_dual_input = self._check_for_multistream_modules(model)
        # Inherit important attributes from base model
        self.stride = getattr(model, 'stride', None)
        self.names = getattr(model, 'names', {})
        
        if self.requires_dual_input:
            LOGGER.info(f"Initialized dual-stream wrapper for model with MultiStream modules")
        else:
            LOGGER.warning(f"No MultiStream modules found in model, will process single stream only")
    
    @staticmethod
    def _check_for_multistream_modules(model):
        """
        Check if the model contains MultiStream modules.
        
        Args:
            model (nn.Module): The model to check.
        
        Returns:
            bool: True if MultiStream modules are found, False otherwise.
        """
        for module in model.modules():
            if 'MultiStream' in str(type(module).__name__):
                return True
        return False
    
    def forward(self, x):
        """
        Forward pass with support for both single and dual-stream inputs.
        
        Args:
            x: Either a single tensor [batch, channels, height, width] or 
               a tuple/list of two tensors [wide_cam, narrow_cam] for dual-stream mode.
        
        Returns:
            Model outputs (same as the wrapped model would return).
        """
        if self.requires_dual_input:
            # For dual-stream models, check input format
            if isinstance(x, (list, tuple)) and len(x) == 2:
                # Proper dual-stream input: [wide_cam, narrow_cam]
                return self.model(x)
            elif isinstance(x, torch.Tensor):
                # Single tensor provided for dual-stream model
                LOGGER.warning("Single input tensor provided to dual-stream model. "
                               "For optimal results, provide [wide_cam, narrow_cam]")
                # Create a duplicate of the same tensor as fallback
                return self.model([x, x])
            else:
                raise ValueError(f"Unsupported input type for dual-stream model: {type(x)}")
        else:
            # For regular models, use normal forward pass
            return self.model(x)
    
    def predict(self, x, *args, **kwargs):
        """Pass-through to model.predict with dual-stream handling."""
        if hasattr(self.model, 'predict'):
            return self.model.predict(x, *args, **kwargs)
        else:
            return self(x)
    
    def train(self, mode=True):
        """Set training mode for the model."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode for the model."""
        self.model.eval()
        return self
    
    def fuse(self, *args, **kwargs):
        """Fuse model Conv2d and BatchNorm2d layers."""
        if hasattr(self.model, 'fuse'):
            return self.model.fuse(*args, **kwargs)
        return self
        
    def modules(self):
        """Return all modules in the model."""
        return self.model.modules()
