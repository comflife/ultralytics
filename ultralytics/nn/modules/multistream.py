# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Multi-stream modules for multi-sensor fusion."""

import torch
import torch.nn as nn

from .conv import Conv
from .block import C3, Bottleneck

class MultiStreamConv(nn.Module):
    """
    Multi-stream standard convolution module using the Conv module from ultralytics.
    Designed to process multiple input streams with the same architecture.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize MultiStreamConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, d, act)
        self.out_channels = c2  # Track the output channels for model parsing

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.
        
        For multi-stream models, this is called separately on each stream.
        The actual fusion happens in the Fusion module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(x)

class MultiStreamMaxPool2d(nn.Module):
    """Multi-stream MaxPool2d module for multi-sensor inputs."""

    def __init__(self, k=2, s=2):
        """
        Initialize MultiStreamMaxPool2d with given kernel size and stride.
        
        Args:
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=k, stride=s)

    def forward(self, x):
        """
        Apply max pooling to input tensor.
        
        For multi-stream models, this is called separately on each stream.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.pool(x)

class MultiStreamC3(C3):
    """Multi-stream CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize multi-stream C3 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        # No need to override anything as the process is the same
        # Each stream will use the same C3 architecture

class Fusion(nn.Module):
    """
    Fusion module to combine features from multiple streams.
    
    Supports different fusion strategies: 'concat', 'add', 'max', 'weighted_sum'
    """

    def __init__(self, fusion_type='concat', scale_factor=1.0):
        """
        Initialize Fusion module.
        
        Args:
            fusion_type (str): Type of fusion - 'concat', 'add', 'max', 'weighted_sum'
            scale_factor (float): Scale factor for the output (multiplier). For 'concat', 
                                 output channels will be multiplied by this value.
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.scale_factor = scale_factor
        
        # For weighted sum, create learnable weights
        if fusion_type == 'weighted_sum':
            self.weights = nn.Parameter(torch.ones(2))  # Initialize with equal weights
            
    def forward(self, x):
        """
        Fuse multiple input streams.
        
        Args:
            x (list): List of input tensors from multiple streams.
            
        Returns:
            (torch.Tensor): Fused output tensor.
        """
        assert isinstance(x, list) and len(x) >= 2, "Fusion requires at least 2 input streams"
        
        if self.fusion_type == 'concat':
            # Concatenate along channel dimension
            output = torch.cat(x, dim=1)
        elif self.fusion_type == 'add':
            # Element-wise addition
            output = sum(x)
        elif self.fusion_type == 'max':
            # Element-wise maximum
            output = torch.maximum(x[0], x[1])
            for i in range(2, len(x)):
                output = torch.maximum(output, x[i])
        elif self.fusion_type == 'weighted_sum':
            # Weighted sum with learnable weights
            normalized_weights = torch.softmax(self.weights, dim=0)
            output = sum(normalized_weights[i] * tensor for i, tensor in enumerate(x[:len(normalized_weights)]))
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
        return output
