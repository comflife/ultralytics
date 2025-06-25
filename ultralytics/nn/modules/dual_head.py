"""
Dual-stream detection head for RGBT (RGB-Thermal) models.
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv


class DualStreamDetect(Detect):
    """
    YOLO Dual-Stream Detect head for RGBT detection models.
    
    This head processes features from both wide and narrow camera streams,
    fuses them, and generates detections.
    """

    def __init__(self, nc=80, ch=(), fusion_method='concat'):
        """
        Initialize Dual-Stream detection head.
        
        Args:
            nc (int): Number of classes
            ch (tuple): Channel dimensions from backbone 
            fusion_method (str): Feature fusion method ('concat', 'add', 'attention')
        """
        # Initialize parent class with concatenated channels if using concat fusion
        if fusion_method == 'concat':
            ch_fused = tuple(c * 2 for c in ch)  # Double channels for concatenation
        else:
            ch_fused = ch
        
        super().__init__(nc, ch_fused)
        
        self.fusion_method = fusion_method
        self.original_ch = ch
        
        # Fusion modules for each detection layer
        if fusion_method == 'attention':
            self.fusion_modules = nn.ModuleList([
                AttentionFusion(ch[i]) for i in range(len(ch))
            ])
        elif fusion_method == 'conv':
            # Use 1x1 conv to reduce concatenated features back to original size
            self.fusion_modules = nn.ModuleList([
                Conv(ch[i] * 2, ch[i], 1) for i in range(len(ch))
            ])
        else:
            self.fusion_modules = nn.ModuleList()  # Empty ModuleList for other methods

    def forward(self, x):
        """
        Forward pass for dual-stream detection.
        
        Args:
            x (tuple): Tuple of (wide_features, narrow_features)
                      Each is a list of feature maps from different scales
                      
        Returns:
            Same as parent Detect.forward()
        """
        if isinstance(x, tuple) and len(x) == 2:
            wide_features, narrow_features = x
            
            # Fuse features from both streams
            fused_features = []
            for i in range(len(wide_features)):
                wide_feat = wide_features[i]
                narrow_feat = narrow_features[i]
                
                if self.fusion_method == 'concat':
                    # Simple concatenation along channel dimension
                    fused_feat = torch.cat([wide_feat, narrow_feat], dim=1)
                    
                elif self.fusion_method == 'add':
                    # Element-wise addition
                    fused_feat = wide_feat + narrow_feat
                    
                elif self.fusion_method == 'attention':
                    # Attention-based fusion
                    fused_feat = self.fusion_modules[i](wide_feat, narrow_feat)
                    
                elif self.fusion_method == 'conv':
                    # Concat then conv to reduce channels
                    concat_feat = torch.cat([wide_feat, narrow_feat], dim=1)
                    fused_feat = self.fusion_modules[i](concat_feat)
                    
                else:
                    raise ValueError(f"Unknown fusion method: {self.fusion_method}")
                
                fused_features.append(fused_feat)
            
            # Use fused features for detection
            return super().forward(fused_features)
        else:
            # Fallback to single stream
            return super().forward(x)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion module for dual-stream features.
    """
    
    def __init__(self, channels):
        """
        Initialize attention fusion module.
        
        Args:
            channels (int): Number of input channels
        """
        super().__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels * 2, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention  
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion_conv = Conv(channels * 2, channels, 1)
        
    def forward(self, wide_feat, narrow_feat):
        """
        Forward pass for attention fusion.
        
        Args:
            wide_feat (torch.Tensor): Wide camera features
            narrow_feat (torch.Tensor): Narrow camera features
            
        Returns:
            torch.Tensor: Fused features
        """
        # Concatenate features
        concat_feat = torch.cat([wide_feat, narrow_feat], dim=1)
        
        # Channel attention
        ca_weights = self.channel_attention(concat_feat)
        concat_feat = concat_feat * ca_weights
        
        # Spatial attention
        avg_pool = torch.mean(concat_feat, dim=1, keepdim=True)
        max_pool, _ = torch.max(concat_feat, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa_weights = self.spatial_attention(spatial_input)
        concat_feat = concat_feat * sa_weights
        
        # Final fusion
        fused_feat = self.fusion_conv(concat_feat)
        
        return fused_feat


class DualStreamFusion(nn.Module):
    """
    Simple fusion module that can be inserted between backbone and head.
    """
    
    def __init__(self, fusion_method='concat'):
        """
        Initialize fusion module.
        
        Args:
            fusion_method (str): Fusion method ('concat', 'add', 'weighted_add')
        """
        super().__init__()
        self.fusion_method = fusion_method
        
        if fusion_method == 'weighted_add':
            # Learnable weights for weighted addition
            self.wide_weight = nn.Parameter(torch.ones(1))
            self.narrow_weight = nn.Parameter(torch.ones(1))
            
    def forward(self, wide_features, narrow_features):
        """
        Fuse features from wide and narrow streams.
        
        Args:
            wide_features (list): Features from wide camera stream
            narrow_features (list): Features from narrow camera stream
            
        Returns:
            list: Fused features
        """
        fused_features = []
        
        for wide_feat, narrow_feat in zip(wide_features, narrow_features):
            if self.fusion_method == 'concat':
                fused_feat = torch.cat([wide_feat, narrow_feat], dim=1)
            elif self.fusion_method == 'add':
                fused_feat = wide_feat + narrow_feat
            elif self.fusion_method == 'weighted_add':
                fused_feat = self.wide_weight * wide_feat + self.narrow_weight * narrow_feat
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")
                
            fused_features.append(fused_feat)
            
        return fused_features
