# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class MultiStreamConv(nn.Module):
    """Multi-stream convolution for processing dual camera inputs."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize MultiStreamConv with dual stream processing capability."""
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g=g, d=d, act=act)

    def forward(self, x):
        """
        Forward pass for multi-stream convolution.
        
        Args:
            x (torch.Tensor): Input tensor. Can be:
                - 4D: [B, C, H, W] (single stream)
                - 5D: [B, 2, C, H, W] (dual stream)
        
        Returns:
            torch.Tensor: Output tensor in dual stream format [B, 2, C_out, H_out, W_out]
        """
        if x.dim() == 5 and x.shape[1] == 2:
            # Dual stream input: [B, 2, C, H, W]
            B, streams, C, H, W = x.shape
            
            # Reshape to process both streams: [B*2, C, H, W]
            x_reshaped = x.view(B * streams, C, H, W)
            
            # Apply convolution
            out = self.conv(x_reshaped)  # [B*2, C_out, H_out, W_out]
            
            # Reshape back to dual stream format: [B, 2, C_out, H_out, W_out]
            _, C_out, H_out, W_out = out.shape
            out = out.view(B, streams, C_out, H_out, W_out)
            
            return out
            
        elif x.dim() == 4:
            # Single stream input: [B, C, H, W]
            # Convert to dual stream by duplicating
            out = self.conv(x)  # [B, C_out, H_out, W_out]
            # Duplicate for dual stream: [B, 2, C_out, H_out, W_out]
            out = out.unsqueeze(1).repeat(1, 2, 1, 1, 1)
            return out
            
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D input with shape {x.shape}")


class SpatialAlignedMultiStreamConv(nn.Module):
    """공간적으로 정렬된 듀얼 스트림 Conv"""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.cv_wide = Conv(c1, c2, k, s, autopad(k, p, d), g=g, d=d, act=act)
        self.cv_narrow = Conv(c1, c2, k, s, autopad(k, p, d), g=g, d=d, act=act)
        
        # Narrow FOV가 wide image에서 보이는 위치 (YOLO format)
        self.narrow_bbox = {
            'center_x': 0.499289,
            'center_y': 0.499912,
            'width': 0.286041,
            'height': 0.291975
        }

    def place_narrow_in_wide_space(self, narrow_tensor, target_size):
        """Narrow tensor를 wide space의 올바른 위치에 배치"""
        B, C, H_narrow, W_narrow = narrow_tensor.shape
        H_wide, W_wide = target_size
        
        # print(f"DEBUG: Placing narrow {narrow_tensor.shape} into wide space {target_size}")
        
        # 출력 텐서 초기화 (zero padding)
        aligned_narrow = torch.zeros(B, C, H_wide, W_wide, 
                                   device=narrow_tensor.device, 
                                   dtype=narrow_tensor.dtype)
        
        # YOLO bbox -> 픽셀 좌표 변환
        center_x = int(self.narrow_bbox['center_x'] * W_wide)
        center_y = int(self.narrow_bbox['center_y'] * H_wide)
        bbox_w = int(self.narrow_bbox['width'] * W_wide)
        bbox_h = int(self.narrow_bbox['height'] * H_wide)
        
        # 배치할 위치 계산
        x1 = max(0, center_x - bbox_w // 2)
        y1 = max(0, center_y - bbox_h // 2)
        x2 = min(W_wide, x1 + bbox_w)
        y2 = min(H_wide, y1 + bbox_h)
        
        # print(f"DEBUG: Narrow bbox in wide space: ({x1}, {y1}) to ({x2}, {y2})")
        # print(f"DEBUG: Narrow bbox size: {x2-x1}x{y2-y1}")
        
        # Narrow tensor를 bbox 크기로 resize
        target_h = y2 - y1
        target_w = x2 - x1
        
        if target_h > 0 and target_w > 0:
            # Narrow를 target 크기로 resize
            narrow_resized = F.interpolate(narrow_tensor, 
                                         size=(target_h, target_w), 
                                         mode='bilinear', 
                                         align_corners=False)
            
            # Wide space의 해당 위치에 배치
            aligned_narrow[:, :, y1:y2, x1:x2] = narrow_resized
            
            # print(f"DEBUG: Successfully placed narrow at ({x1}:{x2}, {y1}:{y2})")
        # else:
            # print(f"DEBUG: ❌ Invalid target size: {target_w}x{target_h}")
        
        return aligned_narrow

    def forward(self, x):
        # print(f"DEBUG: SpatialAlignedMultiStreamConv input shape: {x.shape}")
        
        if x.dim() == 5 and x.shape[1] == 2:  # [B, 2, C, H, W]
            # print("DEBUG: ✅ Processing dual stream with spatial alignment")
            
            wide_stream = x[:, 0]    # [B, C, H, W]
            narrow_stream = x[:, 1]  # [B, C, H, W]
            
            # print(f"DEBUG: Wide stream shape: {wide_stream.shape}")
            # print(f"DEBUG: Narrow stream shape: {narrow_stream.shape}")
            
            # Wide stream 처리 (그대로)
            wide_out = self.cv_wide(wide_stream)
            
            # Narrow stream을 wide space에 올바른 위치에 배치
            H_wide, W_wide = wide_stream.shape[2], wide_stream.shape[3]
            narrow_aligned = self.place_narrow_in_wide_space(narrow_stream, (H_wide, W_wide))
            
            # 정렬된 narrow stream 처리
            narrow_out = self.cv_narrow(narrow_aligned)
            
            # print(f"DEBUG: Wide output shape: {wide_out.shape}")
            # print(f"DEBUG: Narrow aligned output shape: {narrow_out.shape}")
            
            # Dual stream 형태로 재결합
            output = torch.stack([wide_out, narrow_out], dim=1)  # [B, 2, C_out, H_out, W_out]
            
            # print(f"DEBUG: SpatialAligned output shape: {output.shape}")
            return output
        else:
            # Single stream 처리
            return self.cv_wide(x)


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
            scale_factor (float): Scale factor for the output (multiplier).
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
            x (torch.Tensor or list): Input tensor with dual streams [B, 2, C, H, W] 
                                     or list of tensors from multiple streams.
            
        Returns:
            (torch.Tensor): Fused output tensor.
        """
        # Handle tensor input with dual streams: [B, 2, C, H, W]
        if isinstance(x, torch.Tensor) and x.dim() == 5 and x.shape[1] == 2:
            # Split dual stream tensor into list
            stream1 = x[:, 0]  # [B, C, H, W]
            stream2 = x[:, 1]  # [B, C, H, W]
            streams = [stream1, stream2]
        # Handle list input (multiple separate tensors)
        elif isinstance(x, list) and len(x) >= 2:
            streams = x
        else:
            # Single stream input - just return as is (no fusion needed)
            return x
        
        if self.fusion_type == 'concat':
            # Concatenate along channel dimension
            output = torch.cat(streams, dim=1)
        elif self.fusion_type == 'add':
            # Element-wise addition
            output = sum(streams)
        elif self.fusion_type == 'max':
            # Element-wise maximum
            output = torch.maximum(streams[0], streams[1])
            for i in range(2, len(streams)):
                output = torch.maximum(output, streams[i])
        elif self.fusion_type == 'weighted_sum':
            # Weighted sum with learnable weights
            normalized_weights = torch.softmax(self.weights, dim=0)
            output = sum(normalized_weights[i] * tensor for i, tensor in enumerate(streams[:len(normalized_weights)]))
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
        return output


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

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
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

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
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            (torch.Tensor): Fused kernel
            (torch.Tensor): Fused bias
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Select and return a particular index from input.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
