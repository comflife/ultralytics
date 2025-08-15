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
    "MultiStreamConv",
)



# --- Helper Functions and Basic Building Blocks ---

def autopad(k, p=None, d=1):
    """
    Pad to 'same' shape outputs.
    Calculates padding for a convolution layer to maintain spatial dimensions with stride=1.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# --- Custom Backbone Modules ---

class MultiStreamConv(nn.Module):
    """
    NPU-friendly dual-stream convolution.
    Uses standard tensor slicing for channel splitting.
    """
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        if c1 % 2 != 0 or c2 % 2 != 0:
            raise ValueError("Input and output channels must be divisible by 2 for MultiStreamConv.")

        self.c1_half = c1 // 2
        c2_half = c2 // 2
        
        # 🔴 Conv로 구현했던 Split 부분을 완전히 제거
        
        # 각 스트림을 처리할 Conv 레이어만 정의
        self.conv1 = Conv(self.c1_half, c2_half, k, s, p=None, g=g, d=d, act=act)
        self.conv2 = Conv(self.c1_half, c2_half, k, s, p=None, g=g, d=d, act=act)

    def forward(self, x):
        # 🔴 표준 텐서 슬라이싱으로 6채널 입력을 3채널씩 두 개로 분리
        # x의 형태: [Batch, 6, H, W]
        stream1 = x[:, :self.c1_half, :, :]  # 앞쪽 3개 채널
        stream2 = x[:, self.c1_half:, :, :]  # 뒤쪽 3개 채널
        
        out1 = self.conv1(stream1)
        out2 = self.conv2(stream2)
        
        return torch.cat([out1, out2], dim=1)

# class MultiStreamConv(nn.Module):
#     """
#     Custom dual-stream convolution that splits channels, processes them in parallel,
#     and concatenates the results. Uses standard 'same' padding.
#     """
#     def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
#         super().__init__()
#         if c1 % 2 != 0 or c2 % 2 != 0:
#             raise ValueError("Input and output channels must be divisible by 2 for MultiStreamConv.")

#         c1_half = c1 // 2
#         c2_half = c2 // 2

#         # Fixed 1x1 convs for channel splitting (NPU-friendly)
#         w1 = torch.zeros(c1_half, c1, 1, 1)
#         w2 = torch.zeros(c1_half, c1, 1, 1)
#         w1[:, :c1_half, 0, 0] = torch.eye(c1_half)
#         w2[:, c1_half:, 0, 0] = torch.eye(c1_half)

#         self.split1 = nn.Conv2d(c1, c1_half, kernel_size=1, stride=1, padding=0, bias=False)
#         self.split2 = nn.Conv2d(c1, c1_half, kernel_size=1, stride=1, padding=0, bias=False)
#         self.split1.weight = nn.Parameter(w1, requires_grad=False)
#         self.split2.weight = nn.Parameter(w2, requires_grad=False)

#         # Processing convs with standard 'same' padding
#         self.conv1 = Conv(c1_half, c2_half, k, s, p=None, g=g, d=d, act=act)
#         self.conv2 = Conv(c1_half, c2_half, k, s, p=None, g=g, d=d, act=act)

#     def forward(self, x):
#         stream1 = self.split1(x)
#         stream2 = self.split2(x)
#         out1 = self.conv1(stream1)
#         out2 = self.conv2(stream2)
#         return torch.cat([out1, out2], dim=1)


class SpatialAlignedMultiStreamConv(nn.Module):
    """
    NPU-friendly spatially-aligned dual-stream conv. (REVISED STATIC VERSION)
    - Replaced F.conv_transpose2d with nn.Upsample + nn.Conv2d.
    - Removed dynamic mask slicing, simplifying the fusion to element-wise addition.
    - This module does NOT downsample (stride is fixed to 1).
    """
    def __init__(self, c1, c2, max_hw, k=1, p=None, d=1, act=True): # max_hw는 더 이상 사용되지 않지만, YAML 호환성을 위해 남겨둠
        super().__init__()
        if c1 % 2 != 0:
            raise ValueError("Input channels must be divisible by 2 for SpatialAlignedMultiStreamConv.")
        
        self.c_half = c1 // 2
        
        # 1. 채널 분리 (기존과 동일)
        w_wide = torch.zeros(self.c_half, c1, 1, 1, dtype=torch.float32)
        w_wide[:, :self.c_half, 0, 0] = torch.eye(self.c_half)
        self.split_wide = nn.Conv2d(c1, self.c_half, 1, 1, 0, bias=False)
        self.split_wide.weight = nn.Parameter(w_wide, requires_grad=False)

        w_narrow = torch.zeros(self.c_half, c1, 1, 1, dtype=torch.float32)
        w_narrow[:, self.c_half:, 0, 0] = torch.eye(self.c_half)
        self.split_narrow = nn.Conv2d(c1, self.c_half, 1, 1, 0, bias=False)
        self.split_narrow.weight = nn.Parameter(w_narrow, requires_grad=False)
        
        # 2. 각 스트림 독립 처리 (기존과 동일)
        self.wide_processor   = Conv(self.c_half, self.c_half, k=3, s=1, p=1, g=1, d=d, act=act)
        self.narrow_processor = Conv(self.c_half, self.c_half, k=3, s=1, p=1, g=1, d=d, act=act)

        # 3. [수정] Upsampling 방식 변경: conv_transpose2d 대신 표준 Upsample + Conv 사용
        #    이 방식이 NPU 호환성이 훨씬 높습니다.
        self.upsampler = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(self.c_half, self.c_half, k=3, s=1, p=1, act=act)
        )
        
        # 4. [수정] 최종 Fusion Conv (기존과 유사)
        #    입력 채널이 wide_proc(c_half)와 upsampler(c_half)의 합이므로 c1이 됩니다.
        self.fusion_conv = Conv(self.c_half, c2, k=k, s=1, p=p, g=1, d=d, act=act)

    def forward(self, x):
        wide = self.split_wide(x)
        narrow = self.split_narrow(x)
        
        wide_proc = self.wide_processor(wide)
        narrow_proc = self.narrow_processor(narrow)
        
        # 5. [수정] 동적 마스킹 및 정렬 로직을 단순한 Upsample + Add로 대체
        #    가장 확실하고 NPU 친화적인 퓨전 방식입니다.
        narrow_upsampled = self.upsampler(narrow_proc)
        
        # 해상도를 맞추기 위한 Crop (만약 upsample 결과가 1픽셀 크다면)
        if narrow_upsampled.shape[2:] != wide_proc.shape[2:]:
            target_h, target_w = wide_proc.shape[2:]
            narrow_upsampled = narrow_upsampled[:, :, :target_h, :target_w]

        fused = wide_proc + narrow_upsampled
        
        out = self.fusion_conv(fused)
        
        return out

# class SpatialAlignedMultiStreamConv(nn.Module):
#     """
#     NPU-friendly spatially-aligned dual-stream conv. This module only performs
#     feature fusion and does NOT downsample (stride is fixed to 1).
#     Downsampling should be handled by a subsequent standard Conv layer in the YAML file.
#     """
#     def __init__(self, c1, c2, max_hw, k=1, p=None, d=1, act=True):
#         super().__init__()
#         if c1 % 2 != 0:
#             raise ValueError("Input channels must be divisible by 2 for SpatialAlignedMultiStreamConv.")
        
#         self.c_half = c1 // 2
        
#         # Channel splitting for wide/narrow streams
#         w_wide = torch.zeros(self.c_half, c1, 1, 1, dtype=torch.float32)
#         w_wide[:, :self.c_half, 0, 0] = torch.eye(self.c_half)
#         self.split_wide = nn.Conv2d(c1, self.c_half, 1, 1, 0, bias=False)
#         self.split_wide.weight = nn.Parameter(w_wide, requires_grad=False)

#         w_narrow = torch.zeros(self.c_half, c1, 1, 1, dtype=torch.float32)
#         w_narrow[:, self.c_half:, 0, 0] = torch.eye(self.c_half)
#         self.split_narrow = nn.Conv2d(c1, self.c_half, 1, 1, 0, bias=False)
#         self.split_narrow.weight = nn.Parameter(w_narrow, requires_grad=False)
        
#         # Independent processors for each stream
#         self.wide_processor   = Conv(self.c_half, self.c_half, k=3, s=1, p=1, g=1, d=d, act=act)
#         self.narrow_processor = Conv(self.c_half, self.c_half, k=3, s=1, p=1, g=1, d=d, act=act)

#         # Alignment components
#         self.downscaler = nn.AvgPool2d(kernel_size=2, stride=2)
        
#         w_place = torch.zeros((self.c_half, 1, 2, 2), dtype=torch.float32)
#         w_place[:, 0, 0, 0] = 1.0
#         self.register_buffer("place_weight", w_place, persistent=True)
        
#         # Pre-calculates a large static mask based on max_hw
#         max_H, max_W = int(max_hw[0]), int(max_hw[1])
#         narrow_bbox = {
#             'center_x': 0.499289, 'center_y': 0.499912,
#             'width': 0.286041, 'height': 0.291975
#         }
#         cx, cy = narrow_bbox['center_x'] * max_W, narrow_bbox['center_y'] * max_H
#         bw, bh = narrow_bbox['width'] * max_W, narrow_bbox['height'] * max_H
#         left = max(0, int(round(cx - bw / 2)))
#         top = max(0, int(round(cy - bh / 2)))
#         right = min(max_W, int(round(left + bw)))
#         bottom = min(max_H, int(round(top + bh)))
        
#         mask = torch.zeros(1, 1, max_H, max_W, dtype=torch.float32)
#         if top < bottom and left < right:
#             mask[:, :, top:bottom, left:right] = 1.0
#         self.register_buffer("full_res_mask", mask, persistent=True)

#         # Final fusion conv, always with stride=1 to preserve resolution
#         self.fusion_conv = Conv(self.c_half, c2, k=k, s=1, p=p, g=1, d=d, act=act)

#     def forward(self, x):
#         wide = self.split_wide(x)
#         narrow = self.split_narrow(x)
        
#         wide_proc = self.wide_processor(wide)
#         narrow_proc = self.narrow_processor(narrow)
        
#         narrow_down = self.downscaler(narrow_proc)
        
#         target_h, target_w = wide_proc.shape[2:]
#         down_h, down_w = narrow_down.shape[2:]
        
#         raw_out_h, raw_out_w = (down_h - 1) * 2 + 2, (down_w - 1) * 2 + 2
#         output_padding_h, output_padding_w = target_h - raw_out_h, target_w - raw_out_w
        
#         aligned_raw = F.conv_transpose2d(
#             narrow_down, self.place_weight, bias=None, stride=2, padding=0, 
#             output_padding=(output_padding_h, output_padding_w), groups=self.c_half
#         )
        
#         # Crops the runtime_mask from the large static mask
#         H, W = aligned_raw.shape[2:]
#         max_H, max_W = self.full_res_mask.shape[2:]
        
#         if H > max_H or W > max_W:
#             raise ValueError(
#                 f"Runtime feature map size ({H}, {W}) is larger than max_hw ({max_H}, {max_W}) defined in YAML. "
#                 f"Please increase the max_hw value for this layer in your YAML file."
#             )
        
#         start_h = (max_H - H) // 2
#         start_w = (max_W - W) // 2
        
#         runtime_mask = self.full_res_mask[:, :, start_h : start_h + H, start_w : start_w + W]
        
#         aligned = aligned_raw * runtime_mask
#         fused = wide_proc + aligned
#         out = self.fusion_conv(fused)
        
#         return out


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
    def __init__(self, fusion_type='concat', scale_factor=1.0):
        super().__init__()
        self.fusion_type = fusion_type
        self.scale_factor = scale_factor
        
        # For weighted sum, create learnable weights
        if fusion_type == 'weighted_sum':
            self.weights = nn.Parameter(torch.ones(2))
            
    def forward(self, x):
        # print(f"DEBUG: ===== FUSION LAYER DEBUG =====")
        # print(f"DEBUG: Fusion input type: {type(x)}")
        # print(f"DEBUG: Fusion input shape: {x.shape if hasattr(x, 'shape') else 'N/A'}")
        
        # Handle tensor input with dual streams: [B, 2, C, H, W]
        if isinstance(x, torch.Tensor) and x.dim() == 5 and x.shape[1] == 2:
            # print(f"DEBUG: Processing dual stream tensor: {x.shape}")
            # Split dual stream tensor into list
            stream1 = x[:, 0]  # [B, C, H, W]
            stream2 = x[:, 1]  # [B, C, H, W]
            streams = [stream1, stream2]
            print(f"DEBUG: Stream1 shape: {stream1.shape}")
            print(f"DEBUG: Stream2 shape: {stream2.shape}")
        elif isinstance(x, list) and len(x) >= 2:
            streams = x
            # print(f"DEBUG: Processing list input with {len(streams)} streams")
        # else:
            # print(f"DEBUG: Single stream input, returning as-is")
            return x
        
        if self.fusion_type == 'concat':
            # Concatenate along channel dimension
            output = torch.cat(streams, dim=1)
            # print(f"DEBUG: Concat output shape: {output.shape}")
        elif self.fusion_type == 'add':
            # Element-wise addition
            output = sum(streams)
            # print(f"DEBUG: Add output shape: {output.shape}")
        elif self.fusion_type == 'max':
            # Element-wise maximum
            output = torch.maximum(streams[0], streams[1])
            for i in range(2, len(streams)):
                output = torch.maximum(output, streams[i])
            # print(f"DEBUG: Max output shape: {output.shape}")
        elif self.fusion_type == 'weighted_sum':
            # Weighted sum with learnable weights
            normalized_weights = torch.softmax(self.weights, dim=0)
            output = sum(normalized_weights[i] * tensor for i, tensor in enumerate(streams[:len(normalized_weights)]))
            # print(f"DEBUG: Weighted sum output shape: {output.shape}")
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
        # print(f"DEBUG: Final fusion output shape: {output.shape}")
        # print(f"DEBUG: Final fusion output range: {output.min():.6f} ~ {output.max():.6f}")
        # print(f"DEBUG: ===== END FUSION LAYER DEBUG =====")
        
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

    # default_act = nn.SiLU()  # default activation
    default_act = nn.ReLU()

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
        # NPU 호환성을 위해 항상 groups=1 사용
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=1, dilation=d, bias=False)
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

    # default_act = nn.SiLU()  # default activation
    default_act = nn.ReLU()

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

    # default_act = nn.SiLU()  # default activation
    default_act = nn.ReLU()

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
