# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C3Ghost,
    C3x,
    GhostBottleneck,
    HGBlock,
    HGStem,
    Proto,
    RepC3,
    ResNetLayer,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

from .CoreV8.Backbone.emo import C3_RMB, CSRMBC, C2f_RMB, CPNRMB, ReNLANRMB
from .CoreV8.Backbone.biformer import CSCBiF, ReNLANBiF, CPNBiF, C3_Biformer, C2f_Biformer
from .CoreV8.Backbone.CFNet import CSCFocalNeXt, ReNLANFocalNeXt, CPNFocalNeXt, C3_FocalNeXt, C2f_FocalNeXt
from .CoreV8.Backbone.FasterNet import FasterNeXt, CSCFasterNeXt, ReNLANFasterNeXt, C3_FasterNeXt, C2f_FasterNeXt
from .CoreV8.Backbone.Ghost import CPNGhost, CSCGhost, ReNLANGhost, C3_Ghost, C2f_Ghost
from .CoreV8.Backbone.EfficientRep import RepVGGBlock, SimConv, RepBlock, Transpose

__all__ = (
    "Conv",
    "Conv2",
    #------------------
    "C3_RMB", "CSRMBC", "C2f_RMB", "CPNRMB", "ReNLANRMB",
    "CSCBiF", "ReNLANBiF", "CPNBiF", "C3_Biformer", "C2f_Biformer",
    "CSCFocalNeXt", "ReNLANFocalNeXt", "CPNFocalNeXt", "C3_FocalNeXt", "C2f_FocalNeXt",
    "CSCFasterNeXt", "ReNLANFasterNeXt", "FasterNeXt", "C3_FasterNeXt", "C2f_FasterNeXt",
    "CPNGhost", "CSCGhost", "ReNLANGhost", "C3_Ghost", "C2f_Ghost",
    "RepVGGBlock", "SimConv", "RepBlock", "Transpose",
    #------------------
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
)
