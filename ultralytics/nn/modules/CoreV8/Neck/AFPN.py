import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    # 🎈YOLOv8-SSFF 改进==👇'
    # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见 https://github.com/iscyy/ultralyticsPro/wiki/YOLOv8
    pass
        
class Downsample_x2(nn.Module):
    # 🎈YOLOv8-SSFF 改进==👇'
    # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见 https://github.com/iscyy/ultralyticsPro/wiki/YOLOv8
    pass
    
class Downsample_x4(nn.Module):
    # 🎈YOLOv8-SSFF 改进==👇'
    # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见 https://github.com/iscyy/ultralyticsPro/wiki/YOLOv8
    pass
    
class Downsample_x8(nn.Module):
    # 🎈YOLOv8-SSFF 改进==👇'
    # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见 https://github.com/iscyy/ultralyticsPro/wiki/YOLOv8
    pass
    
class BasicBlock(nn.Module):
    # 🎈YOLOv8-SSFF 改进==👇'
    # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见 https://github.com/iscyy/ultralyticsPro/wiki/YOLOv8
    pass
class ASFF_2(nn.Module):
    def __init__(self, c1, c2, level=0):
        super(ASFF_2, self).__init__()
        c1_l, c1_h = c1[0], c1[1]
        self.level = level
        self.dim = [
            c1_l,
            c1_h
        ]
        self.inter_dim = self.dim[self.level] # 0
        compress_c = 8

        if level == 0: 
            self.stride_level_1 = Upsample(c1_h, self.inter_dim) # c1_l
        if level == 1:
            self.stride_level_0 = Downsample_x2(c1_l, self.inter_dim) # c1_h
        
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1 = x[0], x[1]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1


        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weights_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :]
        out = self.conv(fused_out_reduced)

        return out

class ASFF_3(nn.Module):
    # 🎈YOLOv8-SSFF 改进==👇'
    # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见 https://github.com/iscyy/ultralyticsPro/wiki/YOLOv8
    pass