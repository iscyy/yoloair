
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CARAFE(nn.Module):     
    #CARAFE: Content-Aware ReAssembly of FEatures  # AIEAGNY      https://arxiv.org/pdf/1905.02188.pdf
    def __init__(self, c1, c2, kernel_size=9, up_factor=3):
        super(CARAFE, self).__init__()
        # 🎈YOLOv8 改进==👇'
        # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见 https://github.com/iscyy/ultralyticsPro/wiki/YOLOv8
        pass

