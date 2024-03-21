# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

'''
YOLOv8 + 各类 损失函数" 改进，如下说明
- 只需要加上对应改进的核心损失函数模块，该项目代码就可以直接运行各种`YOLOv8-xxx.yaml`网络配置文件，乐高式创新改进，一键运行即可
使用 各类 损失函数 进行实验改进
- 项目相关改进可以支持 答疑 服务。详情见 ⭐⭐⭐ https://github.com/iscyy/ultralyticsPro/wiki/YOLOv8 ⭐⭐⭐ 说明
'''

'''
GIoU
DIoU
CIoU
EIoU
SIoU
WIoU
FocalerIoU
'''
def bbox_multi_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, WIoU=False, FocalerIoU=False, eps=1e-7):
    # ...code
    iou = ''
    pass
    return iou  # 🚀IoU

'''
FocalLoss_GIoU
FocalLoss_DIoU
FocalLoss_CIoU
FocalLoss_EIoU
FocalLoss_SIoU
FocalLoss_WIoU
FocalLoss_FocalerIoU
'''
def bbox_focal_multi_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, WIoU=False, FocalLoss_= 'none', eps=1e-7):
        # ...code
    iou = ''
    pass
    return iou  # 🚀IoU

