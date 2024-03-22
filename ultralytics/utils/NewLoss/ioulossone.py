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
    FocalerIoU 改进各类Loss 可以结合多种进行使用, 已经更新如下超过10+种
    Focaler_PIoU/Focaler_PIoUv2
    Focaler_GIoU
    Focaler_DIoU
    Focaler_CIoU
    Focaler_EIoU
    Focaler_SIoU
    Focaler_WIoU
    Focal_Focaler_PIoU/Focal_Focaler_PIoUv2
    Focal_Focaler_GIoU
    Focal_Focaler_DIoU
    Focal_Focaler_CIoU
    Focal_Focaler_EIoU
    Focal_Focaler_SIoU
    Focal_Focaler_WIoU
    替换参数即可
'''

# 详情见链接：ultralytics\utils\loss.py文件夹


def bbox_shape_iou(box1, box2, xywh=True, scale=0.7, eps=1e-7):
    # ...code
    iou = ''
    pass
    return iou  # 🚀IoU


def bbox_mpdiou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, MPDIoU=False, eps=1e-7):
    # ...code
    iou = ''
    pass
    return iou  # 🚀IoU

'''
    Inner-IoU 改进各类Loss 可以结合多种进行使用, 已经更新如下超过10+种
    Focal_Inner_PIoU/Focal_Inner_PIoUv2
    Focal_Inner_GIoU
    Focal_Inner_DIoU
    Focal_Inner_CIoU
    Focal_Inner_EIoU
    Focal_Inner_SIoU
    Focal_Inner_WIoU
    Inner_PIoU/Inner_PIoUv2
    Inner_GIoU
    Inner_DIoU
    Inner_CIoU
    Inner_EIoU
    Inner_SIoU
    Inner_WIoU
    替换参数即可
'''
def bbox_inner_multi_iou(box1, box2, ratio = 0.8, xywh=True, eps=1e-7, Inner_GIoU=False, Inner_DIoU=False, Inner_CIoU=False, Inner_EIoU=False, Inner_SIoU=False, Inner_WIoU=False, FocalLoss_=False): 
    # ...code
    iou = ''
    pass
    return iou  # 🚀IoU


def bbox_piou(box1, box2, xywh=True, PIoU=False,PIoU2=False,Lambda=1.3,eps=1e-7):
    # ...code
    iou = ''
    pass
    return iou  # 🚀IoU


def bbox_xiou(box1, box2, xywh=True, eps=1e-7):
    # ...code
    iou = ''
    pass
    return iou  # 🚀IoU

def nwdiou(box1, box2, xywh=True, eps=1e-7):
    # ...code
    iou = ''
    pass
    return iou  # 🚀IoU

def bbox_effciou(box1, box2, xywh=True, eps=1e-7):
    # ...code
    iou = ''
    pass
    return iou  # 🚀IoU

def bbox_xiou(box1, box2, xywh=True, eps=1e-7):
    # ...code
    iou = ''
    pass
    return iou  # 🚀IoU