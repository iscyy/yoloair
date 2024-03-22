import math
import torch

red = 'orangered'
orange = 'darkorange'
yellow = 'gold'
green = 'greenyellow'
cyan = 'aqua'
blue = 'deepskyblue'
purple = 'mediumpurple'
pink = 'violet'

COLORS = [purple, blue, green, yellow, orange]


def xywh_to_ltrb(attr):
    attr[..., :2] -= attr[..., 2: 4] / 2
    attr[..., 2: 4] += attr[..., :2]
    return attr


def ltrb_to_xywh(attr):
    attr[..., 2: 4] -= attr[..., :2]
    attr[..., :2] += attr[..., 2: 4] / 2
    return attr
 
class IoU_Cal:
    ''' pred, target: x0,y0,x1,y1
        monotonous: {
            None: origin
            True: monotonic FM
            False: non-monotonic FM
        }
        momentum: The momentum of running mean'''
    iou_mean = 1.
    momentum = 1 - pow(0.5, exp=1 / 7000)
    _is_train = True
 
    def __init__(self, pred, target, monotonous = False):
        self.pred, self.target = pred, target
        self.monotonous = monotonous
        self._fget = {
            # x,y,w,h
            'pred_xy': lambda: (self.pred[..., :2] + self.pred[..., 2: 4]) / 2,
            'pred_wh': lambda: self.pred[..., 2: 4] - self.pred[..., :2],
            'target_xy': lambda: (self.target[..., :2] + self.target[..., 2: 4]) / 2,
            'target_wh': lambda: self.target[..., 2: 4] - self.target[..., :2],
            # x0,y0,x1,y1
            'min_coord': lambda: torch.minimum(self.pred[..., :4], self.target[..., :4]),
            'max_coord': lambda: torch.maximum(self.pred[..., :4], self.target[..., :4]),
            # The overlapping region
            'wh_inter': lambda: self.min_coord[..., 2: 4] - self.max_coord[..., :2],
            's_inter': lambda: torch.prod(torch.relu(self.wh_inter), dim=-1),
            # The area covered
            's_union': lambda: torch.prod(self.pred_wh, dim=-1) +
                               torch.prod(self.target_wh, dim=-1) - self.s_inter,
            # The smallest enclosing box
            'wh_box': lambda: self.max_coord[..., 2: 4] - self.min_coord[..., :2],
            's_box': lambda: torch.prod(self.wh_box, dim=-1),
            'l2_box': lambda: torch.square(self.wh_box).sum(dim=-1),
            # The central points' connection of the bounding boxes
            'd_center': lambda: self.pred_xy - self.target_xy,
            'l2_center': lambda: torch.square(self.d_center).sum(dim=-1),
            # IoU
            'iou': lambda: 1 - self.s_inter / self.s_union
        }
        self._update(self)
 
    def __setitem__(self, key, value):
        self._fget[key] = value
 
    def __getattr__(self, item):
        if callable(self._fget[item]):
            self._fget[item] = self._fget[item]()
        return self._fget[item]
 
    @classmethod
    def train(cls):
        cls._is_train = True
 
    @classmethod
    def eval(cls):
        cls._is_train = False
 
    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls.momentum) * cls.iou_mean + \
                                         cls.momentum * self.iou.detach().mean().item()
 
    def _scaled_loss(self, loss, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                loss *= (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                loss *= beta / alpha
        return loss
 
    @classmethod
    def IoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        return self.iou
 
    @classmethod
    def WIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        dist = torch.exp(self.l2_center / self.l2_box.detach())
        return self._scaled_loss(dist * self.iou)
