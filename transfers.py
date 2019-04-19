import random

import torch
from PIL import Image


class YTransfer:
    def __init__(self, y_encoder, input_size=None):
        self.y_encoder = y_encoder
        self.input_size = input_size

    def __call__(self, img, labels, markers):
        cls_targets, loc_targets = self.y_encoder.encode(
            labels, markers, self.input_size)
        return img, cls_targets, loc_targets


class Resize:
    def __init__(self, size, max_size=1500):
        '''
        args:
            size：图片的大小，如果是tuple则指的是(w, h)
        '''
        self.size = size
        self.max_size = max_size

    def __call__(self, img, labels, markers):
        w, h = img.size
        if isinstance(self.size, int):
            size_min = min(w, h)
            size_max = max(w, h)
            sw = sh = float(self.size) / size_min
            if sw * size_max > self.max_size:
                sw = sh = float(self.max_size) / size_max
            ow = int(w * sw + 0.5)
            oh = int(h * sh + 0.5)
        else:
            ow, oh = self.size
            sw = float(ow) / w
            sh = float(oh) / h
        return (
            img.resize((ow, oh), Image.BILINEAR),
            labels,
            markers * torch.tensor([sw, sh, sw, sh])
        )


class RandomFlipLeftRight:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, labels, markers):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - markers[:, 2]
            xmax = w - markers[:, 0]
            markers[:, 0] = xmin
            markers[:, 2] = xmax
        return img, labels, markers


class RandomFlipTopBottom:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, labels, markers):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            h = img.height
            ymin = h - markers[:, 3]
            ymax = h - markers[:, 1]
            markers[:, 1] = ymin
            markers[:, 3] = ymax
        return img, labels, markers


class OnlyImage:
    def __init__(self, transfer):
        self.transfer = transfer

    def __call__(self, img, labels, markers):
        img = self.transfer(img)
        return img, labels, markers


class MultiCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

