import random

import torch
from PIL import Image, ImageFilter
from torchvision.transforms import Compose


class YTransfer:
    '''
    将YEncoder包装成一个transform（但实际代码中并没有使用）
    '''
    def __init__(self, y_encoder, input_size=None):
        '''
        args:
            y_encoder，YEncoder实例
            input_size，处理的图像的大小，w x h
        '''
        self.y_encoder = y_encoder
        self.input_size = input_size

    def __call__(self, img, labels, markers):
        cls_targets, loc_targets = self.y_encoder.encode(
            labels, markers, self.input_size)
        return img, cls_targets, loc_targets


class Resize:
    '''
    将图像Resize成特定大小并尽量保持其宽高比例，其中还改变了标定框的大小
    '''
    def __init__(self, size, max_size=1500):
        '''
        args:
            size：图片的大小，如果是int则指的是短边要resize到的大小，
                如果是tuple则指的是(w, h)
            max_size：其resize后长边不能超过max_size
        '''
        self.size = size
        self.max_size = max_size

    def __call__(self, alls):
        '''
        args:
            img，PIL Image对象；
            labels，(#objs,)大小的tensor，表示这张图片上的objects的labels，在此
                并没有进行改变；
            markers，(#objs, 4)大小的tensor，表示图片上的objects的框，xyxy格式
        '''
        img, labels, markers = alls
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
        return [
                img.resize((ow, oh), Image.BILINEAR),
                labels,
                markers * torch.tensor([sw, sh, sw, sh])
            ]


class RandomFlipLeftRight:
    '''
    随机进行水平翻转
    '''
    def __init__(self, p=0.5):
        '''
        args:
            p，是否进行随机翻转的概率
        '''
        self.p = p

    def __call__(self, alls):
        '''
        同Resize
        '''
        img, labels, markers = alls
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - markers[:, 2]
            xmax = w - markers[:, 0]
            markers[:, 0] = xmin
            markers[:, 2] = xmax
        return [img, labels, markers]


class RandomFlipTopBottom:
    '''
    随机进行垂直翻转
    '''
    def __init__(self, p=0.5):
        '''
        args:
            p，是否进行随机翻转的概率
        '''
        self.p = p

    def __call__(self, alls):
        '''
        同Resize
        '''
        img, labels, markers = alls
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            h = img.height
            ymin = h - markers[:, 3]
            ymax = h - markers[:, 1]
            markers[:, 1] = ymin
            markers[:, 3] = ymax
        return [img, labels, markers]


class RandomBlur:
    '''
    随机进行模糊操作（github原代码中给blur设定了一个大小(5,5)，这里使用的PIL没
        有这个参数）
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.BLUR)
        return img


class OnlyImage:
    '''
    将只应用在image上的transforms进行特殊的调整，这样这个transfoms输入和输出
    都是image和labels了，便于和其他label也会发成变化的transforms对接
    '''
    def __init__(self, transfers, return_tuple=False):
        '''
        args:
            transfers: 多个transforms对象或一个transforms对象；
            return_tuple: 如果是true，则返回的是tuple；
        '''
        if isinstance(transfers, (tuple, list)):
            self.transfers = Compose(transfers)
        else:
            self.transfers = transfers
        self.return_tuple = return_tuple

    def __call__(self, inpts):
        img, others = inpts[0], inpts[1:]
        img = self.transfers(img)
        oupts = [img] + list(others)
        if self.return_tuple:
            return tuple(oupts)
        return oupts
