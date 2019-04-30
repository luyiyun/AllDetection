from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import resnet50


class FPN(nn.Module):
    '''
    FPN网络，通过增加额外的top-down feature map和横向连接来提高对于小物体预测的
    能力。
    '''
    def __init__(
        self, backbone=resnet50, normal_init=True, feature_maps=(3, 4, 5, 6, 7)
    ):
        '''
        args:
            backbone，torchvision.models中的预训练的模型，现在仅支持resnet
                的网络架构。
        '''
        super(FPN, self).__init__()
        self.backbone = backbone(True)
        self.feature_maps = feature_maps
        # 另外增加的预测大物体的两个feature maps
        if 6 in self.feature_maps:
            self.conv6 = nn.Conv2d(
                2048, 256, kernel_size=3, stride=2, padding=1)
        if 7 in self.feature_maps:
            self.conv7 = nn.Conv2d(
                256, 256, kernel_size=3, stride=2, padding=1)
        # Lateral layers
        if len(set([5, 4, 3]).intersection(set(self.feature_maps))) != 0:
            self.latlayer1 = nn.Conv2d(
                2048, 256, kernel_size=1, stride=1, padding=0)
        if len(set([4, 3]).intersection(set(self.feature_maps))) != 0:
            self.latlayer2 = nn.Conv2d(
                1024, 256, kernel_size=1, stride=1, padding=0)
        if 3 in self.feature_maps:
            self.latlayer3 = nn.Conv2d(
                512, 256, kernel_size=1, stride=1, padding=0)
        # Top-down layers
        if len(set([4, 3]).intersection(set(self.feature_maps))) != 0:
            self.toplayer1 = nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1)
        if 3 in self.feature_maps:
            self.toplayer2 = nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1)
        if normal_init:
            # 对新增加的layers使用weight=normal(0, 0.01)、bias=0进行初始化
            self.init()

    def forward(self, x):
        results = [None] * 7
        # Bottom-up
        c1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        c1 = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(c1)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        if 6 in self.feature_maps:
            p6 = self.conv6(c5)
            results[-2] = p6
        if 7 in self.feature_maps:
            p7 = self.conv7(F.relu(p6))
            results[-1] = p7

        # Top-down
        if len(set([5, 4, 3]).intersection(set(self.feature_maps))) != 0:
            p5 = self.latlayer1(c5)
            if 5 in self.feature_maps:
                results[-3] = p5
        if len(set([4, 3]).intersection(set(self.feature_maps))) != 0:
            p4 = self._upsample_add(p5, self.latlayer2(c4))
            p4 = self.toplayer1(p4)
            if 4 in self.feature_maps:
                results[-4] = p4
        if 3 in self.feature_maps:
            p3 = self._upsample_add(p4, self.latlayer3(c3))
            p3 = self.toplayer2(p3)
            results[-5] = p3
        return tuple([p for p in results if p is not None])

    def _upsample_add(self, x, y):
        '''
        对x进行上采样(使用双线性差值)到y的大小，并和y相加
        '''
        _, _, H, W = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def init(self):
        ''' 对新增加的layers进行normal和0初始化 '''
        for n, m in self.named_children():
            if n != 'backbone':
                init.normal_(m.weight, 0, 0.01)


class RetinaNet(nn.Module):
    '''
    RetinaNet
    '''
    def __init__(
        self, backbone=resnet50, num_class=2, num_anchors=9, head_bn=False,
        normal_init=True, cls_bias_init=True, ps=(3, 4, 5, 6, 7), test=False
    ):
        '''
        args:
            backbone，使用的torchvision的预训练模型；
            num_class，前景类共有几类；
            num_anchors，每个输出网格共对应几个anchors；
        '''
        super(RetinaNet, self).__init__()
        self.test = test
        self.head_bn = head_bn

        self.fpn = FPN(backbone, normal_init=normal_init, feature_maps=ps)
        self.num_class = num_class
        self.num_anchors = num_anchors
        self.loc_head = self._make_head(self.num_anchors * 4)
        if cls_bias_init:
            bias_init = -log(0.99 * 0.01)
        else:
            bias_init = None
        self.cls_head = self._make_head(
                self.num_anchors * self.num_class, bias_init=bias_init
        )
        if normal_init:
            self.init()

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(
                x.size(0), -1, self.num_class)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        if self.test:
            return cls_preds, loc_preds
        return torch.cat(cls_preds, 1), torch.cat(loc_preds, 1)

    def _make_head(self, out_planes, bias_init=None):
        '''
        创建分类和回归头
        args:
            out_planes，输出的channels的数量；
            bias_init，如果不是None，则会对最后一层conv的bias进行以此为值的
                constant init；
        returns:
            head，输出一个nn.moudules，代表分类头或回归头；
        '''
        layers = []
        for _ in range(4):
            layers.append(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            if self.head_bn:
                layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(True))

        last_layer = nn.Conv2d(
                256, out_planes, kernel_size=3, stride=1, padding=1)
        if bias_init is not None:
            nn.init.constant_(last_layer.bias, bias_init)
        layers.append(last_layer)
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''
        固定所有的bn层为eval状态
        '''
        # modules方法返回所有的后代modules，但可能有些modules用了好几次，但
        #   也只返回一次，这正适合我们固定bn的任务
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def init(self):
        '''
        对所有的新建的conv使用normal和0初始化
        '''
        for n, m in self.named_modules():
            if 'fpn' not in n and isinstance(m, nn.Conv2d):
                init.normal_(m.weight, 0, 0.01)


def main():
    net = RetinaNet(ps=(6, 7), test=True)
    # for n, m in net.named_modules():
    #     print(n)
    cls_preds, loc_preds = net(torch.rand(2, 3, 224, 224))
    print([i.size() for i in cls_preds])
    print([i.size() for i in loc_preds])


if __name__ == "__main__":
    main()
