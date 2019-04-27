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
    def __init__(self, backbone=resnet50, normal_init=True):
        '''
        args:
            backbone，torchvision.models中的预训练的模型，现在仅支持resnet
                的网络架构。
        '''
        super(FPN, self).__init__()
        self.backbone = backbone(True)
        # 另外增加的预测大物体的两个feature maps
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(
            2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(
            1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(
            512, 256, kernel_size=1, stride=1, padding=0)
        # Top-down layers
        self.toplayer1 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        if normal_init:
            # 对新增加的layers使用weight=normal(0, 0.01)、bias=0进行初始化
            self.init()

    def forward(self, x):
        # Bottom-up
        c1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        c1 = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(c1)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))

        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        return p3, p4, p5, p6, p7

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
        normal_init=True
    ):
        '''
        args:
            backbone，使用的torchvision的预训练模型；
            num_class，前景类共有几类；
            num_anchors，每个输出网格共对应几个anchors；
        '''
        super(RetinaNet, self).__init__()
        self.head_bn = head_bn

        self.fpn = FPN(backbone, normal_init=normal_init)
        self.num_class = num_class
        self.num_anchors = num_anchors
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(
                self.num_anchors * self.num_class, bias_init=-log(0.99 * 0.01)
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
    net = RetinaNet()
    for n, m in net.named_modules():
        print(n)
    loc_preds, cls_preds = net(torch.rand(2, 3, 224, 224))
    print(loc_preds.size())
    print(cls_preds.size())


if __name__ == "__main__":
    main()
