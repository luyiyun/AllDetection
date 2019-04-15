import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101


class FPN(nn.Module):
    def __init__(self, backbone=resnet50):
        super(FPN, self).__init__()
        self.backbone = backbone(True)

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
        对x进行上采样到y的大小，并和有相加
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y


class RetinaNet(nn.Module):
    def __init__(self, backbone=resnet50, num_class=2, num_anchors=9):
        super(RetinaNet, self).__init__()
        self.fpn = FPN(backbone)
        self.num_class = num_class
        self.num_anchors = 9
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_class)

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

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(
            nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        # modules方法返回所有的后代modules，但可能有些modules用了好几次，但
        #   也只返回一次，这正适合我们固定bn的任务
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


def main():
    net = RetinaNet()
    loc_preds, cls_preds = net(torch.rand(2, 3, 224, 224))
    print(loc_preds.size())
    print(cls_preds.size())


if __name__ == "__main__":
    main()
