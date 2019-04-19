import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot


class FocalLoss(nn.Module):
    def __init__(self, num_class=2, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, cls_targets, loc_targets, cls_preds, loc_preds):
        batch_size, num_boxes = cls_targets.size()
        # 0是背景类，1是要忽略的anchors，得到[N, #anchors]的boolean
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()

        # smoothl1loss, loc_loss, 做bbr，只使用分配了object的anchors进行
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # 先做成4的长度，便于mask
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # 可以看做是anchor是样本
        masked_loc_targets = loc_targets[mask].view(-1, 4)
        loc_loss = self.smooth_l1_loss(masked_loc_targets, masked_loc_preds)

        # focal loss, cls_loss
        pos_neg = cls_targets > -1  # 要忽略的anchors去除
        num_pos_neg = pos_neg.data.long().sum()
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_class)
        # 现在preds是[#anchors, num_class]
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        # 直接用两个均数相加不行吗，这样实际上cls_loss使用的样本数量不是num_pos
        # ？？？？？
        loss = (loc_loss + cls_loss)/num_pos
        return loss, cls_loss / num_pos_neg, loc_loss / num_pos

    def focal_loss(self, x, y):
        # 这样做，使得背景anchor的标签是[0,0]，两类的标签是[0,1]和[1,0]
        t = one_hot(y.data, 1+self.num_class, device=torch.device('cuda:0'))
        t = t[:, 1:]
        # 使用sigmoid值，则相当于对于每一anchor，每一个类和背景类作logistic，
        #   自然此分数可以认为是这一类相对于背景类的得分，而在决定使用那一类作
        #   为此anchor的类时，直接进行比较即可。
        #   softmax考虑类间效应的功能就没有了？？？
        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)
        w = self.alpha * t + (1 - self.alpha) * t
        w = w * (1 - pt).pow(self.gamma)
        return F.binary_cross_entropy_with_logits(
            x, t, w.detach(), reduction='sum')

    def focal_loss_alt(self, x, y):
        # 这样做，使得背景anchor的标签是[0,0]，两类的标签是[0,1]和[1,0]
        t = one_hot(y.data, 1+self.num_class)
        t = t[:, 1:]

        xt = x * (2 * t - 1)  # x乘的是[-1, -1]，[-1, 1]和[1, -1]
        pt = (2 * xt + 1).sigmoid()

        w = self.alpha * t + (1 - self.alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()
