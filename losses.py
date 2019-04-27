import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot


class FocalLoss(nn.Module):
    '''
    RetinaNet使用的loss，包括两个部分，即loc loss和cls loss部分，其中loc loss
    使用的是普通的bounding box regression（smooth l1 loss），cls loss则使用其
    发明的focal loss。
    '''
    def __init__(self, num_class=2, alpha=0.25, gamma=2):
        '''
        args:
            num_class，对于object的分类有几类，默认是2；
            alpha，对背景类和前景类的loss采取不同的权重，默认是0.25？；
            gamma，focal loss使用的gamma值，越大则对易分样本关注越少；
        '''
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, cls_targets, loc_targets, cls_preds, loc_preds):
        '''
        计算总的loss。
        args:
            cls_targets，(#imgs, #anchors,)，编码后得到的每张图片每个anchor所对
                应gtbb的标签，-1代表IoU在中间的部分（不用），0代表背景类，1,2,..
                .表示各个前景类；
            loc_targets，(#imgs, #anchors, 4)，编码后得到的每张图片每个anchor
                相对于gtbb（这个anchor对应的）的位置偏倚量，其中前两列是中心点的
                两个方向的平移量，后两列是w和h的缩放量；
            cls_preds，(#imgs, #anchors, #classes)，是每张图片在每个anchor上
                每一类相对于背景类来说其logit是多少，即其不是一个softmax，而是
                以每一类和背景类来作logistic，而且此概率还没有进行sigmoid变换；
            loc_preds，(#imgs, #anchors, 4)，预测的每个anchor上的偏移量
        returns:
            loss，总loss；
            cls_loss：即focal loss，loss中的分类部分；
            loc_loss：即bbr的loss，loss中的回归部分；
        '''
        batch_size, num_boxes = cls_targets.size()
        # 0是背景类，1是要忽略的anchors，得到[N, #anchors]的boolean
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()
        pos_neg = cls_targets > -1  # 要忽略的anchors去除
        # num_pos_neg = pos_neg.data.long().sum()

        # smoothl1loss, loc_loss, 做bbr，只使用分配了object的anchors进行
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # 先做成4的长度，便于mask
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # 可以看做是anchor是样本
        masked_loc_targets = loc_targets[mask].view(-1, 4)
        loc_loss = self.smooth_l1_loss(masked_loc_targets, masked_loc_preds)

        # focal loss, cls_loss
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_class)
        # 现在preds是[#anchors, num_class]
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        # 直接用两个均数相加不行吗，这样实际上cls_loss使用的样本数量不是num_pos
        # ？？？？？
        loc_loss = loc_loss / num_pos
        cls_loss = cls_loss / num_pos
        loss = loc_loss + cls_loss
        return loss, cls_loss, loc_loss

    def focal_loss(self, x, y):
        '''
        计算focal loss的函数
        args:
            x，(#all_anchors, #classes)，预测的在多个类上的logit，这是去除了IoU
                在0.4-0.5之间的部分（-1类）后剩下的那些anchors，是一个batch的所
                有图片的anchors的综合；
            y，(#all_ancors, )，去除了-1类后每个anchor对应的gtbb的标签，值是
                0,1,2,...；
        returns:
            focal_loss，scalar，一个batch上所有anchors的focal loss，这里没有进行
                平均；
        '''
        # 这样做，使得背景anchor的标签是[0,0]，两类的标签是[0,1]和[1,0]
        t = one_hot(y.data, 1+self.num_class, device=torch.device('cuda:0'))
        t = t[:, 1:]
        # 使用sigmoid值，则相当于对于每一anchor，每一个类和背景类作logistic，
        #   自然此分数可以认为是这一类相对于背景类的得分，而在决定使用那一类作
        #   为此anchor的类时，直接进行比较即可。
        #   softmax考虑类间效应的功能就没有了？？？
        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)
        # 这里是对背景类有更高的惩罚？
        w = self.alpha * t + (1 - self.alpha) * (1 - t)
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
