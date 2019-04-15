import torch


def meshgrid(x, y, row_major=True):
    '''
    Return meshgrid in range x & y.
    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.
    Returns:
      (tensor) meshgrid, sized [x*y,2]
    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]
    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0, x)
    b = torch.arange(0, y)
    xx = a.repeat(y).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, x).view(-1, 1)
    return torch.cat([xx, yy], 1).to(torch.float) \
        if row_major else torch.cat([yy, xx], 1).to(torch.float)


def change_box_order(boxes, order):
    '''
    在xyxy和xywh两种格式间进行转换
    '''
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a + b) / 2, b - a + 1], 1)
    else:
        return torch.cat([(a - b) / 2, a + b / 2], 1)


def box_iou(box1, box2, order='xyxy'):
    '''
    计算两组boxes的IoU
    returns：
        ious, shape是[len(box1), len(box2)]
    '''
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    # box1加None将是在其对应的位置加上一个维度，其shape为1
    # 这里使用了广播操作，其先对其最后一维，这里自动将box2的第一维广播到了
    #   box1的第二个维度上去了
    # 得到了len(box1) * len(box2) * 2，储存的是xmin和ymin在每对box1、box2组
    #   合交集的上界
    # 下面那个同理，求的是交集的下界
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt + 1).clamp(min=0)  # ????
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    area2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''
    非最大抑制（non maximum suppression），即如果有许多的bboxes都重叠在一起，则
        取其中score（就是预测的那一类的得分）最大的那个。
    args:
        bboxes: 预测的框的xyxy坐标；
        scores: 预测的框的得分，是sigmoid得分，是预测的那一类的得分；
        threshold: IoU界限，如果和最高得分的bbox的IoU超过这个值，则认为两者实际
            上预测的是同一个obj，则得分较低的删除；
        mode: 如果是union，则计算的是IoU，如果是min，则计算交集和两个框较小的那
            个的比值；
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        # 接下来的工作实际上就是计算和评分最高的bbox的IoU
        #   还有其他的mode，但可以考虑是否利用上面编写的iou函数
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s' % mode)

        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


class FocalLoss(nn.Module):
    def __init__(self, num_class=2, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.

    def forward(self, pred, true):
        pass

    def focal_loss(self, x, y):

