import math

import torch

from utils import meshgrid, change_box_order, box_iou, box_nms


class YEncoder:

    def __init__(
        self, anchor_areas=[2. ** i for i in range(5, 10)],
        aspect_ratios=[1/2., 1/1., 1/1.],
        scale_ratios=[1., pow(2, 1/3.), pow(2, 2/3.)],
        iou_thre=0.5, ignore_thres=(0.4, 0.5), nms_thre=0.5,
        input_size=None
    ):
        self.anchor_areas = anchor_areas
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.iou_thre = iou_thre
        self.ignore_thres = ignore_thres
        self.nms_thre = nms_thre
        self.input_size = torch.tensor([input_size, input_size]) \
            if isinstance(input_size, int) else torch.tensor(input_size)
        self.num_anchor_per_cell = len(aspect_ratios) * len(scale_ratios)

        self.anchor_wh = self._get_anchor_wh()

    def encode(self, boxes, labels, input_size=None):
        '''
        编码xml中的object格式为bounding boxes regression的格式
        args:
            boxes: tensor, ground truth bounding boxes，
                (xmin, ymin, xmax, ymax)，size是[#box, 4]
            labels: tensor, 每个gtbb的标签，size是[#box,]
            input_size：int/tuple，输入图像的大小
        returns:
            loc_targets: tensor，每个anchor被赋予的bbr的标签，size是
                [#anchors, 4]，#anchors是所有特征图上的所有anchors
            cls_targets: tensor，每个anchor被赋予的标签，size是[#anchors, ]，
                其中的值0代表背景类，1-k表示k个分类，-1表示忽略的anchors
        '''
        if input_size is None:
            input_size = self.input_size
        else:
            input_size = torch.tensor([input_size, input_size]) \
                if isinstance(input_size, int) else torch.tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')
        # 计算每个anchor和每个gtbb间的iou，根据此来给标签
        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]
        # 计算bbr的偏移量，即bbr的标签
        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = 1 + labels[max_ids]  # 加1是为了空出0来给背景类
        # 规定背景类，规定忽略的anchors
        cls_targets[max_ious < self.iou_thre] = 0
        ignore = (max_ious > self.ignore_thres[0]) & \
            (max_ious < self.ignore_thres[1])
        cls_targets[ignore] = -1  # 这些anchors是不用的
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size=None):
        '''
        将网络的输出转化为正常的人们所能理解的标签和boxes
        '''
        # 根据图像的大小和anchor设定来计算出所有anchor的信息
        if input_size is None:
            input_size = self.input_size
        else:
            input_size = torch.tensor([input_size, input_size]) \
                if isinstance(input_size, int) else torch.tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        # 取出预测的中心偏移量和宽高缩放量
        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]
        # 利用anchor的信息和bbr的效果（在每个anchor上需要再调整）来得到每个
        #   最后的预测框的loc，并转换成xyxy mode
        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)  # xyxy format
        # 将preds进行sigmoid，变成概率，并去除那些得分（最大）比较低的框
        score, labels = cls_preds.sigmoid().max(1)
        ids = score > self.iou_thre
        ids = ids.nonzero().squeeze()  # [#obj, ]
        obj_boxes, obj_score = boxes[ids], score[ids]
        # 再对剩下的预测框进行nms，得到的即是最后的结果
        keep = box_nms(obj_boxes, obj_score, threshold=self.nms_thre)
        return obj_boxes[keep], obj_score[keep]

    def _get_anchor_wh(self):
        '''
        对每个feature map计算anchor宽和高
        returns：
            anchor_wh：tensor，size是[
                feature map的数量,
                feature map上每一个空间点上有多少个anchor(
                    num_aspect_ration x num_scale_ratio),
                import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
                2(宽和高)
            ]
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            anchor_wh_per_cell = []
            for ar in self.aspect_ratios:
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:
                    anchor_h = h * sr
                    anchor_w = w * sr
                    anchor_wh_per_cell.append([anchor_w, anchor_h])
            anchor_wh.append(anchor_wh_per_cell)
        return torch.tensor(anchor_wh)

    def _get_anchor_boxes(self, input_size):
        '''
        计算每个anchor其所对应的boxes的坐标
        args:
            input_size: tensor, 输入图像的大小，(w, h)
        return:
            boxes: 所有特征图的anchor的中心点坐标及其宽高，shape是
                [anchors的总数, 4]
        '''
        num_fms = len(self.anchor_areas)
        boxes = []
        for i in range(num_fms):
            # 每个特征图的大小，这里认为每个特征图依次下采样了2倍
            fm_size = (input_size / pow(2., i+3)).ceil()
            # 每个空间点应该对应原图中多大的范围
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            # 计算的坐标是每个空间点在原图对应的矩形的中心点坐标
            xy = meshgrid(fm_w, fm_h) + 0.5
            # 这样的计算本质上等价于fm_w * grid_size + 0.5 * grad_size
            # 进行view的时候是先横向排的，所以，w被排到了axis=1上
            # 并将每个位置复制9份，因为每个中心点对应9个anchor
            xy = (xy * grid_size).view(fm_h, fm_w, 1, 2).expand(
                fm_h, fm_w, self.num_anchor_per_cell, 2)
            wh = self.anchor_wh[i].view(
                1, 1, self.num_anchor_per_cell, 2
            ).expand(
                fm_h, fm_w, 9, 2
            )
            box = torch.cat([xy, wh], dim=3)  # 这样每个位置点的是[x, y, w, h]
            boxes.append(box.view(-1, 4))

        return torch.cat(boxes, 0)


def main():
    pass


if __name__ == "__main__":
    main()
