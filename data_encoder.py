import math

import torch

from utils import meshgrid, change_box_order, box_iou, box_nms


class YEncoder:
    '''
    用于将objects形式的标签（每张图片有不同数量的objects，其中每个objects有
        4个坐标xyxy，和一个scalar表示其类别(0, 1)）和anchors形式的标签（
        每张图片有相同数量的anchors(大约18万个)，其中每个anchors有4个实数值表
        示其相对于其match的object的位置偏移量，和一个scalar来表示其属于哪一类，
        0表示是背景类，-1表示在接下来的训练中不考虑的anchors，其他(1-2)表示每个
        前景类）的转换
    '''
    def __init__(
        self, ps=[3, 4, 5, 6, 7],
        aspect_ratios=[1/2., 2/1., 1/1.],
        scale_ratios=[1., pow(2, 1/3.), pow(2, 2/3.)],
        iou_thre=0.5, ignore_thres=(0.4, 0.5), cls_thre=0.5, nms_thre=0.5,
        input_size=None
    ):
        '''
        args:
            anchor_areas，每个anchor所覆盖的在原图上的大小，是面积；
            aspect_ratios，每个anchor所拥有的宽高比的种类；
            scale_ratios，每个anchor所拥有的在原始anchor大小的基础上的扩充大小，
                是在宽和高上进行的扩充；
            iou_thre，在对anchor匹配object的时候，和object的IoU大于此的anchor被
                认为是前景类的anchor；
            ignore_thres，2-tuple，和object的最大IoU在此区间内的anchor被标记-1，
                在之后的训练中会被丢弃；
            cls_thre，预测时，如果score小于此，则认为这个框被预测为背景；
            nms_thres，float，进行nms时使用的阈值；
            input_size：输入图像的大小，是w x h的格式，或者是单个int，则会被认为
                是h，然后进行保持宽高比的变换；
        '''
        self.ps = ps
        anchor_areas = [(2. ** (i+2)) ** 2 for i in ps]
        self.anchor_areas = anchor_areas
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.iou_thre = iou_thre
        self.ignore_thres = ignore_thres
        self.cls_thre = cls_thre
        self.nms_thre = nms_thre
        # tensor的计算的时候，如果分子是int，那么进行除法的时候实际上使用的
        #   是整除，所以为了避免在之后计算特征图大小（而这里是ceil）的时候出错，
        #   这里先对其float变换一下
        if input_size is None:
            input_size = (1920, 1200)
        elif isinstance(input_size, int):
            input_size = (1920*input_size/1200, input_size)
        self.input_size = torch.tensor(input_size, dtype=torch.float)
        self.num_anchor_per_cell = len(aspect_ratios) * len(scale_ratios)
        self.anchor_wh = self._get_anchor_wh()
        self.anchor_boxes = self._get_anchor_boxes(self.input_size)

    def encode(self, labels, boxes, input_size=None, test=False):
        '''
        编码xml中的object格式为bounding boxes regression的格式
        tx = (x - anchor_x) / anchor_w
        ty = (y - anchor_y) / anchor_h
        tw = log(w / anchor_w)
        th = log(h / anchor_h)
        注意，这个方法输入的是单张图片的objects，所以使用的时候必须一张图片一张
            图片的输入
        args:
            labels: tensor, 每个gtbb的标签，size是[#box,]
            boxes: tensor, ground truth bounding boxes，
                (xmin, ymin, xmax, ymax)，size是[#box, 4]
            input_size：int/tuple，输入图像的大小
            test: 测试时使用；
        returns:
            cls_targets: tensor，每个anchor被赋予的标签，size是[#anchors, ]，
                其中的值0代表背景类，1-k表示k个分类，-1表示忽略的anchors
            loc_targets: tensor，每个anchor被赋予的bbr的标签，size是
                [#anchors, 4]，#anchors是所有特征图上的所有anchors
        '''
        if input_size is None:
            input_size = self.input_size
            anchor_boxes = self.anchor_boxes
        else:
            if isinstance(input_size, int):
                input_size = (1920*input_size/1200, input_size)
            input_size = torch.tensor(input_size, dtype=torch.float)
            anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')
        # 计算每个anchor和每个gtbb间的iou，根据此来给标签
        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]
        if test:
            _, orders = max_ious.sort(0, True)
            loc_targets = change_box_order(anchor_boxes, 'xywh2xyxy')[orders]
        else:
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
        if test:
            cls_targets = cls_targets[orders]
        return cls_targets, loc_targets

    def decode(
        self, cls_preds, loc_preds, input_size=None,
        device=torch.device('cuda:0')
    ):
        '''
        将网络的输出转化为正常的人们所能理解的标签和boxes
        这里的cls_preds和loc_preds因为是网络的输出，所以其第一个维度是batch
        这里需要注意，即其可以直接输入多张图片；
        args:
            cls_preds: tensor，每个anchor被预测的标签logits，size是[batch,
                #anchors, #classes]
            loc_preds: tensor，每个anchor被预测的bbr的偏移量，size是[batch,
                #anchors, 4]，#anchors是所有特征图上的所有anchors
            input_size：int/tuple，输入图像的大小，可以是None，此时使用实例化
                YEncoder对象时候输入的input_size；
            cuda：anchor_boxes所在的设备，默认是在GPU上，如果是测试和在cpu上进行
                预测可能需要更改；
        returns:
            labels: list of tensors, 每个tensors的size是[#boxes_i, #classes]，
                表示的是一张图片中预测框在每一类的logits值；
            boxes: list of tensors，每个tensors的size是[#boxes_i, 4]，是一张图
                片中预测框的位置(xmin, ymin, xmax, ymax)
        '''
        # 根据图像的大小和anchor设定来计算出所有anchor的信息
        if input_size is None:
            input_size = self.input_size
            anchor_boxes = self.anchor_boxes
            anchor_boxes = anchor_boxes.to(device)
        else:
            if isinstance(input_size, int):
                input_size = (1920*input_size/1200, input_size)
            input_size = torch.tensor(input_size, dtype=torch.float)
            anchor_boxes = self._get_anchor_boxes(input_size)
            anchor_boxes = anchor_boxes.to(device)
        if cls_preds.dim() == 3:
            anchor_boxes = anchor_boxes.unsqueeze(0).expand_as(loc_preds)
        # 取出预测的中心偏移量和宽高缩放量
        loc_xy = loc_preds[..., :2]
        loc_wh = loc_preds[..., 2:]
        # 利用anchor的信息和bbr的效果（在每个anchor上需要再调整）来得到每个
        #   最后的预测框的loc，并转换成xyxy mode
        xy = loc_xy * anchor_boxes[..., 2:] + anchor_boxes[..., :2]
        wh = loc_wh.exp() * anchor_boxes[..., 2:]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 2)  # xyxy format
        # 将preds进行sigmoid，变成概率，并去除那些得分（最大）比较低的框
        cls_preds = cls_preds.sigmoid()
        score, labels = cls_preds.max(2)
        ids = score > self.cls_thre
        # ids = ids.nonzero().squeeze()  # [#obj, ]
        result_boxes = []
        result_score = []
        for i in range(cls_preds.size(0)):
            obj_boxes, obj_score = boxes[i][ids[i]], score[i][ids[i]]
            objs_score = cls_preds[i][ids[i]]
            # 再对剩下的预测框进行nms，得到的即是最后的结果
            keep = box_nms(obj_boxes, obj_score, threshold=self.nms_thre)
            result_boxes.append(obj_boxes[keep])
            result_score.append(objs_score[keep])
        # 经过nms后，每张图片得到的预测框的数量是不一样的，所以无法将其都
        #   stack到一个tensor中，因为预测框的数量要作为dim=1来存在，只能
        #   使用list来保存
        return result_score, result_boxes

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
            for ar in self.aspect_ratios:  # w / h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:
                    anchor_h = h * sr
                    anchor_w = w * sr
                    anchor_wh_per_cell.append([anchor_w, anchor_h])
            anchor_wh.append(anchor_wh_per_cell)
        return torch.tensor(anchor_wh)

    def _get_anchor_boxes(self, input_size, concat=True):
        '''
        计算每个anchor其所对应的boxes的坐标
        args:
            input_size: tensor, 输入图像的大小，(w, h)；
            concat：boolean，如果False则返回list代表不同大小的feature map的
                anchors，这主要用于下面的test用；
        return:
            boxes: 所有特征图的anchor的中心点坐标及其宽高，shape是
                [anchors的总数, 4]
        '''
        num_fms = len(self.anchor_areas)
        boxes = []
        for i, p in zip(range(num_fms), self.ps):
            # 每个特征图的大小，这里认为每个特征图依次下采样了2倍
            # 这里要和anchor boxes的大小区分开，anchor的数量主要通过其使用的
            #   feature map上空间点的数量来确定，而anchor boxes的大小是看这个
            #   feature的感受野大小来确定的
            grid_size = torch.tensor([pow(2., p)] * 2, dtype=torch.float)
            fm_size = (input_size / grid_size).ceil()
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
            ).expand(fm_h, fm_w, self.num_anchor_per_cell, 2)
            box = torch.cat([xy, wh], dim=3)  # 这样每个位置点的是[x, y, w, h]
            boxes.append(box.view(-1, 4))
        if concat:
            return torch.cat(boxes, 0)
        else:
            return boxes


def test():
    import random

    import argparse
    from PIL import Image, ImageDraw

    from utils import one_hot

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-is', '--input_size', default=(1920, 1200), nargs=2, type=int,
        help='图像的大小，wxh，默认是1920x1200'
    )
    parser.add_argument(
        '-m', '--mode', default='show_wh',
        help=(
            'show_wh是展示每个位置不同feature map出现的的anchor（默认），'
            'show_anchor_center是将图像中所有的anchor point都画出，'
            'show_encode是根据给定（或随机）的一个gtbb来看不同anchor box的标'
            '签，show_decode是先使用decode来编码后再使用decode来解码看能不能'
            '得到原来的结果，以此来测试decode方法'
        )
    )
    parser.add_argument(
        '-b', '--box', default=None, nargs=4, type=int,
        help='当是show_encode时，如果此不为None，则使用这个来作为gtbb'
    )
    parser.add_argument(
        '-tk', '--top_k', default=10, type=int,
        help='当使用show_encode的时候，只显示IoU最大的几个anchor boxes'
    )
    args = parser.parse_args()
    y_encoder = YEncoder(input_size=args.input_size, ps=[3, 4, 5])
    if args.mode == 'show_wh':
        cen = torch.tensor(args.input_size, dtype=torch.float) / 2
        lt = cen - y_encoder.anchor_wh / 2
        rb = cen + y_encoder.anchor_wh / 2
        xyxy = torch.cat([lt, rb], dim=2)
        for i in range(len(y_encoder.anchor_wh)):
            img = Image.new(size=args.input_size, color='black', mode='RGB')
            draw = ImageDraw.Draw(img)
            xyxyi = xyxy[i]
            for j in range(len(xyxyi)):
                draw.rectangle(xyxyi[j].tolist(), outline='red')
            img.show(title='anchor level: %d' % i)
    elif args.mode == 'show_anchor_center':
        anchorss = y_encoder._get_anchor_boxes(
            torch.tensor(args.input_size, dtype=torch.float), concat=False)
        for anchors in anchorss:
            arr = anchors[:, :2].unique(dim=0)
            img = Image.new(size=args.input_size, color='black', mode='RGB')
            draw = ImageDraw.Draw(img)
            arr = arr.view(-1)
            draw.point(arr.tolist(), fill='red')
            img.show()
    elif args.mode == 'show_encode':
        if args.box is None:
            lt = [random.randint(1, i) for i in args.input_size]
            rd = [
                random.randint(i1, i2) for i1, i2 in zip(lt, args.input_size)]
            xyxy = lt + rd
        else:
            xyxy = args.box
        cls_target, loc_target = y_encoder.encode(
            torch.tensor([1]), torch.tensor([xyxy], dtype=torch.float),
            test=True
        )
        cls_target = cls_target[:args.top_k]
        loc_target = loc_target[:args.top_k]
        img = Image.new(size=args.input_size, color='black', mode='RGB')
        draw = ImageDraw.Draw(img)
        boxes2 = loc_target[cls_target == 2]
        boxes_1 = loc_target[cls_target == -1]
        # boxes0 = loc_target[cls_target == 0]
        for b in boxes_1:
            draw.rectangle(b.tolist(), outline='white')
        for b in boxes2:
            draw.rectangle(b.tolist(), outline='green')
        draw.rectangle(xyxy, outline='red')
        img.show()
    elif args.mode == 'show_decode':
        if args.box is None:
            lt = [random.randint(1, i) for i in args.input_size]
            rd = [
                random.randint(i1, i2) for i1, i2 in zip(lt, args.input_size)]
            xyxy = lt + rd
        else:
            xyxy = args.box
        cls_target, loc_target = y_encoder.encode(
            torch.tensor([1]), torch.tensor([xyxy], dtype=torch.float),
        )
        cls_pred = cls_target + 1
        loc_pred = loc_target
        cls_pred = one_hot(cls_pred, 4)[:, 2:]
        cls_pred = cls_pred.to(dtype=torch.float)
        cls_pred, loc_pred = cls_pred.unsqueeze(0), loc_pred.unsqueeze(0)
        score, box = y_encoder.decode(
            cls_pred, loc_pred, device=torch.device('cpu'))
        print('begin:%s, end:%s' % (str(xyxy), str(box)))
        print(
            'sigmoid 1:%s, end:%s' % (
                str(torch.tensor([1.]).sigmoid()), str(score[0][:, 1])
            )
        )


if __name__ == "__main__":
    test()
