import sys
import os
import platform

import numpy as np
import torch
import sklearn.metrics as skm
import matplotlib.pyplot as plt

from utils import box_iou, one_hot
from data_loader import get_data_df, AllDetectionDataset, draw_rectangle


def AP2(true, pred):
    p, r, _ = skm.precision_recall_curve(true, pred)
    return skm.auc(r, p)


def mAP(
    true_cls, true_loc, pred_cls, pred_loc, iou_thre=0.5, num_class=2,
    ap_func=AP2
):
    '''
    计算mAP
    args:
        true_cls，list of tensors，每个tensor的维度是(#obj_i,)，值是0,1,...，
            代表真实的每张图片的多个objs的类别；
        true_loc，list of tensors，每个tensor的维度是(#obj, 4)，mode=xyxy，
            真实的每张图片的每个ground truth bounding boxes的坐标；
        pred_cls，list of tensor，每个tensor的维度是(#anchor_remain, #class)，
            预测的在每张图片上的预测框的得分，anchor_remain表示这是经过卡阈
            值、nms等剩下的预测框；
        pred_loc，list of tensor，每个tensor的维度是(#anchor_remain, 4)，
            mode=xyxy，预测的框的loc，注意，这里list的len就是图片的数量；
        iou_thre，iou_thre，默认是0.5，用于匹配预测框和gtbb；
    returns:
        mAP，输出的是一个float的scalar。
    '''
    # 得到图片数
    num_imgs = len(true_cls)
    # 每一张图片的预测框和gtbb进行匹配，这样给每个预测框的每个类上匹配一个新的
    #   label，如果预测框和某个类的gtbb的IoU超过0.5则认为此框在此类上是1，否则
    #   是0，并将不同图片的匹配的结果都concat到一起
    true_cls_for_pred = []
    for i in range(num_imgs):
        t_cls = true_cls[i].cuda()
        t_loc = true_loc[i].cuda()
        p_loc = pred_loc[i]
        iou_matrix = box_iou(p_loc, t_loc)
        match_matrix = (iou_matrix > iou_thre).float()
        one_hot_t = one_hot(t_cls, num_class).cuda().float()
        t_cls_for_pred = match_matrix.mm(one_hot_t) > 0
        true_cls_for_pred.append(t_cls_for_pred)
    true_cls_for_pred = torch.cat(true_cls_for_pred, dim=0)
    # 然后计算每个类上的AP(sklearn)，并进行平均（使用average=macro）
    pred_cls = torch.cat(pred_cls, dim=0).cpu().numpy()
    true_cls_for_pred = true_cls_for_pred.cpu().numpy()
    # mAP = average_precision_score(
    #     true_cls_for_pred, pred_cls, average='macro')
    aps = []
    for t, p in zip(true_cls_for_pred.T, pred_cls.T):
        if len(t) == 0:
            aps.append(0.)
        else:
            aps.append(ap_func(t, p))
    return aps, np.mean(aps)


'''
以下是测试的代码
'''


def random_rectangle(num, img_size=(1200, 1920)):
    imgh, imgw = img_size
    xmin = torch.randint(imgh, size=(num,))
    ymin = torch.randint(imgw, size=(num,))
    w_h = torch.normal(
        mean=torch.tensor([[imgh/2]*2]*num),
        std=100
    ).floor().long()
    xmax = (xmin + w_h[:, 0]).clamp(max=imgh)
    ymax = (ymin + w_h[:, 1]).clamp(max=imgw)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def simulate_labels(true_labels, pos_min=0.4, pos_max=1.0):
    proba_interval = pos_max - pos_min
    simu_proba = torch.rand(len(true_labels), dtype=torch.float) *\
        proba_interval + pos_min
    simu_proba = simu_proba * true_labels.float() +\
        (1 - true_labels).float() * (1 - simu_proba)
    simu_proba = torch.stack([1. - simu_proba, simu_proba], dim=1)
    return simu_proba


def simulate_markers(true_markers, shape_ratio=(0.8, 1.2)):
    obj_wh = true_markers[:, 2:] - true_markers[:, :2]
    ratio_interval = shape_ratio[1] - shape_ratio[0]
    simu_ratio = torch.rand_like(obj_wh) * ratio_interval + shape_ratio[0]
    simu_wh = obj_wh.float() * simu_ratio
    simu_max = true_markers[:, :2] + simu_wh
    return torch.cat([true_markers[:, :2], simu_max], dim=1)


def test():
    if platform.system() == 'Windows':
        root_dir = 'E:/Python/AllDetection/label_boxes'
    else:
        root_dir = '/home/dl/deeplearning_img/AllDet/label_boxes'
    img_label_dir_pair = []
    for d in os.listdir(root_dir):
        img_dir = os.path.join(root_dir, d)
        label_dir = os.path.join(root_dir, d, 'outputs')
        img_label_dir_pair.append((img_dir, label_dir))

    data_df = get_data_df(img_label_dir_pair, check=True)
    if len(sys.argv) == 1:
        dataset = AllDetectionDataset(data_df, input_size=(1200, 1920))
        for i in range(99, 109):
            img, labels, markers = dataset[i]
            print(labels)
            print(markers)
            img = draw_rectangle(img, labels.numpy(), markers.numpy())
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(np.asarray(img))
            plt.show()
            if i == 4:
                break
    else:
        if sys.argv[1] == 'simulate_good':
            score_ratio = (0.4, 1.0)
            loc_ratio = (0.8, 1.2)
        elif sys.argv[1] == 'simulate_bad_score':
            score_ratio = (0.4, 0.6)
            loc_ratio = (0.8, 1.2)
        elif sys.argv[1] == 'simulate_bad_loc':
            score_ratio = (0.4, 1.0)
            loc_ratio = (0.4, 2.0)
        dataset = AllDetectionDataset(data_df, input_size=(1200, 1920))
        imgs = []
        true_labels = []
        true_markers = []
        simu_labels = []
        simu_markers = []
        for i in range(99, 109):
            img, labels, markers = dataset[i]
            imgs.append(img)
            true_labels.append(labels.cuda())
            true_markers.append(markers.cuda())

            simu_labels.append(simulate_labels(labels, *score_ratio).cuda())
            simu_markers.append(simulate_markers(markers, loc_ratio).cuda())

            # img = draw_rectangle(
            # img, simu_labels[-1].cpu().numpy(),
            # simu_markers[-1].cpu().numpy())
            # fig, ax = plt.subplots(figsize=(20, 10))
            # ax.imshow(np.asarray(img))
            # plt.show()
        map_score = mAP(
            true_labels, true_markers, simu_labels, simu_markers,
            iou_thre=0.5, num_class=2)
        print(map_score)


if __name__ == "__main__":
    test()

