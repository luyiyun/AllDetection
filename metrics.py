import sys
import os
import platform

import numpy as np
import torch
import sklearn.metrics as skm

from utils import box_iou
from data_loader import get_data_df, ColabeledDataset


def compute_ap(recall, precision):
    '''
    使用得到的recall和precision计算ap（即曲线下面积），这个得到的结果会比使用
    sklearn中的auc函数得到的值要高。
    args：
        recall，得到的一系列的recall值，是x轴；
        precision，得到的一系列的precision值，是y轴；
        （以上两个都没有包含(0, 0)和(0, 1)这两个点，这两个点应该是pr曲线中一定存
        在的两个点，所以在下面的程序中会补上）
    returns：
        ap，计算得到的average precision，也可以看做pr曲线下面积；
    '''
    # 加上(0, 0)和(0, 1)两个点
    mrec = np.concatenate([[0.], recall, [1.]])
    mpre = np.concatenate([[0.], precision, [0.]])
    # 这样使得整个pr曲线是递减的，也是这里的缘故使得此函数的结果要比auc函数的结
    #   果要好（auc计算的梯形面积）
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    # 可能相邻的两个点的recall是一样的，那这样我们选择前一个点
    # 这样得到的是所有后一个点和此点的recall不相同的点
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # 计算矩形面积，但注意的是使用的高是小梯形右边的边长，这个边长是较小的一个
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def mAP(
    true_cls, true_loc, pred_cls, pred_loc, iou_thre=0.5, num_class=2,
    ap_func=compute_ap
):
    '''
    计算mAP，参考的是
        https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/
        utils/eval.py
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
        num_class，分类个数；
        ap_func，使用precision和recall计算ap的函数，默认使用的是compute_ap，这个
            相比于sklearn.metrics.auc得到的结果要好一些；
    returns:
        APs，每个类的AP值；
        mAP，输出的是一个float的scalar，所有类的平均AP。
    '''
    # 把true objects转到pred objects相关的设备中
    device = pred_cls[0].device
    true_cls = [tc.to(device) for tc in true_cls]
    true_loc = [tl.to(device) for tl in true_loc]
    # 储存每一类的ap值
    aps = []
    num_imgs = len(true_cls)
    # 因为输入的是预测的每个类的分数，需要取最大得到预测的类和这个类的score
    pred_score = []
    pred_class = []
    for t in pred_cls:
        # 如果存在空的pred_cls（即没有预测框，直接使用max方法会报错）
        if len(t) == 0:
            pred_score.append(t.new_empty(0))
            pred_class.append(
                torch.zeros(0, dtype=torch.long, device=t.device))
        else:
            t_s, t_c = t.max(dim=1)
            pred_score.append(t_s)
            pred_class.append(t_c)
    all_scores = torch.cat(pred_score, dim=0)
    all_classes = torch.cat(pred_class, dim=0)
    # 对每个类分别计算ap
    for c in range(num_class):
        # 创建ndarray来储存是否是一个true positive的预测
        tp = np.zeros((0,))
        # 记录这一类一共有多少个true objects，用于计算recall
        num_true_objs = 0.0
        for i in range(num_imgs):
            # 得到一张图片中这一类的所有预测框
            true_c_mask = true_cls[i] == c
            num_true_objs += true_c_mask.sum()
            pred_c_mask = pred_class[i] == c
            true_loc_i_c = true_loc[i][true_c_mask]
            pred_loc_i_c = pred_loc[i][pred_c_mask]
            # 用于记录已经匹配到的gtbb的序号，每张图片重新记录
            detected_true_boxes = []
            # 如果这个类的预测框数量为0，则此循环不会运行，fp和tp是长度为0的
            #   array；如果此类的预测框数量不是0而是N，则会使fp和tp的长度增加
            #   至N。
            for d in pred_loc_i_c:
                # 如果这张图片上gtbb并没有这个类，则所有这类的预测都被看做是
                #   false positive
                if true_loc_i_c.size(0) == 0:
                    tp = np.append(tp, 0)
                    continue
                # 计算预测框和所有gtbb的IoU，并去最大的一个作为此预测框的预测对象
                ious = box_iou(d.unsqueeze(0), true_loc_i_c).squeeze(0)
                max_iou, max_idx = ious.max(dim=0)
                # 如果这个最大的IoU大于thre，则认为此预测框针对的正是这个gtbb，
                #   则认为这是个true postive（实际上认为是true postive还有一个
                #   条件是这个预测框的scores大于指定的阈值，但我们需要移动这个阈
                #   值来构建不同的recall和其对应的precision，所以这里先认为
                #   只要是匹配上了就是1，当之后变化阈值的时候只要把score小于阈值
                #   tp设为0、fp设为1即可，而本来都没有匹配上的永远是0）
                # 另外，记录这一张图片上已经匹配过的gtbb，之后再进行匹配的时候就
                #   不进行匹配了（这里有一些小问题，即我们考虑的时候没有先考虑score
                #   比较大的框，这样可能导致因为score比较小的框先把gtbb给占了而
                #   导致可能匹配的更好的score更高的预测框被认为是false postive，
                #   这样可能会拉低ap）
                if max_iou >= iou_thre and max_idx not in detected_true_boxes:
                    tp = np.append(tp, 1)
                    detected_true_boxes.append(max_idx)
                else:
                    tp = np.append(tp, 0)

        # 如果对于某一类，所有图片的gtbb中都没有这一类，则认为此类的ap是0，？
        if num_true_objs == 0.0:
            aps.append(0.)
            continue
        # 依据score进行排序
        _, order = all_scores[all_classes == c].sort(dim=0, descending=True)
        order = order.cpu().numpy()
        tp = tp[order]
        fp = 1 - tp
        # 逐个计算array的前n个元素中fp和tp的个数，这个可以看做在每个元素的间隔间
        #   变化阈值来使的低于此阈值的所有都被预测是0，则计算postive（不管是fp
        #   还是tp）只需要考虑前面就可以了。
        fp = fp.cumsum()
        tp = tp.cumsum()
        # 计算recall和precision
        recall = tp / num_true_objs.item()
        # --这里可能出现预测的里没有postive（比如我们把阈值卡的特别高的时候），
        #   当然这是tp也是0，但分母=0会使得无法计算，所以需要加一个eps来避免
        precision = tp / np.maximum((tp + fp), np.finfo(np.float64).eps)
        aps.append(ap_func(recall, precision))
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
        root_dir = '/home/dl/deeplearning_img/Detection/ALL'
    data_df, label_set = get_data_df(
        root_dir, check=False, check_labels=True)
    label_mapper = {l: i for i, l in enumerate(label_set)}
    print(label_mapper)
    if sys.argv[1] == 'simulate_good':
        score_ratio = (0.4, 1.0)
        loc_ratio = (0.8, 1.2)
    elif sys.argv[1] == 'simulate_bad_score':
        score_ratio = (0.4, 0.6)
        loc_ratio = (0.8, 1.2)
    elif sys.argv[1] == 'simulate_bad_loc':
        score_ratio = (0.4, 1.0)
        loc_ratio = (0.4, 2.0)
    dataset = ColabeledDataset(data_df, input_size=(600, 960))
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
    map_score1 = mAP(
        true_labels, true_markers, simu_labels, simu_markers,
        iou_thre=0.5, num_class=2, ap_func=compute_ap)
    map_score2 = mAP(
        true_labels, true_markers, simu_labels, simu_markers,
        iou_thre=0.5, num_class=2, ap_func=skm.auc)
    print(map_score1, map_score2)


if __name__ == "__main__":
    test()

