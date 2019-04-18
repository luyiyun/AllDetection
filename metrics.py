import torch
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score

from utils import box_iou, one_hot


def mAP(
    true_cls, true_loc, pred_cls, pred_loc, iou_thre=0.5,
):
    '''
    计算mAP
    args:
        true_cls，tensor，(#imgs, #obj,)，值是0,1,...，代表类别；
        true_loc，tensor，(#imgs, #obj, 4)，mode=xyxy，真实的每个ground truth
            bounding boxes的坐标；
        pred_cls，tensor，(#imgs, #anchor_remain, #class)，每个类在每个预测
            框上的得分，anchor_remain表示这是经过卡阈值、nms等剩下的预测；
        pred_loc，tensor，(#imgs, #anchor_remain, 4)，mode=xyxy，预测的框的
            loc；
        iou_thre，iou_thre，默认是0.5，用于匹配预测框和gtbb；
    returns:
        mAP，输出的是一个float的scalar。
    '''
    # 得到图片数和分类数
    num_imgs = true_cls.size(0)
    num_class = true_cls.max().item() + 1
    # 每一张图片的预测框和gtbb进行匹配，这样给每个预测框的每个类上匹配一个新的
    #   label，如果预测框和某个类的gtbb的IoU超过0.5则认为此框在此类上是1，否则
    #   是0，并将不同图片的匹配的结果都concat到一起
    true_cls_for_pred = []
    for i in range(num_imgs):
        t_cls = true_cls[i]
        t_loc = true_loc[i]
        p_loc = pred_loc[i]
        iou_matrix = box_iou(p_loc, t_loc)
        match_matrix = (iou_matrix > 0.5).float()
        one_hot_t = one_hot(t_cls, num_class).float()
        t_cls_for_pred = match_matrix.mm(one_hot_t) > 0
        true_cls_for_pred.append(t_cls_for_pred)
    true_cls_for_pred = torch.cat(true_cls_for_pred, dim=0)
    # 然后计算每个类上的AP(sklearn)，并进行平均（使用average=macro）
    pred_cls = pred_cls.cpu().numpy()
    true_cls_for_pred = true_cls_for_pred.cpu().numpy()
    mAP = average_precision_score(
        true_cls_for_pred, pred_cls, average='macro')
    return mAP


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


def test():
    img = Image.new('RGB', (1920, 1200), color='white')
    draw = ImageDraw.Draw(img)
    rectangles1 = random_rectangle(6)
    pred_cls = torch.rand(6, 2)
    rectangles2 = random_rectangle(4)
    true_cls = torch.randint(2, size=(4,))
    for i in range(4):
        rec1 = rectangles1[:, [1, 0, 3, 2]][i]
        rec2 = rectangles2[:, [1, 0, 3, 2]][i]
        draw.rectangle(rec1.tolist(), outline='red', width=3)
        draw.rectangle(rec2.tolist(), outline='blue', width=3)
    img.show()
    print(mAP(true_cls, rectangles2, pred_cls, rectangles1))


if __name__ == "__main__":
    test()

