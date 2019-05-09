import os
import platform

import torch
import argparse
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from progressbar import progressbar as pb
from PIL import ImageDraw, ImageFont

from data_loader import get_data_df, ColabeledDataset
from net import RetinaNet
import transfers
from train import test


TrueColorList = [
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
]
PredColorList = [
    [128, 128, 128],
    [0, 0, 128],
    [0, 128, 0],
    [128, 0, 0]
]


def draw_rectangle(
    img, labels, markers, color_mapper=None, fonts=None, width=4,
    size=40
):
    '''
    在img上画上标记框，如果labels是概率，则会计算其最大的概率，根据最大的列标
        来表颜色，并把最大的概率写在上面；
    args:
        img，PIL的Image对象；
        labels，iterable，是objects的标签，迭代得到的是list；
        markers，iterable，是objects的boxes的loc，迭代得到的是list；
        color_mapper，dict，keys是markers，values是用于draw的颜色；
        fonts，iterable，是要写在预测框上的文字，如果是None则不加这个，
            迭代得到的是list；
    returns:
        img，PIL的Image对象，画上标记框的image；
    '''
    draw = ImageDraw.Draw(img)
    for label, marker in zip(labels, markers):
        if color_mapper is None:
            color = 'red'
        else:
            color = color_mapper[label]
        draw.rectangle(marker, outline=color, width=width)
    if fonts is not None:
        if platform.system() == 'Windows':
            font_type = ImageFont.truetype('calibri', size=size)
        else:
            font_type = ImageFont.truetype('arial.ttf', size=size)
        for label, marker, font in zip(labels, markers, fonts):
            color = color_mapper[label]
            draw.text(
                marker[:2], str(font), fill=color,
                font=font_type
            )
    return img


class NoNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, x):
        mean = self.mean.view(1, -1, 1, 1)
        std = self.std.view(1, -1, 1, 1)

        return x * std + mean


def main():
    from train import default_rd
    # --------------------------- 命令行参数设置 ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', help='model 所在的文件夹'
    )
    parser.add_argument(
        '-bs', '--batch_size', default=2, type=int, help='batch size，默认是2')
    parser.add_argument(
        '-nj', '--n_jobs', default=6, type=int, help='多核并行的核数，默认是6')
    parser.add_argument(
        '-is', '--input_size', default=(960, 600), type=int, nargs=2,
        help=(
            '默认是960, 600（是用于All的detection），因为增加了对于TCT的支持，'
            '这里不再能够使用int和None，必须指定2个'
        )
    )
    parser.add_argument(
        '-rs', '--random_seed', default=1234, type=int,
        help='随机种子数，默认是1234'
    )
    parser.add_argument(
        '-rd', '--root_dir', default=default_rd(),
        help=(
            '数据集所在的根目录，其内部是子文件夹储存图片, 这里是'
            '和system的类型有关，默认是ALL的数据')
    )
    parser.add_argument(
        '--no_normalize', action='store_false',
        help='是否对数据进行标准化，如果使用该参数则不进行标准化'
    )
    parser.add_argument(
        '-bb', '--backbone', default='resnet50',
        help='使用的backbone网络，默认是resnet50'
    )
    parser.add_argument(
        '-ps', default=[3, 4, 5, 6, 7], type=int, nargs='+',
        help='使用FPN中的哪些特征图来构建anchors，默认是p3-p7'
    )
    parser.add_argument(
        '-ph', '--phase', default='valid',
        choices=['all', 'train', 'valid', 'test'],
        help='是对全部的数据集做还是对根据seed分出来的test或valid做，默认是valid'
    )
    parser.add_argument(
        '-sd', '--save_dir', default='valid',
        help=(
            "会创建一个文件夹在模型所在的目录中，输出的图像都保存在其中，默认"
            "名称是valid"
        )
    )
    parser.add_argument(
        '-sr', '--save_root', default='./ALLresults',
        help='结果保存的根目录，默认是./ALLresults'
    )
    parser.add_argument(
        '--top_k', default=0, type=int,
        help="使用的是排名第几的模型进行预测，默认是0"
    )
    parser.add_argument(
        '-lm', '--label_map', default=['异常', '正常'], nargs='+',
        help="用于label mapper的list，按顺序起分别是0、1、...，默认是"
        "['异常', '正常']，即用于ALL分类"
    )
    parser.add_argument(
        '--wh_min', default=None, type=int,
        help="默认是None，用于xml读取，过滤错误的框"
    )
    args = parser.parse_args()

    # --------------------------- 读取文件名称 ---------------------------
    data_df = get_data_df(
        args.root_dir, check=False, check_labels=False)
    label_mapper = {l: i for i, l in enumerate(args.label_map)}
    print(label_mapper)
    # 数据集分割
    if args.phase in ['train', 'valid', 'test']:
        trainval_df, test_df = train_test_split(
            data_df, test_size=0.1, shuffle=True,
            random_state=args.random_seed
        )
        if args.phase == 'test':
            use_dat = test_df
        else:
            train_df, valid_df = train_test_split(
                trainval_df, test_size=1/9, shuffle=True,
                random_state=args.random_seed
            )
            if args.phase == 'train':
                use_dat = train_df
            else:
                use_dat = valid_df
    else:
        use_dat = data_df

    # --------------------------- 数据集建立 ---------------------------
    img_transfer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img_transfer = transfers.OnlyImage(img_transfer)
    resize_transfer = transfers.Resize(args.input_size)
    test_transfers = transforms.Compose([
        resize_transfer, img_transfer
    ])
    y_encoder_args = {'input_size': args.input_size, 'ps': args.ps}
    xml_parse = {}
    if args.wh_min is not None:
        xml_parse['wh_min'] = args.wh_min
    use_data = ColabeledDataset(
        use_dat, transfer=test_transfers, y_encoder_mode='object',
        label_mapper=label_mapper, xml_parse=xml_parse, **y_encoder_args
    )
    use_dataloader = DataLoader(
        use_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_jobs, collate_fn=use_data.collate_fn)

    # --------------------------- 载入训练好的模型 ---------------------------
    if args.backbone == 'resnet50':
        backbone = models.resnet50
    elif args.backbone == 'resnet101':
        backbone = models.resnet101
    net = RetinaNet(backbone=backbone, ps=args.ps, num_class=len(label_mapper))
    bests = torch.load(
        os.path.join(args.save_root, args.model, 'model.pth')
    )
    state_dict = bests[args.top_k]['model_wts']
    net.load_state_dict(state_dict)
    net.eval()

    # --------------------------- 预测 ---------------------------
    (labels_preds, markers_preds), (APs, mAP_score) = test(
        net.cuda(), use_dataloader, evaluate=True, predict=True,
        device=torch.device('cuda:0'), num_class=len(label_mapper)
    )
    for k, v in label_mapper.items():
        print('%s的AP是%.4f' % (k, APs[v]))
    print('mAP是%.4f' % mAP_score)

    # --------------------------- 可视化 ---------------------------
    # 创建文件夹保存结果
    save_dir = os.path.join(args.save_root, args.model, args.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # 设置和图像预处理相反的操作，得到原始图像
    to_pil = transforms.ToPILImage()
    no_norm = NoNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # 遍历结果，并将结果画在图上
    true_color_mapper = {
        i: tuple(TrueColorList[i]) for i in range(len(args.label_map))}
    pred_color_mapper = {
        i: tuple(PredColorList[i]) for i in range(len(args.label_map))}
    for j, ((imgs, labels, markers), labels_pred, markers_pred) in pb(
        enumerate(zip(use_dataloader, labels_preds, markers_preds))
    ):
        imgs = no_norm(imgs)
        for i, (img, label, marker, label_pred, marker_pred) in enumerate(
            zip(imgs, labels, markers, labels_pred, markers_pred)
        ):
            img = to_pil(img)
            img = draw_rectangle(
                img, label.tolist(), marker.tolist(),
                color_mapper=true_color_mapper
            )
            if label_pred.numel() == 0:
                img.save(
                    os.path.join(
                        save_dir, str(j*args.batch_size+i) + '.png'
                    )
                )
            else:
                proba, idx_pred = label_pred.max(dim=1)
                img = draw_rectangle(
                    img, idx_pred.cpu().numpy(), marker_pred.cpu().numpy(),
                    color_mapper=pred_color_mapper,
                    fonts=proba.cpu().numpy().round(4)
                )
                img.save(
                    os.path.join(save_dir, str(j*args.batch_size+i) + '.png'))


if __name__ == "__main__":
    main()
