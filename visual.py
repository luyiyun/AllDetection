import os

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


def draw_rectangle(
    img, labels, markers, color_mapper=None, fonts=None
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
        draw.rectangle(marker, outline=color, width=6)
    if fonts is not None:
        font_type = ImageFont.truetype('calibri', size=50)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', help='model 所在的文件夹'
    )
    parser.add_argument(
        '-bs', '--batch_size', default=2, type=int, help='batch size，默认是2')
    parser.add_argument(
        '-nj', '--n_jobs', default=6, type=int, help='多核并行的核数，默认是6')
    parser.add_argument(
        '-is', '--input_size', default=None, type=int, nargs='+',
        help=(
            '模型接受的输入图片的大小，如果指定了1个，则此为短边会resize到的大小'
            '，在此期间会保证图片的宽高比，如果指定两个，则分别是高和宽，'
            '默认是None，即不进行resize')
    )
    parser.add_argument(
        '-rs', '--random_seed', default=1234, type=int,
        help='随机种子数，默认是1234'
    )
    parser.add_argument(
        '-rd', '--root_dir',
        default='/home/dl/deeplearning_img/AllDet/label_boxes',
        help=(
            '数据集所在的根目录，其内部是子文件夹储存图片, 这里默认是'
            '/home/dl/deeplearning_img/AllDet/label_boxes')
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
        '-sr', '--save_root', default='./results',
        help='结果保存的根目录，默认是./results'
    )
    parser.add_argument(
        '--top_k', default=0, type=int,
        help="使用的是排名第几的模型进行预测，默认是0"
    )
    args = parser.parse_args()
    # 需要对input_size进行比较复杂的处理
    if isinstance(args.input_size, (tuple, list)):
        if len(args.input_size) == 1:
            input_size = args.input_size[0]
        else:
            input_size = list(args.input_size)
    elif args.input_size is None:
        input_size = None
    else:
        raise ValueError('input_size must be one of tuple, list or None')

    data_df, label_set = get_data_df(
        args.root_dir, check=False, check_labels=True)
    label_mapper = {l: i for i, l in enumerate(label_set)}
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

    # 数据集建立
    img_transfer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img_transfer = transfers.OnlyImage(img_transfer)
    # 注意对于PIL，其输入大小时是w,h的格式
    if input_size is None:
        test_transfers = img_transfer
    else:
        # 在PIL中，一般表示图像的两个空间维度时候使用w,h的顺序，但在pytorch中，
        # 一般使用h,w的顺序，这里使用的Resize是pytorch的对象，所以其接受的是h x
        # w的顺序
        resize_transfer = transfers.Resize(input_size)
        test_transfers = transforms.Compose([
            resize_transfer, img_transfer
        ])
    # 这里实际上是送给YEncoder类的参数，需要指定为w x h
    if isinstance(input_size, list):
        y_input_size = input_size[::-1]
    else:
        y_input_size = input_size
    y_encoder_args = {'input_size': y_input_size, 'ps': args.ps}
    use_data = ColabeledDataset(
        use_dat, transfer=test_transfers, y_encoder_mode='object',
        label_mapper=label_mapper, **y_encoder_args
    )
    use_dataloader = DataLoader(
        use_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_jobs, collate_fn=use_data.collate_fn)

    # 载入训练好的模型
    if args.backbone == 'resnet50':
        backbone = models.resnet50
    elif args.backbone == 'resnet101':
        backbone = models.resnet101
    net = RetinaNet(backbone=backbone, ps=args.ps)
    bests = torch.load(
        os.path.join(args.save_root, args.model, 'model.pth')
    )
    state_dict = bests[args.top_k]['model_wts']
    net.load_state_dict(state_dict)
    net.eval()

    # 使用模型进行预测
    (labels_preds, markers_preds), = test(
        net.cuda(), use_dataloader, evaluate=False, predict=True,
        device=torch.device('cuda:0')
    )

    # ..
    save_dir = os.path.join(args.save_root, args.model, args.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    to_pil = transforms.ToPILImage()
    no_norm = NoNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    for j, ((imgs, labels, markers), labels_pred, markers_pred) in pb(
        enumerate(zip(use_dataloader, labels_preds, markers_preds))
    ):
        imgs = no_norm(imgs)
        # for i in range(args.batch_size):
        for i, (img, label, marker, label_pred, marker_pred) in enumerate(
            zip(imgs, labels, markers, labels_pred, markers_pred)
        ):
            # img = imgs[i]
            # label = labels[i]
            # marker = markers[i]
            # label_pred = labels_pred[i]
            # marker_pred = markers_pred[i]
            img = to_pil(img)
            img = draw_rectangle(img.tolist(), label.tolist(), marker.tolist())
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
                    color_mapper={0: 'blue', 1: 'yellow'},
                    fonts=proba.cpu().numpy().round(4)
                )
                img.save(
                    os.path.join(save_dir, str(j*args.batch_size+i) + '.png'))


if __name__ == "__main__":
    main()
