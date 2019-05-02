import os

import torch
import argparse
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from progressbar import progressbar as pb

from data_loader import get_data_df, AllDetectionDataset, draw_rectangle
from net import RetinaNet
import transfers
from train import test


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
        '-nj', '--n_jobs', default=4, type=int, help='多核并行的核数，默认是4')
    parser.add_argument(
        '-is', '--input_size', default=(960, 600), type=int, nargs=2,
        help='模型接受的输入的大小，需要指定两个，即宽x高，默认是(960, 600)'
    )
    parser.add_argument(
        '-rs', '--random_seed', default=1234, type=int,
        help='随机种子数，默认是1234'
    )
    parser.add_argument(
        '-rd', '--root_dir', default='E:/Python/AllDetection/label_boxes',
        help='数据集所在的根目录，其内部是子文件夹储存图片'
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
    args = parser.parse_args()

    img_label_dir_pair = []
    for d in os.listdir(args.root_dir):
        img_dir = os.path.join(args.root_dir, d)
        label_dir = os.path.join(args.root_dir, d, 'outputs')
        img_label_dir_pair.append((img_dir, label_dir))
    data_df = get_data_df(img_label_dir_pair, check=False)

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
                use_dat == train_df
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
    if list(args.input_size) == [1920, 1200]:
        test_transfers = img_transfer
    else:
        resize_transfer = transfers.Resize(args.input_size)
        test_transfers = transforms.Compose([
            resize_transfer, img_transfer
        ])
    y_encoder_args = {
        'input_size': args.input_size,
        'ps': args.ps
    }
    use_data = AllDetectionDataset(
        use_dat, transfer=test_transfers, y_encoder_mode='object',
        **y_encoder_args)
    use_dataloader = DataLoader(
        use_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_jobs, collate_fn=use_data.collate_fn)

    # 载入训练好的模型
    if args.backbone == 'resnet50':
        backbone = models.resnet50
    elif args.backbone == 'resnet101':
        backbone = models.resnet101
    net = RetinaNet(backbone=backbone, ps=args.ps)
    state_dict = torch.load(
        os.path.join(args.save_root, args.model, 'model.pth')
    )
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
        for i in range(args.batch_size):
            img = imgs[i]
            label = labels[i]
            marker = markers[i]
            label_pred = labels_pred[i]
            marker_pred = markers_pred[i]
            img = to_pil(img)
            img = draw_rectangle(img, label.numpy(), marker.numpy())
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
