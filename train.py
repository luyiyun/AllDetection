import os
import copy

import torch
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import argparse
import progressbar as pb

from data_loader import get_data_df, AllDetectionDataset
from data_encoder import YEncoder
import transfers
from net import RetinaNet
from losses import FocalLoss
from metrics import mAP


def train(
    net, criterion, optimizer, dataloaders, epoch, lr_schedular=None,
    data_encoder=None
):
    # 训练开始的时候需要初始化的一些值
    history = {
        'loss': [], 'cls_loss': [], 'loc_loss': [],
        'val_loss': [], 'val_cls_loss': [], 'val_loc_loss': []
    }
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(net.state_dict())
    for e in range(epoch):
        if 'valid' in dataloaders.keys():
            phases = ['train', 'valid']
        else:
            phases = ['train']

        for phase in phases:
            # 计算每个epoch的loss
            epoch_loss = 0.
            epoch_cls_loss = 0.
            epoch_loc_loss = 0.
            # 分不同的phase进行处理
            if phase == 'train':
                net.train()
                if lr_schedular is not None:
                    raise NotImplementedError
                widgets = [
                    'epoch: %d' % e, '| ', pb.Counter(),
                    pb.Bar(), pb.AdaptiveETA(),
                ]
                iterator = pb.progressbar(dataloaders[phase], widgets=widgets)
            else:
                net.eval()
                iterator = dataloaders[phase]
            # 进行一个epoch中所有batches的迭代
            for imgs, labels, markers in iterator:
                imgs = imgs.cuda()
                labels = labels.cuda()
                markers = markers.cuda()
                if phase == 'train':
                    optimizer.zero_grad()
                with torch.set_grad_enable(phase == 'train'):
                    cls_preds, loc_preds = net(imgs)
                    loss, cls_loss, loc_loss = criterion(
                        labels, cls_preds, markers, loc_preds)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                with torch.no_grad():
                    epoch_loss += loss.item()
                    epoch_cls_loss += cls_loss.item()
                    epoch_loc_loss += loc_loss.item()
            # 每个epoch的所有batches都结束，计算在epoch上的loss平均
            with torch.no_grad():
                num_batches = len(dataloaders[phase].dataset)
                epoch_loss /= num_batches
                epoch_cls_loss /= num_batches
                epoch_loc_loss /= num_batches
                if phase == 'train':
                    history['loss'].append(epoch_loss)
                    history['cls_loss'].append(epoch_cls_loss)
                    history['loc_loss'].append(epoch_loc_loss)
                elif phase == 'valid':
                    history['val_loss'].append(epoch_loss)
                    history['val_cls_loss'].append(epoch_cls_loss)
                    history['val_loc_loss'].append(epoch_loc_loss)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(net.state_dict())
            print(
                '%s, loss: %.4f, cls_loss: %.4f, loc_loss: %.4f' %
                (phase, epoch_loss, epoch_cls_loss, epoch_loc_loss)
            )

    net.load_state_dict(best_model_wts)
    print('valid best loss: %.4f' % (best_loss))

    # 如果有test，则再进行test的预测
    if 'test' in dataloaders.keys():
        y_encoder = dataloaders['train'].dataset.transfer.\
            transforms[-1].y_encoder
        _, test_loss, test_map = test(
            net, dataloaders['test'], y_encoder, criterion=criterion)
        return history, (test_loss, test_map)

    return history


def test(net, dataloader, data_encoder, criterion=None, evaluate=True):
    assert evaluate and criterion is not None
    print('Testing ...')
    with torch.no_grad():
        all_loss = 0.
        all_cls_preds = []
        all_loc_preds = []
        all_cls_trues = []
        all_loc_trues = []
        for imgs, labels, markers in dataloader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            markers = markers.cuda()
            cls_preds, loc_preds = net(imgs)
            cls_preds, loc_preds = data_encoder.decode(cls_preds, loc_preds)
            if evaluate or criterion is not None:
                loss = criterion(labels, markers, cls_preds, loc_preds)
                all_loss += loss.item()
            all_cls_preds.append(cls_preds)
            all_loc_preds.append(loc_preds)
            if evaluate:
                all_cls_trues.append(labels)
                all_loc_trues.append(markers)
        all_cls_preds = torch.cat(all_cls_preds, dim=0)
        all_loc_preds = torch.cat(all_loc_preds, dim=0)
        if evaluate:
            all_cls_trues = torch.cat(all_cls_trues, dim=0)
            all_loc_trues = torch.cat(all_loc_trues, dim=0)

        if evaluate or criterion is not None:
            mean_loss = all_loss / len(dataloader.dataset)
        if evaluate:
            map_score = mAP(
                all_cls_trues, all_loc_trues, all_cls_preds, all_loc_preds)
            return (all_cls_preds, all_loc_preds), mean_loss, map_score
        elif criterion is not None:
            return (all_cls_preds, all_loc_preds), mean_loss
        else:
            return all_cls_preds, all_loc_preds


def main():
    # 命令行参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'save', default='exam', nargs='?',
        help='保存的本次训练的文件夹的名称，默认是exam')
    parser.add_argument(
        '-sr', '--save_root', default='./results',
        help='结果保存的根目录，默认是./results'
    )
    parser.add_argument(
        '-bs', '--batch_size', default=2, type=int, help='batch size，默认是2')
    parser.add_argument(
        '-nj', '--n_jobs', default=4, type=int, help='多核并行的核数，默认是4')
    parser.add_argument(
        '-is', '--input_size', default=(600, 960), type=int, nargs=2,
        help='模型接受的输入的大小，需要指定两个，默认是(600, 960)'
    )
    parser.add_argument(
        '-lr', '--learning_rate', type=float, default=1e-3,
        help='学习率，默认是1e-3')
    parser.add_argument(
        '-e', '--epoch', type=int, default=200, help='epoch数，默认是200')
    parser.add_argument(
        '-rs', '--random_seed', default=1234, type=int,
        help='随机种子数，默认是1234'
    )
    parser.add_argument(
        '-rd', '--root_dir', default='E:/Python/AllDetection/label_boxes',
        help='数据集所在的根目录，其内部是子文件夹储存图片'
    )
    args = parser.parse_args()
    # 读取数据根目录，构建data frame
    img_label_dir_pair = []
    for d in os.listdir(args.root_dir):
        img_dir = os.path.join(args.root_dir, d)
        label_dir = os.path.join(args.root_dir, d, 'outputs')
        img_label_dir_pair.append((img_dir, label_dir))
    data_df = get_data_df(img_label_dir_pair, check=False)

    # 数据集分割
    trainval_df, test_df = train_test_split(
        data_df, test_size=0.1, shuffle=True, random_state=args.random_seed)
    train_df, valid_df = train_test_split(
        trainval_df, test_size=1/9, shuffle=True, random_state=args.random_seed
    )

    # 数据集建立
    data_augment = transforms.Compose([
        transfers.RandomFlipLeftRight(),
        transfers.RandomFlipTopBottom(),
        transfers.Resize(args.input_size)
    ])
    img_transfer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img_transfer = transfers.OnlyImage(img_transfer)
    y_encoder = YEncoder(input_size=args.input_size)
    y_transfer = transfers.YTransfer(y_encoder)
    train_transfers = transforms.Compose([
        data_augment, img_transfer, y_transfer
    ])
    test_transfers = transforms.Compose([
        img_transfer, y_transfer
    ])
    datasets = {
        'train': AllDetectionDataset(train_df, transfer=train_transfers),
        'valid': AllDetectionDataset(valid_df, transfer=test_transfers),
        'test': AllDetectionDataset(test_df, transfer=test_transfers)
    }
    dataloaders = {
        k: data.DataLoader(
            v, batch_size=args.batch_size, shuffle=False,
            num_workers=args.n_jobs)
        for k, v in datasets.items()
    }

    # 模型建立
    net = RetinaNet()
    net.cuda()
    criterion = FocalLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=args.learning_rate,
        momentum=0.9, weight_decay=1e-4
    )
    lr_schedular = None

    # 模型训练
    history, (test_loss, test_map) = train(
        net, criterion, optimizer, dataloaders, epoch=args.epoch,
        lr_schedular=lr_schedular
    )

    # 模型保存
    save_dir = os.path.join(args.root_dir, args.save)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    state_dict = copy.deepcopy(net.state_dict())
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))


if __name__ == "__main__":
    main()
