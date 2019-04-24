import os
import copy
import json
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import argparse
import progressbar as pb

from data_loader import get_data_df, AllDetectionDataset
import transfers
from net import RetinaNet
from losses import FocalLoss
from metrics import mAP


def train(
    net, criterion, optimizer, dataloaders, epoch, lr_schedular=None,
):
    '''
    对RetinaNet进行训练，
    args：
        net，实例化的RetinaNet网络，在训练的过程中其发生了改变；
        criterion，loss，即实例化的FocalLoss对象；
        optimizer，优化器；
        dataloaders，dataloaders组成的dict，其key必须有'train'，可以有'valid'，
            'test'，如果有test则会在最后使用训练过程中最好的net进行预测；
        epoch，int，一共需要进行多少个epoch的训练；
        lr_schedular，使用的学习率策略；
    returns：
        history，dict，储存每个epoch train的loss和valid的loss、mAP；
        （另外，实际上输入的net会发生改变，成为在训练过程中valid上最好的net）
        test_loss, test_mAP，如果有dataloaders中还有test，则会返回；
    '''
    # 训练开始的时候需要初始化的一些值
    history = {
        'loss': [], 'cls_loss': [], 'loc_loss': [],
        'val_loss': [], 'val_cls_loss': [], 'val_loc_loss': [],
        'mAP': []
    }
    best_loss = float('inf')
    best_map = 0.
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
            y_encoder = dataloaders[phase].dataset.y_encoder
            all_label_preds = []
            all_marker_preds = []
            all_label_trues = []
            all_marker_trues = []
            for imgs, labels, markers in iterator:
                imgs = imgs.cuda()
                if phase == 'train':
                    cls_trues = labels.cuda()
                    loc_trues = markers.cuda()
                else:
                    labels, cls_trues = labels
                    cls_trues = cls_trues.cuda()
                    markers, loc_trues = markers
                    loc_trues = loc_trues.cuda()

                if phase == 'train':
                    optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    cls_preds, loc_preds = net(imgs)
                    loss, cls_loss, loc_loss = criterion(
                        cls_trues, loc_trues, cls_preds, loc_preds)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        label_preds, marker_preds = y_encoder.decode(
                            cls_preds, loc_preds)
                        all_label_preds.extend(label_preds)
                        all_marker_preds.extend(marker_preds)
                        all_label_trues.extend(labels)
                        all_marker_trues.extend(markers)
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
                    print(
                        '%s, loss: %.4f, cls_loss: %.4f, loc_loss: %.4f' %
                        (phase, epoch_loss, epoch_cls_loss, epoch_loc_loss)
                    )
                elif phase == 'valid':
                    _, map_score = mAP(
                        all_label_trues, all_marker_trues,
                        all_label_preds, all_marker_preds)
                    history['val_loss'].append(epoch_loss)
                    history['val_cls_loss'].append(epoch_cls_loss)
                    history['val_loc_loss'].append(epoch_loc_loss)
                    history['mAP'].append(map_score)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(net.state_dict())
                        best_map = map_score
                    print(
                        ('%s, loss: %.4f, cls_loss: %.4f,'
                         ' loc_loss: %.4f, mAP: %.4f') %
                        (
                            phase, epoch_loss, epoch_cls_loss,
                            epoch_loc_loss, map_score
                        )
                    )

    if 'valid' in dataloaders.keys():
        net.load_state_dict(best_model_wts)
        print('valid best loss: %.4f, best mAP: %.4f' % (best_loss, best_map))

    # 如果有test，则再进行test的预测
    if 'test' in dataloaders.keys():
        test_loss, test_map = test(
            net, dataloaders['test'], criterion=criterion, evaluate=True,
            predict=False
        )
        print('test loss: %.4f, test mAP: %.4f' % (test_loss[0], test_map[1]))
        return history, (test_loss, test_map)

    return history


def test(net, dataloader, criterion=None, evaluate=True, predict=True):
    '''
    使用训练好的net进行预测或评价
    args：
        net，训练好的RetinaNet对象；
        dataloader，需要进行预测或评价的数据集的dataloader；
        criterion，计算损失的函数，如果是None则不计算loss；
        evaluate，如果是True则进行模型的评价（即计算mAP），如果
            是False，则只进行预测或计算loss；
        predict，是否进行预测，如果True则会返回预测框的类别和坐标，如果False，
            则不会返回；
    returns：
        all_labels_pred, all_markers_pred，如果predict=True，则返回预测的每张图
            片的预测框的标签和loc；
        losses, cls_losses, loc_losses，如果criterion不是None，则得到其3个loss；
        APs, mAP_score，如果evaluate=True，则返回每一类的AP值和其mAP；
    '''
    data_encoder = dataloader.dataset.y_encoder
    y_encoder_mode = dataloader.dataset.y_encoder_mode
    assert evaluate and y_encoder_mode in ['obj', 'all']
    assert criterion is not None and y_encoder_mode in ['all', 'anchor']

    print('Testing ...')
    results = []
    with torch.no_grad():
        losses = 0.
        cls_losses = 0.
        loc_losses = 0.
        all_labels_pred = []  # 预测的labels，
        all_markers_pred = []
        all_labels_true = []
        all_markers_true = []
        for imgs, labels, markers in dataloader:
            imgs = imgs.cuda()
            if y_encoder_mode == 'all':
                # 如果需要计算mAP，则除了dataloader除了输出已经编码好的每个
                #   anchor对应的偏移量和对应的标签，还需要没有编码的每张图片
                #   对应的obj loc tensor和obj label tensor，因为每张图片对应的
                #   objs的数量不同，所以只能使用list来储存（通过重写dataloader
                #   的collate_fn）
                # labels是list，每个元素是一个tensor，指代一张图片中所有的obj
                #   的label，[(img1#obj,), (img2#obj,), (img3#obj,), ...]
                # cls_trues是一个tensor，(#imgs, #anchors, #class)，是每张图片
                #   中每个anchor对应的gtbb的类别
                labels, cls_trues = labels
                cls_trues = cls_trues.cuda()  # labels in cpu, cls_trues in gpu
                # markers是list，每个元素是一个tensor，指代一张图片中所有的obj
                #   的xyxy loc，[(img1#obj, 4), (img2#obj, 4), (img3#obj, 4),
                #   ...]
                # loc_trues是一个tensor，(#imgs, #anchors, 4)，是每张图片中每个
                #   anchor和其对应的gtbb的位置偏移量
                markers, loc_trues = markers
                loc_trues = loc_trues.cuda()  # marker in cpu, loc_trues in gpu
            elif y_encoder_mode == 'anchor':
                # 如果只需要计算loss，则只需要输出编码好的偏移量和anchor的标签
                #   即可
                cls_trues = labels.cuda()
                loc_trues = markers.cuda()

            cls_preds, loc_preds = net(imgs)

            if criterion is not None:
                loss, cls_loss, loc_loss = criterion(
                    cls_trues, loc_trues, cls_preds, loc_preds)
                losses += loss.item()
                cls_losses += cls_loss.item()
                loc_losses += loc_loss.item()
            # 使用decode得到的是list，其每个元素是batch中每个图片的预测框的
            #   tensor
            label_preds, marker_preds = data_encoder.decode(
                cls_preds, loc_preds)
            # 使用append则输出的的list的len是图片的数量，这样的输出更加明白
            all_markers_pred.append(marker_preds)
            all_labels_pred.append(label_preds)
            if evaluate:
                all_labels_true.append(labels)
                all_markers_true.append(markers)
        if predict:
            results.append((all_labels_pred, all_markers_pred))
        if criterion is not None:
            losses = losses / len(dataloader.dataset)
            cls_losses = cls_losses / len(dataloader.dataset)
            loc_losses = loc_losses / len(dataloader.dataset)
            results.append((losses, cls_losses, loc_losses))
        if evaluate:
            # 使用chain将两层的嵌套list变成一层，符合mAP函数的输出要求
            APs, mAP_score = mAP(
                list(chain.from_iterable(all_labels_true)),
                list(chain.from_iterable(all_markers_true)),
                list(chain.from_iterable(all_labels_pred)),
                list(chain.from_iterable(all_markers_pred)),
            )
            results.append((APs, mAP_score))
    return tuple(results)


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
        '-is', '--input_size', default=(960, 600), type=int, nargs=2,
        help='模型接受的输入的大小，需要指定两个，即宽x高，默认是(960, 600)'
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
    data_augment = transfers.MultiCompose([
        transfers.RandomFlipLeftRight(),
        transfers.RandomFlipTopBottom(),
    ])
    # 注意对于PIL，其输入大小时是w,h的格式
    resize_transfer = transfers.Resize(args.input_size)
    img_transfer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img_transfer = transfers.OnlyImage(img_transfer)
    train_transfers = transfers.MultiCompose([
        data_augment, resize_transfer, img_transfer
    ])
    test_transfers = transfers.MultiCompose([
        resize_transfer, img_transfer
    ])
    datasets = {
        'train': AllDetectionDataset(
            train_df, transfer=train_transfers, y_encoder_mode='anchor',
            input_size=(600, 960)),
        'valid': AllDetectionDataset(
            valid_df, transfer=test_transfers,
            y_encoder_mode='all', input_size=(600, 960)
        ),
        'test': AllDetectionDataset(
            test_df, transfer=test_transfers, y_encoder_mode='all',
            input_size=(600, 960))
    }
    dataloaders = {
        k: data.DataLoader(
            v, batch_size=args.batch_size, shuffle=False,
            num_workers=args.n_jobs, collate_fn=v.collate_fn)
        for k, v in datasets.items()
    }

    # 模型建立
    net = RetinaNet()
    net.cuda()
    # net.freeze_bn()  # ???
    criterion = FocalLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=args.learning_rate,
        momentum=0.9,  # weight_decay=1e-4
    )
    # optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    lr_schedular = None

    # 模型训练
    history, (test_loss, test_map) = train(
        net, criterion, optimizer, dataloaders, epoch=args.epoch,
        lr_schedular=lr_schedular
    )

    # 模型保存
    save_dir = os.path.join(args.save_root, args.save)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    state_dict = copy.deepcopy(net.state_dict())
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

    test_res = {
            k: v for k, v in zip(['loss', 'cls_loss', 'loc_loss'], test_loss)}
    test_res['APs'] = test_map[0]
    test_res['mAP'] = test_map[1]
    with open(os.path.join(save_dir, 'test.json'), 'w') as f:
        json.dump(test_res, f)

    train_df = pd.DataFrame(history)
    train_df.to_csv(os.path.join(save_dir, 'train.csv'))


if __name__ == "__main__":
    main()
