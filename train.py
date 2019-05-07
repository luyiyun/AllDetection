import os
import copy
import json
import platform
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from sklearn.model_selection import train_test_split
import argparse
import progressbar as pb

from data_loader import get_data_df, ColabeledDataset
import transfers
from net import RetinaNet
from losses import FocalLoss
from metrics import mAP


def default_rd():
    if platform.system() == 'Windows':
        return 'E:/Python/AllDetection/label_boxes'
    return '/home/dl/deeplearning_img/AllDet/label_boxes'


class History:
    '''
    用于储存训练时期结果的对象，其中其有两个最重要的属性：history和best
    history，是一个dict或df，用于储存在训练时期loss和mAP的信息；
    best，用于储存当前训练阶段最好的t个模型及其metrics；
    '''
    def __init__(self, best_num=3, best_metric='mAP', min_better=False):
        self.best_metric = best_metric
        self.verse_f = np.less if min_better else np.greater
        self.history = {
            'loss': [], 'cls_loss': [], 'loc_loss': [],
            'val_loss': [], 'val_cls_loss': [], 'val_loc_loss': [],
            'mAP': []
        }
        self.hist_keys = self.history.keys()
        self.best = [
            {
                'mAP': 0., 'loss': float('inf'), 'cls_loss': float('inf'),
                'loc_loss': float('inf'), 'epoch': -1, 'model_wts': None
            } for _ in range(best_num)
        ]
        self.best_keys = set(self.best[0].keys())
        # self.best_keys_alt = copy.deepcopy(self.best_keys)
        # self.best_keys_alt.remove('model_wts')
        # self.best_keys_alt.add('model')

    def update_best(self, **kwargs):
        '''
        将本次得到的结果更新到best属性中
        args是self.best的keys，包括mAP、loss、cls_loss、loc_loss、epoch和
            model_wts，其中model_wts是model的state_dict的deepcopy值
        这里需要注意的是，需要将所有要更新的参数都写上，不然会报错。
        '''
        self.compare_keys(kwargs.keys(), 'best')  # 检查输入是否符合格式
        best_scores = [b[self.best_metric] for b in self.best]
        new_score = kwargs[self.best_metric]
        for i, bs in enumerate(best_scores):
            if self.verse_f(new_score, bs):
                self.best.insert(i, kwargs)
                self.best.pop()
                break

    def update_hist(self, **kwargs):
        '''
        更新history属性
        '''
        # self.compare_keys(kwargs.keys(), 'hist')
        # 因为hist的内容需要分别在train和valid阶段进行更新，所以一次更新是无法
        #   完成的，就没有设置检查机制
        for k, v in kwargs.items():
            self.history[k].append(v)

    def compare_keys(self, new_keys, func='best'):
        '''
        检查输入是否符合格式，即是否是我们要求的那些keys；
        args:
            new_keys，是输入的参数名；
            func，我们要比较的是哪个函数的参数s，可以选择best or hist；
        '''
        new_keys = set(new_keys)
        if func == 'best':
            old_keys = self.best_keys
        elif func == 'hist':
            old_keys = self.hist_keys
        diff = len(new_keys.symmetric_difference(old_keys))
        if diff > 0:
            raise ValueError(
                '输入的参数和要求的不一致，要求的参数是%s' % str(old_keys)
            )


def train(
    net, criterion, optimizer, dataloaders, epoch, lr_schedular=None,
    clip_norm=None, best_num=1, best_metric='mAP', history=History()
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
        clip_norm, 使用的梯度截断的参数，如果是None则不进行梯度截断；
        best_num，要保存的最好的模型的数目；
        best_metric，来判断模型好坏的指标，可以选择mAP或loss；
        history，History对象，用于储存训练时期的结果，默认是History()的默认配置；
    returns：
        history，History对象，储存每个epoch train的loss和valid的loss、mAP和最好
            的几个模型；
        （另外，实际上输入的net会发生改变，成为在训练过程中valid上最好的net）
        test_loss, test_mAP，如果有dataloaders中还有test，则会返回；
    '''
    # 训练开始的时候需要初始化的一些值
    # history = {
    #     'loss': [], 'cls_loss': [], 'loc_loss': [],
    #     'val_loss': [], 'val_cls_loss': [], 'val_loc_loss': [],
    #     'mAP': []
    # }
    # best_loss = float('inf')
    # best_map = 0.
    # best_model_wts = copy.deepcopy(net.state_dict())
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
                    lr_schedular.step()
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
                        if clip_norm != 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                net.parameters(), clip_norm)
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
                    history.update_hist(
                        loss=epoch_loss, cls_loss=epoch_cls_loss,
                        loc_loss=epoch_loc_loss
                    )
                    print(
                        '%s, loss: %.4f, cls_loss: %.4f, loc_loss: %.4f' %
                        (phase, epoch_loss, epoch_cls_loss, epoch_loc_loss)
                    )
                elif phase == 'valid':
                    _, map_score = mAP(
                        all_label_trues, all_marker_trues,
                        all_label_preds, all_marker_preds)
                    history.update_hist(
                        val_loss=epoch_loss, val_cls_loss=epoch_cls_loss,
                        val_loc_loss=epoch_loc_loss, mAP=map_score
                    )
                    if map_score is np.nan:
                        map_score = 0
                    history.update_best(
                        mAP=map_score, loss=epoch_loss,
                        cls_loss=epoch_cls_loss, loc_loss=epoch_loc_loss,
                        epoch=e, model_wts=copy.deepcopy(net.state_dict())
                    )
                    print(
                        ('%s, loss: %.4f, cls_loss: %.4f,'
                         ' loc_loss: %.4f, mAP: %.4f') %
                        (
                            phase, epoch_loss, epoch_cls_loss,
                            epoch_loc_loss, map_score
                        )
                    )

    if 'valid' in dataloaders.keys():
        net.load_state_dict(history.best[0]['model_wts'])
        print(
            'valid best loss: %.4f, best mAP: %.4f' %
            (history.best[0]['loss'], history.best[0]['mAP'])
        )

    # 如果有test，则再进行test的预测
    if 'test' in dataloaders.keys():
        test_loss, test_map = test(
            net, dataloaders['test'], criterion=criterion, evaluate=True,
            predict=False
        )
        print('test loss: %.4f, test mAP: %.4f' % (test_loss[0], test_map[1]))
        return history, (test_loss, test_map)

    return history


def test(
    net, dataloader, criterion=None, evaluate=True, predict=True,
    device=torch.device('cuda:0')
):
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
    assert (evaluate and y_encoder_mode in ['object', 'all']) or not evaluate
    assert (criterion is not None and y_encoder_mode in ['all', 'anchor']) or \
        criterion is None

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
        for imgs, labels, markers in pb.progressbar(dataloader):
            imgs = imgs.to(device)
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
                # labels in cpu, cls_trues in gpu
                cls_trues = cls_trues.to(device)
                # markers是list，每个元素是一个tensor，指代一张图片中所有的obj
                #   的xyxy loc，[(img1#obj, 4), (img2#obj, 4), (img3#obj, 4),
                #   ...]
                # loc_trues是一个tensor，(#imgs, #anchors, 4)，是每张图片中每个
                #   anchor和其对应的gtbb的位置偏移量
                markers, loc_trues = markers
                # marker in cpu, loc_trues in gpu
                loc_trues = loc_trues.to(device)
            elif y_encoder_mode == 'anchor':
                # 如果只需要计算loss，则只需要输出编码好的偏移量和anchor的标签
                #   即可
                cls_trues = labels.to(device)
                loc_trues = markers.to(device)

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
                cls_preds, loc_preds, device=device)
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
        '-nj', '--n_jobs', default=6, type=int, help='多核并行的核数，默认是6')
    parser.add_argument(
        '-is', '--input_size', default=None, type=int, nargs='+',
        help=(
            '模型接受的输入图片的大小，如果指定了1个，则此为短边会resize到的大小'
            '，在此期间会保证图片的宽高比，如果指定两个，则分别是高和宽，'
            '默认是None，即不进行resize')
    )
    parser.add_argument(
        '-lr', '--learning_rate', type=float, default=5e-4,
        help='学习率，默认是0.0005')
    parser.add_argument(
        '-e', '--epoch', type=int, default=100, help='epoch数，默认是100')
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
        '--bn_freeze', action='store_true',
        help=(
            '是否固定RetinaNet中的BatchNorm中的参数不训练，'
            '如果使用此命令，则BatchNorm中的参数将不会训练'
        )
    )
    parser.add_argument(
        '--no_normalize', action='store_false',
        help='是否对数据进行标准化，如果使用该参数则不进行标准化'
    )
    parser.add_argument(
        '-ls', '--lr_schedular', default=None, type=int, nargs='+',
        help=(
            '使用multistep的lr schedule的参数，默认是None，'
            '如果是None则不进行lr schedule，如果不是None，则此为lr_schedular'
            '的表示在哪个epoch进行lr降低的参数'
        )
    )
    parser.add_argument(
        '-bb', '--backbone', default='resnet50',
        help='使用的backbone网络，默认是resnet50'
    )
    parser.add_argument(
        '--normal_init', action='store_true',
        help=(
            '是否使用normal分布来初始化新增加的layer，如果使用此参数，'
            '则进行normal初始化'
        )
    )
    parser.add_argument(
        '--alpha', default=0.25, type=float,
        help='focal loss的参数之一，默认是0.25'
    )
    parser.add_argument(
        '--no_bias_init', action='store_false',
        help='如果使用这个参数，则分类头最后一次的bias将不使用特殊的初始化模式'
    )
    parser.add_argument(
        '-cn', '--clip_norm', default=1.0, type=float,
        help='进行梯度截断的参数，默认是1.0，如果是0.0则不进行梯度截断'
    )
    parser.add_argument(
        '-ps', default=[3, 4, 5, 6, 7], type=int, nargs='+',
        help='使用FPN中的哪些特征图来构建anchors，默认是p3-p7'
    )
    parser.add_argument(
        '--best_num', default=3, type=int,
        help="保存最好的模型的数目，默认是3"
    )
    parser.add_argument(
        '--best_metric', default='mAP',
        help="判断模型好坏的指标，可以选择的指标有mAP和loss，默认是mAP"
    )
    parser.add_argument(
        '-o', '--optimizer', default='adam',
        help="使用的迭代器，默认是adam"
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
    # 读取数据根目录，构建data frame
    data_df, label_set = get_data_df(
        args.root_dir, check=False, check_labels=True)
    label_mapper = {l: i for i, l in enumerate(label_set)}
    print(label_mapper)
    # 数据集分割
    trainval_df, test_df = train_test_split(
        data_df, test_size=0.1, shuffle=True, random_state=args.random_seed)
    train_df, valid_df = train_test_split(
        trainval_df, test_size=1/9, shuffle=True, random_state=args.random_seed
    )

    # 数据集建立
    if args.no_normalize:
        img_transfer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        img_transfer = transforms.Compose([
            transforms.ToTensor(),
        ])
    img_transfer = transfers.OnlyImage(img_transfer)
    data_augment = transforms.Compose([
        transfers.RandomFlipLeftRight(),
        transfers.RandomFlipTopBottom(),
    ])
    # 注意对于PIL，其输入大小时是w,h的格式
    if input_size is None:
        train_transfers = transforms.Compose([
            data_augment, img_transfer
        ])
        test_transfers = img_transfer
    else:
        # 在PIL中，一般表示图像的两个空间维度时候使用w,h的顺序，但在pytorch中，
        # 一般使用h,w的顺序，这里使用的Resize是pytorch的对象，所以其接受的是h x
        # w的顺序
        resize_transfer = transfers.Resize(input_size)
        train_transfers = transforms.Compose([
            data_augment, resize_transfer, img_transfer
        ])
        test_transfers = transforms.Compose([
            resize_transfer, img_transfer
        ])
    # 这里实际上是送给YEncoder类的参数，需要指定为w x h
    if isinstance(input_size, list):
        y_input_size = input_size[::-1]
    else:
        y_input_size = input_size
    y_encoder_args = {'input_size': y_input_size, 'ps': args.ps}
    datasets = {
        'train': ColabeledDataset(
            train_df, transfer=train_transfers, y_encoder_mode='anchor',
            label_mapper=label_mapper, **y_encoder_args
        ),
        'valid': ColabeledDataset(
            valid_df, transfer=test_transfers, label_mapper=label_mapper,
            y_encoder_mode='all', **y_encoder_args
        ),
        'test': ColabeledDataset(
            test_df, transfer=test_transfers, y_encoder_mode='all',
            label_mapper=label_mapper, **y_encoder_args
        )
    }
    dataloaders = {
        k: data.DataLoader(
            v, batch_size=args.batch_size, shuffle=False,
            num_workers=args.n_jobs, collate_fn=v.collate_fn)
        for k, v in datasets.items()
    }

    # 模型建立
    bb_dict = {
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet34': models.resnet34,
    }
    backbone = bb_dict[args.backbone]
    net = RetinaNet(
        backbone=backbone, normal_init=args.normal_init,
        cls_bias_init=args.no_bias_init, ps=args.ps
    )
    net.cuda()
    if args.bn_freeze:
        net.freeze_bn()  # ???
    criterion = FocalLoss(alpha=args.alpha)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            net.parameters(), lr=args.learning_rate,
            momentum=0.9, weight_decay=1e-4
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    if args.lr_schedular is not None:
        lr_schedular = optim.lr_scheduler.MultiStepLR(
            optimizer, args.lr_scheduler, gamma=0.1
        )
    else:
        lr_schedular = None

    # 模型训练
    history, (test_loss, test_map) = train(
        net, criterion, optimizer, dataloaders, epoch=args.epoch,
        lr_schedular=lr_schedular, clip_norm=args.clip_norm,
        history=History(
            args.best_num, args.best_metric, args.best_metric == 'loss')
    )

    # 模型保存
    save_dir = os.path.join(args.save_root, args.save)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # state_dict = copy.deepcopy(net.state_dict())
    torch.save(history.best, os.path.join(save_dir, 'model.pth'))

    test_res = {
            k: v for k, v in zip(['loss', 'cls_loss', 'loc_loss'], test_loss)}
    test_res['APs'] = test_map[0]
    test_res['mAP'] = test_map[1]
    with open(os.path.join(save_dir, 'test.json'), 'w') as f:
        json.dump(test_res, f)

    train_df = pd.DataFrame(history.history)
    train_df.to_csv(os.path.join(save_dir, 'train.csv'))


if __name__ == "__main__":
    main()
