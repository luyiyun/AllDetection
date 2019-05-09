import os

import numpy as np
import pandas as pd
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from data_encoder import YEncoder


'''
读取xml的数据
'''


def parse_pascal(root):
    ''' 用于解析pascal格式的xml '''
    position_tag = ['xmin', 'ymin', 'xmax', 'ymax']
    # 看图片是否存在一个object element，如果不存在则返回空的列表，表示该图片
    #   中没有对象（即没有进行标记）
    obj1 = root.find('object')
    if obj1 is not None:
        labels = []
        positions = []
        for element in root.iter('object'):
            label = element.findtext('name')
            bndbox = element.find('bndbox')
            position = [int(bndbox.findtext(tag)) for tag in position_tag]
            labels.append(label)
            positions.append(position)
        return labels, positions
    return [], []


def parse_xml(root):
    ''' 用于解析xml格式的xml '''
    position_tag = ['xmin', 'ymin', 'xmax', 'ymax']
    labeled = root.findtext('labeled')
    if labeled == 'true':
        labels = []
        positions = []
        for element in root.iter('item'):
            label = element.findtext('name')
            bndbox = element.find('bndbox')
            position = [int(bndbox.findtext(tag)) for tag in position_tag]
            labels.append(label)
            positions.append(position)
        return labels, positions
    return [], []


def parse_size(root):
    ''' 得到xml文件中的图片大小信息 '''
    element = root.find('size')
    w = int(element.findtext('width'))
    h = int(element.findtext('height'))
    return w, h


def xml_to_markers(
    file_path, label_correct={'ASC_H': 'ASC-H', 'ASC_US': 'ASC-US'},
    size=None, wh_min=None
):
    '''
    从xml文件中读取标签和位置信息，这一个xml文件是对一张图片的标记信息；
    args:
        file_path，xml文件的路径；
        label_correct，有些标签标错了，比如下划线和-不分，这里进行纠正；
        size，有些xml中会存在超出图像边界的标记框，这里提供的size可以用于裁剪，
            如果是None，则需要从xml文件中读取；
        wh_min，用于将低于此的标记框过滤掉，这些是人失误画的；
    returns:
        labels，list，每个元素是这一张图片所有的objs的标签，字符串；
        postitions，list，每个元素是一个4-list，为xyxy格式，int；
    '''
    et = ET.parse(file_path)
    root = et.getroot()
    # 根据root node的tag来判断是哪种类型的xml，并进行读取
    if root.tag == 'doc':
        labels, positions = parse_xml(root)
    elif root.tag == 'annotation':
        labels, positions = parse_pascal(root)
    # 如果没有object，则直接返回空列表
    if len(labels) == 0:
        return labels, positions
    # 根据图片的size对框的大小进行截断，防止出现超出图像边缘的情况出现
    if size is None:
        w, h = parse_size(root)
    else:
        w, h = size
    positions = np.array(positions)
    positions[:, :2] = np.maximum(positions[:, :2], 0)
    positions[:, 2] = np.minimum(positions[:, 2], w)
    positions[:, 3] = np.minimum(positions[:, 3], h)
    # 根据给定的wh_min，将w或h小于此的框删除
    if wh_min is not None:
        wh = positions[:, 2:] - positions[:, :2]
        remain_ = (wh > wh_min).all(axis=1)
        labels = np.array(labels)[remain_]
        positions = positions[remain_]
        labels = labels.tolist()
    positions = positions.tolist()
    # 如果有label correct，则进行更正
    if label_correct is not None:
        new_labels = []
        for l in labels:
            if l in label_correct.keys():
                new_labels.append(label_correct[l])
            else:
                new_labels.append(l)
        labels = new_labels
    return labels, positions


'''
得到每张图片的路径及其对应的xml文件的路径
'''


def _check(img_fs, xml_fs, img_dir):
    '''
    对标记文件的集合和图像文件的集合进行检查，看是否能够完全对上，如果对不上，
        则返回错误；
    args:
        img_fs，图像的文件名列表；
        xml_fs，xml文件的文件名列表；
        img_dir，所在的文件夹路径；
    returns:
        None
    '''
    img_set = set(img_fs)
    label_set = set(xml_fs)
    symmetric_difference = img_set.symmetric_difference(label_set)
    if len(symmetric_difference) > 0:
        raise ValueError(
            '%s的标记和图像文件间存在差异，%s' %
            (img_dir, str(symmetric_difference)))


class LabelChecker:
    '''
    用于检查这个img的label并进行收集
    '''
    def __init__(self, collect=False, drop_nonobjects=True):
        '''
        args:
            collect: 是否收集所有的label；
            drop_nonobjects: 是否将没有object的img去掉；
        '''
        self.collect = collect
        self.drop_nonobjects = drop_nonobjects
        self.imgfs = []
        self.xmlfs = []
        self.labels_set = set()

    def check(self, img, xml):
        '''
        args:
            img, img的路径；
            xml, xml的路径;
        '''
        labels, markers = xml_to_markers(xml)
        if len(labels) > 0:
            # 是否收集所有的类别
            if self.collect:
                self.labels_set = self.labels_set.union(set(labels))
            self.imgfs.append(img)
            self.xmlfs.append(xml)
        elif not self.drop_nonobjects:
            self.imgfs.append(img)
            self.xmlfs.append(xml)


def get_data_df(
    root_dir, check=False, drop_nonobjects=True, img_type=['jpg', 'png'],
    check_labels=False,
):
    '''
    得到一个DataFrame，储存有图片和标签文件的对应信息；
    args:
        root_dir, 储存文件的总目录；
        check，是否对文件夹pair间的文件名进行检查，看标签和图像间是否对的上；
        drop_nonobjects，是否把没有objects的样本丢弃；
        img_type，使用的图片的格式；
        check_labels，如果是True，则会遍历一遍labels看看有哪些类；
    returns:
        set，所有类别组成的set；
        df，2-columns的df；
    '''
    label_checker = LabelChecker(
        collect=check_labels, drop_nonobjects=drop_nonobjects)
    # 遍历root_dir下的所有子文件下的文件，如果该子文件夹存在outputs文件夹，
    #   则认为此文件夹内正是保存图片的文件夹，将其中的jpg文件路径保存到list
    #   另外把该jpg文件对应的xml文件路径进行保存
    for d, folders, files in os.walk(root_dir):
        if 'outputs' in folders:
            xml_dir = os.path.join(d, 'outputs')
            # 当前路径下的jpg文件名和outputs文件夹中的xml文件名是否一致，
            #   如果不一致则返回error
            if check:
                jpg_file_names = [
                    f[:-4] for f in files if f.endswith(tuple(img_type))
                ]
                xml_file_names = [
                    f[:-4] for f in os.listdir(xml_dir)
                    if f.endswith('xml')
                ]
                _check(jpg_file_names, xml_file_names, d)
            for f in files:
                if f.endswith(tuple(img_type)):
                    xml_f = os.path.join(d, 'outputs', f[:-4]+'.xml')
                    img_f = os.path.join(d, f)
                    label_checker.check(img_f, xml_f)
    df = pd.DataFrame(
        {'img': label_checker.imgfs, 'label': label_checker.xmlfs})
    if check_labels:
        return df, label_checker.labels_set
    return df


def pil_loader(img_f):
    '''
    使用PIL来读取图像
    args:
        img_f，图像文件的路径；
    returns:
        Image，返回的是PIL的Image对象；
    '''
    return Image.open(img_f)


class ColabeledDataset(Dataset):
    '''
    建立用于RetinaNet训练的数据集
    '''
    def __init__(
        self, df, img_loader=pil_loader, label_mapper={'正常': 0, '异常': 1},
        transfer=None, y_encoder_mode='object', xml_parse={}, **kwargs
    ):
        '''
        args:
            df：使用的图片文件路径和xml文件路径组成的2-columns的df；
            img_loader：读取图片文件的loader，方式；
            label_mapper：xml中的obj的标签如何映射成数字；
            transfer：用于图片和boxes进行augment、resize等的操作，注意，这里
                为了能够计算mAP，所有是不包括将obj的坐标和标签变成anchor上的
                偏移量和标签的部分的，这个部分作为此dataset的内置功能；
            y_encoder_mode：如果是'anchor'，则输出anchor偏移量和标签，如果是
                'object'，则输出obj的xyxy boxes和标签，如果是'all'，则两者都会输
                出，用于计算mAP；
            xml_parse：dict，用于传递给xml_to_markers函数的参数；
            kwargs：传入到YEncoder中的参数
        '''
        assert y_encoder_mode in ['all', 'object', 'anchor']
        self.img_loader = img_loader
        self.label_mapper = label_mapper
        self.transfer = transfer
        self.df = df
        self.y_encoder_mode = y_encoder_mode
        self.xml_parse = xml_parse
        # 这里隐式的创建YEncoder来对obj进行关于anchor的编码
        self.y_encoder = YEncoder(**kwargs)

    def __getitem__(self, idx):
        '''
        dataset标准方法，
        returns:
            img，返回的图像对象，如果没有transfer，则返回的就是PIL的Image对象；
            labels, cls_targets，这里视y_encoder_mode的值来返回不同的结果，如果
                是anchor，则只返回cls_targts（即编码后的在anchors水平上的标签和
                位置偏移量），如果是object则返回labels（原始的obj的标签和labels
                ），只是把标签给数值化，如果是all则返回两者；
            markers, loc_targets，同上；
        '''
        img_f, label_f = self.df.iloc[idx, :].values
        img = self.img_loader(img_f)
        labels, markers = xml_to_markers(label_f, **self.xml_parse)
        if self.label_mapper is not None:
            labels = torch.tensor([self.label_mapper[l] for l in labels])
            markers = torch.tensor(markers, dtype=torch.float)
        if self.transfer is not None:
            img, labels, markers = self.transfer([img, labels, markers])
        if self.y_encoder_mode == 'all':
            cls_targets, loc_targets = self.y_encoder.encode(labels, markers)
            return img, (labels, cls_targets), (markers, loc_targets)
        elif self.y_encoder_mode == 'anchor':
            cls_targets, loc_targets = self.y_encoder.encode(labels, markers)
            return img, cls_targets, loc_targets
        return img, labels, markers

    def __len__(self):
        '''
        图像的数量
        '''
        return len(self.df)

    def collate_fn(self, batch):
        '''
        因为在不同y_encode_mode下后返回不同类型的结果，比如tensor和list，为了
            能够在合并成batch的时候区别对待，这里编写一个collate_fn用于dataloader
            的collate_fn参数，用于怎么把单个sample合并成batch；
        args:
            batch，多个__getitem__返回的结果组成的list；
        returns:
            img_batch，多个images组成的tensor，(#imgs, #channels, #height,
                #width)；
            label_batch, cls_target_batch，这里视y_encoder_mode的值来返回不同的
                结果，如果是anchor，则只返回cls_targts_batch（即编码后的在
                anchors水平上的标签和位置偏移量），如果是object则返回label_batch
                （原始的obj的标签和labels），只是把标签给数值化，如果是all则两者
                都返回，其中label_batch还是list，cls_target_batch是tensors；
            marker_batch, loc_target_batch，同上；
        '''
        if self.y_encoder_mode == 'anchor':
            return tuple(default_collate(batch))
        elif self.y_encoder_mode == 'object':
            img_batch = [b[0] for b in batch]
            label_batch = [b[1] for b in batch]
            marker_batch = [b[2] for b in batch]
            return default_collate(img_batch), label_batch, marker_batch
        else:
            img_batch = [b[0] for b in batch]
            label_batch = [b[1][0] for b in batch]
            cls_target_batch = [b[1][1] for b in batch]
            marker_batch = [b[2][0] for b in batch]
            loc_target_batch = [b[2][1] for b in batch]
            return (
                default_collate(img_batch),
                (label_batch, default_collate(cls_target_batch)),
                (marker_batch, default_collate(loc_target_batch))
            )


def main():
    import argparse

    from visual import draw_rectangle

    color_list = [
        'red', 'blue', 'yellow', 'pink'
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root_dir', default='E:/Python/AllDetection/label_boxes',
        nargs='?', help='保存数据的根目录，默认是E:/Python/AllDetection/label_boxes'
    )
    parser.add_argument(
        '-is', '--image_size', default=(1200, 1920), nargs=2, type=int,
        help='输入图片的大小，默认是1200、1920，需要输入2个，是H x W'
    )
    parser.add_argument(
        '-c', '--check', action='store_true',
        help="是否进行检查，如果使用此参数则进行检查，看xml文件和img文件是否一致"
    )
    parser.add_argument(
        '-m', '--mode', default='img', choices=['img', 'encode'],
        help="进行哪些的展示，默认是img，即将图片画出并把gtbb画上去"
    )
    parser.add_argument(
        '-it', '--image_type', default='jpg', help='图片后缀，默认是jpg')
    parser.add_argument(
        '-fs', '--figsize', default=(20, 10), nargs='+', type=int,
        help='用plt画图的figsize，默认是20、10'
    )
    parser.add_argument(
        '-dno', '--drop_nonobjects', action='store_true',
        help='如果使用此参数，则不会包括没有标记的图像'
    )
    args = parser.parse_args()

    # 我们并不知道objects一共有多少类，所以这里需要使用check_labels来遍历一般xml
    #   得到所有的objects类别组成的set
    data_df, labels_set = get_data_df(
        args.root_dir, check=args.check, img_type=args.image_type,
        check_labels=True, drop_nonobjects=args.drop_nonobjects
    )
    print(labels_set)
    if args.mode == 'img':
        color_mapper = {l: color_list[i] for i, l in enumerate(labels_set)}
        dataset = ColabeledDataset(
            data_df, label_mapper=None, transfer=None, y_encoder_mode='object',
            input_size=args.image_size
        )
        print(len(dataset))
        for i in range(len(dataset)):
            img, labels, markers = dataset[i]
            print(markers)
            img = draw_rectangle(img, labels, markers, color_mapper, labels)
            img.show()
            action = input('n is next, q is quit:')
            if action == 'q':
                break
    elif args.mode == 'encode':
        label_mapper = {l: i for i, l in enumerate(labels_set)}
        dataset = ColabeledDataset(
            data_df, y_encoder_mode='all', label_mapper=label_mapper,
            transfer=None, input_size=args.image_size,
        )
        img, labels, markers = dataset[1]
        print(labels)
        print(markers)


if __name__ == "__main__":
    main()
