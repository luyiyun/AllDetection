import os

import pandas as pd
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from data_encoder import YEncoder


def xml_to_markers(
    file_path, label_correct={'ASC_H': 'ASC-H', 'ASC_US': 'ASC-US'}
):
    '''
    从xml文件中读取标签和位置信息，这一个xml文件是对一张图片的标记信息；
    args:
        file_path，xml文件的路径；
        label_correct，有些标签标错了，比如下划线和-不分，这里进行纠正；
    returns:
        labels，list，每个元素是这一张图片所有的objs的标签，字符串；
        postitions，list，每个元素是一个4-list，为xyxy格式，int；
    '''
    postition_tag = ['xmin', 'ymin', 'xmax', 'ymax']
    et = ET.parse(file_path)
    root = et.getroot()
    labels = []
    postitions = []
    for element in root.iter('object'):
        label = element.findtext('name')
        bndbox = element.find('bndbox')
        position = [int(bndbox.findtext(tag)) for tag in postition_tag]
        if label in label_correct.keys():
            label = label_correct[label]
        labels.append(label)
        postitions.append(position)
    return labels, postitions


def pil_loader(img_f):
    '''
    使用PIL来读取图像
    args:
        img_f，图像文件的路径；
    returns:
        Image，返回的是PIL的Image对象；
    '''
    return Image.open(img_f)


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
    img_files = []
    xml_files = []
    labels_set = set()
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
                    if drop_nonobjects or check_labels:
                        labels, markers = xml_to_markers(xml_f)
                        # 是否收集所有的类别
                        if check_labels:
                            labels_set = labels_set.union(set(labels))
                        # 是否要把没有标记的样本丢弃
                        if drop_nonobjects:
                            if len(labels) > 0:
                                img_files.append(os.path.join(d, f))
                                xml_files.append(xml_f)
                        else:
                            img_files.append(os.path.join(d, f))
                            xml_files.append(xml_f)
                    else:
                        img_files.append(os.path.join(d, f))
                        xml_files.append(xml_f)
    if check_labels:
        return pd.DataFrame({'img': img_files, 'label': xml_files}), labels_set
    return pd.DataFrame({'img': img_files, 'label': xml_files})


class ColabeledDataset(Dataset):
    '''
    建立用于RetinaNet训练的数据集
    '''
    def __init__(
        self, df, img_loader=pil_loader, label_mapper={'正常': 0, '异常': 1},
        transfer=None, y_encoder_mode='object', **kwargs
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
            kwargs：传入到YEncoder中的参数
        '''
        assert y_encoder_mode in ['all', 'object', 'anchor']
        self.img_loader = img_loader
        self.label_mapper = label_mapper
        self.transfer = transfer
        self.df = df
        self.y_encoder_mode = y_encoder_mode
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
        labels, markers = xml_to_markers(label_f)
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
