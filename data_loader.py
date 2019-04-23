import os
import sys
import platform

import numpy as np
import pandas as pd
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from PIL.ImageDraw import Draw
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt

from data_encoder import YEncoder


def xml_to_markers(file_path):
    '''
    从xml文件中读取标签和位置信息，这一个xml文件是对一张图片的标记信息；
    args:
        file_path，xml文件的路径；
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


def _check(name, label_f, img_dir):
    '''
    对标记文件的集合和图像文件的集合进行检查，看是否能够完全对上，如果对不上，
        则返回错误；
    args:
        name，图像的文件名列表；
        label_f，xml文件的文件名列表；
        img_dir，所在的文件夹路径；
    returns:
        None
    '''
    img_set = set(name)
    label_set = set([f[:-4] for f in label_f])
    symmetric_difference = img_set.symmetric_difference(label_set)
    if len(symmetric_difference) > 0:
        raise ValueError(
            '%s的标记和图像文件间存在差异，%s' %
            (img_dir, str(symmetric_difference)))


def get_data_df(img_label_dir_pair, check=False):
    '''
    得到一个DataFrame，储存有图片和标签文件的对应信息；
    args:
        img_label_dir_pair，其是一个list，每个元素是一个2-tuple，是两个文件夹路
            径，前一个是储存有图片的文件夹路径，后一个是储存有相应标签的文件夹
            路径，可能有多个这样的文件夹pair；
        check，是否对文件夹pair间的文件名进行检查，看标签和图像间是否对的上；
    returns:
        df
    '''
    sample_files = []
    for img_dir, label_dir in img_label_dir_pair:
        names = []
        for f in os.listdir(img_dir):
            if f.endswith('jpg'):
                name = f[:-4]
                names.append(name)
                img_f = os.path.join(img_dir, f)
                label_f = os.path.join(label_dir, name+'.xml')
                sample_files.append((img_f, label_f))
        if check:
            _check(names, list(os.listdir(label_dir)), img_dir)
    return pd.DataFrame(sample_files, columns=['img', 'label'])


class AllDetectionDataset(Dataset):
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
        labels = torch.tensor([self.label_mapper[l] for l in labels])
        markers = torch.tensor(markers, dtype=torch.float)
        if self.transfer is not None:
            img, labels, markers = self.transfer(img, labels, markers)
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
            return default_collate(batch)
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


'''
以下是关于本模块测试的函数
'''


def draw_rectangle(img, labels, markers, color_mapper={0: 'green', 1: 'red'}):
    '''
    在img上画上标记框，如果labels是概率，则会计算其最大的概率，根据最大的列标
        来表颜色，并把最大的概率写在上面；
    args:
        img，PIL的Image对象；
        labels，iterable，是objects的标签；
        markers，iterable，是objects的boxes的loc；
        color_mapper，dict，keys是markers，values是用于draw的颜色；
    returns:
        img，PIL的Image对象，画上标记框的image；
    '''
    draw = Draw(img)
    # font = ImageFont.truetype(font="C:/Windows/Fonts/Calibar", size=5)
    for label, marker in zip(labels, markers):
        if labels.ndim == 1:
            color = color_mapper[label]
        else:
            color = color_mapper[label.argmax()]
            text_position = (marker[[0, 2]].mean(), marker[[1, 3]].mean())
            draw.text(text_position, str(label.max()), fill=color)
        draw.rectangle(marker.tolist(), outline=color, width=5)
    return img


def main():
    if platform.system() == 'Windows':
        root_dir = 'E:/Python/AllDetection/label_boxes'
    else:
        root_dir = '/home/dl/deeplearning_img/AllDet/label_boxes'
    img_label_dir_pair = []
    for d in os.listdir(root_dir):
        img_dir = os.path.join(root_dir, d)
        label_dir = os.path.join(root_dir, d, 'outputs')
        img_label_dir_pair.append((img_dir, label_dir))

    data_df = get_data_df(img_label_dir_pair, check=True)
    if len(sys.argv) == 1:
        dataset = AllDetectionDataset(data_df, input_size=(1200, 1920))
        for i in range(len(dataset)):
            img, labels, markers = dataset[i]
            print(markers)
            img = draw_rectangle(img, labels.numpy(), markers.numpy())
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(np.asarray(img))
            plt.show()
    else:
        dataset = AllDetectionDataset(
            data_df, y_encoder_mode='all', input_size=(1200, 1920))
        img, labels, markers = dataset[1]
        print(labels)
        print(markers)


if __name__ == "__main__":
    main()
