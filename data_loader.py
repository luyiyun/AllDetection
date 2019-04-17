import os
import sys

import numpy as np
import pandas as pd
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from PIL.ImageDraw import Draw
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from data_encoder import YEncoder
from transfers import YTransfer


def xml_to_markers(file_path):
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
    return Image.open(img_f)


def draw_rectangle(img, labels, markers, color_mapper={0: 'green', 1: 'red'}):
    draw = Draw(img)
    for label, marker in zip(labels, markers):
        color = color_mapper[label]
        draw.rectangle(marker.tolist(), outline=color, width=5)
    return img


def _check(name, label_f, img_dir):
    img_set = set(name)
    label_set = set([f[:-4] for f in label_f])
    symmetric_difference = img_set.symmetric_difference(label_set)
    if len(symmetric_difference) > 0:
        raise ValueError(
            '%s的标记和图像文件间存在差异，%s' %
            (img_dir, str(symmetric_difference)))


def get_data_df(img_label_dir_pair, check=False):
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

    def __init__(
        self, df, img_loader=pil_loader, label_mapper={'正常': 0, '异常': 1},
        transfer=None
    ):
        self.img_loader = img_loader
        self.label_mapper = label_mapper
        self.transfer = transfer
        self.df = df

    def __getitem__(self, idx):
        img_f, label_f = self.df.iloc[idx, :].values
        img = self.img_loader(img_f)
        labels, markers = xml_to_markers(label_f)
        labels = torch.tensor([self.label_mapper[l] for l in labels])
        markers = torch.tensor(markers, dtype=torch.float)
        if self.transfer is not None:
            img, labels, markers = self.transfer(img, labels, markers)
        return img, labels, markers

    def __len__(self):
        return len(self.df)


def main():
        root_dir = 'E:/Python/AllDetection/label_boxes'
        img_label_dir_pair = []
        for d in os.listdir(root_dir):
            img_dir = os.path.join(root_dir, d)
            label_dir = os.path.join(root_dir, d, 'outputs')
            img_label_dir_pair.append((img_dir, label_dir))

        data_df = get_data_df(img_label_dir_pair, check=False)
        if len(sys.argv) == 1:
            dataset = AllDetectionDataset(data_df)
            for i in range(len(dataset)):
                img, labels, markers = dataset[i]
                img = draw_rectangle(img, labels.numpy(), markers.numpy())
                fig, ax = plt.subplots(figsize=(20, 10))
                ax.imshow(np.asarray(img))
                plt.show()
        else:
            y_transfer = YTransfer(YEncoder(), (224., 224.))
            dataset = AllDetectionDataset(data_df, transfer=y_transfer)
            img, labels, markers = dataset[1]
            print(labels.size())
            print(markers.size())
            print(markers[:10])


if __name__ == "__main__":
    main()
