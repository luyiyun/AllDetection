import numpy as np


def boxes_info(locs):
    ws = []
    hs = []
    ss = []
    for l in locs:
        w = l[2] - l[0]
        h = l[3] - l[1]
        ws.append(w)
        hs.append(h)
        ss.append(w * h)
    return np.mean(ws), np.mean(hs), np.mean(ss)


def main():
    import argparse
    import pandas as pd
    import progressbar as pb

    from train import default_rd
    from data_loader import get_data_df, ColabeledDataset

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root_dir', default=default_rd(),
        nargs='?', help='输入所在的路径'
    )
    # parser.add_argument(
    #     '-is', '--image_size', default=(1200, 1920), nargs=2, type=int,
    #     help='输入图片的大小，默认是1200、1920，需要输入2个，是H x W'
    # )
    parser.add_argument(
        '-c', '--check', action='store_true',
        help="是否进行检查，如果使用此参数则进行检查，看xml文件和img文件是否一致"
    )
    # parser.add_argument(
    #     '-m', '--mode', default='img', choices=['img', 'encode'],
    #     help="进行哪些的展示，默认是img，即将图片画出并把gtbb画上去"
    # )
    # parser.add_argument(
    #     '-it', '--image_type', default='jpg', help='图片后缀，默认是jpg')
    parser.add_argument(
        '-dno', '--drop_nonobjects', action='store_true',
        help='如果使用此参数，则不会包括没有标记的图像'
    )
    parser.add_argument(
        '--wh_min', default=None, type=int,
        help="默认是None，用于xml读取，过滤错误的框"
    )
    args = parser.parse_args()

    # 我们并不知道objects一共有多少类，所以这里需要使用check_labels来遍历一般xml
    #   得到所有的objects类别组成的set
    data_df, labels_set = get_data_df(
        args.root_dir, check=args.check, check_labels=True,
        drop_nonobjects=args.drop_nonobjects
    )
    print(labels_set)
    xml_parse = {}
    if args.wh_min is not None:
        xml_parse['wh_min'] = args.wh_min
    dataset = ColabeledDataset(
        data_df, label_mapper=None, transfer=None, y_encoder_mode='object',
        input_size=(2048, 2048), xml_parse=xml_parse
    )
    print(len(dataset))
    ws = []
    hs = []
    img_id = []
    obj_label = []
    for i in pb.progressbar(range(len(dataset))):
        img, labels, markers = dataset[i]
        if len(labels) > 0:
            markers = np.array(markers)
            w, h = (markers[:, 2:] - markers[:, :2]).T.tolist()
            ws.extend(w)
            hs.extend(h)
            img_id.extend([img.filename] * len(w))
            obj_label.extend(labels)
        else:
            ws.append(None)
            hs.append(None)
            img_id.append(img.filename)
            obj_label.append(None)
    df = pd.DataFrame(dict(ws=ws, hs=hs, img_id=img_id, label=obj_label))
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT


if __name__ == "__main__":
    main()
