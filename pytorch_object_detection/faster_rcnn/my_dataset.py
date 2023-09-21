from typing import Any

import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOCDataSet(Dataset):
    """
    读取解析PASCAL VOC2007/2012数据集
    self.root: 数据集路径 VOCdevkit/VOC2012
    self.img_root: 图片路径 VOCdevkit/VOC2012/JPEGImages
    self.annotations_root: 注解路径 VOCdevkit/VOC2012/Annotations
    self.xml_list: train.txt 或 val.txt 中含目标的文件列表 [...,2007_000027.xml,...]
    self.class_dict: 类别及其对应的索引值 {...,"dog":12,...}
    self.transforms: 图片转换方法
    """

    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 增加容错能力
        if "VOCdevkit" in voc_root:
            self.root = os.path.join(voc_root, f"VOC{year}")
        else:
            self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")

        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            # strip()返回删除字符串前导和尾随空格的字符串副本，这里使用strip()方法去掉换行符
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in read.readlines() if len(line.strip()) > 0]

        self.xml_list = []
        # check file
        for xml_path in xml_list:
            # 检查对应的文件是否存在
            if os.path.exists(xml_path) is False:
                print(f"Warning: not found '{xml_path}', skip this annotation file.")
                continue

            # check for targets
            # 读取文件
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)  # 将xml文件转化为Element对象

            data = self.parse_xml_to_dict(xml)["annotation"]  # 将xml文件解析成字典形式后，读取annotation
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue

            # 文件存在且有对象时写入
            self.xml_list.append(xml_path)

        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)

        # read class_indict
        json_file = './pascal_voc_classes.json'  # 一个存放物体类别及其对应序号的文件 {"类别":序号,...}
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx) -> (Any, {}):
        """
        根据图片索引值获取 image, target
        Args:
            idx: 索引值

        Returns:
            image: 图片信息
            target:图片标签
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)  # 将xml文件转化为Element对象

        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":  # VOC数据集全部都是jpg格式
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []  # 标注框，x,y为标注框的左上角坐标和右下角坐标,一张图片上的所有标注框
        labels = []  # 存储的是索引值(0-20)，不是具体的类（如dog）

        # COCO数据集中是用来决定是RLE格式还是polygon格式 是否与其它目标重叠
        # 在这里简单的表示为是否能检测，0：单目标，好检测，与VOC数据集中的difficult一样
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)

        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])  # 添加标注框物体类别索引值
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        # 转化成tensor格式
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])  # 当前数据对应的索引值
        # boxes = [...,[xmin,ymin,xmax,ymax],[],...]
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # 标注区域面积

        target = {}
        target["boxes"] = boxes  # 标注框信息
        target["labels"] = labels  # 标注框物体类别对应序号值
        target["image_id"] = image_id  # 当前图片在xml_list中对应的索引值，数量为图片个数
        target["area"] = area  # 标注框面积
        target["iscrowd"] = iscrowd  # 标注物体是否难识别或是否与其它目标重叠，0为易识别

        # 对图片进行转换
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx) -> (int, int):
        """
        获取图片的高度和宽度(height, width)
        Args:
            idx: 图片索引值

        Returns:

        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)

        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: 使用lxml.etree.fromstring()方法将xml文件转化为的Element对象

        Returns:
            保存XML内容的Python字典。
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        """
        将dataset返回的图像信息image和target合并在一起
        通过非关键字参数将 输入数据 输入到zip函数中进行打包，再将打包信息转成元组形式
        Args:
            batch:

        Returns:

        """
        return tuple(zip(*batch))

# import transforms
# from draw_box_utils import draw_objs
# from PIL import Image
# import json
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random
#
# # read class_indict
# category_index = {}
# try:
#     json_file = open('./pascal_voc_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {str(v): str(k) for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# # train_data_set = VOCDataSet(os.getcwd(), "2012", data_transform["train"], "train.txt")
# train_data_set = VOCDataSet("E:/VOCdevkit", "2012", data_transform["train"], "train.txt")
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     plot_img = draw_objs(img,
#                          target["boxes"].numpy(),
#                          target["labels"].numpy(),
#                          np.ones(target["labels"].shape[0]),
#                          category_index=category_index,
#                          box_thresh=0.5,
#                          line_thickness=3,
#                          font='arial.ttf',
#                          font_size=20)
#     plt.imshow(plot_img)
#     plt.show()
