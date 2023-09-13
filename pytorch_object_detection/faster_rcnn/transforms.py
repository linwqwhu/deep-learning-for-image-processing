import random
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:  # 使用random.random()随机生成一个概率，小于prob才翻转
            height, width = image.shape[-2:]  # [N, C, H, W]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]  # [ [box1],[box2], [box3]]
            # bbox: [ [xmin, ymin, xmax, ymax],...]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            # 翻转时y值不变，变的是x值
            # x'_min = W - x_max
            # x'_max = W - x_min

            target["boxes"] = bbox
        return image, target
