from typing import List, Tuple
from torch import Tensor


class ImageList(object):
    """
    将图像列表（可能大小不同）作为单个tensor的结构。

    通过将图像填充到相同的大小，并将每个图像的原始大小存储在字段中来实现的

    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        """
        Arguments:
            tensors (tensor) :padding后的图像数据
            image_sizes (list[tuple[int, int]]) : padding前的图像尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)
