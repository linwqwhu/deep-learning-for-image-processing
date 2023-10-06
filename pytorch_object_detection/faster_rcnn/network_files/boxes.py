import torch
from typing import Tuple
from torch import Tensor
import torchvision


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """

    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    Args:
        boxes (Tensor[N, 4]): 即将执行NMS的boxes, [...,[x1, y1, x2, y2],...]
        scores (Tensor[N]): 每个box的得分
        iou_threshold (float): 阈值，丢弃IoU > iou_threshold的所有重叠框

    Returns:
        keep (Tensor): int64 tensor，具有NMS保存的元素的索引，按分数递减顺序排序

    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """
    以批处理方式执行非最大值抑制

    每个索引值对应于一个类别，NMS不会应用于不同类别的元素之间。

    Args:
        boxes (Tensor[N, 4]): 即将执行NMS的boxes, [...,[x1, y1, x2, y2],...]
        scores (Tensor[N]): 每个box的得分
        idxs (Tensor[N]): 每个box的类别索引
        iou_threshold (float): 抛弃阈值，丢弃IoU < iou_threshold的所有重叠框

    Returns:
        keep (Tensor): int64 tensor，具有NMS保存的元素的索引，按分数递减顺序排序

    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # 策略：为了能够按类别独立地执行NMS。给所有的box加一个偏移量。偏移量仅取决于类别idx，并且足够大，因此来自不同类别的框不会重叠
    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()

    # to(): Performs Tensor dtype and/or device conversion
    # 为每一个类别/每一层生成一个很大的偏移量
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes加上对应层的偏移量后，保证不同类别/层之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes, min_size):
    # type: (Tensor, float) -> Tensor
    """
    移除宽或高小于指定阈值min_size的索引

    Remove boxes which contains at least one side smaller than min_size.

    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        min_size (float): minimum size

    Returns:
        keep (Tensor[K]): 宽高都大于min_size的框的索引
            indices of the boxes that have both sides larger than min_size
    """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]  # pred_boxes的宽和高
    # keep = (ws >= min_size) & (hs >= min_size)  # 当满足宽，高都大于给定阈值时为True
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    # nonzero(): Returns a tensor containing the indices of all non-zero elements of input
    # keep = keep.nonzero().squeeze(1)
    keep = torch.where(keep)[0]
    return keep


def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int]) -> Tensor
    """
    裁剪预测的boxes信息，将越界的坐标调整到图片边界上

    Clip boxes so that they lie inside an image of size `size`.

    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2   “...”(ellipsis)操作符，表示其他维度不变
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width)  # 限制x坐标范围在[0,width]之间
        boxes_y = boxes_y.clamp(min=0, max=height)  # 限制y坐标范围在[0,height]之间

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def box_area(boxes):
    """
    计算一组boxes的面积，boxes = [...,[x1, y1, x2, y2],...]

    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): 将为其计算面积的boxes

    Returns:
        area (Tensor[N]): 每个box的面积，(x2-x1)*(y2-y1)
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    返回boxes1与boxes2的IOU

    两组boxes格式均应为(x1, y1, x2, y2)

    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): 包含boxes1和boxes2中每个元素的成对IoU值的NxM矩阵
            the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    # 当形状不匹配时，返回的输出张量的形状遵循广播规则
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]，两张图片的左上角坐标中的最大值
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]，两张图片的右下角坐标中的最小值

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]，重叠面积

    # WHY shape不对
    # area1原本形状为[N,1]，area1[:,None]变为[N,1(新增的维度),1]
    # 再与形状为[M,1]的area2相加，由广播机制，[N,M,1] + [M,1] -> [N,M,M]
    # [N,M,M] + [N,M]-> [N,M,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
