import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from .roi_head import RoIHeads
from .transform import GeneralizedRCNNTransform
from .rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork


class FasterRCNNBase(nn.Module):
    """
    Generalized R-CNN的主要部分

    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        """

        Args:
            backbone: 骨干网络
            rpn: 区域建议生成网络
            roi_heads: ROI pooling + Two MLPHead + FastRcNNPredictor + Postprocess Detections
            transform: 执行从输入到模型的数据转换
        """
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): 需要处理的图像，这里输入的images的大小都是不同的，
                后面会进行预处理，将图片放入同样大小的tensor中打包成一个batch
            targets (list[Dict[Tensor]]): 图像中存在的GT (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): 模型输出.
                在训练过程中，它会返回一个dict[Tensor]，其中包含损失losses。
                在测试过程中，它返回列表[BoxList]，该列表包含其他字段，如“scores”、“labels”和“mask”（用于Mask R-CNN模型）
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:  # 进一步判断传入的target的boxes参数是否符合规定
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(boxes.shape)
                                         )  # N代表一张图像当中有多个目标，而一个目标有4个值x_min,y_min,x_max,y_max
                else:
                    raise ValueError("Expected target boxes to be of type Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]  # shape: [channel, height, width]
            assert len(val) == 2  # 防止输入的是个一维向量
            original_image_sizes.append((val[0], val[1]))
        # original_image_sizes = [img.shape[-2:] for img in images]

        images, targets = self.transform(images, targets)  # 对图像进行预处理，对应GeneralizedRCNNTransform
        # 两步操作：Normalize 和 Resize（限定输入图像的最小边长和最大边长，没有缩放到统一大小）

        # print(images.tensors.shape)
        features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图
        if isinstance(features, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
            features = OrderedDict([('0', features)])  # 若在多层特征层上预测，传入的就是一个有序字典

        # 将特征层以及标注target信息传入rpn中
        # proposals: List[Tensor], Tensor_shape: [num_proposals, 4],
        # 每个proposals是绝对坐标，且为(x1, y1, x2, y2)格式
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

        # if self.training:
        #     return losses
        #
        # return detections


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        # 输入x为roi_pool层的输出

        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Fast R-CNN的标准分类+边界框回归层

    Standard classification + bounding box regression layers for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)  # 针对每一个proposal对N+1个类的预测
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)  # [1024(两张图片，一张图片就有512个proposal),1024],实际这里的展平处理不起作用
        scores = self.cls_score(x)  # [1024,21]
        bbox_deltas = self.bbox_pred(x)  # [1024,84]

        return scores, bbox_deltas


class FasterRCNN(FasterRCNNBase):
    """
    实现 Faster R-CNN.

    模型的输入期待为是list[Tensor]，每个Tensor的形状为[C, H, W]，每个图像对应一个张量，并且应该在0-1范围内。不同的图像可以具有不同的大小。

    模型的行为会发生变化，这取决于它是处于训练模式还是评估模式。

    在训练过程中，模型期望输入格式为tensors和targets为list of dictionary，其中包含：
        - boxes (FloatTensor[N, 4]): [x1, y1, x2, y2]格式的GT，y1、y2值在0和H之间，x1、x2值在0和W之间
        - labels (Int64Tensor[N]): 每个GT的类标签

    该模型在训练期间返回一个Dict[Tensor]，包含RPN和R-CNN的分类及回归损失。

    在推理过程中，模型只需要输入张量，并将处理后的预测返回一个List[Dict[Tensor]]，每张输入图像对应一个Dict[Tensor]。Dict的字段如下：
        - boxes (FloatTensor[N, 4]): [x1, y1, x2, y2]格式的预测框，y1、y1值在0和H之间，x1、x2值在0和W之间
        - labels (Int64Tensor[N]): 每个图像的预测标签
        - scores (Tensor[N]): 每个预测的分数

    Arguments:
        backbone (nn.Module): 用于计算模型特征的网络。
            它应该包含out_channels属性，该属性指示每个特征图具有的输出通道的数量（并且对于所有特征图都应该相同）。
            backbone应返回单个Tensor或OrderedDict[Tensor]。
        num_classes (int): 模型的输出类的数量（包括背景）。如果指定了box_predictor，则num_classes应为None。
        min_size (int): 将图像馈送到backbone之前要重新缩放的图像的最小尺寸
            minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): 将图像馈送到backbone之前要重新缩放的图像的最大尺寸
            maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): 用于输入归一化的平均值。
            通常是主干在其上进行训练的数据集的平均值
        image_std (Tuple[float, float, float]): 用于输入规范化的标准差值。
            通常是对主干进行训练的数据集的标准值
        rpn_anchor_generator (AnchorGenerator): 用于生成一组特征图的anchor的模型
        rpn_head (nn.Module): 根据RPN计算objectness和回归增量regression deltas的模型
        rpn_pre_nms_top_n_train (int): 在训练期间应用NMS之前要保留的建议数量
        rpn_pre_nms_top_n_test (int): 在测试期间应用NMS之前要保留的建议数量
        rpn_post_nms_top_n_train (int): 在训练期间应用NMS后要保留的建议数量
        rpn_post_nms_top_n_test (int): 在测试期间应用NMS后要保留的建议数量
        rpn_nms_thresh (float): 处理RPN proposal的NMS阈值
        rpn_fg_iou_thresh (float): 正样本，anchor和GT box之间的最小IoU
        rpn_bg_iou_thresh (float): 负样本，anchor和GT box之间的最大IoU
        rpn_batch_size_per_image (int): 在RPN训练期间为计算损失而选取的anchor的数量
        rpn_positive_fraction (float): RPN训练期间一个batch正样本anchor的比例
        rpn_score_thresh (float): 在推理过程中，只返回分类分数大于rpn_score_thresh的建议
        box_roi_pool (MultiScaleRoIAlign): 在bbox指示的位置裁剪和调整特征图大小的模块
        box_head (nn.Module): 将裁剪的特征图作为输入的模块
        box_predictor (nn.Module): 获取box_head的输出并返回classification logits和box regression deltas的模块
        box_score_thresh (float): 在推理过程中，只返回分类得分大于box_score_thresh的proposal
        box_nms_thresh (float): the prediction head的NMS阈值。推理过程中使用
        box_detections_per_img (int): 所有类别的每个图像的最大检测次数
        box_fg_iou_thresh (float): the classification head训练期间，正样本， proposal和GT box之间的最小IoU
        box_bg_iou_thresh (float): the classification head训练期间，负样本， proposal和GT box之间的最大IoU
        box_batch_size_per_image (int): the classification head训练期间抽样的proposal数量
        box_positive_fraction (float): the classification head训练期间一个mini-batch中正样本比例
        bbox_reg_weights (Tuple[float, float, float, float]): 用于bbox的编码/解码的权重
    """

    def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=800, max_size=1333,  # 预处理resize时限制的最小尺寸与最大尺寸
                 image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,  # anchor生成器；根据RPN计算objectness和回归增量regression deltas的模块

                 # 这里NMS前后相同主要是针对带有FPN的网络。FPN有多个预测特征层，
                 # 每层在NMS前都保留2000个，总共加起来就上万了，然后再通过NMS保留2000个
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,  # rpn中在nms处理前保留的proposal数(根据score)
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
                 rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn计算损失时，采集正负样本设置的阈值
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时采样的样本数，以及正样本占总样本的比例
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,  # 依次是ROI pooling，Two MLPHead， FastRCNNPredictor
                 # 移除低目标概率；fast rcnn中进行nms处理的阈值；对预测结果根据score排序取前100个目标
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,  # fast rcnn计算误差时，采集正负样本设置的阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
                 bbox_reg_weights=None  # bbox回归权重
                 ):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels)"
            )

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        # 预测特征层的channels
        out_channels = backbone.out_channels

        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))  # anchor面积尺寸，单元素的元组类型后面必须加逗号（,），不然会被识别成int类型
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        # 生成RPN通过滑动窗口预测网络部分
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)
        # rpn_batch_size_per_image RPN在计算损失时采用正负样本的总个数
        # rpn_positive_fraction 正样本占用于计算损失所有样本的比例
        # rpn_pre_nms_top_n NMS处理之前，针对每个预测特征层所保留的目标个数
        # rpn_post_nms_top_n NMS处理之后，针对每个预测特征层剩余的目标个数
        # rpn_nms_thresh NMS处理时的阈值

        #  Multi-scale RoIAlign pooling
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                output_size=[7, 7],
                sampling_ratio=2)

        # fast RCNN中roi pooling后的展平处理两个全连接层部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认等于7
            representation_size = 1024  # 全连接层Fc1的节点个数
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        # 在box_head的输出上预测部分
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)  # num_classes = 20 + 1

        # 将roi pooling, box_head以及box_predictor结合在一起
        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)
