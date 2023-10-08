from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict


class IntermediateLayerGetter(nn.ModuleDict):
    """
    返回模型的中间层的模块包装器

    使用前提：模块已按照与 使用的 相同顺序注册到模型中。
    这意味着，不应该在forward中重复使用同一个nn.Module两次。
    而且，它只能查询直接分配给模型的子模块。因此，如果传递了“model”，
    则可以返回“model.feature1”，但不能返回“model.feature1.layer2”。

    Arguments:
        model (nn.Module): 提取特征的模型
        return_layers (Dict[name, new_name]): key=将返回激活的模块的名称，value=返回激活的名称（用户可以指定）。
    """
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构，舍去之后不用的结构
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer1, layer2, layer3, layer4的输出
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FeaturePyramidNetwork(nn.Module):
    """
    从一组特征图的顶部添加FPN的模块。

    这是基于“用于对象检测的特征金字塔网络”<https://arxiv.org/abs/1612.03144>`_。
    特征图在输入时应按深度递增的顺序排列好
    输入类型为OrderedDict[Tensor]，包含将在其上添加FPN的特征图。

    Arguments:
        in_channels_list (list[int]): 传递到模块的每个特征图的通道数
        out_channels (int): FPN呈现的通道数
        extra_blocks (ExtraFPNBlock or None): 如果提供，将执行额外的操作。
            将fpn特征、原始特征 和 原始特征的名称 作为输入，并返回一个新的特征图列表及其对应的名称
    """

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()
        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        这相当于self.inner_blocks[idx](x)，但torchscript还不支持这一点

        Args:
            x:
            idx:

        Returns:

        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        这相当于self.layer_blocks[idx](x)，但torchscript还不支持这一点

        Args:
            x:
            idx:

        Returns:

        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        计算一组特征图的FPN

        Arguments:
            x (OrderedDict[Tensor]): 每个特征层的特征图

        Returns:
            results (OrderedDict[Tensor]): FPN层后的特征图，从最高分辨率开始排序的（降序）。
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        # 将resnet layer4的channel调整到指定的out_channels
        # last_inner = self.inner_blocks[-1](x[-1])
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        # result中保存着每个预测特征层
        results = []
        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        # results.append(self.layer_blocks[-1](last_inner))
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(torch.nn.Module):
    """
    在最后一个特征层的顶部应用max_pool2d

    Applies a max_pool2d on top of the last feature map
    """

    def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  # input, kernel_size, stride, padding
        return x, names


class BackboneWithFPN(nn.Module):
    """
    在模型顶部添加FPN

    使用torchvision.models._utils.IntermediateLayerGetter来提取一个子模型，该子模型返回return_layers中指定的特征图。

    IntermediatLayerGetter的限制也适用于此。

    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): key=将返回激活的模块的名称，value=返回激活的名称（用户可以指定）
        in_channels_list (List[int]): 返回的每个特征图的通道数，按它们在OrderedDict中的出现顺序
        out_channels (int): FPN中的通道数
        extra_blocks: ExtraFPNBlock
    Attributes:
        out_channels (int): FPN中的通道数
    """

    def __init__(self,
                 backbone: nn.Module,
                 return_layers=None,
                 in_channels_list=None,
                 out_channels=256,
                 extra_blocks=None,
                 re_getter=True):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if re_getter is True:
            assert return_layers is not None
            # 这个是官方提供的，只能获取module.children()，并不能获取module.children().children()或更下的
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            self.body = backbone

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )

        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
