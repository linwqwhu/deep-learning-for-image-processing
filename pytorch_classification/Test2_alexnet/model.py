import torch.nn as nn
import torch


class AlexNet(nn.Module):
    """
    这是一个根据AlexNet网络架构实现的一个模型
    有修改
    """

    def __init__(self, num_classes=1000, init_weights=False):
        """
        初始化
        :param num_classes: 类别总数
        :param init_weights: 是否初始化权重
        """
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(

            # padding=i 说明在图片的左右各填充两列零、上下各填充i行零
            # padding=(i,j) 说明在图片的左右各填充i列零、上下各填充j行零
            # 若想实现左右或上下填充不同的列（或行）需要使用 nn.ZeroPad2d() 方法
            # 如，nn.ZeroPad2d(i, j, k, l) 说明在图片的左边填充i列零、右边填充j列零，上方填充k行零、下方填充l行零

            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]   ----(N, C, H, W) 这里忽略了N

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        初始化权重
        :return: None
        """
        for m in self.modules():    # 迭代每一个定义的网络神经结构
            if isinstance(m, nn.Conv2d):    # 判断 m 是不是 nn.Conv2d 这个结构
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
