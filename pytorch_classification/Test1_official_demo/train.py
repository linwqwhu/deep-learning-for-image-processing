import numpy as np
import torch
import torchvision
import torch.nn as nn
import matplotlib
import torch.optim as optim
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from model import LeNet
from torch.utils.data import DataLoader

matplotlib.use('TkAgg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=36,
                              shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    batch_size = 10000
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # 转换成可迭代的迭代器，get some random training images
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)
    val_image.to(device)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #
    # # 查看训练图片
    # def imshow(img):
    #     """
    #     functions to show an image
    #     :param img:
    #     :return: None
    #     """
    #     img = img / 2 + 0.5  # unnormalize，这里是由于前面对图像进行了标准化，需要把图片还原回来
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # # print labels
    # batch_size = 4  # 太大显示不了
    # print(' '.join(f'{classes[val_label[j]]:5s}' for j in range(batch_size)))
    # # show images
    # imshow(torchvision.utils.make_grid(val_image))

    net = LeNet()
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)


    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 上面三行可以使用这一句代替
            # inputs, labels = data[0].to(device), data[1].to(device)


            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # 每500次就验证一下
            # if step % 500 == 499:  # print every 500 mini-batches
            with torch.no_grad():
                val_image = val_image.to(device)
                val_label = val_label.to(device)
                outputs = net(val_image)  # [batch, 10]
                # predict_y = torch.max(outputs, dim=1)[1]
                predict_y = outputs.argmax(1)
                accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()


# 测试结果
# cuda
# Files already downloaded and verified
# Files already downloaded and verified
# [1,   500] train_loss: 1.777  test_accuracy: 0.444
# [1,  1000] train_loss: 1.415  test_accuracy: 0.509
# [2,   500] train_loss: 1.224  test_accuracy: 0.559
# [2,  1000] train_loss: 1.160  test_accuracy: 0.585
# [3,   500] train_loss: 1.034  test_accuracy: 0.612
# [3,  1000] train_loss: 1.045  test_accuracy: 0.628
# [4,   500] train_loss: 0.939  test_accuracy: 0.639
# [4,  1000] train_loss: 0.935  test_accuracy: 0.653
# [5,   500] train_loss: 0.853  test_accuracy: 0.656
# [5,  1000] train_loss: 0.874  test_accuracy: 0.655
# Finished Training
