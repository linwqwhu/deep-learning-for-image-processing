import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('./img/airplane.png')
    im.convert("RGB")  # png是四通道，除了RGB之外还有一个透明度通道，因此这里 选择保留颜色通道
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]  # 增加一个新的纬度 与 image = torch.reshape(image, (1, 3, 32, 32)) 一样

    with torch.no_grad():
        im = im.to("cpu")  # 这里不知道为什么不用处理成 "cuda"
        outputs = net(im)

        # predict = torch.max(outputs, dim=1)[1].numpy()

        predict = torch.softmax(outputs, dim=1)
        print(predict)
        c = predict.argmax(1)

    # print(classes[int(predict)])
    print(classes[int(c)])


if __name__ == '__main__':
    main()
