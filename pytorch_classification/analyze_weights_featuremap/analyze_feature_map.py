import torch
from alexnet_model import AlexNet
from resnet_model import resnet34
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
matplotlib.use("TkAgg")

# data_transform = transforms.Compose(
#     [transforms.Resize((224, 224)),
#      transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model
# model = AlexNet(num_classes=5)
model = resnet34(num_classes=5)
# load model weights
# model_weight_path = "./AlexNet.pth"  # "./resNet34.pth"
# model_weight_path = "../Test2_alexnet/AlexNet.pth"  # "./resNet34.pth"
model_weight_path = "../Test5_resnet/resNet34.pth"

model.load_state_dict(torch.load(model_weight_path))
print(model)

# load image
img = Image.open("../img_1.png")
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        # [H, W, C]
        plt.imshow(im[:, :, i], cmap='gray')
    plt.show()

