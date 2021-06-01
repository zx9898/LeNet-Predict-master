import torch
import cv2
import torch.nn.functional as F
from model import LeNet  ##重要，虽然显示灰色(即在次代码中没用到)，但若没有引入这个模型代码，加载模型时会找不到模型
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r'F:\\')#your filepath

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式
    filename = input("please input your pic:")  # "dog.jpg"
    img = cv2.imread(filename)  # 读取要预测的图片
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    dst = cv2.resize(img, (32, 32))
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    dst = trans(dst)
    dst = dst.to(device)
    dst = dst.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    # 扩展后，为[1，1，28，28]
    output = model(dst)
    prob = F.softmax(output, dim=1)  # prob是10个分类的概率
    print(prob)
    value, predicted = torch.max(output.data, 1)
    print(predicted.item())
    print(value)
    pred_class = classes[predicted.item()]
    print(pred_class)

    plt.figure()
    pltimg = plt.imread(filename)
    fig, ax = plt.subplots()
    ax.imshow(pltimg, extent=[1, 10, 2, 11])
    plt.text(5, 5, pred_class, fontsize=15)
    plt.show()

