#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time    : 2019/10/20 10:43 
# @Author  : CongXiaofeng 
# @File    : load_data.py 
# @Software: PyCharm


import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils import data
import matplotlib.pyplot as plt


def get_data_loader(basic_dir, batch_size, image_size, shuffle=True):
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.RandomVerticalFlip())
    transform.append(T.ToTensor())
    # 归一化操作
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    dataset = ImageFolder(basic_dir, transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  pin_memory=True, shuffle=shuffle,
                                  num_workers=0, drop_last=True)
    return data_loader


if __name__ == "__main__":
    batch_size = 3
    data_loader = get_data_loader(basic_dir="data/apple2orange/train_A", batch_size=batch_size,
                                  image_size=(256, 256), shuffle=True)
    data_loader = iter(data_loader)
    image, _ = next(data_loader)
    # 将数据先转为numpy，再转置
    # pytorch默认为（batch_size, channel, H, W）
    # 需要转为 （batch_size, H, W, channel）
    image = image.numpy().transpose(0, 2, 3, 1)
    plt.figure(figsize=(10, 4))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        plt.axis("off")
        plt.imshow((image[i, :, :, :]+1)/2)
    plt.show()
