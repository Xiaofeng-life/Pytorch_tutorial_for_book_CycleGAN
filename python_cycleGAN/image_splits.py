#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time    : 2019/11/5 14:01 
# @Author  : CongXiaofeng 
# @File    : image_splits.py 
# @Software: PyCharm


from PIL import Image
import os
import numpy as np


def split(image_dir, target_dir_left, target_dir_right):
    """
    将拼接的图像分为左右两个图，放入两个文件夹中
    :param image_dir: 原始拼接图的存放地址
    :param target_dir_left: 左边图像存放地址
    :param target_dir_right: 右边图片的存放地址
    :return: 运行完成返回True
    """
    image_files = os.listdir(image_dir)
    for i, image in enumerate(image_files):
        im = Image.open(image_dir + image)
        im = np.array(im)
        # 获取左右子图
        h, w = im.shape[0], im.shape[1]
        im_left = Image.fromarray(im[:, 0:int(w / 2), :])
        im_right = Image.fromarray(im[:, int(w / 2):, :])

        im_left.save(target_dir_left + str(i) + ".jpg")
        im_right.save(target_dir_right + str(i) + ".jpg")

    print(" save done!!!")
    return True


image_dir = "test_edge/val/"
target_dir_left = "test_edge/testA/"
target_dir_right = "test_edge/testB/"

split(image_dir, target_dir_left, target_dir_right)
