#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time    : 2019/10/30 15:51 
# @Author  : CongXiaofeng 
# @File    : show.py 
# @Software: PyCharm


import matplotlib.pyplot as plt
import imageio
import os
images_list = os.listdir("plot/")
print(images_list)
plt.figure()

for i in range(16):
    image = imageio.imread("plot/"+images_list[i])
    plt.subplot(4, 4, i+1)
    plt.axis("off")
    plt.imshow(image/255)

plt.savefig("zhihu/images.jpg")
plt.show()