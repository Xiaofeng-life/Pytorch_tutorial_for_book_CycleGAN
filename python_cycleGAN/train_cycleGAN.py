#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/29 16:05
# @Author  : CongXiaofeng
# @File    : train.py
# @Software: PyCharm

import itertools
import torch.optim as optim
from load_data import get_data_loader
import models
import torch.nn as nn
import torch
from loss_utils import LossWriter
from utils import make_project_dir, save_image
import os


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# 训练和验证超参数定义
BETA1 = 0.5
BETA2 = 0.999
EPOCHS = 100
IMAGE_SIZE = 256
LR = 0.0001
BATCH_SIZE = 1
RESULTS_DIR = "results"
IMG_SAVE_FREQ = 100
PTH_SAVE_FREQ = 2

# 生成器和判别器通道数基础值
NGF = 64
NDF = 64
# 生成器残差块数量
NUM_RES = 6
# 判别器卷积层数
NUM_LAYERS = 3

VAL_FREQ = 1
VAL_BATCH_SIZE = 1

device = torch.device('cuda')

# 2：准备两个图像域的训练和验证数据
dataloader_train_A = get_data_loader("data/apple2orange/train_A", BATCH_SIZE,
                                     (IMAGE_SIZE, IMAGE_SIZE))
dataloader_train_B = get_data_loader("data/apple2orange/train_B", BATCH_SIZE,
                                     (IMAGE_SIZE, IMAGE_SIZE))

dataloader_val_A = get_data_loader("data/apple2orange/val_A", VAL_BATCH_SIZE,
                                   (IMAGE_SIZE, IMAGE_SIZE))
dataloader_val_B = get_data_loader("data/apple2orange/val_B", VAL_BATCH_SIZE,
                                   (IMAGE_SIZE, IMAGE_SIZE))

# 创建标签数据，real_label代表正类标签，fake_label代表负类标签
# 这里的real_label和fake_label也可以使用标签平滑策略来构建
real_label = torch.ones(size=(BATCH_SIZE, 1, 32, 32), requires_grad=False).to(device)
fake_label = torch.zeros(size=(BATCH_SIZE, 1, 32, 32), requires_grad=False).to(device)


# 3: 定义生成器网络G(generator_x2y)，生成器网络F(generator_y2x)
# 定义判别器网络DX(discriminator_x)，判别器网络DY(discriminator_y)
generator_x2y = models.Generator(in_ch=3, out_ch=3, ngf=NGF, num_res=NUM_RES).to(device)
generator_y2x = models.Generator(in_ch=3, out_ch=3, ngf=NGF, num_res=NUM_RES).to(device)
discriminator_x = models.Discriminator(in_ch=3, ndf=NGF, n_layers=NUM_LAYERS).to(device)
discriminator_y = models.Discriminator(in_ch=3, ndf=NDF, n_layers=NUM_LAYERS).to(device)


# 4: 给生成器和判别器分别定义优化器，
optimizer_G = optim.Adam(itertools.chain(generator_x2y.parameters(), generator_y2x.parameters()),
                         lr=LR, betas=(BETA1, BETA2))
optimizer_D = optim.Adam(itertools.chain(discriminator_x.parameters(), discriminator_y.parameters()),
                         lr=LR, betas=(BETA1, BETA2))
# optimizer_D_y = optim.Adam(discriminator_y.parameters(),
#                            lr=cfg_cyc_train["lr"], betas=(cfg_cyc_train["beta1"], cfg_cyc_train["beta2"]))

# 循环一致损失cycle_loss可以使用L1损失或者MSE损失
# 对抗损失gan_loss采用MSE损失
cycle_loss = nn.L1Loss()
identity_loss = nn.L1Loss()
gan_loss = nn.MSELoss()

loss_writer = LossWriter(os.path.join(RESULTS_DIR, "loss"))
make_project_dir(RESULTS_DIR, RESULTS_DIR)


iteration = 0
for epo in range(EPOCHS):
    # 遍历两个图像域的dataloder中含有的图像
    for data_X, data_Y in zip(dataloader_train_A, dataloader_train_B):
        # 1：数据准备，分别取出图像文件
        image_x = data_X[0].to(device)
        image_y = data_Y[0].to(device)

        # #################################################
        # 第一部分：训练生成器
        optimizer_G.zero_grad()
        # 计算生成器G_X2Y伪造样本损失
        generated_x2y = generator_x2y(image_x)
        d_out_fake_x2y = discriminator_y(generated_x2y)
        g_loss_x2y = gan_loss(d_out_fake_x2y, real_label)
        identity_x = generator_y2x(image_x)
        identity_x_loss = identity_loss(identity_x, image_x) * 5

        # 计算生成器G_Y2X伪造样本损失
        generated_y2x = generator_y2x(image_y)
        d_out_fake_y2x = discriminator_x(generated_y2x)
        g_loss_y2x = gan_loss(d_out_fake_y2x, real_label)
        identity_y = generator_x2y(image_y)
        identity_y_loss = identity_loss(identity_y, image_y) * 5

        # ##########################################################
        # 计算两个图像域的循环一致损失
        cycle_x2y2x = generator_y2x(generated_x2y)
        g_loss_cycle_x2y2x = cycle_loss(cycle_x2y2x, image_x) * 10
        cycle_y2x2y = generator_x2y(generated_y2x)
        g_loss_cycle_y2x2y = cycle_loss(cycle_y2x2y, image_y) * 10

        # 计算生成器总体损失
        g_loss = g_loss_cycle_x2y2x + g_loss_cycle_y2x2y + g_loss_y2x + g_loss_x2y + identity_x_loss + identity_y_loss

        # 更新生成器
        set_requires_grad([discriminator_x, discriminator_y], False)
        g_loss.backward()
        optimizer_G.step()

        # #################################################
        set_requires_grad([discriminator_x, discriminator_y], True)
        optimizer_D.zero_grad()

        # 第二部分：训练判别器DX
        # 计算判别器DX对真实样本给出为真的loss
        d_out_real_x = discriminator_x(image_x)
        d_real_loss_x = gan_loss(d_out_real_x, real_label)

        # 计算判别器DX对伪造样本的损失
        d_out_fake_y2x_ = discriminator_x(generated_y2x.detach())
        d_fake_loss_y2x_ = gan_loss(d_out_fake_y2x_, fake_label)

        # 计算判别器DX总损失
        d_loss_x = (d_real_loss_x + d_fake_loss_y2x_) * 0.5
        d_loss_x.backward()
        # optimizer_D_x.step()

        # ##############################################
        # 第三部分：训练判别器 DY
        # 计算判别器DY对真实样本给出为真的loss
        d_out_real_y = discriminator_y(image_y)
        d_real_loss_y = gan_loss(d_out_real_y, real_label)

        # 计算判别器DY对伪造样本的损失
        d_out_fake_x2y_ = discriminator_y(generated_x2y.detach())
        d_fake_loss_x2y_ = gan_loss(d_out_fake_x2y_, fake_label)

        # 计算判别器DY总损失
        d_loss_y = (d_real_loss_y + d_fake_loss_x2y_) * 0.5
        d_loss_y.backward()

        # 更新判别器参数
        optimizer_D.step()

        print("iter: {}, G loss: {:.4f}, D loss X: {:.4f}, "
              "D loss Y: {:.4f}".format(iteration, g_loss.item(),
                                        d_loss_x.item(), d_loss_y.item()))
        loss_writer.add("G loss", g_loss.item(), iteration)
        loss_writer.add("D loss X", d_loss_x.item(), iteration)
        loss_writer.add("D loss Y", d_loss_y.item(), iteration)

        # 4：自加
        iteration += 1

        # #################################################
        # 5：打印损失，保存图片
        if iteration % IMG_SAVE_FREQ == 0:
                result = torch.cat((image_x, generated_x2y), dim=3)
                save_image(result[0].squeeze(),
                           os.path.join(RESULTS_DIR, "train_images", str(iteration) + "_A2B.png"))
                result = torch.cat((image_y, generated_y2x), dim=3)
                save_image(result[0].squeeze(),
                           os.path.join(RESULTS_DIR, "train_images", str(iteration) + "_B2A.png"))

    if epo % PTH_SAVE_FREQ == 0:
        torch.save(generator_x2y.state_dict(),
                   os.path.join(RESULTS_DIR, "pth", str(epo) + "_x2y.pth"))
        torch.save(generator_y2x.state_dict(),
                   os.path.join(RESULTS_DIR, "pth", str(epo) + "_y2x.pth"))

    if epo %VAL_FREQ == 0:
        # 将生成器generator_x2y和generator_y2x均切换到eval模式，并关闭梯度计算

        generator_x2y.eval()
        generator_y2x.eval()
        with torch.no_grad():
            for i, (data_X, data_Y) in enumerate(zip(dataloader_val_A, dataloader_val_B)):
                image_x = data_X[0].to(device)
                image_y = data_Y[0].to(device)

                # 首先将图像域X中的image_x转换到图像域Y中的image_x2y
                # 然后将image_x2y重构为图像域X中的image_x2y2x
                image_x2y = generator_x2y(image_x)
                image_x2y2x = generator_y2x(image_x2y)
                result = torch.cat((image_x, image_x2y, image_x2y2x), dim=3)
                save_image(result[0].squeeze(),
                           os.path.join(RESULTS_DIR,
                                        "val_images", str(i) + "_A2B2A.png"))

                # 首先将图像域Y中的image_y转换到图像域Y中的image_y2x
                # 然后将image_y2x重构为图像域Y中的image_y2x2y
                image_y2x = generator_y2x(image_y)
                image_y2x2y = generator_x2y(image_y2x)
                result = torch.cat((image_y, image_y2x, image_y2x2y), dim=3)
                save_image(result[0].squeeze(),
                           os.path.join(RESULTS_DIR,
                                        "val_images", str(i) + "_B2A2B.png"))

        generator_x2y.train()
        generator_y2x.train()
