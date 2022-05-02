import models
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF

if __name__ == "__main__":
    device = torch.device("cuda")

    # 创建并加载两个生成器
    generator_x2y = models.Generator(in_ch=3, out_ch=3, ngf=64, num_res=6).to(device)
    generator_y2x = models.Generator(in_ch=3, out_ch=3, ngf=64, num_res=6).to(device)
    generator_x2y.load_state_dict(torch.load("results/pth/60_x2y.pth"))
    generator_y2x.load_state_dict(torch.load("results/pth/60_y2x.pth"))

    # 切换掉评估模式
    generator_x2y.eval()
    generator_y2x.eval()

    # 输入的苹果和橘子图像
    image_y_path = "data/apple2orange/train_A/1/n07740461_1164.jpg"
    image_x_path = "data/apple2orange/train_B/1/n07749192_183.jpg"

    with torch.no_grad():
        image_x = Image.open(image_x_path)
        image_y = Image.open(image_y_path)

        # 执行图像域X的转换
        image_x = image_x.resize((256, 256))
        image_x = TF.to_tensor(image_x)
        image_x_ori = image_x.clone()
        image_x = TF.normalize(image_x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image_x = image_x.unsqueeze(0).to(device)
        x2y = generator_x2y(image_x).squeeze()
        x2y = (x2y + 1) / 2

        # 执行图像域Y的转换
        image_y = image_y.resize((256, 256))
        image_y = TF.to_tensor(image_y)
        image_y_ori = image_y.clone()
        image_y = TF.normalize(image_y, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image_y = image_y.unsqueeze(0).to(device)
        y2x = generator_y2x(image_y).squeeze()
        y2x = (y2x + 1) / 2

        # 获取转换后的图像，并执行拼接操作
        result_a = torch.cat((image_x_ori.cpu(), x2y.cpu()), dim=2) * 255
        result_b = torch.cat((image_y_ori.cpu(), y2x.cpu()), dim=2) * 255
        result_a = result_a.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        result_b = result_b.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        result = np.concatenate((result_a, result_b), axis=0)

        plt.imshow(result)
        plt.savefig("demo.jpg", dpi=500, bbox_inches="tight")
        plt.show()