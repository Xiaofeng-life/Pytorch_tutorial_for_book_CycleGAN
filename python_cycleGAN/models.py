import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf, num_res=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_ch, ngf, 7, 1, 0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(inplace=True)]

        # Downsampling
        for _ in range(2):
            model += [nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * 2),
                      nn.ReLU(inplace=True)]
            ngf = ngf * 2

        # Residual blocks
        for _ in range(num_res):
            model += [ResidualBlock(ngf)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.Upsample(scale_factor=2),
                      nn.Conv2d(ngf, ngf // 2, 3, 1, 1),
                      # nn.ConvTranspose2d(ngf, ngf // 2, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(ngf // 2),
                      nn.ReLU(inplace=True)]
            ngf = ngf // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, out_ch, 7, 1, 0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()

        # 定义首层卷积
        model = [nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]
        # 定义中间层卷积
        for i in range(1, n_layers):
            model += [nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1),
                      nn.BatchNorm2d(ndf * 2),
                      nn.LeakyReLU(0.2, inplace=True)]
            ndf = ndf * 2

        # model += [nn.Conv2d(ndf, ndf * 8, 4, stride=1, padding=1),
        #           nn.BatchNorm2d(ndf * 8),
        #           nn.LeakyReLU(0.2, inplace=True)]

        # 定义最后一层卷积，输出通道数为1
        model += [nn.Conv2d(ndf, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    import torch
    x = torch.rand(size=(1, 3, 256, 256))
    # dis = Discriminator(3)
    # pred = dis(x)
    # print(dis)
    gen = Generator(3, 3, 64)
    out = gen(x)
    print(out.size())