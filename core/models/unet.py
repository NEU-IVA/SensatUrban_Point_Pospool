import math
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from core.modules.unet_block import DoubleConv, Down, CompressConv, Up, OutConv


class UNet_alt(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_alt, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        pretrained_model1 = resnet.__dict__['resnet{}'.format(34)](pretrained=True)
        pretrained_model2 = resnet.__dict__['resnet{}'.format(18)](pretrained=True)
        # rgb pipeline
        self.inc_rgb = DoubleConv(3, 64)
        self.resnet1_rgb = pretrained_model1._modules['layer1']  # 64 as input
        self.resnet2_rgb = pretrained_model1._modules['layer2']  # 128 as output
        # self.down1_rgb = Down(64, 128)
        self.down2_rgb = Down(128, 256)
        self.down3_rgb = Down(256, 512)
        # thermal pipeline
        self.inc_t = DoubleConv(1, 64)
        self.resnet1_t = pretrained_model2._modules['layer1']  # 64 as input
        self.resnet2_t = pretrained_model2._modules['layer2']  # 128 as output
        # self.down1 = Down(64, 128)
        self.down2_t = Down(128, 128)  # 128/256
        self.down3_t = Down(128, 128)  # 256/512
        # fusion seg model
        self.down4 = Down(512, 512)
        self.compress1 = CompressConv(128, 64)
        self.compress2 = CompressConv(256, 128)
        self.compress3 = CompressConv(384, 128)  # (512, 256)
        self.compress4 = CompressConv(640, 512)  # (1024, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(384, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # split two branches: rgb, t

        # rgb feature extraction
        x_rgb = x[:, 0:3, :, :]
        x1_rgb = self.inc_rgb(x_rgb)
        # rgb backbone
        x1_rgb = self.resnet1_rgb(x1_rgb)
        x2_rgb = self.resnet2_rgb(x1_rgb)
        # x2_rgb = self.down1_rgb(x1_rgb)
        x3_rgb = self.down2_rgb(x2_rgb)
        x4_rgb = self.down3_rgb(x3_rgb)

        # thermal feature extraction
        x_t = x[:, 3, :, :].unsqueeze(1)
        x1_t = self.inc_t(x_t)
        # thermal backbone
        x1_t = self.resnet1_t(x1_t)
        x2_t = self.resnet2_t(x1_t)
        # x2_t = self.down1_t(x1_t)
        x3_t = self.down2_t(x2_t)
        x4_t = self.down3_t(x3_t)

        # fusion by concat
        x1 = torch.cat((x1_rgb, x1_t), 1)
        x2 = torch.cat((x2_rgb, x2_t), 1)
        x3 = torch.cat((x3_rgb, x3_t), 1)
        x4 = torch.cat((x4_rgb, x4_t), 1)
        x1 = self.compress1(x1)
        x2 = self.compress2(x2)
        x3 = self.compress3(x3)
        x4 = self.compress4(x4)
        # post-fusion processing
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_eca(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_eca, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        pretrained_model1 = resnet.__dict__['resnet{}'.format(34)](pretrained=True)
        pretrained_model2 = resnet.__dict__['resnet{}'.format(18)](pretrained=True)

        # rgb pipeline
        self.inc_rgb = DoubleConv(3, 64)
        self.resnet1_rgb = pretrained_model1._modules['layer1']  # 64 as input
        self.resnet2_rgb = pretrained_model1._modules['layer2']  # 128 as output
        # self.down1_rgb = Down(64, 128)
        self.down2_rgb = Down(128, 256)
        self.down3_rgb = Down(256, 512)
        # thermal pipeline
        self.inc_t = DoubleConv(1, 64)
        self.resnet1_t = pretrained_model2._modules['layer1']  # 64 as input
        self.resnet2_t = pretrained_model2._modules['layer2']  # 128 as output
        # self.down1 = Down(64, 128)
        self.down2_t = Down(128, 128)  # 128/256
        self.down3_t = Down(128, 128)  # 256/512
        # fusion seg model
        self.down4 = Down(512, 512)
        self.compress1 = CompressConv(128, 64)
        self.compress2 = CompressConv(256, 128)
        self.compress3 = CompressConv(384, 128)  # (512, 256)
        self.compress4 = CompressConv(640, 512)  # (1024, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(384, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def eca_layer(self, x, gamma=2, b=1):
        # eca-net
        # ???????????????GPA??????????????????????????????1*1*C?????????????????????1???conv??????????????????
        N, C, H, W = x.size()
        t = int(abs((math.log(C, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        avg_pool_eca = nn.AdaptiveAvgPool2d(1)
        conv1d_eca = nn.Conv1d(1, 1, kernel_size=k_size, padding=int(k_size // 2), bias=False)
        sigmoid_eca = nn.Sigmoid()
        y = avg_pool_eca(x)
        y = y.cpu()
        y = conv1d_eca(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = sigmoid_eca(y)
        y = y.cuda()
        return x * y.expand_as(x)

    def forward(self, x):
        # split two branches: rgb, t

        # rgb feature extraction
        x_rgb = x[:, 0:3, :, :]
        x1_rgb = self.inc_rgb(x_rgb)
        # rgb backbone
        x1_rgb = self.resnet1_rgb(x1_rgb)
        x2_rgb = self.resnet2_rgb(x1_rgb)
        # x2_rgb = self.down1_rgb(x1_rgb)
        x3_rgb = self.down2_rgb(x2_rgb)
        x4_rgb = self.down3_rgb(x3_rgb)

        # thermal feature extraction
        x_t = x[:, 3, :, :].unsqueeze(1)
        x1_t = self.inc_t(x_t)
        # thermal backbone
        x1_t = self.resnet1_t(x1_t)
        x2_t = self.resnet2_t(x1_t)
        # x2_t = self.down1_t(x1_t)
        x3_t = self.down2_t(x2_t)
        x4_t = self.down3_t(x3_t)

        # fusion by concat
        x1 = torch.cat((x1_rgb, x1_t), 1)
        x2 = torch.cat((x2_rgb, x2_t), 1)
        x3 = torch.cat((x3_rgb, x3_t), 1)  # 512 -> 384
        x4 = torch.cat((x4_rgb, x4_t), 1)  # 1024 -> 640
        x1 = self.eca_layer(x1)
        x2 = self.eca_layer(x2)
        x3 = self.eca_layer(x3)
        x4 = self.eca_layer(x4)
        x1 = self.compress1(x1)
        x2 = self.compress2(x2)
        x3 = self.compress3(x3)
        x4 = self.compress4(x4)
        # post-fusion processing
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    model = UNet_eca(4, 13).cuda()
    model1 = UNet_alt(4, 13).cuda()
    a = torch.ones((1, 4, 400, 400)).cuda()
    y = model(a)
    y1 = model1(a)
    print(y)
    print(y1)
