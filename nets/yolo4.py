# coding: utf-8
import torch
import torch.nn as nn
from collections import OrderedDict
from nets.CSPdarknet import darknet53

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1))
    ]))

def convx3(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1)
    )

    return m

def convx5(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1)
    )

    return m

def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1)
    )

    return m

class SPP(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SPP, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=pool_sizes[0], stride=1, padding=pool_sizes[0]//2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=pool_sizes[1], stride=1, padding=pool_sizes[1]//2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=pool_sizes[2], stride=1, padding=pool_sizes[2]//2)

    def forward(self, x):
        features1 = self.maxpool1(x)
        features2 = self.maxpool2(x)
        features3 = self.maxpool3(x)
        y = torch.cat([features1, features2, features3, x], dim=1)

        return y

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.conv = conv2d(in_channels, out_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)

        return x

class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        final_out_filter = num_anchors * (5 + num_classes)
        # backbone
        self.backbone = darknet53(None)

        self.conv1 = convx3([512, 1024], 1024)
        self.spp = SPP()
        self.conv2 = convx3([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_p4 = conv2d(512, 256, 1)
        self.convx5_p41 = convx5([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_p3 = conv2d(256, 128, 1)
        self.convx5_p3 = convx5([128, 256], 256)
        self.yolo_head3 = yolo_head([256, final_out_filter], 128)

        self.down3 = conv2d(128, 256, 3, 2)
        self.convx5_p42 = convx5([256, 512], 512)
        self.yolo_head4 = yolo_head([512, final_out_filter], 256)

        self.down4 = conv2d(256, 512, 3, 2)
        self.convx5_p5 = convx5([512, 1024], 1024)
        self.yolo_head5 = yolo_head([1024, final_out_filter], 512)

    def forward(self, x):
        # backbone
        x2, x1, x0 = self.backbone(x)

        p5 = self.conv1(x0)
        p5 = self.spp(p5)
        p5 = self.conv2(p5)

        p5_up = self.upsample1(p5)
        p4 = self.conv_p4(x1)
        p4 = torch.cat([p4, p5_up], dim=1)
        p4 = self.convx5_p41(p4)

        p4_up = self.upsample2(p4)
        p3 = self.conv_p3(x2)
        p3 = torch.cat([p3, p4_up], dim=1)
        p3 = self.convx5_p3(p3)

        p3_down = self.down3(p3)
        p4 = torch.cat([p3_down, p4], dim=1)
        p4 = self.convx5_p42(p4)

        p4_down = self.down4(p4)
        p5 = torch.cat([p4_down, p5], dim=1)
        p5 = self.convx5_p5(p5)

        out2 = self.yolo_head3(p3)
        out1 = self.yolo_head4(p4)
        out0 = self.yolo_head5(p5)

        return out0, out1, out2