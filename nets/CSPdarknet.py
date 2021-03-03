# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * torch.tanh(F.softplus(x))

        return x

# 卷积块
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

# CSPdarknet结构块组成部分
class Resblock(nn.Module):
    def __init__(self, inplanes, outplanes=None):
        super(Resblock, self).__init__()
        if outplanes is None:
            outplanes = inplanes

        self.block = nn.Sequential(
            BasicConv(inplanes, outplanes, 1),
            BasicConv(outplanes, inplanes, 3)
        )

    def forward(self, x):
        return x + self.block(x)

# CSPdarknet结构块
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        self.conv1 = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(out_channels, out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                self._make_layer(num_blocks, out_channels//2),
                BasicConv(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def _make_layer(self, num_blocks, inplanes):
        layers = []
        for i in range(num_blocks):
            layers.append(("residual_{}".format(i), Resblock(inplanes)))
        return nn.Sequential(OrderedDict(layers))


    def forward(self, x):
        x = self.conv1(x)
        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)

        return x

class CSPDarknet(nn.Module):
    def __init__(self, layers):
        super(CSPDarknet, self).__init__()
        self.inplanes = 32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.layers = nn.ModuleList([
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers[0](x)
        x = self.layers[1](x)
        out3 = self.layers[2](x)
        out4 = self.layers[3](out3)
        out5 = self.layers[4](out4)

        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    model = CSPDarknet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model