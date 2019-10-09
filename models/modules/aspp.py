import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.batchnorm import SynchronizedBatchNorm2d

BN_MOMENTUM = 0.1

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

        # self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes, use_spconv3=True):
        super(ASPP, self).__init__()
        self.use_spconv3 = use_spconv3
        if use_spconv3:
            self.conv6 = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)

            self.conv6 = nn.Conv2d(in_channels=64, out_channels=outplanes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        if self.use_spconv3:
            x = self.conv6(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)

        return x


def build_aspp(inplanes, outplanes, use_spconv3=True):
    return ASPP(inplanes, outplanes, use_spconv3)