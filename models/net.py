import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable


import torch.nn as nn
import torch, math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pylab
from PIL import Image


def save_img(x, dir):
    x = x.cpu().numpy().astype(np.float32)
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    x = Image.fromarray(x * 255)
    if x.mode != 'RGB':
        x = x.convert('RGB')
    x.save(dir)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class DAC(nn.Module):
    def __init__(self, n_channels):
        super(DAC, self).__init__()

        self.mean = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )
        self.std = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )

    def forward(self, observed_feat, referred_feat):
        assert (observed_feat.size()[:2] == referred_feat.size()[:2])
        size = observed_feat.size()
        referred_mean, referred_std = calc_mean_std(referred_feat)
        observed_mean, observed_std = calc_mean_std(observed_feat)

        normalized_feat = (observed_feat - observed_mean.expand(
            size)) / observed_std.expand(size)
        referred_mean = self.mean(referred_mean)
        referred_std = self.std(referred_std)
        output = normalized_feat * referred_std.expand(size) + referred_mean.expand(size)
        return output


class MSHF(nn.Module):
    def __init__(self, n_channels, kernel=3):
        super(MSHF, self).__init__()

        pad = int((kernel - 1) / 2)

        self.grad_xx = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_yy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_xy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)

        for m in self.modules():
            if m == self.grad_xx:
                m.weight.data.zero_()
                m.weight.data[:, :, 1, 0] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, 1, -1] = 1
            elif m == self.grad_yy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 1] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, -1, 1] = 1
            elif m == self.grad_xy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 0] = 1
                m.weight.data[:, :, 0, -1] = -1
                m.weight.data[:, :, -1, 0] = -1
                m.weight.data[:, :, -1, -1] = 1

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

    def forward(self, x):
        fxx = self.grad_xx(x)
        fyy = self.grad_yy(x)
        fxy = self.grad_xy(x)
        hessian = ((fxx + fyy) + ((fxx - fyy) ** 2 + 4 * (fxy ** 2)) ** 0.5) / 2
        return hessian


class rcab_block(nn.Module):
    def __init__(self, n_channels, kernel, bias=False, activation=nn.ReLU(inplace=True)):
        super(rcab_block, self).__init__()

        block = []

        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1, bias=bias))
        block.append(activation)
        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1, bias=bias))

        self.block = nn.Sequential(*block)

        self.calayer = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        residue = self.block(x)
        chnlatt = F.adaptive_avg_pool2d(residue, 1)
        chnlatt = self.calayer(chnlatt)
        output = x + residue * chnlatt

        return output


class DiEnDec(nn.Module):
    def __init__(self, n_channels, act=nn.ReLU(inplace=True)):
        super(DiEnDec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
        )
        self.gate = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        output = self.gate(self.decoder(self.encoder(x)))
        return output


class SingleModule(nn.Module):
    def __init__(self, n_channels, n_blocks, act, attention):
        super(SingleModule, self).__init__()
        res_blocks = [rcab_block(n_channels=n_channels, kernel=3, activation=act) for _ in range(n_blocks)]
        self.body_block = nn.Sequential(*res_blocks)
        self.attention = attention
        if attention:
            self.coder = nn.Sequential(DiEnDec(3, act))
            self.dac = nn.Sequential(DAC(n_channels))
            self.hessian3 = nn.Sequential(MSHF(n_channels, kernel=3))
            self.hessian5 = nn.Sequential(MSHF(n_channels, kernel=5))
            self.hessian7 = nn.Sequential(MSHF(n_channels, kernel=7))

    def forward(self, x):
        sz = x.size()
        resin = self.body_block(x)

        if self.attention:
            hessian3 = self.hessian3(resin)
            hessian5 = self.hessian5(resin)
            hessian7 = self.hessian7(resin)
            hessian = torch.cat((torch.mean(hessian3, dim=1, keepdim=True),
                                 torch.mean(hessian5, dim=1, keepdim=True),
                                 torch.mean(hessian7, dim=1, keepdim=True))
                                , 1)
            hessian = self.coder(hessian)
            attention = torch.sigmoid(self.dac[0](hessian.expand(sz), x))
            resout = resin * attention
        else:
            resout = resin

        output = resout + x

        return output


class Generator(nn.Module):
    def __init__(self, n_channels, n_blocks, n_modules, act=nn.ReLU(True), attention=True, scale=(2, 3, 4)):
        super(Generator, self).__init__()
        self.n_modules = n_modules
        self.input = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=True)
        if n_modules == 1:
            self.body = nn.Sequential(SingleModule(n_channels, n_blocks, act, attention))
        else:
            self.body = nn.Sequential(*[SingleModule(n_channels, n_blocks, act, attention) for _ in range(n_modules)])

        self.tail = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)

        self.upscale = nn.ModuleList([UpScale(n_channels=n_channels, scale=s, act=False) for s in scale])

        self.output = nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):

        body_input = self.input(x)
        body_output = self.body(body_input)
        if self.n_modules == 1:
            sr_high = self.upscale[0](body_output)
        else:
            sr_high = self.upscale[0](self.tail(body_output) + body_input)
        results = self.output(sr_high)

        return results


class UpScale(nn.Sequential):
    def __init__(self, n_channels, scale, bn=False, act=nn.ReLU(inplace=True), bias=False):
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers.append(nn.Conv2d(in_channels=n_channels, out_channels=4 * n_channels, kernel_size=3, stride=1,
                                        padding=1, bias=bias))
                layers.append(nn.PixelShuffle(2))
                if bn: layers.append(nn.BatchNorm2d(n_channels))
                if act: layers.append(act)
        elif scale == 3:
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=9 * n_channels, kernel_size=3, stride=1,
                                    padding=1, bias=bias))
            layers.append(nn.PixelShuffle(3))
            if bn: layers.append(nn.BatchNorm2d(n_channels))
            if act: layers.append(act)
        else:
            raise NotImplementedError

        super(UpScale, self).__init__(*layers)





def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = SingleModule(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = SingleModule(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = SingleModule(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

