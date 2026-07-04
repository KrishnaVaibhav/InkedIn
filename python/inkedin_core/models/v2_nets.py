"""Vendored inference architecture of manga-colorization-v2 (qweasdd), which itself
derives from AlacGAN (orashi) and Tag2Pix (blandocs). Kept verbatim where it affects
weight compatibility. Training-only parts (discriminator, spectral norm blocks) removed.

Source: https://github.com/qweasdd/manga-colorization-v2 (no license file — weights
and this architecture are for local/research use; see models/manifests/NOTICE.md).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Selayer(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out


class BottleneckX_Origin(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(
            planes * 2, planes * 2, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.selayer = Selayer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.selayer(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class SEResNeXt_Origin(nn.Module):
    def __init__(self, block, layers, input_channels=3, cardinality=32, num_classes=1000):
        super().__init__()
        self.cardinality = cardinality
        self.inplanes = 64
        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, self.cardinality, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        return x1, x2, x3, x4


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super().__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(
            D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
            groups=cardinality, bias=False,
        )
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module("shortcut", nn.AvgPool2d(2, stride=2))
        self.selayer = Selayer(out_channels)

    def forward(self, x):
        bottleneck = F.leaky_relu(self.conv_reduce(x), 0.2, True)
        bottleneck = F.leaky_relu(self.conv_conv(bottleneck), 0.2, True)
        bottleneck = self.selayer(self.conv_expand(bottleneck))
        return self.shortcut(x) + bottleneck


class Generator(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
        self.encoder = SEResNeXt_Origin(BottleneckX_Origin, [3, 4, 6, 3], num_classes=370, input_channels=1)

        self.to0 = self._make_encoder_block_first(5, 32)
        self.to1 = self._make_encoder_block(32, 64)
        self.to2 = self._make_encoder_block(64, 92)
        self.to3 = self._make_encoder_block(92, 128)
        self.to4 = self._make_encoder_block(128, 256)

        self.deconv_for_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, output_padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
        )

        tunnel4 = nn.Sequential(*[ResNeXtBottleneck(512, 512, cardinality=32, dilate=1) for _ in range(20)])
        self.tunnel4 = nn.Sequential(
            nn.Conv2d(1024 + 128, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            tunnel4,
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
        )

        depth = 2
        tunnel = [ResNeXtBottleneck(256, 256, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(256, 256, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(256, 256, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [
            ResNeXtBottleneck(256, 256, cardinality=32, dilate=2),
            ResNeXtBottleneck(256, 256, cardinality=32, dilate=1),
        ]
        self.tunnel3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Sequential(*tunnel),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
        )

        tunnel = [ResNeXtBottleneck(128, 128, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(128, 128, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(128, 128, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [
            ResNeXtBottleneck(128, 128, cardinality=32, dilate=2),
            ResNeXtBottleneck(128, 128, cardinality=32, dilate=1),
        ]
        self.tunnel2 = nn.Sequential(
            nn.Conv2d(128 + 256 + 64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Sequential(*tunnel),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
        )

        tunnel = [
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=1),
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=2),
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=4),
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=2),
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=1),
        ]
        self.tunnel1 = nn.Sequential(
            nn.Conv2d(64 + 32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Sequential(*tunnel),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
        )

        self.exit = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        )

    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, sketch):
        x0 = self.to0(sketch)
        aux_out = self.to1(x0)
        aux_out = self.to2(aux_out)
        aux_out = self.to3(aux_out)

        x1, x2, x3, x4 = self.encoder(sketch[:, 0:1])

        out = self.tunnel4(torch.cat([x4, aux_out], 1))
        x = self.tunnel3(torch.cat([out, x3], 1))
        x = self.tunnel2(torch.cat([x, x2, x1], 1))
        x = torch.tanh(self.exit(torch.cat([x, x0], 1)))
        decoder_output = self.deconv_for_decoder(out)
        return x, decoder_output
