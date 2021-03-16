# -*- coding: utf-8 -*-

import torch
from torch import nn
from steganogan.spectral import SpectralNorm

class Adv(nn.Module):
    """
    The BasicDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.

    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """
    def _conv3d_tro(self, in_channels, out_channels):
        return nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1,1,1),
            stride=(2,1,1),
            padding=(0,0,0)

        )
    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
    def _conv3d_1(self, in_channels, out_channels):
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3,3,3),
            padding=(0,1,1)
        )
    def _conv3d_2(self, in_channels, out_channels):
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2,3,3),
            padding=(0,1,1),
            stride=(2,1,1)
        )
    def _conv3d_3(self, in_channels, out_channels):
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3,3,3),
            padding=(0,1,1),
            stride=(1,1,1)
        )
    def _conv3d(self, in_channels, out_channels):
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3,3,3),
            padding=(1,1,1)
        )
    def _conv3d_k1(self, in_channels, out_channels):
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1,1,1),
            padding=(0,0,0)
        )
    def _conv2d_k1(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0
        )
    def _build_models(self):
        self.layers = nn.Sequential(
            self._conv3d_1(4, 16),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)

        )
        self.layer1 = nn.Sequential(
            self._conv3d_2(16, 32),

            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            self._conv3d_1(32, 32),

            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            self._conv3d(32, 32),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
            )
        self.layer4 = nn.Sequential(
            self._conv3d(32, 32),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
            )
        self.layer5 = nn.Sequential(
            SpectralNorm(self._conv3d_k1(32,32)))
        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))


            # self._conv3d(64, 32),
            # nn.LeakyReLU(inplace=True),
            # nn.BatchNorm3d(32),
            # self._conv3d(32, 16),
            # nn.LeakyReLU(inplace=True),
            # nn.BatchNorm3d(16),
            # self._conv3d(16, 3)




        return self.layers,self.layer1,self.layer2,self.layer3,self.layer4,self.layer5,self.layer6

    def __init__(self):
        super().__init__()
        self.version = '1'

        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self._models = [self.layers,self.layer1,self.layer2,self.layer3,self.layer4,self.layer5,self.layer6]

            self.version = '1'

    def forward(self, x):
    #
        layers = self._models[0](x)
        layer1 = self._models[1](layers)
        layer2 = self._models[2](layer1)
        layer3 = self._models[3](layer2)
        layer4 = self._models[4](layer3 )
        x = self._models[5](layer4)

        x = torch.squeeze(x,dim=2)




        x = self._models[6](x)

        x = torch.mean(x.view(x.size(1), -1), dim=1)




    #     return layer0,layer1,layer2,layer3,layer4,layer5,layer6


        return x

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self._models = [
                self.self.layers,
                self.self.layer1,
                self.self.layer2,
                self.self.layer3,
                self.self.layer4,
                self.self.layer5,
                self.self.layer6,
            ]

            self.version = '1'
