# -*- coding: utf-8 -*-

import torch
from torch import nn


class Encoder(nn.Module):

    add_image = False

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
    def _conv3d(self, in_channels, out_channels):
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1,3,3),
            padding=(0,1,1)

        )
    def _conv3d_tro(self, in_channels, out_channels):
        return nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1,2,2),
            stride=(1,2,2)

        )
    def _conv3d_k1(self, in_channels, out_channels):
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1,1,1),
            padding=(0,0,0),

        )

    def _conv3d_old(self, in_channels, out_channels):
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)

        )
    def _build_models(self):
        self.features = nn.Sequential(
            self._conv3d(4, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size),
        )
        self.layers = nn.Sequential(
            self._conv3d_k1(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size),
            self._conv3d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size),
            self._conv3d(self.hidden_size, 4),

            nn.Tanh(),
        )
        return self.features, self.layers

    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, image, data):
        print(image.shape)
        print(data.shape)
        x = self._models[0](torch.cat([image] + [data], dim=1))
        print("loading")
        x_1 = self._models[1](x)

        x_2 = self._models[2](x_1)

        x_3 = self._models[3](x_2)

        x_4 = self._models[4](x_3)

        x_4_cat = torch.cat([x_4,x_2], dim=1)

        x_7 = self._models[7](x_4_cat)
        x_5 = self._models[5](x_7)

        x_5_cat = torch.cat([x_5,x_1], dim=1)
        x_8 = self._models[8](x_5_cat)

        x_6 = self._models[6](x_8)
        x_6_cat = torch.cat([x_6,x,data], dim=1)

        x_9 = self._models[9](x_6_cat)
        x_9 = torch.cat([x_9,data],dim=1)
        x = self._models[10](x_9)

        x = self._models[11](x)




        if self.add_image:
            x = image + x

        return x


