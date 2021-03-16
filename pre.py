# -*- coding: utf-8 -*-

import torch
from torch import nn


class BasicCritic(nn.Module):
    """
    The BasicCritic module takes an image and predicts whether it is a cover
    image or a steganographic image (N, 1).

    Input: (N, 3, H, W)
    Output: (N, 1)
    """

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3
        )
    def _conv3d(self, in_channels, out_channels):
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3,3,3),
            padding=(1,1,1)
        )
    def _build_models(self):
        return nn.Sequential(
            self._conv3d(4, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size),

            self._conv3d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size),

            self._conv3d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size),

            self._conv3d(self.hidden_size, 1)
        )

    def __init__(self, hidden_size):
        super().__init__()
        self.version = '1'
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self._models = self.layers
            self.version = '1'

    def forward(self, x):
        x = self._models(x)
        x = torch.mean(x.view(x.size(0), -1), dim=1)

        x = self._models[0](x)
        for layer in self._models[1:]:
            x = layer(x)

        return x
class XuCritic(BasicCritic):
    def _build_models(self):
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels=4,
                out_channels=8,
                kernel_size=(3, 5, 5),
                padding=(1, 1, 1)
            ))

            self.conv2 = nn.Sequential(
                nn.BatchNorm3d(8),
                nn.Tanh(),
                nn.AvgPool3d(kernel_size=(1, 5, 5), stride=(1,2,2), padding=(0,2,2)),

                nn.Conv3d(8, 16, (3, 5, 5), padding=(1,2,2)),
                nn.BatchNorm3d(16),
                nn.Tanh(),
                nn.AvgPool3d(kernel_size=(3, 5, 5), stride=(1,2,2), padding=(1,2,2)),

                nn.Conv3d(16, 32, 1, padding=0),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.AvgPool3d(kernel_size=(1, 5, 5), stride=(1,2,2), padding=(0,2,2)),

                nn.Conv3d(32, 64, 1, padding=0),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.AvgPool3d(kernel_size=(3, 5, 5), stride=(1,2,2), padding=(1,2,2)),

                nn.Conv3d(64, 128, 1, padding=0),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.AvgPool3d((3, 16, 16), stride=1, padding=0),

                nn.Conv3d(128, 1, 1, padding=(0,1,1)),
            )
            return self.conv1, self.conv2

    def forward(self, x):
        x = self._models[0](x)
        x = torch.abs(x)
        x = self._models[1](x)
        x = torch.mean(x.view(x.size(0), -1), dim=1)

        return x

class Pre_network(BasicCritic):
    """
    预处理网络：针对时域进行上采样，目的一类似论文HIDE对图像特征进行提取，目的二把1为的图像变成与GIF图像同样维度
    """
    def _build_models(self):
            self.conv1 = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=(3, 1, 1),
                    stride=(1, 1, 1)

                ),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm3d(32))

            self.conv2 = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(2, 1, 1),
                    stride=(2, 1, 1)
                ),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm3d(32))

            self.conv3 = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(3, 1, 1),
                    stride=(1, 1, 1)
            ),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm3d(32))

            self.conv4 = nn.Sequential(
                nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm3d(64))

            self.conv5 = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm3d(32))

            self.conv6 = nn.Sequential(
                nn.Conv3d(in_channels=32, out_channels=7, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                )
            self.conv7 = nn.Sequential(
                nn.Tanh()
            )


            return self.conv1, self.conv2,self.conv3, self.conv4,self.conv5, self.conv6,self.conv7

    def forward(self, x):

        x = self._models[0](x)
        for layer in self._models[1:]:
            x = layer(x)
        return x