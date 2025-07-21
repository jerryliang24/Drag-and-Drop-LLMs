from functools import partial, reduce
from operator import mul

import torch
from torch import nn
from torch.nn import functional as F

cumprod = partial(reduce, mul)


class HyperConv3d(nn.Module):
    def __init__(self, in_features: tuple, out_features: tuple, kernel_size: int):
        super().__init__()
        sequence_length, in_height, in_width = in_features
        sequence_length, out_height, out_width = out_features
        # init parameters
        self.norm = nn.LayerNorm(list(in_features[1:3]))
        self.conv1w = nn.Conv2d(in_width, out_width, kernel_size, 1, "same", bias=False)
        self.conv1h = nn.Conv2d(in_height, out_height, kernel_size, 1, "same", bias=False)
        self.conv2h = nn.Conv2d(in_height, out_height, kernel_size, 1, "same", bias=False)
        self.conv2w = nn.Conv2d(in_width, out_width, kernel_size, 1, "same", bias=False)
        self.bias = nn.Parameter(torch.empty(out_height, out_width))
        nn.init.uniform_(self.bias, -1.0 / self.bias.numel(), 1.0 / self.bias.numel())

    def forward(self, x):
        x = self.norm(x)
        outer_shape, inner_shape = x.shape[:-3], x.shape[-3:]
        x = torch.flatten(x, end_dim=-4)
        x = (self.conv_width_first(x) + self.conv_height_first(x) + self.bias) / 3
        x = torch.unflatten(x, dim=0, sizes=outer_shape)
        return x

    def conv_width_first(self, x):
        x = x.transpose(-1, -3)  # B, N, L, C -> B, C, L, N
        x = self.conv1w(x)
        x.transpose_(-1, -3)
        x.transpose_(-2, -3)
        x = self.conv1h(x)
        x.transpose_(-2, -3)
        return x

    def conv_height_first(self, x):
        x = x.transpose(-2, -3)  # B, N, L, C -> B, L, N, C
        x = self.conv2h(x)
        x.transpose_(-2, -3)
        x.transpose_(-1, -3)
        x = self.conv2w(x)
        x.transpose_(-1, -3)
        return F.tanh(x)


class HyperConvLayer(nn.Module):
    def __init__(self, in_features: tuple, out_features: tuple, kernel_size: int):
        super().__init__()
        self.linear1 = HyperConv3d(in_features, out_features, kernel_size=kernel_size)
        self.linear2 = nn.Conv2d(in_features[0], out_features[0], kernel_size, 1, padding="same")

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class HyperConvBlock(nn.Module):
    def __init__(self, features: list[tuple], kernel_size: int):
        super().__init__()
        in_features_list = features[:-1]
        out_features_list = features[1:]
        module_list = []
        for i, o in zip(in_features_list, out_features_list):
            module_list.append(HyperConvLayer(i, o, kernel_size))
        self.module_list = nn.ModuleList(module_list)

    def forward(self, x):
        for module in self.module_list:
            y = module(x)
            x = x + y if y.shape == x.shape else y
        return x


class HyperConvDecoder(nn.Module):
    def __init__(
        self,
        features: list[tuple],
        condition_dim: tuple,
        kernel_size: int,
    ):
        super().__init__()
        assert condition_dim == features[0]
        self.to_condition = nn.Identity()
        self.decoder = HyperConvBlock(features, kernel_size)

    @property
    def device(self):
        return self.decoder.module_list[0].bias.devcie

    def forward(self, condition):
        # condition: (batch_size, condition_dim)
        # x: (batch_size, length, height, width)
        x = self.to_condition(condition)
        x = self.decoder(x)
        return x
