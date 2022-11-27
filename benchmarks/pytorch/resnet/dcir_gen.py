#!/usr/bin/python3

# Desc: Generates MLIR code for DCIR of the ResNet benchmark
# Usage: python3 dcir.py

import torch
import torch_mlir
from torch import nn


# Load model
def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return aa_layer(stride) if issubclass(
        aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 cardinality=1,
                 base_width=64,
                 reduce_first=1,
                 dilation=1,
                 first_dilation=None,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d,
                 attn_layer=None,
                 aa_layer=None,
                 drop_block=None,
                 drop_path=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2
                                           or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes,
                               first_planes,
                               kernel_size=3,
                               stride=1 if use_aa else stride,
                               padding=first_dilation,
                               dilation=first_dilation,
                               bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block(
        ) if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer,
                            channels=first_planes,
                            stride=stride,
                            enable=use_aa)

        self.conv2 = nn.Conv2d(first_planes,
                               outplanes,
                               kernel_size=3,
                               padding=dilation,
                               dilation=dilation,
                               bias=False)
        self.bn2 = norm_layer(outplanes)

        # self.se = create_attn(attn_layer, outplanes)
        self.se = None

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        # x = self.conv1(x)
        # x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        # x = self.conv2(x)
        # x = self.bn2(x)

        # if self.se is not None:
        #     x = self.se(x)

        # if self.drop_path is not None:
        #     x = self.drop_path(x)

        # if self.downsample is not None:
        #     shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.act2(x)

        return x


model = BasicBlock(3, 3)
model.eval()

# Compile model
data = torch.rand(8, 32, 224, 224)

module = torch_mlir.compile(model,
                            data,
                            output_type=torch_mlir.OutputType.MHLO)

# Output MLIR code
print(module)
