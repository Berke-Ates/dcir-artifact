#!/usr/bin/python3

# Desc: Runs the ResNet SDFG benchmark for DCIR
# Usage: ./bench_dcir.py <Input SDFG> <Repetitions> <Test Output (T/F)>

import sys
import dace
import torch
from torch import nn
import numpy as np

if len(sys.argv) != 4:
    print("DCIR PyTorch Benchmarking Tool (ResNet)")
    print("Arguments:")
    print("  Input SDFG: The SDFG to benchmark")
    print("  Repetitions: How many times to run the benchmark")
    print("  Test Output (T/F): If 'T', tests the output against PyTorch")
    exit(1)

input_file = sys.argv[1]
repetitions = int(sys.argv[2])
test_output = sys.argv[3] == 'T'

# Load and compile SDFG
sdfg = dace.SDFG.from_file(input_file)
obj = sdfg.compile()

# Run Benchmark
for i in range(repetitions):
    arg_dict = {}

    for argName, argType in sdfg.arglist().items():
        arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
        arg_dict[argName] = arr

    obj(**arg_dict)

# Print output
if test_output:

    def create_aa(aa_layer, channels, stride=2, enable=True):
        if not aa_layer or not enable:
            return nn.Identity()
        return aa_layer(stride) if issubclass(
            aa_layer, nn.AvgPool2d) else aa_layer(channels=channels,
                                                  stride=stride)

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

    model = nn.Sequential(BasicBlock(3, 3)).to(torch.device('cpu'))
    model.eval()

    data = torch.zeros(8, 32, 224, 224)

    for i in range(8):
        for j in range(32):
            for k in range(224):
                for l in range(224):
                    data[i, j, k, l] = (i + j + k + l) / (8 + 32 + 224 + 224)

    prediction_pytorch = model.forward(data).numpy()

    # Get output from SDFG
    prediction_dcir = dace.ndarray(shape=data.numpy().shape,
                                   dtype=data.numpy().dtype)
    obj(_arg0=data.numpy().copy(), _arg1=prediction_dcir)

    # Compare
    if not np.allclose(
            prediction_pytorch, prediction_dcir, rtol=1e-5, atol=1e-8):
        exit(1)
