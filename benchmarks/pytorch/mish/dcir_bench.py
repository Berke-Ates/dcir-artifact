#!/usr/bin/python3

# Desc: Runs the MISH SDFG benchmark for DCIR
# Usage: ./bench_dcir.py <Input SDFG> <Repetitions> <Test Output (T/F)>

import sys
import dace
import torch
from torch import nn
import numpy as np

if len(sys.argv) != 4:
    print("DCIR PyTorch Benchmarking Tool")
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

    class Mish(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.log(1 + torch.exp(x))
            return x

    model = nn.Sequential(Mish()).to(torch.device('cpu'))
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
