#!/usr/bin/python3

# Desc: Runs the Mish benchmark using Pytorch
# Usage: python3 pytorch.py <Repetitions> <Test Output (T/F)>

import sys
import numpy as np
import torch
from torch import nn
import time
import torch_mlir

from torch_mlir_e2e_test.mhlo_backends.linalg_on_tensors import LinalgOnTensorsMhloBackend

if len(sys.argv) != 3:
    print("PyTorch Mish Benchmarking Tool")
    print("Arguments:")
    print("  Repetitions: How many times to run the benchmark")
    print("  Test Output (T/F): If 'T', tests the output against Torch-MLIR")
    exit(1)

repetitions = int(sys.argv[1])
test_output = sys.argv[2] == 'T'


# Load model
class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.log(1 + torch.exp(x))
        return x


model = nn.Sequential(Mish()).to(torch.device('cpu'))
model.eval()

# Benchmark
for i in range(repetitions):
    data = torch.rand(8, 32, 224, 224)
    start = time.time()
    model.forward(data)
    runtime = time.time() - start
    print(runtime * 1000)

# Test output
if test_output:
    data = torch.zeros(8, 32, 224, 224)

    for i in range(8):
        for j in range(32):
            for k in range(224):
                for l in range(224):
                    data[i, j, k, l] = (i + j + k + l) / (8 + 32 + 224 + 224)

    prediction_pytorch = model.forward(data).numpy()

    # Generate MHLO
    backend = LinalgOnTensorsMhloBackend()

    module = torch_mlir.compile(model,
                                data,
                                output_type=torch_mlir.OutputType.MHLO)

    compiled = backend.compile(module)
    jit_module = backend.load(compiled)

    prediction_torch_mlir = jit_module.forward(data.numpy())

    # Compare
    if not np.allclose(
            prediction_pytorch, prediction_torch_mlir, rtol=1e-5, atol=1e-8):
        exit(1)
