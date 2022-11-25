#!/usr/bin/python3

# Desc: Runs the Mish benchmark using Torch-MLIR
# Usage: python3 torch-mlir.py <Repetitions> <Print Output (T/F)>

import sys
import numpy as np
import time
import torch
import torch_mlir

from torch_mlir_e2e_test.mhlo_backends.linalg_on_tensors import LinalgOnTensorsMhloBackend

if len(sys.argv) != 3:
    print("Torch-MLIR Mish Benchmarking Tool")
    print("Arguments:")
    print("  Warmup: How many rounds of warmup to run")
    print("  Repetitions: How many times to run the benchmark")
    print("  Print Output (T/F): If 'T', prints the output to standard error")
    exit(1)

repetitions = int(sys.argv[1])
print_output = sys.argv[2] == 'T'


# Load model
class Mish(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.log(1 + torch.exp(x))
        return x


model = Mish()
model.eval()

# Compile model
data = torch.rand(8, 32, 224, 224)
backend = LinalgOnTensorsMhloBackend()

module = torch_mlir.compile(model,
                            data,
                            output_type=torch_mlir.OutputType.MHLO)

compiled = backend.compile(module)
jit_module = backend.load(compiled)
jit_func = jit_module.forward

# Benchmark
for i in range(repetitions):
    data = torch.rand(8, 32, 224, 224)
    dnp = data.numpy()
    start = time.time()
    jit_func(dnp)
    runtime = time.time() - start
    print(runtime)

# Output
if print_output:
    data = torch.zeros(8, 32, 224, 224)

    for i in range(8):
        for j in range(32):
            for k in range(224):
                for l in range(224):
                    data[i, j, k, l] = (i + j + k + l) / (8 + 32 + 224 + 224)

    dnp = data.numpy()
    prediction = jit_func(dnp)
    print(prediction, file=sys.stderr)
