#!/usr/bin/python3

# Desc: Runs the Mish benchmark using Pytorch. The output contains any
#       intermediate results and the times in the CSV format
# Usage: python3 pytorch.py <Warmup> <Repetitions> <Print Output (T/F)>

import sys
import torch
from torch import nn
import time

if len(sys.argv) != 4:
    print("PyTorch Mish Benchmarking Tool")
    print("Arguments:")
    print("  Warmup: How many rounds of warmup to run")
    print("  Repetitions: How many times to run the benchmark")
    print("  Print Output (T/F): If 'T', prints the output to standard error")
    exit(1)

warmup = int(sys.argv[1])
repetitions = int(sys.argv[2])
print_output = sys.argv[3] == 'T'


# Load model
class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.log(1 + torch.exp(x))
        return x


model = nn.Sequential(Mish()).to(torch.device('cpu'))
model.eval()

# Warmup
for i in range(warmup):
    data = torch.rand(8, 32, 224, 224)
    model.forward(data)

# Benchmark
for i in range(repetitions):
    data = torch.rand(8, 32, 224, 224)
    start = time.time()
    model.forward(data)
    runtime = time.time() - start
    print(runtime)

if print_output:
    data = torch.zeros(8, 32, 224, 224)

    for i in range(8):
        for j in range(32):
            for k in range(224):
                for l in range(224):
                    data[i, j, k, l] = (i + j + k + l) / (8 + 32 + 224 + 224)

    prediction = model.forward(data)
    print(prediction, file=sys.stderr)
