#!/usr/bin/python3

# Desc: Runs a pytorch SDFG benchmark for DCIR
# Usage: ./bench_dcir.py <Input SDFG> <Repetitions> <Print Output (T/F)>

import sys
import dace
import torch

if len(sys.argv) != 4:
    print("DCIR PyTorch Benchmarking Tool")
    print("Arguments:")
    print("  Input SDFG: The SDFG to benchmark")
    print("  Repetitions: How many times to run the benchmark")
    print("  Print Output (T/F): If 'T', prints the output to standard error")
    exit(1)

input_file = sys.argv[1]
repetitions = int(sys.argv[2])
print_output = sys.argv[3] == 'T'

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
if print_output:
    data = torch.zeros(8, 32, 224, 224)

    for i in range(8):
        for j in range(32):
            for k in range(224):
                for l in range(224):
                    data[i, j, k, l] = (i + j + k + l) / (8 + 32 + 224 + 224)
    # TODO: Generate prediction & print
