#!/usr/bin/python3

# Desc: Runs a Snippet SDFG benchmark for DCIR
# Usage: ./bench_dcir.py <Input SDFG> <Repetitions> <Print Output (T/F)>

import sys
import dace

if len(sys.argv) != 4:
    print("DCIR Snippet Benchmarking Tool")
    print("Arguments:")
    print("  Input SDFG: The SDFG to benchmark")
    print("  Repetitions: How many times to run the benchmark")
    print(
        "  Print Output (T/F): If 'T', prints the output of the latest run to standard out"
    )
    exit(1)

input_file = sys.argv[1]
repetitions = int(sys.argv[2])
print_output = sys.argv[3] == 'T'

# Load and compile SDFG
sdfg = dace.SDFG.from_file(input_file)
obj = sdfg.compile()

latest_results = {}

# Run Benchmark
for i in range(repetitions):
    arg_dict = {}

    for argName, argType in sdfg.arglist().items():
        arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
        arg_dict[argName] = arr

    obj(**arg_dict)

    latest_results = arg_dict

# Print output
if print_output:
    for argName, arr in latest_results.items():
        for elem in arr:
            print(elem)
