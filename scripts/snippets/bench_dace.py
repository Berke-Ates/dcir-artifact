#!/usr/bin/python3

# Desc: Runs a Snippet SDFG benchmark for DaCe
# Usage: ./bench_sdfg.py <Input SDFG> <Repetitions>

import sys
import dace

if len(sys.argv) != 3:
    print("DaCe Snippet Benchmarking Tool")
    print("Arguments:")
    print("  Input SDFG: The SDFG to benchmark")
    print("  Repetitions: How many times to run the benchmark")
    exit(1)

input_file = sys.argv[1]
repetitions = int(sys.argv[2])

# Load and compile SDFG
sdfg = dace.SDFG.from_file(input_file)
obj = sdfg.compile()

# Run Benchmark
for i in range(repetitions):
    argv_loc = dace.ndarray(shape=(0, ), dtype=dace.dtypes.int8)
    obj(argc_loc=0, _argcount=0, argv_loc=argv_loc)
