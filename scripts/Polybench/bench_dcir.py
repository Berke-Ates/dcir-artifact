#!/usr/bin/python3

# Desc: Runs a Polybench SDFG benchmark for DCIR
# Usage: ./bench_dcir.py <Input SDFG> <Repetitions> <Print Output (T/F)>

import sys
import dace

if len(sys.argv) != 4:
    print("DCIR Polybench Benchmarking Tool")
    print("Arguments:")
    print("  Input SDFG: The SDFG to benchmark")
    print("  Repetitions: How many times to run the benchmark")
    print(
        "  Print Output (T/F): If 'T', prints the output of the latest run to standard error"
    )
    exit(1)

input_file = sys.argv[1]
repetitions = int(sys.argv[2])
print_output = sys.argv[3] == 'T'


# General Printer
def printArray(arr, offset, depth):
    if (depth > 0):
        for dimIdx, dim in enumerate(arr):
            offsetFac = len(arr) if depth > 1 else 1
            printArray(dim, offsetFac * (offset + dimIdx), depth - 1)
    else:
        if offset % 20 == 0:
            print("", file=sys.stderr)
        print("%.4f " % arr, end='', file=sys.stderr)


# Printer for the Deriche benchmark
def printDeriche(arr):
    W = 4096
    H = 2160
    for i in range(W):
        for j in range(H):
            if (i * H + j) % 20 == 0:
                print("", file=sys.stderr)
            print("%.4f " % arr[i, j], end='', file=sys.stderr)


# Printer for the Doitgen benchmark
def printDoitgen(arr):
    NQ = 140
    NR = 150
    NP = 160
    for i in range(NR):
        for j in range(NQ):
            for k in range(NP):
                if (i * NQ * NP + j * NP + k) % 20 == 0:
                    print("", file=sys.stderr)
                print("%.4f " % arr[i, j, k], end='', file=sys.stderr)


# Printer for the Cholesky benchmark
def printCholesky(arr):
    N = 2000
    for i in range(N):
        for j in range(i + 1):
            if (i * N + j) % 20 == 0:
                print("", file=sys.stderr)
            print("%.4f " % arr[i, j], end='', file=sys.stderr)


# Printer for the Gramschmidt benchmark
def printGramschmidt(arr, useN):
    M = 1000
    N = 1200
    iUB = N if useN else M

    for i in range(iUB):
        for j in range(N):
            if (i * N + j) % 20 == 0:
                print("", file=sys.stderr)
            print("%.4f " % arr[i, j], end='', file=sys.stderr)


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
    print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)

    for argName, arr in latest_results.items():
        print("begin dump: %s" % argName, end='', file=sys.stderr)
        if "cholesky" in input_file:
            printCholesky(arr)
        elif "gramschmidt" in input_file:
            printGramschmidt(arr, argName == "_arg1")
        elif "deriche" in input_file:
            printDeriche(arr)
        elif "doitgen" in input_file:
            printDoitgen(arr)
        else:
            printArray(arr, 0, len(arr.shape))

        print("\nend   dump: %s" % argName, file=sys.stderr)

    print("==END   DUMP_ARRAYS==", file=sys.stderr)
