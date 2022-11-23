#!/usr/bin/python3

# Desc: Returns the runtime [s] of the N-th run of the provided SDFG
# Usage: ./get_sdfg_times.py <Input SDFG> <Index N>

import sys
import dace

if len(sys.argv) != 3:
    print("SDFG Runtime [s] Extracting Tool")
    print("Arguments:")
    print("  Input SDFG: The SDFG to extract the runtime [s] from")
    print("  Index: The index of the run")
    exit(1)

input_file = sys.argv[1]
n = int(sys.argv[2])

sdfg = dace.SDFG.from_file(input_file)

times = list(
    list(list(sdfg.get_latest_report().durations.values())[0].values())
    [0].values())[0]

print(times[n] / 1000)
