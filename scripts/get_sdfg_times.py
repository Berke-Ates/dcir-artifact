#!/usr/bin/python3

# Desc: Returns the runtime of the N-th run of the provided SDFG
# Usage: ./get_sdfg_times.py <Input SDFG> <Index N> <Use seconds (T/F)>

import sys
import dace

if len(sys.argv) != 4:
    print("SDFG Runtime [s] Extracting Tool")
    print("Arguments:")
    print("  Input SDFG: The SDFG to extract the runtime [s] from")
    print("  Index: The index of the run")
    print("  Use seconds: Outputs the time in seconds instead of milliseconds")
    exit(1)

input_file = sys.argv[1]
n = int(sys.argv[2])
use_seconds = sys.argv[3] == 'T'

sdfg = dace.SDFG.from_file(input_file)

times = list(
    list(list(sdfg.get_latest_report().durations.values())[0].values())
    [0].values())[0]

if use_seconds:
    print(times[n] / 1000)
else:
    print(times[n])
