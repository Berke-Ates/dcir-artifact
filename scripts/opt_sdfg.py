#!/usr/bin/python3

# Desc: Optimizes a SDFG
# Usage: ./opt_sdfg.py <Input SDFG> <Output SDFG> <Optimization Level (0-3)> <Add Timing Instrumentation (T/F)>

import sys
import dace
from dace.transformation.passes.scalar_to_symbol import promote_scalars_to_symbols
from dace.transformation.auto.auto_optimize import auto_optimize, move_small_arrays_to_stack
from dace.transformation.interstate import StateFusion
from dace.transformation.passes.optional_arrays import OptionalArrayInference
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace import SDFG

if len(sys.argv) != 4:
    print("SDFG Optimizing Tool")
    print("Arguments:")
    print("  Input SDFG: The SDFG to optimize")
    print("  Output SDFG: The filepath of the optimized SDFG")
    print(
        "  Optimization Level (0-3): Determines how many optimizations to apply"
    )
    print(
        "  Add Timing Instrumentation (T/F): If 'T', measures execution time")
    exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
opt_lvl = int(sys.argv[3])
add_timing = sys.argv[4] == 'T'

# Load and validate initial SDFG
sdfg = SDFG.from_file(input_file)
sdfg.validate()

# Apply Optimizations

if opt_lvl == 1:
    for i in range(5):
        OptionalArrayInference().apply_pass(sdfg, dict())
        ConstantPropagation().apply_pass(sdfg, dict())
        sdfg.apply_transformations_repeated([StateFusion])

if opt_lvl >= 2:
    sdfg.simplify()

if opt_lvl > 0:
    move_small_arrays_to_stack(sdfg)  # TODO: Can this be moved?

if opt_lvl == 3:
    auto_optimize(sdfg, dace.DeviceType.CPU)

# Ensure a sequential execution
for node, parent in sdfg.all_nodes_recursive():
    if isinstance(node, dace.nodes.MapEntry):
        node.schedule = dace.ScheduleType.Sequential

# Add instrumentations
if add_timing:
    sdfg.instrument = dace.InstrumentationType.Timer

# Save the optimized SDFG
sdfg.save(output_file)
