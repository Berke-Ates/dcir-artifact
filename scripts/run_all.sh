#!/bin/bash

# Desc: Runs all the benchmarks. The output contains any intermediate results
# and the times in the CSV format as well as all the plots.
# Usage: ./run_all.sh <Output Dir> <Repetitions>

# Check args
if [ $# -ne 2 ]; then
  echo "Usage: ./run_all.sh <Output Dir> <Repetitions>"
  exit 1
fi

# Read args
output_dir=$1
repetitions=$2

# Create output directory
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir
fi

# Helpers
scripts_dir=$(dirname $0)

# Run benchmarks
./$scripts_dir/Polybench/run_all.sh "$output_dir" "$repetitions"
./$scripts_dir/pytorch/run_all.sh "$output_dir" "$repetitions"
./$scripts_dir/snippets/run_all.sh "$output_dir" "$repetitions"
