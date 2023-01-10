#!/bin/bash

# Desc: Runs all snippet benchmarks using GCC, Clang, Polygeist + MLIR, DCIR
# and DaCe. The output contains any intermediate results and the times in the
# CSV format as well as all the plots.
# Usage: ./run_all.sh <Output Dir> <Repetitions>

# Be safe
set -e          # Fail script when subcommand fails
set -u          # Disallow using undefined variables
set -o pipefail # Prevent errors from being masked

# Check args
if [ $# -ne 2 ]; then
  echo "Usage: ./run_all.sh <Output Dir> <Repetitions>"
  exit 1
fi

# Read args
output_dir=$1
repetitions=$2

# Create output directory
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Silence Python warnings
export PYTHONWARNINGS="ignore"

# Helpers
runners_dir=$(dirname "$0")
scripts_dir=$(dirname "$0")/..
benchmarks_dir=$(dirname "$0")/../../benchmarks/snippets

# Run benchmarks
benchmarks=$(find "$benchmarks_dir"/* -name '*.c' -not -name '*-chrono.c')
total=$(echo "$benchmarks" | wc -l)

runners="$runners_dir/gcc.sh $runners_dir/clang.sh $runners_dir/dace.sh \
        $runners_dir/mlir.sh $runners_dir/dcir.sh"

for runner in $runners; do
  count=0
  echo "Running with: $runner"

  for benchmark in $benchmarks; do
    bname="$(basename "$benchmark" .c)"
    count=$((count + 1))
    diff=$((total - count))
    percent=$((count * 100 / total))

    prog=''
    for _ in $(seq 1 $count); do
      prog="$prog#"
    done

    for _ in $(seq 1 $diff); do
      prog="$prog-"
    done

    echo -ne "\033[2K\r"
    echo -ne "$prog ($percent%) ($bname) "

    $runner "$benchmark" "$output_dir" "$repetitions"
  done

  echo ""
done

for benchmark in $benchmarks; do
  bname="$(basename "$benchmark" .c)"

  if [[ "$bname" == "demo" ]]; then
    fig_num=2
  fi

  if [[ "$bname" == "mem_line" ]]; then
    fig_num=8
  fi

  if [[ "$bname" == "congrad_multi_field" ]]; then
    fig_num=9
  fi

  if [[ "$bname" == "memory" ]]; then
    fig_num=10
  fi

  mv "$output_dir/${bname}_timings.csv" "$output_dir/fig${fig_num}.csv"
  python3 "$scripts_dir"/single_plot.py "$output_dir/fig${fig_num}.csv" \
    "$output_dir"/fig"$fig_num".pdf
done
