#!/bin/bash

# Desc: Runs a Polybench benchmark using GCC. The output contains any
#       intermediate results and the times in the CSV format
# Usage: ./gcc.sh <Benchmark File> <Output Dir> <Repetitions>

# Be safe
set -e          # Fail script when subcommand fails
set -u          # Disallow using undefined variables
set -o pipefail # Prevent errors from being masked

# Check args
if [ $# -ne 3 ]; then
  echo "Usage: ./gcc.sh <Benchmark File> <Output Dir> <Repetitions>"
  exit 1
fi

# Read args
input_file=$1
output_dir=$2
repetitions=$3

# Check tools
check_tool() {
  if ! command -v "$1" &>/dev/null; then
    echo "$1 could not be found"
    exit 1
  fi
}

check_tool gcc
check_tool clang
check_tool python3

# Create output directory
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Helpers
input_name=$(basename "${input_file%.*}")
input_dir=$(dirname "$input_file")
utils_dir=$input_dir/../utilities
scripts_dir=$(dirname "$0")/..
timings_file=$output_dir/${input_name}_timings.csv
touch "$timings_file"
reference=$output_dir/${input_name}_reference.txt
actual=$output_dir/${input_name}_actual_gcc.txt

# Adds a value to the timings file, jumps to the next row after a write
csv_line=1
add_csv() {
  while [[ $(grep -c ^ "$timings_file") -lt $csv_line ]]; do
    echo '' >>"$timings_file"
  done

  if [ -n "$(sed "${csv_line}q;d" "$timings_file")" ]; then
    sed -i "${csv_line}s/$/,/" "$timings_file"
  fi

  sed -i "${csv_line}s/$/$1/" "$timings_file"
  csv_line=$((csv_line + 1))
}

# Flags for the benchmark
flags="-DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_TIME -fPIC -march=native"
opt_lvl_cc=3 # Optimization level for the control-centric optimizations

# Lower optimization level for specific benchmarks
if [[ "$input_name" == "gramschmidt" ]]; then
  opt_lvl_cc=1
fi

# Compile
# shellcheck disable=SC2086
gcc -I "$utils_dir" -O$opt_lvl_cc $flags -o "$output_dir"/"${input_name}"_gcc.out \
  "$input_file" "$utils_dir"/polybench.c -lm

# Check output
# shellcheck disable=SC2086
gcc -I "$utils_dir" -O$opt_lvl_cc $flags -DPOLYBENCH_DUMP_ARRAYS \
  -o "$output_dir"/"${input_name}"_gcc_dump.out "$input_file" "$utils_dir"/polybench.c -lm

# shellcheck disable=SC2086
clang -I "$utils_dir" -O0 $flags -DPOLYBENCH_DUMP_ARRAYS \
  -o "$output_dir"/"${input_name}"_clang_ref.out "$input_file" "$utils_dir"/polybench.c -lm

"$output_dir"/"${input_name}"_gcc_dump.out 2>"$actual" 1>/dev/null
"$output_dir"/"${input_name}"_clang_ref.out 2>"$reference" 1>/dev/null

if ! python3 "$scripts_dir"/../polybench-comparator/comparator.py "$reference" "$actual"; then
  echo "Output incorrect!"
  exit 1
fi

# Running the benchmark
add_csv "GCC"

for _ in $(seq 1 "$repetitions"); do
  time=$(OMP_NUM_THREADS=1 taskset -c 0 ./"$output_dir"/"${input_name}"_gcc.out)
  add_csv "$time"
done
