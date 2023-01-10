#!/bin/bash

# Desc: Runs a snippet benchmark using DaCe. The output contains any
#       intermediate results and the times in the CSV format
# Usage: ./dace.sh <Benchmark File> <Output Dir> <Repetitions>

# Be safe
set -e          # Fail script when subcommand fails
set -u          # Disallow using undefined variables
set -o pipefail # Prevent errors from being masked

# Check args
if [ $# -ne 3 ]; then
  echo "Usage: ./dace.sh <Benchmark File> <Output Dir> <Repetitions>"
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

check_tool clang
check_tool clang++
check_tool python3

# Create output directory
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Clear .dacecache
rm -rf .dacecache

# Helpers
input_name=$(basename "${input_file%.*}")
input_dir=$(dirname "$input_file")
current_dir=$(dirname "$0")
scripts_dir=$(dirname "$0")/..
timings_file=$output_dir/${input_name}_timings.csv
touch "$timings_file"

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
opt_lvl_cc=3 # Optimization level for the control-centric optimizations
opt_lvl_dc=3 # Optimization level for the data-centric optimizations

# Lower optimization level for specific benchmarks
if [[ "$input_name" == "congrad_multi_field" ]]; then
  opt_lvl_dc=1
fi

if [[ "$input_name" == "demo" ]]; then
  opt_lvl_dc=2
fi

# Dace Settings
DACE_compiler_cpu_executable="$(which clang++)"
export DACE_compiler_cpu_executable
CC=$(which clang)
export CC
CXX=$(which clang++)
export CXX
export DACE_compiler_cpu_openmp_sections=0
export DACE_instrumentation_report_each_invocation=0
export DACE_compiler_cpu_args="-fPIC -O$opt_lvl_cc -march=native"
export PYTHONWARNINGS="ignore"

# Optimizing data-centrically with DaCe
python3 "$scripts_dir"/opt_sdfg.py "$input_dir"/"${input_name}"_c2dace.sdfg \
  "$output_dir"/"${input_name}"_c2dace_opt.sdfg $opt_lvl_dc T

# Running the benchmark
OMP_NUM_THREADS=1 taskset -c 0 python3 "$current_dir"/bench_dace.py \
  "$output_dir"/"${input_name}"_c2dace_opt.sdfg "$repetitions"

add_csv "DaCe"

for i in $(seq 1 "$repetitions"); do
  time=$(python3 "$scripts_dir"/get_sdfg_times.py \
    "$output_dir"/"${input_name}"_c2dace_opt.sdfg $((i - 1)) F)
  add_csv "$time"
done
