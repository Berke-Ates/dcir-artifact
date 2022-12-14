#!/bin/bash

# Desc: Runs a snippet benchmark using DCIR. The output contains any
#       intermediate results and the times in the CSV format
# Usage: ./dcir.sh <Benchmark File> <Output Dir> <Repetitions>

# Be safe
set -e          # Fail script when subcommand fails
set -u          # Disallow using undefined variables
set -o pipefail # Prevent errors from being masked

# Check args
if [ $# -ne 3 ]; then
  echo "Usage: ./dcir.sh <Benchmark File> <Output Dir> <Repetitions>"
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
check_tool clang-13
check_tool cgeist
check_tool mlir-opt
check_tool sdfg-opt
check_tool sdfg-translate
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
input_chrono="$input_dir/$input_name-chrono.c"
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
flags="-fPIC -march=native"
opt_lvl_cc=3 # Optimization level for the control-centric optimizations
opt_lvl_dc=3 # Optimization level for the data-centric optimizations

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

# Generating MLIR from C using Polygeist
# shellcheck disable=SC2086
cgeist -resource-dir="$(clang-13 -print-resource-dir)" -S --memref-fullrank \
  -O$opt_lvl_cc --raise-scf-to-affine $flags "$input_file" \
  1>"$output_dir"/"${input_name}"_cgeist.mlir 2>/dev/null

# Optimizing with MLIR
mlir-opt --affine-loop-invariant-code-motion "$output_dir"/"${input_name}"_cgeist.mlir |
  mlir-opt --affine-scalrep | mlir-opt --lower-affine |
  mlir-opt --cse --inline >"$output_dir"/"${input_name}"_opt.mlir

# Converting to DCIR
sdfg-opt --convert-to-sdfg "$output_dir"/"${input_name}"_opt.mlir \
  >"$output_dir"/"${input_name}"_sdfg.mlir

# Translating to SDFG
sdfg-translate --mlir-to-sdfg "$output_dir"/"${input_name}"_sdfg.mlir \
  >"$output_dir"/"$input_name".sdfg

# Optimizing data-centrically with DaCe
python3 "$scripts_dir"/opt_sdfg.py "$output_dir"/"$input_name".sdfg \
  "$output_dir"/"${input_name}"_opt.sdfg $opt_lvl_dc T

# Check output
actual=$(python3 "$current_dir"/bench_dcir.py "$output_dir"/"${input_name}"_opt.sdfg 1 T)

# shellcheck disable=SC2086
clang -O0 $flags -o "$output_dir"/"${input_name}"_clang_ref.out "$input_chrono" -lm &>/dev/null
set +e
"$output_dir"/"${input_name}"_clang_ref.out &>/dev/null
reference=$?
set -e

if [ "$actual" -ne "$reference" ]; then
  echo "Output incorrect!"
  exit 1
fi

# Running the benchmark
OMP_NUM_THREADS=1 taskset -c 0 python3 "$current_dir"/bench_dcir.py \
  "$output_dir"/"${input_name}"_opt.sdfg "$repetitions" F

add_csv "DCIR"

for i in $(seq 1 "$repetitions"); do
  time=$(python3 "$scripts_dir"/get_sdfg_times.py \
    "$output_dir"/"${input_name}"_opt.sdfg $((i - 1)) F)
  add_csv "$time"
done
