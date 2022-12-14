#!/bin/bash

# Desc: Runs a Polybench benchmark using DCIR. The output contains any
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
check_tool gcc
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
utils_dir=$input_dir/../utilities
current_dir=$(dirname "$0")
scripts_dir=$(dirname "$0")/..
timings_file=$output_dir/${input_name}_timings.csv
touch "$timings_file"
reference=$output_dir/${input_name}_reference.txt
actual=$output_dir/${input_name}_actual_dcir.txt

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
flags="-DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS -fPIC -march=native"
opt_lvl_cc=3 # Optimization level for the control-centric optimizations
opt_lvl_dc=3 # Optimization level for the data-centric optimizations

# Lower optimization level for specific benchmarks
if [[ "$input_name" == "gramschmidt" ]]; then
  opt_lvl_cc=2
fi

if [[ "$input_name" == "durbin" ]] ||
  [[ "$input_name" == "gemver" ]] ||
  [[ "$input_name" == "doitgen" ]]; then
  opt_lvl_dc=2
fi

if [[ "$input_name" == "floyd-warshall" ]]; then
  opt_lvl_dc=1
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

# Generating MLIR from C using Polygeist
# shellcheck disable=SC2086
cgeist -resource-dir="$(clang-13 -print-resource-dir)" -I "$utils_dir" \
  -S --memref-fullrank -O$opt_lvl_cc --raise-scf-to-affine $flags "$input_file" \
  >"$output_dir"/"${input_name}"_cgeist.mlir

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
# shellcheck disable=SC2086
gcc -I "$utils_dir" -O0 $flags -DPOLYBENCH_DUMP_ARRAYS \
  -o "$output_dir"/"${input_name}"_gcc_ref.out "$input_file" "$utils_dir"/polybench.c -lm

python3 "$current_dir"/bench_dcir.py "$output_dir"/"${input_name}"_opt.sdfg 1 T 2>"$actual" 1>/dev/null
"$output_dir"/"${input_name}"_gcc_ref.out 2>"$reference" 1>/dev/null

## Obtain array names
touch "$output_dir"/arr_names.txt

grep "begin dump:" "$reference" | while read -r line; do
  # shellcheck disable=SC2206
  arr_tmp=($line)
  arr_name=${arr_tmp[2]}
  echo -n "$arr_name " >>"$output_dir"/arr_names.txt
done

mapfile -t -d " " arr_names < <(cat "$output_dir"/arr_names.txt)
rm "$output_dir"/arr_names.txt

## Remove Warnings from output
sed -i '0,/^==BEGIN DUMP_ARRAYS==$/d' "$actual"
content_actual=$(cat "$actual")
printf '%s\n%s\n' "==BEGIN DUMP_ARRAYS==" "$content_actual" >"$actual"

## Use original array names
idx=0
grep "begin dump:" "$actual" | while read -r line; do
  # shellcheck disable=SC2206
  arr_tmp=($line)
  arr_name=${arr_tmp[2]}
  rep_arr_name=${arr_names[idx]}
  sed -i -e "s/$arr_name/$rep_arr_name/g" "$actual"
  idx=$((idx + 1))
done

## Compare the outputs
set +e
if ! python3 "$scripts_dir"/../polybench-comparator/comparator.py "$reference" "$actual"; then
  echo "Output incorrect!"
  exit 1
fi
set -e

# Running the benchmark
OMP_NUM_THREADS=1 taskset -c 0 python3 "$current_dir"/bench_dcir.py \
  "$output_dir"/"${input_name}"_opt.sdfg "$repetitions" F &>/dev/null

add_csv "DCIR"

for i in $(seq 1 "$repetitions"); do
  time=$(python3 "$scripts_dir"/get_sdfg_times.py \
    "$output_dir"/"${input_name}"_opt.sdfg $((i - 1)) T)
  add_csv "$time"
done
