#!/bin/bash

# Desc: Runs a pytorch benchmark using DCIR. The output contains any
#       intermediate results and the times in the CSV format
# Usage: ./dcir.sh <Benchmark File> <Output Dir> <Repetitions>

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
  if ! command -v $1 &>/dev/null; then
    echo "$1 could not be found"
    exit 1
  fi
}

check_tool clang
check_tool cgeist
check_tool mlir-opt
check_tool sdfg-opt
check_tool sdfg-translate
check_tool python3
check_tool icc

# Create output directory
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir
fi

# Clear .dacecache
rm -rf .dacecache

# Helpers
input_dir=$(dirname $input_file)
input_name=$(basename $input_dir)
gen_file=$input_dir/dcir_gen.py
bench_file=$input_dir/dcir_bench.py
current_dir=$(dirname $0)
scripts_dir=$(dirname $0)/..
timings_file=$output_dir/${input_name}_timings.csv
touch $timings_file

# Adds a value to the timings file, jumps to the next row after a write
csv_line=1
add_csv() {
  while [[ $(grep -c ^ $timings_file) < $csv_line ]]; do
    echo '' >>$timings_file
  done

  if [ ! -z "$(sed "${csv_line}q;d" $timings_file)" ]; then
    sed -i "${csv_line}s/$/,/" $timings_file
  fi

  sed -i "${csv_line}s/$/$1/" "$timings_file"
  csv_line=$((csv_line + 1))
}

# Dace Settings
export DACE_compiler_cpu_executable="$(which icc)"
export CC=$(which icc)
export CXX=$(which icc)
export DACE_compiler_cpu_openmp_sections=0
export DACE_instrumentation_report_each_invocation=0
export DACE_compiler_cpu_args="-fPIC -O3 -march=native"
export PYTHONWARNINGS="ignore"

# Generating MLIR using Torch-MLIR
python3 $gen_file >$output_dir/${input_name}_mhlo.mlir

# Renaming `forward` to `main` (required by DCIR)
sed -i -e "s/forward/main/g" $output_dir/${input_name}_mhlo.mlir

# Converting MHLO to standard dialects
$scripts_dir/mhlo2std.sh $output_dir/${input_name}_mhlo.mlir \
  >$output_dir/${input_name}_std.mlir

# Optimizing with MLIR
mlir-opt --affine-loop-invariant-code-motion $output_dir/${input_name}_std.mlir |
  mlir-opt --affine-scalrep | mlir-opt --lower-affine |
  mlir-opt --cse --inline >$output_dir/${input_name}_opt.mlir

# Converting to DCIR
sdfg-opt --convert-to-sdfg $output_dir/${input_name}_opt.mlir \
  >$output_dir/${input_name}_sdfg.mlir

# Translating to SDFG
sdfg-translate --mlir-to-sdfg $output_dir/${input_name}_sdfg.mlir \
  >$output_dir/$input_name.sdfg

# Optimizing data-centrically with DaCe
python3 $scripts_dir/opt_sdfg.py $output_dir/$input_name.sdfg \
  $output_dir/${input_name}_opt.sdfg 3 T

# Check output
if ! python3 $bench_file $output_dir/${input_name}_opt.sdfg 0 T; then
  echo "Output incorrect!"
  exit 1
fi

# Running the benchmark
OMP_NUM_THREADS=1 taskset -c 0 python3 $bench_file \
  $output_dir/${input_name}_opt.sdfg $repetitions F

add_csv "DCIR"

for i in $(seq 1 $repetitions); do
  time=$(python3 $scripts_dir/get_sdfg_times.py \
    $output_dir/${input_name}_opt.sdfg $((i - 1)) F)
  add_csv "$time"
done
