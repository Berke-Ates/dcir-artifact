#!/bin/bash

# Desc: Runs a Polybench benchmark using DCIR. The output contains any
#       intermediate results and the times int the CSV format
# Usage: ./dcir.sh <Benchmark File> <Output Dir> <Repetitions>

# Check args
if [ $# -ne 3 ]; then
  echo "Usage: ./dcir.sh <Benchmark File> <Output Dir> <Repetitions>"
  exit 1
fi

# Read args
input_file=$1
output_dir=$2
reps=$3

# Check tools
check_tool(){
  if ! command -v $1 &> /dev/null; then
      echo "$1 could not be found"
      exit 1
  fi
}

check_tool clang
check_tool clang++
check_tool cgeist
check_tool mlir-opt
check_tool sdfg-opt
check_tool python3

# Helpers
input_name=$(basename ${input_file%.*})
input_dir=$(dirname $input_file)
utils_dir=$input_dir/../utilities
scripts_dir=$(dirname $0)/..

# Create output directory
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir;
fi

# Flags for the benchmark
flags="-DMINI_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS -fPIC -march=native"
opt_lvl_cc=3 # Optimization level for the control-centric optimizations
opt_lvl_dc=3 # Optimization level for the data-centric optimizations

# Lower optimization level for specific benchmarks
if [[ "$input_name" == "gramschmidt" ]]; then
  opt_lvl_cc=2
fi

if [[ "$input_name" == "durbin" ]] || \
   [[ "$input_name" == "gemver" ]] || \
   [[ "$input_name" == "doitgen" ]]; then
  opt_lvl_dc=2
fi

if [[ "$input_name" == "floyd-warshall" ]]; then
  opt_lvl_dc=1
fi

# Dace Settings
export DACE_compiler_cpu_executable="$(which clang++)"
export CC=`which clang`
export CXX=`which clang++`
export DACE_compiler_cpu_openmp_sections=0
export DACE_instrumentation_report_each_invocation=0
export DACE_compiler_cpu_args="-fPIC -O$opt_lvl_cc -march=native"

# Generating MLIR from C using Polygeist
cgeist -resource-dir=$(clang -print-resource-dir) -I $utils_dir \
  -S --memref-fullrank -O$opt_lvl_cc --raise-scf-to-affine $flags $input_file \
  > $output_dir/${input_name}_cgeist.mlir

# Optimizing with MLIR
mlir-opt --affine-loop-invariant-code-motion $output_dir/${input_name}_cgeist.mlir | \
mlir-opt --affine-scalrep | mlir-opt --lower-affine | \
mlir-opt --cse --inline > $output_dir/${input_name}_opt.mlir

# Converting to DCIR
sdfg-opt --convert-to-sdfg $output_dir/${input_name}_opt.mlir \
  > $output_dir/${input_name}_sdfg.mlir

# Translating to SDFG
sdfg-translate --mlir-to-sdfg $output_dir/${input_name}_sdfg.mlir \
  > $output_dir/$input_name.sdfg

# Optimizing data-centrically with DaCe
python3 $scripts_dir/opt_sdfg.py $output_dir/$input_name.sdfg \
  $output_dir/${input_name}_opt.sdfg $opt_lvl_dc T


