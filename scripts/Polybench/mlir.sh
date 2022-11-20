#!/bin/bash

# Desc: Runs a Polybench benchmark using Polygeist + MLIR. The output contains 
#       any intermediate results and the times in the CSV format
# Usage: ./mlir.sh <Benchmark File> <Output Dir> <Repetitions>

# Check args
if [ $# -ne 3 ]; then
  echo "Usage: ./mlir.sh <Benchmark File> <Output Dir> <Repetitions>"
  exit 1
fi

# Read args
input_file=$1
output_dir=$2
repetitions=$3

# Check tools
check_tool(){
  if ! command -v $1 &> /dev/null; then
      echo "$1 could not be found"
      exit 1
  fi
}

check_tool llc
check_tool clang
check_tool cgeist
check_tool mlir-opt
check_tool mlir-translate
check_tool python3

# Create output directory
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir;
fi

# Helpers
input_name=$(basename ${input_file%.*})
input_dir=$(dirname $input_file)
utils_dir=$input_dir/../utilities
scripts_dir=$(dirname $0)/..
timings_file=$output_dir/${input_name}_timings.csv; touch $timings_file
reference=$output_dir/${input_name}_reference.txt
actual=$output_dir/${input_name}_actual_mlir.txt

# Adds a value to the timings file, jumps to the next row after a write
csv_line=1
add_csv(){
  while [[ $(grep -c ^ $timings_file) < $csv_line ]]; do
    echo '' >> $timings_file
  done

  if [ ! -z "$(sed "${csv_line}q;d" $timings_file)" ]; then
    sed -i "${csv_line}s/$/,/" $timings_file
  fi

  sed -i "${csv_line}s/$/$1/" "$timings_file"
  csv_line=$((csv_line+1))
}

# Flags for the benchmark
flags="-DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_TIME -fPIC -march=native"
opt_lvl_cc=3 # Optimization level for the control-centric optimizations

# Lower optimization level for specific benchmarks
if [[ "$input_name" == "gramschmidt" ]]; then
  opt_lvl_cc=2
fi

# Compiling with MLIR
compile_with_mlir(){
  additional_flags=$1
  output_name=$2

  # Generating MLIR from C using Polygeist
  cgeist -resource-dir=$(clang -print-resource-dir) -I $utils_dir \
    -S --memref-fullrank -O$opt_lvl_cc --raise-scf-to-affine $flags \
    $additional_flags $input_file > $output_dir/${output_name}_cgeist.mlir

  # Optimizing with MLIR
  mlir-opt --affine-loop-invariant-code-motion \
    $output_dir/${output_name}_cgeist.mlir | \
  mlir-opt --affine-scalrep | mlir-opt --lower-affine | \
  mlir-opt --cse --inline > $output_dir/${output_name}_opt.mlir

  # Lower to LLVM dialect
  mlir-opt --convert-scf-to-cf --convert-func-to-llvm --convert-cf-to-llvm \
    --convert-math-to-llvm --lower-host-to-llvm --reconcile-unrealized-casts \
    $output_dir/${output_name}_opt.mlir > $output_dir/${output_name}_ll.mlir

  # Translate
  mlir-translate --mlir-to-llvmir $output_dir/${output_name}_ll.mlir \
    > $output_dir/${output_name}.ll

  # Compile
  llc -O$opt_lvl_cc --relocation-model=pic $output_dir/${output_name}.ll \
    -o $output_dir/${output_name}.s

  # Assemble
  clang -O$opt_lvl_cc $flags $additional_flags $output_dir/${output_name}.s \
    $utils_dir/polybench.c -o $output_dir/${output_name}.out -lm
}

# Compile
compile_with_mlir "" "${input_name}_mlir"

# Check output
compile_with_mlir "-DPOLYBENCH_DUMP_ARRAYS" "${input_name}_mlir_dump"

clang -I $utils_dir -O0 $flags -DPOLYBENCH_DUMP_ARRAYS \
  -o $output_dir/${input_name}_clang_ref.out $input_file $utils_dir/polybench.c -lm

$output_dir/${input_name}_mlir_dump.out 2> $actual 1> /dev/null
$output_dir/${input_name}_clang_ref.out 2> $reference 1> /dev/null

python3 $scripts_dir/../polybench-comparator/comparator.py $reference $actual

if [ $? -ne 0 ]; then
  echo "Output incorrect!"
  exit 1
fi

# Running the benchmark
add_csv "Polygeist + MLIR"

for i in $(seq 1 $repetitions); do
  time=$(./$output_dir/${input_name}_mlir.out)
  add_csv "$time"
done
