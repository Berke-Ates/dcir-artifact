#!/bin/bash

# Desc: Runs a snippet benchmark using Clang. The output contains any
#       intermediate results and the times in the CSV format
# Usage: ./clang.sh <Benchmark File> <Output Dir> <Repetitions>

# Check args
if [ $# -ne 3 ]; then
  echo "Usage: ./clang.sh <Benchmark File> <Output Dir> <Repetitions>"
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

check_tool clang
check_tool gcc
check_tool python3

# Create output directory
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir;
fi

# Helpers
input_name=$(basename ${input_file%.*})
input_dir=$(dirname $input_file)
input_chrono="$input_dir/$input_name-chrono.c"
scripts_dir=$(dirname $0)/..
timings_file=$output_dir/${input_name}_timings.csv; touch $timings_file

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
flags="-fPIC -march=native"
opt_lvl_cc=3 # Optimization level for the control-centric optimizations

# Compile
clang -O$opt_lvl_cc $flags -o $output_dir/${input_name}_clang.out $input_chrono -lm

# Check output
gcc -O0 $flags -o $output_dir/${input_name}_gcc_ref.out $input_chrono -lm

$output_dir/${input_name}_clang.out &> /dev/null
actual=$?
$output_dir/${input_name}_gcc_ref.out &> /dev/null
reference=$?

if [ "$actual" -ne "$reference" ]; then
  echo "Output incorrect!"
  exit 1
fi

# Running the benchmark
add_csv "Clang"

for i in $(seq 1 $repetitions); do
  time=$(OMP_NUM_THREADS=1 taskset -c 0 ./$output_dir/${input_name}_clang.out)
  add_csv "$time"
done
