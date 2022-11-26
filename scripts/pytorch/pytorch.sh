#!/bin/bash

# Desc: Runs a pytorch benchmark using Pytorch. The output contains any
#       intermediate results and the times in the CSV format
# Usage: ./pytorch.sh <Benchmark File> <Output Dir> <Repetitions>

# Check args
if [ $# -ne 3 ]; then
  echo "Usage: ./pytorch.sh <Benchmark File> <Output Dir> <Repetitions>"
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

check_tool python3

# Create output directory
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir;
fi

# Helpers
input_dir=$(dirname $input_file)
input_name=$(basename ${input_dir%.*})
input_file=$input_dir/pytorch.py
current_dir=$(dirname $0)
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

# Check output
if ! python3 $input_file 0 T; then
  echo "Output incorrect!"
  exit 1
fi

# Running the benchmark
runtimes=$(OMP_NUM_THREADS=1 taskset -c 0 python3 $input_file $repetitions F) 

add_csv "PyTorch"

for i in $runtimes; do
  add_csv "$i"
done

