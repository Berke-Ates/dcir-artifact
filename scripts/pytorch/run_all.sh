#!/bin/bash

# Desc: Runs all pytorch benchmarks using PyTorch, Torch-MLIR and DCIR. The 
# output contains any intermediate results and the times in the CSV format as 
# well as all the plots.
# Usage: ./run_all.sh <Output Dir> <Repetitions>

# Check args
if [ $# -ne 2 ]; then
  echo "Usage: ./run_all.sh <Output Dir> <Repetitions>"
  exit 1
fi

# Read args
output_dir=$1
repetitions=$2

# Create output directory
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir;
fi

# Helpers
runners_dir=$(dirname $0)
scripts_dir=$(dirname $0)/..
benchmarks_dir=$(dirname $0)/../../benchmarks/pytorch

# Run benchmarks
benchmarks=$(find $benchmarks_dir/* -name 'pytorch.py')
total=$(echo "$benchmarks" | wc -l)

runners="$runners_dir/pytorch.sh $runners_dir/torch-mlir.sh $runners_dir/dcir.sh"

for runner in $runners; do
  count=0
  echo "Running with: $runner"

  for benchmark in $benchmarks; do
      bname="$(basename $benchmark .c)"
      count=$((count+1))
      diff=$(($total - $count))
      percent=$(($count * 100 / $total))

      prog=''
      for i in $(seq 1 $count); do
        prog="$prog#"
      done

      for i in $(seq 1 $diff); do
        prog="$prog-"
      done

      echo -ne "\033[2K\r"
      echo -ne "$prog ($percent%) ($bname) "
      
      $runner $benchmark $output_dir $repetitions
  done

  echo ""
done

for benchmark in $benchmarks; do
    bench_dir=$(dirname $benchmark)
    bench_name=$(basename $bench_dir)
    mv "$output_dir/${bench_name}_timings.csv" "$output_dir/${bench_name}.csv"
    python3 $scripts_dir/single_plot.py "$output_dir/${bench_name}.csv" \
      $output_dir/$bench_name.pdf
done
