#!/bin/bash

# Desc: Converts the mhlo dialect to builtin dialect for further conversion
# Usage: ./mhlo2std.sh <MLIR File>

# Be safe
set -e          # Fail script when subcommand fails
set -u          # Disallow using undefined variables
set -o pipefail # Prevent errors from being masked

# Check args
if [ $# -ne 1 ]; then
  echo "Usage: ./mhlo2std.sh <MLIR File>"
  exit 1
fi

# Read args
input_file=$1

# Check tools
check_tool() {
  if ! command -v "$1" &>/dev/null; then
    echo "$1 could not be found"
    exit 1
  fi
}

check_tool mlir-hlo-opt

# Execute conversion
mlir-hlo-opt --hlo-legalize-to-linalg "$input_file" |
  mlir-hlo-opt --linalg-bufferize |
  mlir-hlo-opt --convert-linalg-to-affine-loops |
  mlir-hlo-opt --computeop-and-func-bufferize --final-bufferize=alignment=128
