#!/bin/bash

# Desc: Converts the mhlo dialect to builtin dialect for further conversion
# Usage: ./mhlo2std.sh <MLIR File>

# Check args
if [ $# -ne 1 ]; then
  echo "Usage: ./mhlo2std.sh <MLIR File>"
  exit 1
fi

# Read args
input_file=$1

# Execute conversion
mlir-hlo-opt --hlo-legalize-to-linalg $input_file `# mhli -> linalg` | \
mlir-hlo-opt --linalg-bufferize `# bufferize linalg` | \
mlir-hlo-opt --convert-linalg-to-affine-loops `# linalg -> affine` | \
mlir-hlo-opt --computeop-and-func-bufferize \
  --final-bufferize=alignment=128 # tensor -> memref
