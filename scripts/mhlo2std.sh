#!/bin/bash

# Desc: Converts the mhlo dialect to builtin dialect for further conversion
# Usage: ./mhlo2std.sh <MLIR File>

input_file=$1

mlir-hlo-opt --hlo-legalize-to-linalg $input_file | \ # mhli -> linalg
mlir-hlo-opt --linalg-bufferize | \ # bufferize linalg
mlir-hlo-opt --convert-linalg-to-affine-loops | \ # linalg -> affine
mlir-hlo-opt --computeop-and-func-bufferize \
  --final-bufferize=alignment=128 # tensor -> memref
