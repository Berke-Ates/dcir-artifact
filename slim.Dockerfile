# This Dockerfile builds a minified image solely containing the binaries and 
# Python packages

# Base
FROM dcir-cgo23:latest as base

# Slim
FROM ubuntu:latest

# User directory
ENV USER=user
ENV HOME=/home/user
WORKDIR $HOME

# Move dotfiles
RUN mv /root/.bashrc .
RUN mv /root/.profile .

# Make terminal colorful
ENV TERM=xterm-color

# Install dependencies
RUN apt update -y && \ 
  apt install -y \
  clang \
  lld \
  python3-pip

# Launch bash shell at home
ENTRYPOINT cd $HOME && bash

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade "jax[cpu]"
RUN pip install --upgrade seaborn

# Copy scripts
WORKDIR $HOME/scripts
COPY --from=base $HOME/scripts ./

# Copy Python packages
WORKDIR $HOME/python_packages
COPY --from=base $HOME/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir ./torch_mlir
COPY --from=base $HOME/torch-mlir/examples ./examples
COPY --from=base $HOME/torch-mlir/requirements.txt ./requirements.txt
COPY --from=base $HOME/torch-mlir/pytorch-requirements.txt ./pytorch-requirements.txt
RUN pip install -r requirements.txt

WORKDIR $HOME/python_packages/dace
COPY --from=base $HOME/dace ./
RUN pip install --editable .

ENV PYTHONPATH=$HOME/python_packages/torch_mlir:$PYTHONPATH
ENV PYTHONPATH=$HOME/python_packages/examples:$PYTHONPATH

# Copy binaries
WORKDIR $HOME/bin
COPY --from=base $HOME/mlir-dace/llvm-project/build/bin/mlir-opt ./
COPY --from=base $HOME/mlir-dace/llvm-project/build/bin/mlir-translate ./
COPY --from=base $HOME/mlir-dace/llvm-project/build/bin/llc ./
COPY --from=base $HOME/mlir-dace/build/bin/sdfg-opt ./
COPY --from=base $HOME/mlir-dace/build/bin/sdfg-translate ./
COPY --from=base $HOME/mlir-hlo/build/bin/mlir-hlo-opt ./
COPY --from=base $HOME/Polygeist/build/bin/cgeist ./
COPY --from=base $HOME/Polygeist/build/bin/polygeist-opt ./

ENV PATH=$HOME/bin:$PATH

# Copy benchmarks
WORKDIR $HOME/benchmarks
COPY --from=base $HOME/benchmarks ./

# Copy Polybench comparator
WORKDIR $HOME/polybench-comparator
COPY --from=base $HOME/polybench-comparator ./
