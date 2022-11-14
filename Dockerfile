################################################################################
### GENERAL SETUP
################################################################################

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
  git \
  cmake \
  ninja-build \
  clang \
  lld \
  python3-pip

# Launch bash shell at home
ENTRYPOINT cd $HOME && bash

# Copy all files
COPY . .

# Make sure submodules are initialized
RUN git submodule update --init --recursive --depth 1

################################################################################
### Install DaCe
################################################################################

WORKDIR $HOME/dace
RUN pip install --editable .

################################################################################
### Install mlir-dace
################################################################################

# Build MLIR
WORKDIR $HOME/mlir-dace/llvm-project/build

RUN cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_INSTALL_UTILS=ON

RUN ninja

# Add binaries to PATH
ENV PATH=$HOME/mlir-dace/llvm-project/build/bin:$PATH

# Build mlir-dace
WORKDIR $HOME/mlir-dace/build

RUN cmake -G Ninja .. \
  -DMLIR_DIR=$HOME/mlir-dace/llvm-project/build/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$HOME/mlir-dace/llvm-project/build/bin/llvm-lit \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release

RUN ninja

# Add binaries to PATH
ENV PATH=$HOME/mlir-dace/build/bin:$PATH

################################################################################
### Install torch-mlir
################################################################################

# Install python dependencies
WORKDIR $HOME/torch-mlir
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Build torch-mlir in-tree
WORKDIR $HOME/torch-mlir/build

RUN cmake -G Ninja ../externals/llvm-project/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD"/.. \
  -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR="$PWD"/../externals/llvm-external-projects/torch-mlir-dialects \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLIBTORCH_SRC_BUILD=ON \
  -DLIBTORCH_VARIANT=shared

RUN ninja

# Add binaries to PATH
ENV PATH=$HOME/torch-mlir/build/bin:$PATH
ENV PYTHONPATH=$PWD/tools/torch-mlir/python_packages/torch_mlir:$PYTHONPATH
ENV PYTHONPATH=$PWD/../examples:$PYTHONPATH

################################################################################
### Install mlir-hlo
################################################################################

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade "jax[cpu]"

# Build MLIR
WORKDIR $HOME/mlir-hlo

RUN git clone https://github.com/llvm/llvm-project.git

WORKDIR $HOME/mlir-hlo/llvm-project
RUN git checkout $(cat ../build_tools/llvm_version.txt)
WORKDIR $HOME/mlir-hlo 

RUN build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build

WORKDIR $HOME/mlir-hlo/build

RUN cmake .. -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir

RUN ninja

# Add binaries to PATH
ENV PATH=$HOME/mlir-hlo/build/bin:$PATH

################################################################################
### Install Polygeist
################################################################################

WORKDIR $HOME/Polygeist/build

RUN cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release

RUN ninja

# Add binaries to PATH
ENV PATH=$HOME/Polygeist/build/bin:$PATH
