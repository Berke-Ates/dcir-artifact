################################################################################
### GENERAL SETUP
################################################################################

FROM ubuntu:latest as llvm

# User directory
ENV USER=user
ENV HOME=/home/user
WORKDIR $HOME

# Install dependencies
RUN apt update -y && \ 
  apt install -y \
  wget \
  git \
  cmake \
  ninja-build \
  clang \
  lld \
  python3-pip

################################################################################
### Install mlir-dace
################################################################################

# Make sure submodules are initialized
RUN git clone --depth 1 --branch cgo23 https://github.com/spcl/mlir-dace.git
WORKDIR $HOME/mlir-dace
RUN git submodule update --init --recursive --depth 1

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

# Install and clear build folder
RUN mkdir -p $HOME/llvm-dcir
RUN DESTDIR=$HOME/llvm-dcir ninja install

# Add binaries to PATH
ENV PATH=$HOME/llvm-dcir/usr/local/bin:$PATH

# Build mlir-dace
WORKDIR $HOME/mlir-dace/build

RUN cmake -G Ninja .. \
  -DMLIR_DIR=$HOME/llvm-dcir/usr/local/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$HOME/llvm-dcir/usr/local/bin/llvm-lit \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release

RUN ninja

RUN DESTDIR=$HOME/llvm-dcir ninja install

WORKDIR $HOME/bin
RUN cp $HOME/mlir-dace/build/bin/* .

# Clean up build folders for space
RUN rm -rf $HOME/mlir-dace $HOME/llvm-dcir

# Go home
WORKDIR $HOME

################################################################################
### Install mlir-hlo
################################################################################

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade "jax[cpu]"

# Get MLIR-HLO
RUN git clone --depth 1 --branch cgo23 https://github.com/Berke-Ates/mlir-hlo.git
WORKDIR $HOME/mlir-hlo

# Build MLIR
RUN git clone https://github.com/llvm/llvm-project.git

WORKDIR $HOME/mlir-hlo/llvm-project
RUN git checkout $(cat ../build_tools/llvm_version.txt)
WORKDIR $HOME/mlir-hlo 

RUN build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build

# Build MLIR-HLO
WORKDIR $HOME/mlir-hlo/build

RUN cmake .. -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir

RUN DESTDIR=$HOME/llvm-mlir-hlo ninja install

# Copy binaries
WORKDIR $HOME/bin
RUN cp $HOME/mlir-hlo/build/bin/mlir-hlo-opt .

# Clean up build folders for space
RUN rm -rf $HOME/mlir-hlo $HOME/llvm-mlir-hlo

# Go home
WORKDIR $HOME

################################################################################
### Install Polygeist
################################################################################

# Make sure submodules are initialized
RUN git clone --depth 1 --branch cgo23 https://github.com/Berke-Ates/Polygeist.git
WORKDIR $HOME/Polygeist
RUN git submodule update --init --recursive --depth 1

# Build Polygeist
WORKDIR $HOME/Polygeist/build

RUN cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release

RUN DESTDIR=$HOME/llvm-polygeist ninja install

# Copy binaries
WORKDIR $HOME/bin
RUN cp $HOME/llvm-polygeist/usr/local/bin/cgeist .
RUN cp $HOME/llvm-polygeist/usr/local/bin/mlir-opt .
RUN cp $HOME/llvm-polygeist/usr/local/bin/mlir-translate .

# Clean up build folders for space
RUN rm -rf $HOME/Polygeist $HOME/llvm-polygeist

# Go home
WORKDIR $HOME

################################################################################
### Reduce Image Size
################################################################################

# Copy binaries
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
  wget \
  git \
  clang-13 \
  lld \
  python3-pip

# Launch bash shell at home
ENTRYPOINT cd $HOME && bash

# Dependencies for plotting
RUN pip install --upgrade pip
RUN pip install --upgrade seaborn

# Copy Binaries
COPY --from=llvm $HOME/bin $HOME/bin

# Add clang copy
RUN cp `which clang-13` $HOME/bin/clang
RUN cp `which clang-13` $HOME/bin/clang++

# Add binaries to PATH
ENV PATH=$HOME/bin:$PATH

# Get Polybench comparator
RUN git clone --depth 1 --branch cgo23 https://github.com/Berke-Ates/polybench-comparator.git

################################################################################
### Install torch-mlir
################################################################################

# Make sure submodules are initialized
RUN git clone --depth 1 --branch cgo23 https://github.com/Berke-Ates/torch-mlir.git
WORKDIR $HOME/torch-mlir
RUN git submodule update --init --recursive --depth 1

# Install python dependencies
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

# Go home
WORKDIR $HOME

# Add binaries to PATH
ENV PATH=$HOME/torch-mlir/build/bin:$PATH
ENV PYTHONPATH=$HOME/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir:$PYTHONPATH
ENV PYTHONPATH=$HOME/torch-mlir/build/../examples:$PYTHONPATH

################################################################################
### Install ICC
################################################################################

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor \
  | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | tee /etc/apt/sources.list.d/oneAPI.list

RUN apt update -y && apt install -y intel-hpckit

RUN echo "source /opt/intel/oneapi/compiler/2022.2.1/env/vars.sh" >> $HOME/.bashrc

################################################################################
### Install dace
################################################################################

RUN git clone --depth 1 --branch cgo23 https://github.com/Berke-Ates/dace.git

WORKDIR $HOME/dace
RUN git submodule update --init --recursive --depth 1

RUN pip install --editable .

WORKDIR $HOME

################################################################################
### Copy files over
################################################################################

COPY ./benchmarks ./benchmarks
COPY ./output ./output
COPY ./scripts ./scripts
COPY ./README.md ./README.md
COPY ./LICENSE.txt ./LICENSE.txt
