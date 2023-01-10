################################################################################
### GENERAL SETUP
################################################################################

FROM ubuntu:22.04 as llvm

# User directory
ENV USER=user
ENV HOME=/home/user
WORKDIR $HOME

# Install dependencies
RUN apt-get update -y && \ 
  apt-get install -y --no-install-recommends \
  wget=1.21.2-2ubuntu1 \
  git=1:2.34.1-1ubuntu1.5 \
  cmake=3.22.1-1ubuntu1.22.04.1 \
  ninja-build=1.10.1-1 \
  clang=1:14.0-55~exp2 \
  lld=1:14.0-55~exp2 \
  python3-pip=22.0.2+dfsg-1 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

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
  -DLLVM_INSTALL_UTILS=ON && \
  ninja

# Build mlir-dace
WORKDIR $HOME/mlir-dace/build

RUN cmake -G Ninja .. \
  -DMLIR_DIR=$HOME/mlir-dace/llvm-project/build/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$HOME/mlir-dace/llvm-project/build/bin/llvm-lit \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release && \
  ninja && \
  mkdir -p $HOME/bin && \
  cp $HOME/mlir-dace/build/bin/* $HOME/bin && \
  rm -rf $HOME/mlir-dace

# Go home
WORKDIR $HOME

################################################################################
### Install mlir-hlo
################################################################################

# Install python dependencies
RUN pip install --upgrade --no-cache-dir pip==22.3.1 && \
  pip install --upgrade --no-cache-dir "jax[cpu]"==0.4.1

# Get MLIR-HLO
RUN git clone --depth 1 --branch cgo23 https://github.com/Berke-Ates/mlir-hlo.git
WORKDIR $HOME/mlir-hlo

# Build MLIR
RUN git clone https://github.com/llvm/llvm-project.git

WORKDIR $HOME/mlir-hlo/llvm-project
RUN git checkout "$(cat ../build_tools/llvm_version.txt)"
WORKDIR $HOME/mlir-hlo 

RUN build_tools/build_mlir.sh "$PWD"/llvm-project/ "$PWD"/llvm-build

# Build MLIR-HLO
WORKDIR $HOME/mlir-hlo/build

RUN cmake .. -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DMLIR_DIR="$PWD"/../llvm-build/lib/cmake/mlir && \
  ninja && \
  cp $HOME/mlir-hlo/build/bin/mlir-hlo-opt $HOME/bin && \
  rm -rf $HOME/mlir-hlo 

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
  -DCMAKE_BUILD_TYPE=Release && \
  ninja && \
  cp $HOME/Polygeist/build/bin/cgeist $HOME/bin && \
  cp $HOME/Polygeist/build/bin/mlir-opt $HOME/bin && \
  cp $HOME/Polygeist/build/bin/mlir-translate $HOME/bin && \
  rm -rf $HOME/Polygeist

# Go home
WORKDIR $HOME

################################################################################
### Reduce Image Size
################################################################################

# Copy binaries
FROM ubuntu:22.04

# User directory
ENV USER=user
ENV HOME=/home/user
WORKDIR $HOME

# Move dotfiles
RUN mv /root/.bashrc . && mv /root/.profile .

# Make terminal colorful
ENV TERM=xterm-color

# Install dependencies
RUN apt-get update -y && \ 
  apt-get install -y --no-install-recommends \
  wget=1.21.2-2ubuntu1  \
  git=1:2.34.1-1ubuntu1.5 \
  cmake=3.22.1-1ubuntu1.22.04.1 \
  make=4.3-4.1build1 \
  ninja-build=1.10.1-1 \
  clang-13=1:13.0.1-2ubuntu2.1 \
  gcc=4:11.2.0-1ubuntu1 \
  lld=1:14.0-55~exp2  \
  python3-pip=22.0.2+dfsg-1 \
  python3=3.10.6-1~22.04 \
  python3-dev=3.10.6-1~22.04 \
  gpg=2.2.27-3ubuntu2.1 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Launch bash shell at home
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["cd $HOME && bash"]

# Python dependencies
RUN pip install --upgrade --no-cache-dir pip==22.3.1 && \
  pip install --upgrade --no-cache-dir seaborn==0.12.2

# Get Polybench comparator
RUN git clone --depth 1 --branch cgo23 https://github.com/Berke-Ates/polybench-comparator.git

# Copy Binaries
COPY --from=llvm $HOME/bin $HOME/bin

# Add clang copy
RUN cp "$(which clang-13)" $HOME/bin/clang && \ 
  cp "$(which clang-13)" $HOME/bin/clang++

# Add binaries to PATH
ENV PATH=$HOME/bin:$PATH

################################################################################
### Install torch-mlir
################################################################################

# Make sure submodules are initialized
RUN git clone --depth 1 --branch cgo23 https://github.com/Berke-Ates/torch-mlir.git
WORKDIR $HOME/torch-mlir
RUN git submodule update --init --recursive --depth 1

# Install python dependencies
RUN pip install --upgrade --no-cache-dir pip==22.3.1 && \
  sed -i -e 's/1.14.0.dev20221109/2.0.0.dev20221231/g' pytorch-requirements.txt && \
  pip install --no-cache-dir -r requirements.txt

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
  -DLIBTORCH_VARIANT=shared && \
  ninja

# Go home
WORKDIR $HOME

# Add binaries to PATH
ENV PATH=$HOME/torch-mlir/build/bin:$PATH
ENV PYTHONPATH=$HOME/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir:$PYTHONPATH
ENV PYTHONPATH=$HOME/torch-mlir/build/../examples:$PYTHONPATH

################################################################################
### Install ICC
################################################################################

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN wget --progress=dot:giga -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor \
  | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | tee /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update -y && \
  apt-get install -y --no-install-recommends \
  intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic=2023.0.0-25370 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/intel/oneapi/compiler/latest/env/vars.sh" >> $HOME/.bashrc

# Go home
WORKDIR $HOME

################################################################################
### Install dace
################################################################################

RUN git clone --depth 1 --branch cgo23 https://github.com/Berke-Ates/dace.git

WORKDIR $HOME/dace
RUN git submodule update --init --recursive --depth 1 && \
  pip install --no-cache-dir --editable . && \
  pip install --no-cache-dir mxnet-mkl==1.6.0 numpy==1.23.1

# Go home
WORKDIR $HOME

################################################################################
### Copy files over
################################################################################

COPY ./benchmarks ./benchmarks
COPY ./output ./output
COPY ./scripts ./scripts
COPY ./README.md ./README.md
COPY ./LICENSE.txt ./LICENSE.txt
