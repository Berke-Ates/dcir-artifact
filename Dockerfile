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
  command-not-found \
  git

# Update command-not-found database and launch bash shell at home
ENTRYPOINT apt -qq update && cd $HOME && bash

# Copy all files
COPY . .

# Make sure submodules are initialized
RUN git submodule update --init --recursive --depth 1

################################################################################
### Install mlir-dace
################################################################################

WORKDIR $HOME/mlir-dace

################################################################################
### Install torch-mlir
################################################################################
