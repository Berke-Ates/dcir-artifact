# DCIR Artifact
This repository contains the artifact for "Bridging Control-Centric and Data-Centric Optimization", submitted to CGO'23.

# Requirements
Running and building the Docker container requires an installation of docker
and a running instance of the Docker daemon.

The requirements for a manual setup are listed in the `Dockerfile`.

# Setup
There are three options to run the benchmarks (`sudo` is not always necessary):

## Option 1: Pull the docker image
Pull the docker image with:
```sh
sudo docker pull berkeates/dcir-cgo23:latest
```

And run it:
```sh
sudo docker run -it --rm berkeates/dcir-cgo23
```

## Option 2: Manually build the docker image
### Check out the code
```sh
git clone [--recurse-submodules] --depth 1 --shallow-submodules https://github.com/Berke-Ates/dcir-artifact
cd dcir-artifact
```
### Build the image
```sh
sudo docker build -t dcir-cgo23 .
```
### Run the container
```sh
sudo docker run -it --rm dcir-cgo23
```

## Option 3: Manual Setup
For a manual setup follow the instructions in the `Dockerfile` on your machine.

# Running
To run all benchmarks execute the following script _inside the docker container_:
```sh
./scripts/run_all.sh <Output directory> <Number of repetitions>
```

The outputs will then be in CSV format in the output directory, organized by
figure number or result name. The results are also plotted to PDF files.

If running in a container, you can mount a local folder to view the results outside it.
To do so, use the `-v` flag when running the container, as follows:
```sh
sudo docker run -it --rm -v <local folder>:<container folder> berkeates/dcir-cgo23
```

For example:
```
$ mkdir results
$ sudo docker run -it --rm -v /path/to/local/results:/home/user/ae berkeates/dcir-cgo23
# ./scripts/run_all.sh ae 10
```

The raw results from the original paper can be found in the `output` subdirectory.

To run a single benchmark use the runners in the subdirectories:
- `scripts/Polybench`: For Polybench benchmarks
- `scripts/pytorch`: For PyTorch benchmarks
- `scripts/snippets`: For snippet benchmarks

An example command can be seen in this line:
```sh
./scripts/Polybench/dcir.sh ./benchmarks/Polybench/2mm/2mm.c <Output directory> <Number of repetitions>
```

The subdirectories also contain a `run_all.sh`, which executes all benchmarks of their group.

# File Structure
- `benchmarks`: Folder containing all benchmarks files
  - `Polybench`: Folder containing all Polybench benchmarks. The subfolders contain the edited C benchmarks and the generated SDFG files for the DaCe benchmarks.
  - `pytorch`: Folder containing all PyTorch benchmarks.
  - `snippets`: Folder containing all snippet benchmarks. The subfolders contain the adjusted C benchmarks, the C benchmarks with timer attached (suffix: `-chrono.c`) and the generated SDFG files for the DaCe benchmarks.
- `dace`: Folder containing the DaCe project
- `mlir-dace`: Folder containing the MLIR-DaCe project
- `mlir-hlo`: Folder containing the MLIR-HLO project
- `output`: Raw outputs from the paper
- `polybench-comparator`: Folder containing a python script to compare Polybench outputs
- `Polygeist`: Folder containing the Polygeist project
- `scripts`: Folder containing all scripts to run the benchmarks. Every script prints its usage, if it's called without arguments
  - `Polybench`: Folder containing Polybench specific run scripts
  - `pytorch`: Folder containing PyTorch specific run scripts
  - `snippets`: Folder containing snippets specific run scripts
  - `get_sdfg_times.py`: Script retrieving the runtime of a specific SDFG run
  - `mhlo2std.sh`: Script to convert the MHLO dialect to builtin dialects
  - `multi_plot.py`: Script to plot multiple CSV files
  - `opt_sdfg.py`: Script to optimize a SDFG file
  - `run_all.sh`: Script to run all benchmarks and generate the plots
  - `single_plot.py`: Script to plot a single CSV file
- `torch-mlir`: Folder containing the Torch-MLIR project
- `Dockerfile`: Dockerfile to build the image
- `README.md`: This file
