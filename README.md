# Setup

## Check out the code
```sh
git clone [--recurse-submodules] --depth 1 --shallow-submodules <URL>
cd dcir-artifact
```

## Docker
### Build the container
```sh
sudo docker build -t dcir-cgo23 .
```
### Run the container
```sh
sudo docker run -it --rm dcir-cgo23
```

## Manual Setup
For a manual setup follow the instructions in the `Dockerfile` on your machine.

# Requirements
Running and building the Docker container requires an installation of docker
and a running instance of the Docker daemon.

The requirements for a manual setup are listed in the `Dockerfile`.

# Running

# File Structure
