#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
mkdir -p /tmp/mps-pipe
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe
nvidia-smi -c 3
nvidia-cuda-mps-control -d
