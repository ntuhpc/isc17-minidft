#!/bin/bash

rank=$((${PMI_RANK} % 4))
case ${rank} in
[0]) CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 ./mini_dft -in pe-23.LOCAL.in;;
[1]) CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 ./mini_dft -in pe-23.LOCAL.in;;
[2]) CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 ./mini_dft -in pe-23.LOCAL.in;;
[3]) CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 ./mini_dft -in pe-23.LOCAL.in;
esac
