#!/bin/bash

module load pgi64 openmpi/1.10.2 GCC
source /home/public/minidft/source.sh

OMP_NUM_THREADS=11 ./mini_dft -in pe-23.LOCAL.in -nbgrp 2
