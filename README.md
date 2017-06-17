# ISC17 coding challenge - miniDFT

This is a version optimized by SCC team from Nanyang Technological University for the ISC17 Student Cluster Competition.

## Contributors

- [Liu Siyuan](https://github.com/koallen)

## Acknowledgement

Thanks [Shao Yiyang](https://github.com/Allen-Shao) and [Lu Shengliang](https://github.com/lushl9301) for helping me solve bugs when optimizing and porting the code.

## Compilation

Make sure you have the following dependencies

- MAGMA (without OpenMP)
- Intel compilers and MPI
- CUDA (with Fortran thunking cuBLAS interface)
- Nvidia MPS server

Go into folder `src`

```bash
$ module load CUDA OpenMPI
$ source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
$ make comp=intel # make sure you have fortran_thunking.o
```

Optimization macros

- `__CUDA` enables CUDA based optimization
- `__MAGMA` enables MAGMA to solve diagonalization
- `__CUBLAS` enables cuBLAS to solve ZGEMM
- `__NONBLOCKING_FFT` enables non-blocking fft_scatter
- `__ZHEGVD` enables MAGMA call to magmaf_zhegvd

### Known issues

- OpenMP seems to be problematic, please disable OpenMP.

## Usage

Go into folder `benchmark`

```bash
$ sudo nvidia-smi -c 3
$ sudo nvidia-cuda-mps-control -d
$ mpirun -np 88 -ppn 44 -hosts compute0,compute1 bash run.sh
```
