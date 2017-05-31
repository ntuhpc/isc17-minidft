# ISC17 coding challenge - miniDFT

This is the version optimized by SCC team from Nanyang Technological University.

## Compilation

Go into folder `src`

```bash
$ module load CUDA OpenMPI
$ source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
$ make # make sure you have fortran_thunking.o
```

Macros

- `__CUDA` enables CUDA based optimization
- `__MAGMA` enables MAGMA to solve diagonalization
- `__CUBLAS` enables cuBLAS to solve ZGEMM
- `__NONBLOCKING_FFT` enables non-blocking fft_scatter
- `__ZHEGVD` enables MAGMA call to magmaf_zhegvd

## Usage

Go into folder `benchmark`

```bash
$ mpirun -np 88 -ppn 44 -hosts compute0,compute1 bash run.sh
```
