# macros
DFLAGS = -D__INTEL -D__FFTW -D_SCALAPACK -D__MPI #-D__OPENMP -D__MPI #-D__NONBLOCKING_FFT
#DFLAGS = -D__INTEL -D_SCALAPACK -D__OPENMP -D__MPI #-D__NONBLOCKING_FFT

# compiler specific settings
comp = pgi
include make.$(comp)

# rules
arch = gpu
include Makefile.$(arch)
