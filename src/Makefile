# math libraries
MKLROOT   = /opt/intel/mkl

#FFTW_INCL = -I/${MKLROOT}/include/fftw/
#FFTW_LIBS = -lfftw3 -lopenblas -lscalapack
FFTW_LIBS = ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
			${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a \
			-Wl,--end-group -liomp5 -lpthread -lm -ldl

DFLAGS = -D__INTEL -D__FFTW -D_SCALAPACK -D__OPENMP -D__MPI -D__CUDA

# C
CC = mpiicc
CFLAGS = -O3 -I./ -I${MKLROOT}/include

# Fortran
FC = mpiifort
FFLAGS = -O3 -fpp -I./ -I${MKLROOT}/include -qopenmp

# Linker
LD = mpiifort
LDFLAGS = -qopenmp -lcudart -lcublas

include Makefile.base
