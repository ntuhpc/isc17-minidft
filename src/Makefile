MKLROOT   = /opt/intel/mkl

FFTW_LIBS = ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a \
            -Wl,--start-group \
              ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
              ${MKLROOT}/lib/intel64/libmkl_sequential.a \
              ${MKLROOT}/lib/intel64/libmkl_core.a \
              ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a \
            -Wl,--end-group -lpthread -lm -ldl

DFLAGS = -D__INTEL -D__FFTW -D_SCALAPACK -D__NONBLOCKING_FFT

CC = mpiicc
CFLAGS = -O2 -I./

FC = mpiifort
FFLAGS = -O3 -fpp -I./

LD = mpiifort

#ifort -openmp defines _OPENMP instead of__OPENMP
#DFLAGS    += -D__OPENMP
#
#FFLAGS    += -qopenmp
#LDFLAGS   += -qopenmp

include Makefile.base
