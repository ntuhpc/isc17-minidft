MKLROOT   = /opt/intel/mkl
#MKLROOT   = ${MKLROOT}

#FFTW_INCL = -I/${MKLROOT}/include/fftw/
#FFTW_LIBS =  -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl
#FFTW_LIBS = -lfftw3 -lopenblas -lscalapack
#FFTW_LIBS = $(MKLROOT)/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
               $(MKLROOT)/lib/intel64/libmkl_intel_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_blacs_intelmpi_lp64.a \
               -Wl,--end-group -lpthread -lm ${IPM}
FFTW_LIBS = ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
			${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a \
			-Wl,--end-group -liomp5 -lpthread -lm -ldl

DFLAGS = -D__INTEL -D__FFTW -D_SCALAPACK

CC = mpiicc
CFLAGS = -O3 -I./ -I${MKLROOT}/include

#FC = mpif90
FC = mpiifort
FFLAGS = -O3 -fpp -I./ -I${MKLROOT}/include
#FFLAGS = -O3 -cpp -I./

#LD = mpif90
LD = mpiifort

#ifort -openmp defines _OPENMP instead of__OPENMP
DFLAGS    += -D__OPENMP -D__CUDA

FFLAGS    += -qopenmp
LDFLAGS   += -qopenmp -lcudart -lcublas

include Makefile.base
