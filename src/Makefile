# math libraries
MKLROOT   = /opt/intel/mkl

#MATH_LIBS = ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
#			${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a \
#			-Wl,--end-group -liomp5 -lpthread -lm -ldl
MATH_LIBS =  ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group -lpthread -lm -ldl

# macros
DFLAGS = -D__INTEL -D__FFTW -D_SCALAPACK -D__OPENMP -D__MPI

# compilers and flags
comp = intel
include make.$(comp)

# rules
LIBS = $(MATH_LIBS)
include Makefile.base
