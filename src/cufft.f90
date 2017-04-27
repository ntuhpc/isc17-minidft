!-------------------------------------------
! The interface between Fortran 90 and CUFFT
!-------------------------------------------
MODULE cufft

  INTEGER, PUBLIC :: CUFFT_FORWARD = -1 
  INTEGER, PUBLIC :: CUFFT_INVERSE = 1 
  INTEGER, PUBLIC :: CUFFT_R2C = Z'2a' ! Real to Complex (interleaved) 
  INTEGER, PUBLIC :: CUFFT_C2R = Z'2c' ! Complex (interleaved) to Real 
  INTEGER, PUBLIC :: CUFFT_C2C = Z'29' ! Complex to Complex, interleaved 
  INTEGER, PUBLIC :: CUFFT_D2Z = Z'6a' ! Double to Double-Complex 
  INTEGER, PUBLIC :: CUFFT_Z2D = Z'6c' ! Double-Complex to Double 
  INTEGER, PUBLIC :: CUFFT_Z2Z = Z'69' ! Double-Complex to Double-Complex

  !-----------------------------------------------------------------------------
  ! cufftResult 
  !   cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed,
  !     int istride, int idist, int *onembed, int ostride,
  !     int odist, cufftType type, int batch, size_t *workSize);
  !-----------------------------------------------------------------------------
  INTERFACE cufftMakePlanMany
    SUBROUTINE cufftMakePlanMany(plan, rank, n, inembed, &
        istride, idist, onembed, ostride, &
        odist, type, batch, workSize) BIND(C, name='cufftMakePlanMany')
      USE iso_c_binding
      INTEGER(C_INT) :: n, inembed, onembed
      INTEGER(C_INT), VALUE :: plan, rank, istride, idist, ostride, odist, type, batch
      INTEGER(C_SIZE_T) :: workSize
    END SUBROUTINE cufftMakePlanMany
  END INTERFACE cufftMakePlanMany

  !---------------------------------------------
  ! cufftResult 
  !   cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata, 
  !     cufftDoubleComplex *odata, int direction);
  !---------------------------------------------
  INTERFACE cufftExecZ2Z
    SUBROUTINE cufftExecZ2Z(plan, in, out, direction) BIND(C, name='cufftExecZ2Z')
      USE iso_c_binding
      INTEGER(C_INT), VALUE :: plan, direction
      COMPLEX(DP), DEVICE :: in(*), out(*)
    END SUBROUTINE cufftExecZ2Z
  END INTERFACE cufftExecZ2Z

  INTERFACE cufftDestroy 
    SUBROUTINE cufftDestroy(plan) BIND(C, name='cufftDestroy') 
      USE iso_c_binding 
      INTEGER(C_INT), VALUE :: plan 
    END SUBROUTINE cufftDestroy 
  END INTERFACE cufftDestroy

END MODULE cufft