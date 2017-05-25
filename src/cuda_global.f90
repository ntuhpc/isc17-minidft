MODULE cuda_global

    USE mp_global, ONLY : mpime
    USE cudafor

    IMPLICIT NONE

    CONTAINS
        SUBROUTINE cuda_startup ( )
            INTEGER :: num_of_gpu, cuda_err
            cuda_err = cudaGetDeviceCount( num_of_gpu )
            cuda_err = cudaSetDevice( mod(mpime, num_of_gpu) )
        END SUBROUTINE cuda_startup

END MODULE cuda_global
