MODULE cuda_global

    USE mp_global, ONLY : mpime

    IMPLICIT NONE

    CONTAINS
        SUBROUTINE cuda_startup ( )
            INTEGER :: num_of_gpu, cuda_err
            cuda_err = cudaGetDeviceCount( num_of_gpu )
            CALL cudaSetDevice( mod(mpime, num_of_gpu) )
            CALL magmaf_init()
        END SUBROUTINE cuda_startup

END MODULE cuda_global
