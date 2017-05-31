MODULE cuda_global

    USE mp_global, ONLY : mpime
#if defined(__CUDA) && defined(__PGI)
    USE cudafor
#endif

    IMPLICIT NONE

    CONTAINS
        SUBROUTINE cuda_startup ( )
            INTEGER :: num_of_gpu, cuda_err
#if defined(__CUDA) && defined(__PGI)
            cuda_err = cudaGetDeviceCount( num_of_gpu )
            cuda_err = cudaSetDevice( mod(mpime, num_of_gpu) )
#elif defined(__CUDA) && defined(__INTEL)
            CALL cuInit()
            CALL cudaGetDeviceCount( num_of_gpu )
            CALL cudaSetDevice( mod(mpime, num_of_gpu) )
            !CALL cudaFree(0)
#endif
        END SUBROUTINE cuda_startup

END MODULE cuda_global
