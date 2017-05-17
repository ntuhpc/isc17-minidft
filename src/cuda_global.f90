MODULE cuda_global

    USE mp_global, ONLY : mpime

    IMPLICIT NONE

    CONTAINS
        SUBROUTINE cuda_startup ( )
            INTEGER :: deviceToBond(0:1)
            deviceToBond = (/ mpime /)
            CALL cudaSetDevice( mpime )
            ! initialize phiGEMM
            CALL phiGemmInit( 1, 0, 0, deviceToBond, 0 )
        END SUBROUTINE cuda_startup

END MODULE cuda_global
