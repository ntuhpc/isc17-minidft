MODULE cuda_global

    USE mp_global, ONLY : mpime

    IMPLICIT NONE

    CONTAINS
        SUBROUTINE cuda_startup ( )
            CALL cudaSetDevice( mpime )
        END SUBROUTINE cuda_startup

END MODULE cuda_global
