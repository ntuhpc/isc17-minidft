!-----------------------------------------------------------------------
SUBROUTINE vloc_psi_k_gpu(lda, n, m, psi_d, v, hpsi_d)
  !-----------------------------------------------------------------------
  !
  ! Calculation of Vloc*psi using dual-space technique - k-points
  !
  USE parallel_include
  USE kinds,   ONLY : DP
  USE gvecs, ONLY : nls, nlsm
  USE wvfct,   ONLY : igk
  USE mp_global,     ONLY : me_pool, me_bgrp
  USE fft_base,      ONLY : dffts, tg_gather
  USE fft_interfaces_gpu ,ONLY : fwfft_gpu, invfft_gpu
#if defined(__CUDA) && defined(__CUFFT)
  USE wavefunctions_module,  ONLY: psic_d, psic
#else
  USE wavefunctions_module,  ONLY: psic
#endif
  !
  IMPLICIT NONE
  !
  INTEGER, INTENT(in) :: lda, n, m
  COMPLEX(DP), INTENT(in), DEVICE    :: psi_d(lda, m)
  COMPLEX(DP), INTENT(inout), DEVICE :: hpsi_d(lda, m)
  REAL(DP), INTENT(in), DEVICE :: v(dffts%nnr)
  !
  INTEGER :: ibnd, j, incr
  !
  LOGICAL :: use_tg
  ! Task Groups
  REAL(DP),    ALLOCATABLE :: tg_v(:)
  COMPLEX(DP), ALLOCATABLE :: tg_psic(:)
  INTEGER :: v_siz, idx, ioff
  !
  !
  ! The following is dirty trick to prevent usage of task groups if
  ! the number of bands is smaller than the number of task groups 
  ! 
  use_tg = dffts%have_task_groups
  dffts%have_task_groups  = dffts%have_task_groups .and. ( m >= dffts%nogrp )
  !
  incr = 1
  !
!   IF( dffts%have_task_groups ) THEN
!      !
!      v_siz =  dffts%tg_nnr * dffts%nogrp
!      !
!      ALLOCATE( tg_v   ( v_siz ) )
!      ALLOCATE( tg_psic( v_siz ) )
!      !
!      CALL tg_gather( dffts, v, tg_v )
!      incr = dffts%nogrp
!      !
!   ENDIF
  !
  ! the local potential V_Loc psi. First bring psi to real space
  !
  DO ibnd = 1, m, incr
     !
     IF ( dffts%have_task_groups ) THEN
        !
!         tg_psic = (0.d0, 0.d0)
!         ioff   = 0
!         !
!         DO idx = 1, dffts%nogrp

!            IF( idx + ibnd - 1 <= m ) THEN
! !$omp parallel do
!               DO j = 1, n
!                  tg_psic(nls (igk(j))+ioff) =  psi_d(j,idx+ibnd-1)
!               ENDDO
! !$omp end parallel do
!            ENDIF

!            ioff = ioff + dffts%tg_nnr

!         ENDDO
!         !
!         CALL  invfft ('Wave', tg_psic, dffts)
        !
     ELSE
        !
        psic_d(:) = (0.d0, 0.d0)
        !$cuf kernel do <<<*,*>>>
        DO j = 1, n
          psic_d(nls (igk(j))) = psi_d(j, ibnd)
        END DO
        !
        CALL invfft ('Wave', psic_d, dffts)
        !
     ENDIF
     !
     !   fft to real space
     !   product with the potential v on the smooth grid
     !   back to reciprocal space
     !
     IF ( dffts%have_task_groups ) THEN
!         !
! !$omp parallel do
!         DO j = 1, dffts%nr1x*dffts%nr2x*dffts%tg_npp( me_bgrp + 1 )
!            tg_psic (j) = tg_psic (j) * tg_v(j)
!         ENDDO
! !$omp end parallel do
!         !
!         CALL fwfft ('Wave',  tg_psic, dffts)
        !
     ELSE
        !
        !$cuf kernel do <<<*,*>>>
        DO j = 1, dffts%nnr
           psic_d (j) = psic_d (j) * v(j)
        ENDDO
        !
        CALL fwfft ('Wave', psic_d, dffts)
        !
     ENDIF
     !
     !   addition to the total product
     !
     IF ( dffts%have_task_groups ) THEN
        !
!         ioff   = 0
!         !
!         DO idx = 1, dffts%nogrp
!            !
!            IF( idx + ibnd - 1 <= m ) THEN
! !$omp parallel do
!               DO j = 1, n
!                  hpsi (j, ibnd+idx-1) = hpsi (j, ibnd+idx-1) + tg_psic( nls(igk(j)) + ioff )
!               ENDDO
! !$omp end parallel do
!            ENDIF
!            !
!            ioff = ioff + dffts%nr3x * dffts%nsw( me_bgrp + 1 )
!            !
!         ENDDO
        !
     ELSE
         !$cuf kernel do <<<*,*>>>
        DO j = 1, n
           hpsi_d (j, ibnd)   = hpsi_d (j, ibnd)   + psic_d (nls(igk(j)))
        ENDDO
     ENDIF
     !
  ENDDO
  !
  IF( dffts%have_task_groups ) THEN
     !
    !  DEALLOCATE( tg_psic )
    !  DEALLOCATE( tg_v )
     !
  ENDIF
  dffts%have_task_groups = use_tg
  !
  RETURN
END SUBROUTINE vloc_psi_k_gpu
