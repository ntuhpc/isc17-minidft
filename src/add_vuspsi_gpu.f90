!
! Copyright (C) 2001-2003 PWSCF group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
!----------------------------------------------------------------------------
SUBROUTINE add_vuspsi_gpu( lda, n, m, psi, hpsi )
  !----------------------------------------------------------------------------
  !
  !    This routine applies the Ultra-Soft Hamiltonian to a
  !    vector psi and puts the result in hpsi.
  !    Requires the products of psi with all beta functions
  !    in array becp(nkb,m) (calculated by calbec)
  ! input:
  !     lda   leading dimension of arrays psi, spsi
  !     n     true dimension of psi, spsi
  !     m     number of states psi
  ! output:
  !     hpsi  V_US|psi> is added to hpsi
  !
  USE kinds,         ONLY : DP
  USE ions_base,     ONLY : nat, ntyp => nsp, ityp
  USE lsda_mod,      ONLY : current_spin
  USE uspp,          ONLY : vkb, vkb_d, nkb, deeq, deeq_nc
  USE uspp_param,    ONLY : nh
  USE becmod,        ONLY : bec_type, becp, calbec
  !
  IMPLICIT NONE
  !
  ! ... I/O variables
  !
  integer, parameter :: npol=1 !subsitute for noncollin_module%npol
  INTEGER, INTENT(IN)  :: lda, n, m
  COMPLEX(DP), INTENT(IN) :: psi(lda*npol,m)
  COMPLEX(DP), INTENT(INOUT) :: hpsi(lda*npol,m)  
  !
  ! ... here the local variables
  !
  COMPLEX(DP), ALLOCATABLE, DEVICE :: psi_d(:,:)
  INTEGER :: jkb, ikb, ih, jh, na, nt, ijkb0, ibnd
    ! counters
  !
  !
  CALL start_clock( 'add_vuspsi' )  
  !
     CALL add_vuspsi_k_gpu()
  !
  CALL stop_clock( 'add_vuspsi' )  
  !
  RETURN
  !
  CONTAINS
     !
     !-----------------------------------------------------------------------
     !
     !-----------------------------------------------------------------------
     SUBROUTINE add_vuspsi_k_gpu()
       !-----------------------------------------------------------------------
       !
       IMPLICIT NONE
       COMPLEX(DP), ALLOCATABLE :: ps (:,:)
       COMPLEX(DP), ALLOCATABLE, DEVICE :: becpk_d(:)
       INTEGER :: ierr
       !
       IF ( nkb == 0 ) RETURN
       !
       ALLOCATE (ps (nkb,m), STAT=ierr )
       IF( ierr /= 0 ) &
          CALL errore( ' add_vuspsi_k_gpu ', ' cannot allocate ps ', ABS( ierr ) )
       ps(:,:) = ( 0.D0, 0.D0 )
       !
       ijkb0 = 0
       !

       ! these only need to be copied once
       ALLOCATE(psi_d(lda*npol,m))
       vkb_d = vkb
       psi_d = psi
       ALLOCATE (becpk_d (nkb))
       becpk_d(:) = (0.0D0,0.0D0)
       DO ibnd = 1, m

          ! JRD: Compute becp for just this ibnd here
          !becpk_d = becp%k
          CALL calbec ( n, vkb_d, psi_d, becpk_d, ibnd )
          becp%k = becpk_d
          !write(*,*) 'Computing becp', ibnd

          ijkb0 = 0

          DO nt = 1, ntyp
             DO na = 1, nat
                IF ( ityp(na) == nt ) THEN

                   DO jh = 1, nh(nt)
                      jkb = ijkb0 + jh

                      DO ih = 1, nh(nt)
                         ikb = ijkb0 + ih
                         ps(ikb,ibnd) = ps(ikb,ibnd) + &
                              deeq(ih,jh,na,current_spin) * becp%k(jkb)
                      END DO
                   END DO
                   ijkb0 = ijkb0 + nh(nt)
                END IF
             END DO
          END DO
          !
       END DO
       PRINT *,becp%k(1:10)
       !
       ! IMPROVED GEMM
       IF ( m == 1 ) THEN
          !
          CALL ZGEMV( 'N', n, nkb, ( 1.D0, 0.D0 ), vkb, lda, ps, 1, &
             ( 1.D0, 0.D0 ), hpsi, 1 )
          !
       ELSE
          !
          CALL ZGEMM( 'N', 'N', n, m, nkb, ( 1.D0, 0.D0 ) , vkb, &
                   lda, ps, nkb, ( 1.D0, 0.D0 ) , hpsi, lda )
          !
       ENDIF
       ! ORIGINAL GEMM
       !CALL ZGEMM( 'N', 'N', n, m, nkb, ( 1.D0, 0.D0 ) , vkb, &
       !            lda, ps, nkb, ( 1.D0, 0.D0 ) , hpsi, lda )
       !
       DEALLOCATE (ps)
       DEALLOCATE (becpk_d)
       !
       RETURN
       !
     END SUBROUTINE add_vuspsi_k_gpu
     !  
     !-----------------------------------------------------------------------
     !  
     !  
END SUBROUTINE add_vuspsi_gpu
