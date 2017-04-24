
! Copyright (C) 2002-2009 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!----------------------------------------------------------------------------
SUBROUTINE h_psi_gpu( lda, n, m, psi_d, hpsi_d )
  !----------------------------------------------------------------------------
  !
  ! ... This routine computes the product of the Hamiltonian
  ! ... matrix with m wavefunctions contained in psi
  !
  ! ... input:
  ! ...    lda   leading dimension of arrays psi, spsi, hpsi
  ! ...    n     true dimension of psi, spsi, hpsi
  ! ...    m     number of states psi
  ! ...    psi_d
  !
  ! ... output:
  ! ...    hpsi_d  H*psi_d
  !
  USE kinds,    ONLY : DP
  USE lsda_mod, ONLY : current_spin
  USE scf,      ONLY : vrs, vrs_d
  USE wvfct,    ONLY : npwx, g2kin
  USE uspp,     ONLY : vkb, nkb
  USE gvect,    ONLY : gstart
  USE fft_base, ONLY : dffts
  USE exx,      ONLY : vexx
  USE funct,    ONLY : exx_is_active
  USE cudafor
  !
  IMPLICIT NONE
  !
  integer, parameter :: npol=1 !substitute for noncollin_module%npol
  INTEGER, INTENT(IN)     :: lda, n, m
  COMPLEX(DP), INTENT(IN), DEVICE  :: psi_d(lda*npol,m) 
  COMPLEX(DP), INTENT(OUT), DEVICE :: hpsi_d(lda*npol,m)
  ! TODO: consider moving it elsewhere?
  REAL(DP), ALLOCATABLE, DEVICE :: g2kin_d(:)
  ALLOCATE( g2kin_d( npwx ) )
  g2kin_d = g2kin
  !
  INTEGER     :: ipol, ibnd, incr
  !
  CALL start_clock( 'h_psi' )
  !  
  ! ... Here we apply the kinetic energy (k+G)^2 psi
  !
  !$cuf kernel <<<*,*>>>
  DO ibnd = 1, m
     hpsi_d (1:n, ibnd) = g2kin_d (1:n) * psi_d (1:n, ibnd)
     hpsi_d (n+1:lda,ibnd) = (0.0_dp, 0.0_dp)
  END DO
  ! TODO: if the above ALLOCATE is moved, move this too
  DEALLOCATE( g2kin_d )
  !
  !
  !
  ! ... the local potential V_Loc psi
  !
  CALL start_clock( 'h_psi:vloc' )
  !
  vrs_d(1, current_spin) = vrs(1, current_spin)
  CALL vloc_psi_k_gpu ( lda, n, m, psi_d, vrs_d(1,current_spin), hpsi_d )
  vrs(1, current_spin) = vrs_d(1, current_spin)
  !
  CALL stop_clock( 'h_psi:vloc' )
  !
  ! ... Here the product with the non local potential V_NL psi
  ! ... (not in the real-space case: it is done together with V_loc)
  !
     !
     CALL start_clock( 'h_psi:vnl' )
     ! JRD: calbec done in add_vuspsi now
     CALL add_vuspsi_gpu( lda, n, m, psi_d, hpsi_d )
     CALL stop_clock( 'h_psi:vnl' )
     !
!JRD
  IF ( exx_is_active() ) CALL vexx_gpu( lda, n, m, psi_d, hpsi_d )
  !
  ! ... electric enthalpy if required
  !
  !
  CALL stop_clock( 'h_psi' )
  !
  RETURN
  !
END SUBROUTINE h_psi_gpu
