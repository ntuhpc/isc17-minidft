#if defined(__CUDA) && defined(__CUFFT) && defined(__PGI)
!
! Copyright (C) 2001-2012 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!--------------------------------------------------------------------------!
! FFT scalar drivers Module - contains machine-dependent routines for:     !
! FFTW, FFTW3, ESSL, LINUX_ESSL, SCSL, SUNPERF, NEC ASL libraries          !
! (both 3d for serial execution and 1d+2d FFTs for parallel execution,     !
! excepted NEC ASL, 3d only, no parallel execution)                        !
! Written by Carlo Cavazzoni, modified by P. Giannozzi, contributions      !
! by Martin Hilgemans, Guido Roma, Pascal Thibaudeau, Stephane Lefranc,    !
! Nicolas Lacorne, Filippo Spiga - Last update Aug 2012                    !
!--------------------------------------------------------------------------!

#include "fft_defs.h"
!=----------------------------------------------------------------------=!
   MODULE fft_scalar_gpu
!=----------------------------------------------------------------------=!
       USE kinds
       USE cufft
       USE iso_c_binding

        IMPLICIT NONE
        SAVE

        PRIVATE
        PUBLIC :: cft_1z_gpu, cft_2xy_gpu
        PUBLIC :: good_fft_dimension, allowed, good_fft_order
        ! PUBLIC :: cft_b

! ...   Local Parameter

        !   ndims   Number of different FFT tables that the module
        !           could keep into memory without reinitialization
        !   nfftx   Max allowed fft dimension

        INTEGER, PARAMETER :: ndims = 3, nfftx = 2049

        !   Workspace that is statically allocated is defined here
        !   in order to avoid multiple copies of the same workspace
        !   lwork:   Dimension of the work space array (if any)


        !  Only FFTW_ESTIMATE is actually used

#define  FFTW_MEASURE  0
#define  FFTW_ESTIMATE 64



!=----------------------------------------------------------------------=!
   CONTAINS
!=----------------------------------------------------------------------=!

!
!=----------------------------------------------------------------------=!
!
!
!
!         FFT along "z"
!
!
!
!=----------------------------------------------------------------------=!
!

   SUBROUTINE cft_1z_gpu(c, nsl, nz, ldz, isign, cout)

!     driver routine for nsl 1d complex fft's of length nz
!     ldz >= nz is the distance between sequences to be transformed
!     (ldz>nz is used on some architectures to reduce memory conflicts)
!     input  :  c(ldz*nsl)   (complex)
!     output : cout(ldz*nsl) (complex - NOTA BENE: transform is not in-place!)
!     isign > 0 : forward (f(G)=>f(R)), isign <0 backward (f(R) => f(G))
!     Up to "ndims" initializations (for different combinations of input
!     parameters nz, nsl, ldz) are stored and re-used if available

     INTEGER, INTENT(IN) :: isign
     INTEGER, INTENT(IN) :: nsl, nz, ldz

     COMPLEX (DP), DEVICE :: c(:), cout(:)
     COMPLEX (DP), ALLOCATABLE :: test(:)

     REAL (DP)  :: tscale
     INTEGER    :: i, err, idir, ip
     INTEGER, SAVE :: zdims( 3, ndims ) = -1
     INTEGER, SAVE :: icurrent = 1
     LOGICAL :: done

#if defined __HPM
     INTEGER :: OMP_GET_THREAD_NUM
#endif
     INTEGER :: tid

     ! ...   Machine-Dependent parameters, work arrays and tables of factors

     !   ltabl   Dimension of the tables of factors calculated at the
     !           initialization stage

#if defined __OPENMP
     INTEGER :: offset, ldz_t
     INTEGER :: omp_get_max_threads
     EXTERNAL :: omp_get_max_threads
#endif



     !   Pointers to the "C" structures containing FFT factors ( PLAN )
     !   C_POINTER is defined in include/fft_defs.h
     !   for 32bit executables, C_POINTER is integer(4)
     !   for 64bit executables, C_POINTER is integer(8)

     INTEGER, SAVE :: fw_planz( ndims ) = 0
     INTEGER, SAVE :: bw_planz( ndims ) = 0


     IF( nsl < 0 ) THEN
       CALL errore(" fft_scalar: cft_1z ", " nsl out of range ", nsl)
     END IF

     !
     !   Here initialize table only if necessary
     !

     DO ip = 1, ndims

        !   first check if there is already a table initialized
        !   for this combination of parameters

        done = ( nz == zdims(1,ip) )

        done = done .AND. ( nsl == zdims(2,ip) ) .AND. ( ldz == zdims(3,ip) )
        IF (done) EXIT
     END DO

     IF( .NOT. done ) THEN

       !   no table exist for these parameters
       !   initialize a new one

       ! WRITE( stdout, fmt="('DEBUG cft_1z, reinitializing tables ', I3)" ) icurrent



       IF( fw_planz( icurrent) /= 0 ) CALL cufftDestroy( fw_planz( icurrent) )
       IF( bw_planz( icurrent) /= 0 ) CALL cufftDestroy( bw_planz( icurrent) )
       idir = -1
       !WRITE(*,*) nz, SIZE(c), SIZE(cout), ldz, nsl
       CALL cufftPlanMany( fw_planz( icurrent), 1, (/nz/), (/SIZE(c)/), 1, ldz, &
            (/SIZE(cout)/), 1, ldz, CUFFT_Z2Z, nsl )
       ! 1 = rank
       ! nz = n
       ! nsl = howmany
       ! c = in
       ! (/SIZE(c)/) = inembed
       ! 1 = istride
       ! ldz = idist
       ! cout = out
       ! (/SIZE(cout)/) = onembed
       ! 1 = ostride
       ! ldz = odist
       idir = 1
       CALL cufftPlanMany( bw_planz( icurrent), 1, (/nz/), (/SIZE(c)/), 1, ldz, &
            (/SIZE(cout)/), 1, ldz, CUFFT_Z2Z, nsl )

       zdims(1,icurrent) = nz; zdims(2,icurrent) = nsl; zdims(3,icurrent) = ldz;
       ip = icurrent
       icurrent = MOD( icurrent, ndims ) + 1

     END IF

     !
     !   Now perform the FFTs using machine specific drivers
     !
#if defined __FFT_CLOCKS
     CALL start_clock( 'cft_1z' )
#endif


     IF (isign < 0) THEN
        CALL cufftExecZ2Z( fw_planz( ip), c, cout, CUFFT_FORWARD )
        tscale = 1.0_DP / nz
        !$cuf kernel do <<<*,*>>>
        DO i = 1, ldz * nsl
          cout( i ) = cout( i ) * tscale
        END DO
     ELSE IF (isign > 0) THEN
        CALL cufftExecZ2Z( bw_planz( ip), c, cout, CUFFT_INVERSE )
        !ALLOCATE(test(SIZE(cout)))
        !test = cout
        !WRITE(*,*) test(:10)
     END IF


#if defined __FFT_CLOCKS
     CALL stop_clock( 'cft_1z' )
#endif

     RETURN
   END SUBROUTINE cft_1z_gpu

!
!
!=----------------------------------------------------------------------=!
!
!
!
!         FFT along "x" and "y" direction
!
!
!
!=----------------------------------------------------------------------=!
!
!

   SUBROUTINE cft_2xy_gpu(r, nzl, nx, ny, ldx, ldy, isign, pl2ix)

!     driver routine for nzl 2d complex fft's of lengths nx and ny
!     input : r(ldx*ldy)  complex, transform is in-place
!     ldx >= nx, ldy >= ny are the physical dimensions of the equivalent
!     2d array: r2d(ldx, ldy) (x first dimension, y second dimension)
!     (ldx>nx, ldy>ny used on some architectures to reduce memory conflicts)
!     pl2ix(nx) (optional) is 1 for columns along y to be transformed
!     isign > 0 : forward (f(G)=>f(R)), isign <0 backward (f(R) => f(G))
!     Up to "ndims" initializations (for different combinations of input
!     parameters nx,ny,nzl,ldx) are stored and re-used if available

     IMPLICIT NONE

     INTEGER, INTENT(IN) :: isign, ldx, ldy, nx, ny, nzl
     INTEGER, OPTIONAL, INTENT(IN) :: pl2ix(:)
     COMPLEX (DP), DEVICE :: r( : )
     INTEGER :: i, k, j, err, idir, ip, kk
     REAL(DP) :: tscale
     INTEGER, SAVE :: icurrent = 1
     INTEGER, SAVE :: dims( 4, ndims) = -1
     LOGICAL :: dofft( nfftx ), done
     INTEGER, PARAMETER  :: stdout = 6

#if defined __HPM
     INTEGER :: OMP_GET_THREAD_NUM
#endif
#if defined __OPENMP
     INTEGER :: offset
     INTEGER :: nx_t, ny_t, nzl_t, ldx_t, ldy_t
     INTEGER  :: itid, mytid, ntids
     INTEGER  :: omp_get_thread_num, omp_get_num_threads
     EXTERNAL :: omp_get_thread_num, omp_get_num_threads
#endif

     INTEGER, SAVE :: fw_plan( 2, ndims ) = 0
     INTEGER, SAVE :: bw_plan( 2, ndims ) = 0


     dofft( 1 : nx ) = .TRUE.
     IF( PRESENT( pl2ix ) ) THEN
       IF( SIZE( pl2ix ) < nx ) &
         CALL errore( ' cft_2xy ', ' wrong dimension for arg no. 8 ', 1 )
       DO i = 1, nx
         IF( pl2ix(i) < 1 ) dofft( i ) = .FALSE.
       END DO
     END IF

     ! WRITE( stdout,*) 'DEBUG: ', COUNT( dofft )

     !
     !   Here initialize table only if necessary
     !

     DO ip = 1, ndims

       !   first check if there is already a table initialized
       !   for this combination of parameters

       done = ( ny == dims(1,ip) ) .AND. ( nx == dims(3,ip) )
        !   The initialization in ESSL and FFTW v.3 depends on all four parameters
       done = done .AND. ( ldx == dims(2,ip) ) .AND.  ( nzl == dims(4,ip) )
       IF (done) EXIT
     END DO

     IF( .NOT. done ) THEN

       !   no table exist for these parameters
       !   initialize a new one

       ! WRITE( stdout, fmt="('DEBUG cft_2xy, reinitializing tables ', I3)" ) icurrent


       IF ( ldx /= nx .OR. ldy /= ny ) THEN
          IF( fw_plan(2,icurrent) /= 0 )  CALL cufftDestroy( fw_plan(2,icurrent) )
          IF( bw_plan(2,icurrent) /= 0 )  CALL cufftDestroy( bw_plan(2,icurrent) )
          idir = -1
          CALL cufftPlanMany( fw_plan(2,icurrent), 1, (/ny/), (/ldx*ldy/), &
               ldx, 1, (/ldx*ldy/), ldx, 1, CUFFT_Z2Z, 1)
               ! 1 = rank
               ! ny = n
               ! 1 = howmany
               ! r(1:) = in
               ! (/ldx*ldy/) = inembed
               ! ldx = istride
               ! 1 = idist
               ! r(1:) = out
               ! (/ldx*ldy/) = onembed
               ! ldx = ostride
               ! 1 = odist
          idir =  1
          CALL cufftPlanMany( bw_plan(2,icurrent), 1, (/ny/), (/ldx*ldy/), &
               ldx, 1, (/ldx*ldy/), ldx, 1, CUFFT_Z2Z, 1)

          IF( fw_plan(1,icurrent) /= 0 ) CALL cufftDestroy( fw_plan(1,icurrent) )
          IF( bw_plan(1,icurrent) /= 0 ) CALL cufftDestroy( bw_plan(1,icurrent) )
          idir = -1
          CALL cufftPlanMany( fw_plan(1,icurrent), 1, (/nx/), (/ldx*ldy/), &
               1, ldx, (/ldx*ldy/), 1, ldx, CUFFT_Z2Z, ny)
               ! 1 = rank
               ! nx = n
               ! ny = howmany
               ! r(1:) = in
               ! (/ldx*ldy/) = inembed
               ! 1 = istride
               ! ldx = idist
               ! r(1:) = out
               ! (/ldx*ldy/) = onembed
               ! 1 = ostride
               ! ldx = odist
          idir =  1
          CALL cufftPlanMany( bw_plan(1,icurrent), 1, (/nx/), (/ldx*ldy/), &
               1, ldx, (/ldx*ldy/), 1, ldx, CUFFT_Z2Z, ny)
       ELSE
          IF( fw_plan( 1, icurrent) /= 0 ) CALL cufftDestroy( fw_plan( 1, icurrent) )
          IF( bw_plan( 1, icurrent) /= 0 ) CALL cufftDestroy( bw_plan( 1, icurrent) )
          idir = -1
          CALL cufftPlanMany( fw_plan( 1, icurrent), 2, (/nx,ny/), (/nx,ny/), &
               1, nx*ny, (/nx,ny/), 1, nx*ny, CUFFT_Z2Z, nzl)
          ! 2 = rank
          ! (/nx, ny/) = n
          ! nzl = howmany
          ! r(1:) = in
          ! (/nx, ny/) = inembed
          ! 1 = istride
          ! nx*ny = idist
          ! r(1:) = out
          ! (/nx, ny/) = onembed
          ! 1 = ostride
          ! nx*ny = odist
          idir = 1
          CALL cufftPlanMany( bw_plan( 1, icurrent), 2, (/nx,ny/), (/nx,ny/), &
               1, nx*ny, (/nx,ny/), 1, nx*ny, CUFFT_Z2Z, nzl)
       END IF

       dims(1,icurrent) = ny; dims(2,icurrent) = ldx;
       dims(3,icurrent) = nx; dims(4,icurrent) = nzl;
       ip = icurrent
       icurrent = MOD( icurrent, ndims ) + 1

     END IF

     !
     !   Now perform the FFTs using machine specific drivers
     !

#if defined __FFT_CLOCKS
     CALL start_clock( 'cft_2xy' )
#endif



     IF ( ldx /= nx .OR. ldy /= ny ) THEN
        IF( isign < 0 ) THEN
           do j = 0, nzl-1
              CALL cufftExecZ2Z( fw_plan (1, ip), &
                   r(1+j*ldx*ldy:), r(1+j*ldx*ldy:), CUFFT_FORWARD)
           end do
           do i = 1, nx
              do k = 1, nzl
                 IF( dofft( i ) ) THEN
                    j = i + ldx*ldy * ( k - 1 )
                    call cufftExecZ2Z( fw_plan ( 2, ip), r(j:), r(j:), CUFFT_FORWARD)
                 END IF
              end do
           end do
           tscale = 1.0_DP / ( nx * ny )
           !$cuf kernel do <<<*,*>>>
           DO i = 1, ldx * ldy * nzl
             r(i) = dcmplx(tscale,0.0d0) * r(i)
           END DO
           !CALL ZDSCAL( ldx * ldy * nzl, tscale, r(1), 1)
        ELSE IF( isign > 0 ) THEN
           do i = 1, nx
              do k = 1, nzl
                 IF( dofft( i ) ) THEN
                    j = i + ldx*ldy * ( k - 1 )
                    call cufftExecZ2Z( bw_plan ( 2, ip), r(j:), r(j:), CUFFT_INVERSE)
                 END IF
              end do
           end do
           do j = 0, nzl-1
              CALL cufftExecZ2Z( bw_plan( 1, ip), &
                   r(1+j*ldx*ldy:), r(1+j*ldx*ldy:), CUFFT_INVERSE)
           end do
        END IF
     ELSE
        IF( isign < 0 ) THEN
           call cufftExecZ2Z( fw_plan( 1, ip), r(1:), r(1:), CUFFT_FORWARD)
           tscale = 1.0_DP / ( nx * ny )
           !$cuf kernel do <<<*,*>>>
           DO i = 1, ldx * ldy * nzl
             r(i) = dcmplx(tscale,0.0d0) * r(i)
           END DO
           !CALL ZDSCAL( ldx * ldy * nzl, tscale, r(1), 1)
        ELSE IF( isign > 0 ) THEN
           call cufftExecZ2Z( bw_plan( 1, ip), r(1:), r(1:), CUFFT_INVERSE)
        END IF
     END IF


#if defined __FFT_CLOCKS
     CALL stop_clock( 'cft_2xy' )
#endif

     RETURN

   END SUBROUTINE cft_2xy_gpu
!!$
!!$!
!!$!=----------------------------------------------------------------------=!
!!$!
!!$!
!!$!
!!$!   3D parallel FFT on sub-grids, to be called inside OpenMP region
!!$!
!!$!
!!$!
!!$!=----------------------------------------------------------------------=!
!!$!


!
!=----------------------------------------------------------------------=!
!
!
!
!         FFT support Functions/Subroutines
!
!
!
!=----------------------------------------------------------------------=!
!

!
integer &
function good_fft_dimension (n)
  !
  ! Determines the optimal maximum dimensions of fft arrays
  ! Useful on some machines to avoid memory conflicts
  !
  USE kinds, only : DP
  IMPLICIT NONE
  INTEGER :: n, nx
  REAL(DP) :: log2n
  !
  ! this is the default: max dimension = fft dimension
  nx = n
  !
  !
  good_fft_dimension = nx
  return
end function good_fft_dimension


!=----------------------------------------------------------------------=!

function allowed (nr)


  ! find if the fft dimension is a good one
  ! a "bad one" is either not implemented (as on IBM with ESSL)
  ! or implemented but with awful performances (most other cases)

  USE kinds

  implicit none
  integer :: nr

  logical :: allowed
  integer :: pwr (5)
  integer :: mr, i, fac, p, maxpwr
  integer :: factors( 5 ) = (/ 2, 3, 5, 7, 11 /)

  ! find the factors of the fft dimension

  mr  = nr
  pwr = 0
  factors_loop: do i = 1, 5
     fac = factors (i)
     maxpwr = NINT ( LOG( DBLE (mr) ) / LOG( DBLE (fac) ) ) + 1
     do p = 1, maxpwr
        if ( mr == 1 ) EXIT factors_loop
        if ( MOD (mr, fac) == 0 ) then
           mr = mr / fac
           pwr (i) = pwr (i) + 1
        endif
     enddo
  end do factors_loop

  IF ( nr /= ( mr * 2**pwr (1) * 3**pwr (2) * 5**pwr (3) * 7**pwr (4) * 11**pwr (5) ) ) &
     CALL errore (' allowed ', ' what ?!? ', 1 )

  if ( mr /= 1 ) then

     ! fft dimension contains factors > 11 : no good in any case

     allowed = .false.

  else

     
     ! fftw and all other cases: no factors 7 and 11
     
     allowed = ( ( pwr(4) == 0 ) .and. ( pwr(5) == 0 ) )
     

  endif

  return
end function allowed

!=----------------------------------------------------------------------=!

   INTEGER &
   FUNCTION good_fft_order( nr, np )

!
!    This function find a "good" fft order value greater or equal to "nr"
!
!    nr  (input) tentative order n of a fft
!
!    np  (optional input) if present restrict the search of the order
!        in the ensamble of multiples of np
!
!    Output: the same if n is a good number
!         the closest higher number that is good
!         an fft order is not good if not implemented (as on IBM with ESSL)
!         or implemented but with awful performances (most other cases)
!

     IMPLICIT NONE
     INTEGER, INTENT(IN) :: nr
     INTEGER, OPTIONAL, INTENT(IN) :: np
     INTEGER :: new

     new = nr
     IF( PRESENT( np ) ) THEN
       DO WHILE( ( ( .NOT. allowed( new ) ) .OR. ( MOD( new, np ) /= 0 ) ) .AND. ( new <= nfftx ) )
         new = new + 1
       END DO
     ELSE
       DO WHILE( ( .NOT. allowed( new ) ) .AND. ( new <= nfftx ) )
         new = new + 1
       END DO
     END IF

     IF( new > nfftx ) &
       CALL errore( ' good_fft_order ', ' fft order too large ', new )

     good_fft_order = new

     RETURN
   END FUNCTION good_fft_order


!=----------------------------------------------------------------------=!
   END MODULE fft_scalar_gpu
!=----------------------------------------------------------------------=!
#endif
