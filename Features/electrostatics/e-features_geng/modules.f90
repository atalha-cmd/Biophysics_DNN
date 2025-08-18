MODULE COMDATA   
      character(256) :: fname, pathname, file_path
      character(32) :: kappa_str, ef_p_str, ef_L_str
      integer :: lenpath,lenfname       
      INTEGER :: numpars,nMoMs
      integer :: ef_p, ef_L, ef_Nf
      real*8 :: kappa
      REAL*8, DIMENSION(:), ALLOCATABLE :: features
      REAL*8,ALLOCATABLE,DIMENSION(:) :: x,y,z,q   
END MODULE COMDATA



