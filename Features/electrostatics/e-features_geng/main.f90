PROGRAM Electrostatic_Features
USE comdata
USE treecode_procedures

IMPLICIT NONE
REAL*8 xyzminmax(6)
CHARACTER(100) fhead,lineString
CHARACTER(10) c1,c2,c3,c4,c5,c11,rrr
INTEGER err, MEOF, i, level
TYPE(tnode),POINTER :: troot


! Read command-line arguments
CALL get_command_argument(1, file_path)
CALL get_command_argument(2, kappa_str)
CALL get_command_argument(3, ef_p_str)
CALL get_command_argument(4, ef_L_str)

! Convert command-line arguments to appropriate types
READ(kappa_str, *) kappa
READ(ef_p_str, *) ef_p
READ(ef_L_str, *) ef_L

! print *, "File path: ", trim(file_path)
! print *, "Kappa: ", kappa
! print *, "ef_p: ", ef_p
! print *, "ef_L: ", ef_L


! Obtain path
pathname = trim(adjustl(file_path))

! OPEN(101,file="usrdata.in")
! READ(101,*,IOSTAT = MEOF) fhead, fname
! READ(101,*,IOSTAT = MEOF) fhead, kappa 
! READ(101,*,IOSTAT = MEOF) fhead, ef_p 
! READ(101,*,IOSTAT = MEOF) fhead, ef_L
! CLOSE(101)

! !Obtain path
! pathname='test_proteins/'

lenpath = len(pathname)
do while (pathname(lenpath:lenpath) .eq. ' ')
    lenpath = lenpath - 1
enddo  

!Obtain filename
lenfname = len(file_path)
do while (file_path(lenfname:lenfname) .eq. ' ')
    lenfname = lenfname - 1
enddo 

! Find how many particles from the pqr file
! OPEN(102,file=pathname(1:lenpath)//file_path(1:lenfname)//"/pro.pqr")
! OPEN(102, file=pathname, status='old', action='read', iostat=err)
! OPEN(102, file=pathname)

! Check if file is empty
OPEN(102, FILE = trim(pathname), STATUS='OLD', ACTION='READ', IOSTAT=err)
! IF (err /= 0) THEN
!     PRINT *, "Error opening file: ", pathname
!     STOP
! END IF

! READ(102, '(A)', IOSTAT=err) lineString
! IF (err /= 0 .OR. LEN_TRIM(lineString) == 0) THEN
!     PRINT *, "File is empty or cannot be read: ", pathname
!     CLOSE(102)
!     STOP
! END IF

numpars = 0
DO 
    READ(102,'(A)',IOSTAT = MEOF) lineString 
    IF (MEOF .LT. 0) EXIT
    IF (lineString(1:6) == 'REMARK') CYCLE ! Skip the REMARK' line 
    IF (lineString(1:3) == 'TER') CYCLE ! Skip the 'TER' line 
    IF (lineString(1:3) == 'END') EXIT  ! Exit if it starts with 'END'
    numpars = numpars + 1
END DO        

WRITE(*,*) 'information for protein'
WRITE(*,*) 'number of particles = ', numpars 
WRITE(*,*) 'multipoles in ',ef_L,' levels with order ',ef_p
CLOSE(102)

ef_Nf=(ef_p+1)*(ef_p+2)*(ef_p+3)*(8**(ef_L+1)-1)/42
PRINT *,'number of features: ',ef_Nf

ALLOCATE(features(ef_Nf),STAT=err)
IF (err .NE. 0) THEN
    WRITE(6,*) 'Error allocating features!'
    STOP
END IF
features = 0.d0; nMoMs=0

ALLOCATE(x(numpars),y(numpars),z(numpars),q(numpars),STAT=err)
IF (err .NE. 0) THEN
    WRITE(6,*) 'Error allocating x, y, z, q, or orderind!'
    STOP
END IF

x=0.d0; y=0.d0; z=0.d0; q=0.d0

! Read in atomic position and charges information
! OPEN(102,file=pathname(1:lenpath)//file_path(1:lenfname)//"/pro.pqr")
! OPEN(102, file=pathname, status='old', action='read', iostat=err)
! OPEN(102, file=pathname)

! Check if file is empty
OPEN(102, FILE = trim(pathname), STATUS='OLD', ACTION='READ', IOSTAT=err)
! IF (err /= 0) THEN
!     PRINT *, "Error opening file: ", pathname
!     STOP
! END IF

! READ(102, '(A)', IOSTAT=err) lineString
! IF (err /= 0 .OR. LEN_TRIM(lineString) == 0) THEN
!     PRINT *, "File is empty or cannot be read: ", pathname
!     CLOSE(102)
!     STOP
! END IF

i = 0
DO 
    READ(102,'(A)',IOSTAT = MEOF) lineString 
    IF (MEOF .LT. 0) EXIT
    IF (lineString(1:6) == 'REMARK') CYCLE ! Skip the REMARK' line 
    IF (lineString(1:3) == 'TER') CYCLE ! Skip the 'TER' line 
    IF (lineString(1:3) == 'END') EXIT ! Exit if it starts with 'END'
    i = i + 1
    
    !Option 1: Space seperated
    !READ(lineString,*) c1,c2,c3,c4,c5,x(i),y(i),z(i),q(i),rrr,c11 
    
    !Option 2: Column-width controlled (WG: use one of the two options as needed!)
    READ(lineString(31:38),*) x(i)
    READ(lineString(39:46),*) y(i)
    READ(lineString(47:54),*) z(i)
    READ(lineString(55:62),*) q(i) 
END DO        
CLOSE(102)

! Build the tree and compute moments
WRITE(*,*) 'number of read-in particles = ', i 
PRINT *,'obtain particle locations ...'
Call SETUP(x,y,z,q,numpars,xyzminmax)
PRINT *,'total charges:',sum(q),'and dimensions: '
PRINT *,'[',real(xyzminmax),']'

NULLIFY(troot) 
level=0
PRINT *,'create tree ...'
Call CREATE_TREE(troot,1,numpars,x,y,z,q,xyzminmax,level,numpars,ef_p,ef_L)
PRINT *,'number of moments calculated: ',nMoMs

! output features
OPEN(104,file='efeatrue.txt')
Do i=1,ef_Nf 
    WRITE(104,*,IOSTAT = MEOF) features(i)
END DO

PRINT *,'all done, free tree and deallocate variables ...'
CALL CLEANUP(troot)

END PROGRAM Electrostatic_Features