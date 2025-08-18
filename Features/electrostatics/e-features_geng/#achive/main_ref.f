      PROGRAM Electrostatic_Features
      use molecule
      use comdata
      use treecode
      IMPLICIT NONE

C r8 is 8-byte (double precision) real

      INTEGER,PARAMETER :: r8=SELECTED_REAL_KIND(12)

C local variables

      INTEGER :: i,j,k,err,idx(3),ilevel,istep,MEOF,ierr
      REAL(KIND=r8) :: t1,abserr,relerr,absinf_err,relinf_err
      REAL(KIND=r8) :: xxx,yyy,zzz,qqq,rrr,eng_C
      character(100) fhead
      character(10) c1,c2,c3,c4,c5
      REAL(KIND=r8),DIMENSION(3) :: f_inferr,f_relinferr,t

      open(101,file="usrdata.in")
      READ(101,*,IOSTAT = MEOF) fhead, fname
      READ(101,*,IOSTAT = MEOF) fhead, kappa 
      READ(101,*,IOSTAT = MEOF) fhead, order 
      READ(101,*,IOSTAT = MEOF) fhead, maxparnode 
      READ(101,*,IOSTAT = MEOF) fhead, theta
      READ(101,*,IOSTAT = MEOF) fhead, ierr
      READ(101,*,IOSTAT = MEOF) fhead, ef_p 
      READ(101,*,IOSTAT = MEOF) fhead, ef_L
      close(101)

      open(102,file=fname)
      numpars = 0
      DO 
        READ(102,*,IOSTAT = MEOF) xxx, yyy, zzz, qqq, rrr
        IF(MEOF .LT. 0) EXIT
        numpars = numpars + 1
      END DO        
      numpars=numpars-2 
      write(*,*) 'number of particles = ', numpars 
      CLOSE(102)

      ef_Nf=(3*ef_p**2+3*ef_p+2)*(8**(ef_L+1)-1)/14
      print *,'number of features: ',ef_Nf

      ALLOCATE(x(numpars),y(numpars),z(numpars),q(numpars),
     &         orderind(numpars),STAT=err)
      IF (err .NE. 0) THEN
          WRITE(6,*) 'Error allocating x, y, z, q, or orderind!'
          STOP
      END IF

      x=0.d0; y=0.d0; z=0.d0; q=0.d0

      open(103,file=fname)
      Do i=1,numpars 
        READ(103,*,IOSTAT = MEOF) c1,c2,c3,c4,c5, 
     &   x(i), y(i), z(i), q(i), rrr
      END DO
C     do i=1,10
C         write(*,'(5f16.12)') x(i), y(i), z(i), q(i), rrr
C     enddo           
      CLOSE(103)
      
      ALLOCATE(x_copy(numpars),y_copy(numpars),z_copy(numpars),
     &          q_copy(numpars),STAT=err)
      IF (err .NE. 0) THEN
          WRITE(6,*) 'Error allocating copies of x, y, z, q, '
          STOP
      END IF

      x_copy=x; y_copy=y; z_copy=z; q_copy=q
       
      ALLOCATE(tpoten(numpars),dpoten(numpars),STAT=err)
      IF (err .NE. 0) THEN
          WRITE(6,*) 'Error allocating tpoten or dpoten!'
          STOP
      END IF

C compute potential by treecode 
      tpoten=0.d0; dpoten=0.d0

      CALL TREECODE3D_YUKAWA(x,y,z,q,kappa,order,theta,
     &      maxparnode,numpars,orderind,tpoten)

      eng_C=0.d0
      do i=1,numpars
        eng_C=eng_C+tpoten(i) !*q(i)
      enddo
      eng_C=eng_C/2.d0

      print *,'screened coulombic energy by treecode = ',eng_C
      open(106,file='output.txt')
      write(106,*) eng_C       
      CLOSE(106)
     
C compute potential directly
      if (ierr==1) then
      WRITE(6,*) ' '
      WRITE(6,*) 'Computing potential - directly'
C      x=x_copy; y=y_copy; z=z_copy; q=q_copy

      CALL CPU_TIME(timebeg)
      CALL COMPUTE_DIRECT(x,y,z,q,kappa,numpars,dpoten)

      eng_C=0.d0
      do i=1,numpars
        eng_C=eng_C+dpoten(i) !*q(i)
      enddo
      eng_C=eng_C/2.d0
      print *,'screened coulombic energy by dir_sum = ',eng_C
      CALL CPU_TIME(timeend)
      WRITE(6,*) 'Time for direct computation (secs):',timeend-timebeg
      

C compute errors

      WRITE(6,*) ' '
      WRITE(6,*) 'Computing potential error'
      WRITE(6,*) ' '
      
      abserr = 0.0_r8
      relerr = 0.0_r8
      relinf_err = 0.0_r8
      absinf_err = 0.0_r8
      DO j=1,numpars
         t1 = ABS(dpoten(j)-tpoten(j))
         relerr = relerr+t1**2
         abserr = abserr+dpoten(j)*dpoten(j)
         IF (t1 .GT. relinf_err) THEN
            relinf_err = t1
         END IF
         t1 = ABS(dpoten(j))
         IF (t1 .GT. absinf_err) THEN
            absinf_err = t1
         END IF
      END DO
      
      relerr = SQRT(relerr/abserr)
      relinf_err =  relinf_err/absinf_err

      WRITE(6,*) '   Relative L2 and Inf error :',relerr,relinf_err
      WRITE(6,*) ' '
      

      endif
  
      DEALLOCATE(x,y,z,q,orderind,STAT=err)
      IF (err .NE. 0) THEN
          WRITE(6,*) 'Error deallocating x, y, z, q, or orderind!'
          STOP
      END IF
      DEALLOCATE(tpoten,dpoten,STAT=err)
      IF (err .NE. 0) THEN
          WRITE(6,*) 'Error deallocating tpoten or dpoten!'
          STOP
      END IF
      
      END PROGRAM Electrostatic_Features

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE COMPUTE_DIRECT(x,y,z,q,kappa,numpars,dpoten)
                       
      IMPLICIT NONE

      INTEGER,PARAMETER :: r8=SELECTED_REAL_KIND(12)
      INTEGER,INTENT(IN) :: numpars
      REAL(KIND=r8),DIMENSION(numpars),INTENT(IN) :: x,y,z,q
      REAL(KIND=r8),DIMENSION(numpars),INTENT(INOUT) :: dpoten
      REAL(KIND=r8),INTENT(IN) :: kappa

C local variables 
   
      INTEGER :: i,j
      REAL(KIND=r8) :: tx,ty,tz,fx,fy,fz,teng,dist,t1
      REAL(KIND=r8) ::  dpeng,temp,peng,pi
C######################
      pi=acos(-1.d0)
C#######################

      dpoten = 0.0_r8
      DO i = 1,numpars-1
            peng = 0.0_r8
            DO j = i+1,numpars
               tx = x(j)-x(i)
               ty = y(j)-y(i)
               tz = z(j)-z(i)
               dist = SQRT(tx*tx+ty*ty+tz*tz)
C#################################################             
               temp = EXP(-kappa*dist)/dist
C               temp = EXP(-kappa*dist)/dist/4/pi
C#################################################
               peng = peng + q(j)*temp
               dpoten(j) = dpoten(j) + q(i)*temp
            END DO
            dpoten(i) = q(i)*(dpoten(i)+peng)
      END DO
      dpoten(numpars) = q(numpars)*dpoten(numpars)      
      
      RETURN
      END SUBROUTINE COMPUTE_DIRECT     


