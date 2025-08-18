      MODULE treecode3d_procedures
      IMPLICIT NONE

C r8 is 8-byte (double precision) real

      INTEGER,PARAMETER :: r8=SELECTED_REAL_KIND(12)

C global variables for taylor expansions

      INTEGER :: torder,torderlim
      REAL(KIND=r8),ALLOCATABLE,DIMENSION(:) :: cf
      REAL(KIND=r8),ALLOCATABLE,DIMENSION(:) :: cf1,cf2,cf3
      REAL(KIND=r8),ALLOCATABLE,DIMENSION(:,:,:) :: a,b

C global variables to track tree levels 
 
      INTEGER :: minlevel,maxlevel
      
C global variables used when computing potential/force

      INTEGER :: orderoffset
      REAL(KIND=r8),DIMENSION(3) :: tarpos

C global variables for position and charge storage 
C NOTE: arrays ARE NOT COPIED in this version!!  orderarr is still valid

      INTEGER,ALLOCATABLE,DIMENSION(:)  :: orderarr
      REAL(KIND=r8),ALLOCATABLE,DIMENSION(:) :: xcopy,ycopy,zcopy,qcopy

C node pointer and node type declarations

      TYPE tnode_pointer
           TYPE(tnode), POINTER :: p_to_tnode
      END TYPE tnode_pointer
      TYPE tnode
           INTEGER          :: node_idx
           INTEGER          :: numpar,ibeg,iend
           REAL(KIND=r8)    :: x_min,y_min,z_min
           REAL(KIND=r8)    :: x_max,y_max,z_max
           REAL(KIND=r8)    :: x_mid,y_mid,z_mid
           REAL(KIND=r8)    :: radius,aspect
           INTEGER          :: level,num_children,exist_ms
           REAL(KIND=r8),DIMENSION(:,:,:),POINTER :: ms
           TYPE(tnode_pointer), DIMENSION(8) :: child
      END TYPE tnode

      CONTAINS

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE SETUP(x,y,z,q,numpars,order,xyzminmax)
      IMPLICIT NONE
C
C SETUP allocates and initializes arrays needed for the Taylor expansion.
C Also, global variables are set and the Cartesian coordinates of
C the smallest box containing the particles is determined. The particle
C postions and charges are copied so that they can be restored upon exit.
C
      INTEGER,INTENT(IN) :: numpars,order
      REAL(KIND=r8),DIMENSION(numpars),INTENT(IN) :: x,y,z,q
      REAL(KIND=r8),INTENT(INOUT),DIMENSION(6) :: xyzminmax

C local variables

      INTEGER :: err,i,j,k
      REAL(KIND=r8) :: t1

C global integers and reals:  TORDER, TORDERLIM and THETASQ
      
      torder = order
      orderoffset=0
      torderlim = torder+orderoffset

C allocate global Taylor expansion variables

      ALLOCATE(cf(0:torder), STAT=err)
      IF (err .NE. 0) THEN
         WRITE(6,*) 'Error allocationg Taylor variables! '
         STOP
      END IF

      ALLOCATE(cf1(torderlim),cf2(torderlim),cf3(torderlim),STAT=err)
      IF (err .NE. 0) THEN
         WRITE(6,*) 'Error allocating Taylor variables! '
         STOP
      END IF

      ALLOCATE(a(-2:torderlim,-2:torderlim,-2:torderlim),
     &         b(-2:torderlim,-2:torderlim,-2:torderlim),STAT=err)
      IF (err .NE. 0) THEN
         WRITE(6,*) 'Error allocating Taylor variables! '
         STOP
      END IF

      a=0.0_r8
      b=0.0_r8

      DO i=0,torder
         cf(i) = REAL(i,KIND=r8)+1.0_r8
      END DO

      DO i=1,torderlim
         t1=1.0_r8/REAL(i,KIND=r8)
         cf1(i)=t1
         cf2(i)=1.0_r8-0.5_r8*t1
         cf3(i)=1.0_r8-t1
      END DO

C find bounds of Cartesian box enclosing the particles

      xyzminmax(1)=MINVAL(x(1:numpars))
      xyzminmax(2)=MAXVAL(x(1:numpars))
      xyzminmax(3)=MINVAL(y(1:numpars))
      xyzminmax(4)=MAXVAL(y(1:numpars))
      xyzminmax(5)=MINVAL(z(1:numpars))
      xyzminmax(6)=MAXVAL(z(1:numpars))

C Allocate arrays and copy variables. Also create and initialize orderarr.
C
C    DISABLED!!!   Instead ORDERIND array is used to store reordered particle indices.

C      ALLOCATE(xcopy(numpars),ycopy(numpars),zcopy(numpars), 
C     &         qcopy(numpars),orderarr(numpars),STAT=err)
C      IF (err .NE. 0) THEN
C         WRITE(6,*) 'Error allocating copy variables! '
C         STOP
C      END IF    
      ALLOCATE(xcopy(1),ycopy(1),zcopy(1),
     &         qcopy(1),orderarr(numpars),STAT=err)
      IF (err .NE. 0) THEN
         WRITE(6,*) 'Error allocating copy variables! '
         STOP
      END IF
C      xcopy=x(1:numpars)
C      ycopy=y(1:numpars)
C      zcopy=z(1:numpars)
C      qcopy=q(1:numpars)

      DO i=1,numpars
         orderarr(i)=i
      END DO  

      RETURN
      END SUBROUTINE SETUP

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      RECURSIVE SUBROUTINE CREATE_TREE(p,ibeg,iend,x,y,z,q,maxparnode,
     &                                 xyzmm,level,numpars)
      IMPLICIT NONE
C
C CREATE_TREE recursively create the tree structure. Node P is
C input, which contains particles indexed from IBEG to IEND. After
C the node parameters are set subdivision occurs if IEND-IBEG+1 > MAXPARNODE.
C Real array XYZMM contains the min and max values of the coordinates
C of the particle in P, thus defining the box.   
C
      TYPE(tnode),POINTER :: p
      INTEGER,INTENT(IN) :: ibeg,iend,level,maxparnode,numpars
      REAL(KIND=r8),DIMENSION(numpars),INTENT(INOUT) :: x,y,z,q
      REAL(KIND=r8),DIMENSION(6),INTENT(IN) :: xyzmm
      
C local variables

      REAL(KIND=r8) :: x_mid,y_mid,z_mid,xl,yl,zl,lmax,t1,t2,t3
      INTEGER, DIMENSION(8,2) :: ind
      REAL(KIND=r8), DIMENSION(6,8) :: xyzmms
      INTEGER :: i,j,limin,limax,err,loclev,numposchild
      REAL(KIND=r8), DIMENSION(6) ::  lxyzmm
     
C allocate pointer 

      ALLOCATE(p,STAT=err)
      IF (err .NE. 0) THEN
         WRITE(6,*) 'Error allocating pointer! '
         STOP
      END IF

C set node fields: number of particles, exist_ms
C and xyz bounds 

      
      p%numpar=iend-ibeg+1
      p%exist_ms=0
      
      p%x_min=xyzmm(1)
      p%x_max=xyzmm(2)
      p%y_min=xyzmm(3)
      p%y_max=xyzmm(4)
      p%z_min=xyzmm(5)
      p%z_max=xyzmm(6)        
C compute aspect ratio

      xl=p%x_max-p%x_min
      yl=p%y_max-p%y_min
      zl=p%z_max-p%z_min

      lmax=MAX(xl,yl,zl)
      t1=lmax
      t2=MIN(xl,yl,zl)
      IF (t2 .NE. 0.0_r8) THEN
         p%aspect=t1/t2
      ELSE
         p%aspect=0.0_r8
      END IF

C midpoint coordinates , RADIUS and SQRADIUS 

      p%x_mid=(p%x_max+p%x_min)/2.0_r8
      p%y_mid=(p%y_max+p%y_min)/2.0_r8
      p%z_mid=(p%z_max+p%z_min)/2.0_r8
      t1=p%x_max-p%x_mid
      t2=p%y_max-p%y_mid
      t3=p%z_max-p%z_mid
      p%radius=SQRT(t1*t1+t2*t2+t3*t3)

C set particle limits, tree level of node, and nullify children pointers

      p%ibeg=ibeg
      p%iend=iend
      p%level=level
      IF (maxlevel .LT. level) THEN
         maxlevel=level
      END IF
      p%num_children=0
      DO i=1,8
         NULLIFY(p%child(i)%p_to_tnode)
      END DO

      IF (p%numpar .GT. maxparnode) THEN
         print*, "level:", level
C
C set IND array to 0 and then call PARTITION routine.  IND array holds indices
C of the eight new subregions. Also, setup XYZMMS array in case SHRINK=1
C
         xyzmms(1,1)=p%x_min
         xyzmms(2,1)=p%x_max
         xyzmms(3,1)=p%y_min
         xyzmms(4,1)=p%y_max
         xyzmms(5,1)=p%z_min
         xyzmms(6,1)=p%z_max
         ind=0 !Weihua
         ind(1,1)=ibeg
         ind(1,2)=iend
         x_mid=p%x_mid
         y_mid=p%y_mid
         z_mid=p%z_mid

         CALL PARTITION_8(x,y,z,q,xyzmms,xl,yl,zl,lmax,numposchild,
     &                    x_mid,y_mid,z_mid,ind,numpars)
C
C create children if indicated and store info in parent
C
         loclev=level+1
C         print*, "loclev:", loclev
         DO i=1,numposchild
            IF (ind(i,1) .LE. ind(i,2)) THEN
               p%num_children=p%num_children+1
               lxyzmm=xyzmms(:,i)
               CALL CREATE_TREE(p%child(p%num_children)%p_to_tnode,
     &                          ind(i,1),ind(i,2),x,y,z,q,maxparnode,
     &                          lxyzmm,loclev,numpars)
            END IF
         END DO
      ELSE
         IF (level .LT. minlevel) THEN
            minlevel=level
         END IF
      END IF   

      END SUBROUTINE CREATE_TREE      

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE PARTITION_8(x,y,z,q,xyzmms,xl,yl,zl,lmax,numposchild,
     &                       x_mid,y_mid,z_mid,ind,numpars)
      IMPLICIT NONE
C
C PARTITION_8 determines the particle indices of the eight sub boxes
C containing the particles after the box defined by particles I_BEG
C to I_END is divided by its midpoints in each coordinate direction.
C The determination of the indices is accomplished by the subroutine
C PARTITION. A box is divided in a coordinate direction as long as the
C resulting aspect ratio is not too large. This avoids the creation of
C "narrow" boxes in which Talyor expansions may become inefficient.
C On exit the INTEGER array IND (dimension 8 x 2) contains
C the indice limits of each new box (node) and NUMPOSCHILD the number 
C of possible children.  If IND(J,1) > IND(J,2) for a given J this indicates
C that box J is empty.
C
      INTEGER, INTENT(IN) :: numpars
      REAL(KIND=r8),DIMENSION(numpars),INTENT(INOUT) :: x,y,z,q
      INTEGER, DIMENSION(8,2),INTENT(INOUT) :: ind
      REAL(KIND=r8),DIMENSION(6,8),INTENT(INOUT) :: xyzmms
      REAL(KIND=r8), INTENT(IN) :: x_mid,y_mid,z_mid,xl,yl,zl,lmax
      INTEGER,INTENT(INOUT) :: numposchild

C local variables

      INTEGER :: temp_ind,i
      REAL(KIND=r8) :: critlen

      numposchild=1
      critlen=lmax/sqrt(2.0_r8)

      IF (xl .GE. critlen) THEN
         CALL PARTITION(x,y,z,q,orderarr,ind(1,1),ind(1,2),
     &                  x_mid,temp_ind,numpars)
         ind(2,1)=temp_ind+1
         ind(2,2)=ind(1,2)
         ind(1,2)=temp_ind
         xyzmms(:,2)=xyzmms(:,1)
         xyzmms(2,1)=x_mid
         xyzmms(1,2)=x_mid
         numposchild=2*numposchild
      END IF 
 
      IF (yl .GE. critlen) THEN
         DO i=1,numposchild
            CALL PARTITION(y,x,z,q,orderarr,ind(i,1),ind(i,2),
     &                     y_mid,temp_ind,numpars)
            ind(numposchild+i,1)=temp_ind+1
            ind(numposchild+i,2)=ind(i,2)
            ind(i,2)=temp_ind
            xyzmms(:,numposchild+i)=xyzmms(:,i)
            xyzmms(4,i)=y_mid
            xyzmms(3,numposchild+i)=y_mid
         END DO
         numposchild=2*numposchild
      END IF

      IF (zl .GE. critlen) THEN
         DO i=1,numposchild
            CALL PARTITION(z,x,y,q,orderarr,ind(i,1),ind(i,2),
     &                     z_mid,temp_ind,numpars)
            ind(numposchild+i,1)=temp_ind+1
            ind(numposchild+i,2)=ind(i,2)
            ind(i,2)=temp_ind
            xyzmms(:,numposchild+i)=xyzmms(:,i)
            xyzmms(6,i)=z_mid
            xyzmms(5,numposchild+i)=z_mid
         END DO
         numposchild=2*numposchild
      END IF
      RETURN 
      END SUBROUTINE PARTITION_8

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE TREE_COMPP(p,x,y,z,q,kappa,theta,tpoten,numpars)
      IMPLICIT NONE
C
C TREE_COMPF is the driver routine which calls COMPF_TREE for each
C particle, setting the global variable TARPOS before the call. 
C The current target particle's x coordinate and charge are changed
C so that it does not interact with itself. P is the root node of the tree. 
C

      INTEGER,INTENT(IN) :: numpars
      TYPE(tnode),POINTER :: p  
      REAL(KIND=r8),DIMENSION(numpars),INTENT(INOUT) :: x,q
      REAL(KIND=r8),DIMENSION(numpars),INTENT(IN) :: y,z
      REAL(KIND=r8),DIMENSION(numpars),INTENT(INOUT) :: tpoten
      REAL(KIND=r8),INTENT(IN) :: kappa,theta

C local variables

      INTEGER :: i,j
      REAL(KIND=r8) :: peng,tempx,tempq,pi4
C      pi4=acos(-1.d0)*4.d0
      tpoten(1:numpars)=0.0_r8
      
      call comp_ms_all(p,1)
      
      DO i=1,numpars
         peng=0.0_r8
         tarpos(1)=x(i)
         tarpos(2)=y(i)
         tarpos(3)=z(i)
C         write(*,*) i,real(tarpos),real(q(i))
         tempx=x(i)
         tempq=q(i)
         x(i)=x(i)+100.123456789_r8
         q(i)=0.0_r8
         CALL COMPP_TREE(p,peng,x,y,z,q,kappa,theta,numpars)
         x(i)=tempx
         q(i)=tempq
C  USE order of points imposed by tree creation!
C         tpoten(orderarr(i))=tempq*peng
         tpoten(i)=tempq*peng
		
      END DO
      

      RETURN
      END SUBROUTINE TREE_COMPP

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      RECURSIVE SUBROUTINE COMPP_TREE(p,peng,x,y,z,q,kappa,theta,
     &                                numpars)
      IMPLICIT NONE

      INTEGER,INTENT(IN) :: numpars
      TYPE(tnode),POINTER :: p   
      REAL(KIND=r8),INTENT(INOUT) :: peng     
      REAL(KIND=r8),DIMENSION(numpars),INTENT(IN) :: x,y,z,q
      REAL(KIND=r8),INTENT(IN) :: kappa,theta

C local variables

      REAL(KIND=r8) :: tx,ty,tz,dist,penglocal,pi
      INTEGER :: i,j,k,err

C determine DISTSQ for MAC test
      pi=acos(-1.d0)
      tx=p%x_mid-tarpos(1)
      ty=p%y_mid-tarpos(2)
      tz=p%z_mid-tarpos(3)
      dist=SQRT(tx*tx+ty*ty+tz*tz)

C intialize potential energy  

      peng=0.0_r8

C If MAC is accepted and there is more than 1 particle in the 
C box use the expansion for the approximation.

      IF ((p%radius .LT. dist*theta) .AND.
     &    (p%numpar .GT. 1)) THEN
C         print *,'tree: ibeg,iend',p%ibeg,p%iend
         !IF (p%exist_ms .EQ. 0) THEN
         !    ALLOCATE(p%ms(0:torder,0:torder,0:torder),STAT=err)
         !    IF (err .NE. 0) THEN
         !       WRITE(6,*) 'Error allocating node moments! '
         !       STOP
         !    END IF
         !    CALL COMP_MS(p,x,y,z,q,numpars)
         !    p%exist_ms=1
         !END IF   
         
         CALL COMP_TCOEFF(p,kappa)
         DO k=0,torder
            DO j=0,torder-k
               DO i=0,torder-k-j
                  peng = peng+a(i,j,k)*p%ms(i,j,k)
C                  write (*,'(3i4,3f16.8)') i,j,k,p%ms(i,j,k),
C     &	               	a(i,j,k),peng
C                  pause
               END DO
            END DO
         END DO
C         peng=peng/4/pi          
C         print *,'tree',p%ibeg,p%iend,real(peng)
C#############################################################        
C            CALL COMPP_DIRECT(penglocal,p%ibeg,p%iend,
C     &                        x,y,z,q,kappa,numpars)
C            peng=penglocal
C      print *,'direct in tree, test only',p%ibeg,p%iend,real(penglocal)
C############################################################
C       pause
      ELSE

C If MAC fails check to see if there are children. If not, perform direct 
C calculation.  If there are children, call routine recursively for each.
C
         IF (p%num_children .EQ. 0) THEN
            
C            print *,'direct: ibeg,iend',p%ibeg,p%iend
            CALL COMPP_DIRECT(penglocal,p%ibeg,p%iend,
     &                        x,y,z,q,kappa,numpars)
            peng=penglocal
         ELSE
            DO i=1,p%num_children
               CALL COMPP_TREE(p%child(i)%p_to_tnode,penglocal,
     &                        x,y,z,q,kappa,theta,numpars)
               peng=peng+penglocal
            END DO  
         END IF 
      END IF

      RETURN
      END SUBROUTINE COMPP_TREE


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE COMP_TCOEFF(p,kappa)
      IMPLICIT NONE
C
C COMP_TCOEFF computes the Taylor coefficients of the potential
C using a recurrence formula.  The center of the expansion is the
C midpoint of the node P.  TARPOS and TORDERLIM are globally defined.
C
      TYPE(tnode),POINTER :: p
      REAL(KIND=r8),INTENT(IN)  :: kappa

C local varaibles

      REAL(KIND=r8) :: dx,dy,dz,ddx,ddy,ddz,dist,fac
      REAL(KIND=r8) :: kappax,kappay,kappaz
      INTEGER :: i,j,k

C setup variables

      dx=tarpos(1)-p%x_mid
      dy=tarpos(2)-p%y_mid
      dz=tarpos(3)-p%z_mid

      ddx=2.0_r8*dx
      ddy=2.0_r8*dy
      ddz=2.0_r8*dz

      kappax=kappa*dx
      kappay=kappa*dy
      kappaz=kappa*dz

      dist=dx*dx+dy*dy+dz*dz
      fac=1.0_r8/dist
      dist=SQRT(dist)

C 0th coeff or function val

      b(0,0,0)=EXP(-kappa*dist)
      a(0,0,0)=b(0,0,0)/dist
      
C 2 indices are 0

      b(1,0,0)=kappax*a(0,0,0)
      b(0,1,0)=kappay*a(0,0,0)
      b(0,0,1)=kappaz*a(0,0,0)

      a(1,0,0)=fac*dx*(a(0,0,0)+kappa*b(0,0,0))
      a(0,1,0)=fac*dy*(a(0,0,0)+kappa*b(0,0,0))
      a(0,0,1)=fac*dz*(a(0,0,0)+kappa*b(0,0,0))


      DO i=2,torderlim
         b(i,0,0)=cf1(i)*kappa*(dx*a(i-1,0,0)-a(i-2,0,0))
         b(0,i,0)=cf1(i)*kappa*(dy*a(0,i-1,0)-a(0,i-2,0))
         b(0,0,i)=cf1(i)*kappa*(dz*a(0,0,i-1)-a(0,0,i-2))

         a(i,0,0)=fac*(ddx*cf2(i)*a(i-1,0,0)-cf3(i)*a(i-2,0,0)+
     &            cf1(i)*kappa*(dx*b(i-1,0,0)-b(i-2,0,0)))
         a(0,i,0)=fac*(ddy*cf2(i)*a(0,i-1,0)-cf3(i)*a(0,i-2,0)+
     &            cf1(i)*kappa*(dy*b(0,i-1,0)-b(0,i-2,0)))
         a(0,0,i)=fac*(ddz*cf2(i)*a(0,0,i-1)-cf3(i)*a(0,0,i-2)+
     &            cf1(i)*kappa*(dz*b(0,0,i-1)-b(0,0,i-2)))
      END DO

C 1 index 0, 1 index 1, other >=1

      b(1,1,0)=kappax*a(0,1,0)
      b(1,0,1)=kappax*a(0,0,1)
      b(0,1,1)=kappay*a(0,0,1)

      a(1,1,0)=fac*(dx*a(0,1,0)+ddy*a(1,0,0)+kappax*b(0,1,0))
      a(1,0,1)=fac*(dx*a(0,0,1)+ddz*a(1,0,0)+kappax*b(0,0,1))
      a(0,1,1)=fac*(dy*a(0,0,1)+ddz*a(0,1,0)+kappay*b(0,0,1))

      DO i=2,torderlim-1
         b(1,0,i)=kappax*a(0,0,i)
         b(0,1,i)=kappay*a(0,0,i)
         b(0,i,1)=kappaz*a(0,i,0)
         b(1,i,0)=kappax*a(0,i,0)
         b(i,1,0)=kappay*a(i,0,0)
         b(i,0,1)=kappaz*a(i,0,0)

         a(1,0,i)=fac*(dx*a(0,0,i)+ddz*a(1,0,i-1)-a(1,0,i-2)+
     &            kappax*b(0,0,i)) 
         a(0,1,i)=fac*(dy*a(0,0,i)+ddz*a(0,1,i-1)-a(0,1,i-2)+
     &            kappay*b(0,0,i))
         a(0,i,1)=fac*(dz*a(0,i,0)+ddy*a(0,i-1,1)-a(0,i-2,1)+
     &            kappaz*b(0,i,0))
         a(1,i,0)=fac*(dx*a(0,i,0)+ddy*a(1,i-1,0)-a(1,i-2,0)+
     &            kappax*b(0,i,0))
         a(i,1,0)=fac*(dy*a(i,0,0)+ddx*a(i-1,1,0)-a(i-2,1,0)+
     &            kappay*b(i,0,0))
         a(i,0,1)=fac*(dz*a(i,0,0)+ddx*a(i-1,0,1)-a(i-2,0,1)+
     &            kappaz*b(i,0,0))         
      END DO

C 1 index 0, others >= 2

      DO i=2,torderlim-2
         DO j=2,torderlim-i
            b(i,j,0)=cf1(i)*kappa*(dx*a(i-1,j,0)-a(i-2,j,0))
            b(i,0,j)=cf1(i)*kappa*(dx*a(i-1,0,j)-a(i-2,0,j))
            b(0,i,j)=cf1(i)*kappa*(dy*a(0,i-1,j)-a(0,i-2,j))

            a(i,j,0)=fac*(ddx*cf2(i)*a(i-1,j,0)+ddy*a(i,j-1,0)
     &               -cf3(i)*a(i-2,j,0)-a(i,j-2,0)+
     &               cf1(i)*kappa*(dx*b(i-1,j,0)-b(i-2,j,0)))
            a(i,0,j)=fac*(ddx*cf2(i)*a(i-1,0,j)+ddz*a(i,0,j-1)
     &               -cf3(i)*a(i-2,0,j)-a(i,0,j-2)+
     &               cf1(i)*kappa*(dx*b(i-1,0,j)-b(i-2,0,j)))
            a(0,i,j)=fac*(ddy*cf2(i)*a(0,i-1,j)+ddz*a(0,i,j-1)
     &               -cf3(i)*a(0,i-2,j)-a(0,i,j-2)+
     &               cf1(i)*kappa*(dy*b(0,i-1,j)-b(0,i-2,j)))
         END DO
      END DO

C 2 indices 1, other >= 1
C b(1,1,1) is correct, but a little tricky!
C      b(1,1,1)=5.0*dz*fac*b(1,1,0)

      b(1,1,1)=kappax*a(0,1,1)
      a(1,1,1)=fac*(dx*a(0,1,1)+ddy*a(1,0,1)+ddz*a(1,1,0)+
     &         kappax*b(0,1,1))

      DO i=2,torderlim-2
         b(1,1,i)=kappax*a(0,1,i)
         b(1,i,1)=kappax*a(0,i,1)
         b(i,1,1)=kappay*a(i,0,1)

         a(1,1,i)=fac*(dx*a(0,1,i)+ddy*a(1,0,i)+ddz*a(1,1,i-1)
     &           -a(1,1,i-2)+kappax*b(0,1,i))
         a(1,i,1)=fac*(dx*a(0,i,1)+ddy*a(1,i-1,1)+ddz*a(1,i,0)
     &           -a(1,i-2,1)+kappax*b(0,i,1))
         a(i,1,1)=fac*(dy*a(i,0,1)+ddx*a(i-1,1,1)+ddz*a(i,1,0)
     &           -a(i-2,1,1)+kappay*b(i,0,1))
      END DO

C 1 index 1, others >=2

      DO i=2,torderlim-3
         DO j=2,torderlim-i
            b(1,i,j)=kappax*a(0,i,j)
            b(i,1,j)=kappay*a(i,0,j)
            b(i,j,1)=kappaz*a(i,j,0)

            a(1,i,j)=fac*(dx*a(0,i,j)+ddy*a(1,i-1,j)+ddz*a(1,i,j-1)
     &              -a(1,i-2,j)-a(1,i,j-2)+kappax*b(0,i,j))
            a(i,1,j)=fac*(dy*a(i,0,j)+ddx*a(i-1,1,j)+ddz*a(i,1,j-1)
     &              -a(i-2,1,j)-a(i,1,j-2)+kappay*b(i,0,j))
            a(i,j,1)=fac*(dz*a(i,j,0)+ddx*a(i-1,j,1)+ddy*a(i,j-1,1)
     &              -a(i-2,j,1)-a(i,j-2,1)+kappaz*b(i,j,0))

         END DO
      END DO

C all indices >=2

      DO k=2,torderlim-4
         DO j=2,torderlim-2-k
            DO i=2,torderlim-k-j
               b(i,j,k)=cf1(i)*kappa*(dx*a(i-1,j,k)-a(i-2,j,k))

               a(i,j,k)=fac*(ddx*cf2(i)*a(i-1,j,k)+ddy*a(i,j-1,k)
     &                 +ddz*a(i,j,k-1)-cf3(i)*a(i-2,j,k)
     &                 -a(i,j-2,k)-a(i,j,k-2)+
     &                 cf1(i)*kappa*(dx*b(i-1,j,k)-b(i-2,j,k)))
            END DO
         END DO
      END DO

      RETURN
      END SUBROUTINE COMP_TCOEFF
      

      
C#######################################################################
      RECURSIVE SUBROUTINE comp_ms_all(p,ifirst)
      use treecode, only: x,y,z,q,numpars
      IMPLICIT NONE
C
C REMOVE_NODE recursively removes each node from the
C tree and deallocates its memory for MS array if it
C exits.
C
      TYPE(tnode),POINTER :: p
      integer :: ifirst, counter
C local variables

      INTEGER :: i,err
      counter = 0
      IF (p%exist_ms .EQ. 0 .and. ifirst==0) THEN
         ALLOCATE(p%ms(0:torder,0:torder,0:torder),STAT=err)
         IF (err .NE. 0) THEN
            WRITE(6,*) 'Error allocating node MS! '
            STOP
         END IF
C         print*,"torder", torder
C         print*,"p%ms size", size(p%ms, DIM = 1)
C         print*,"dd", size(p%ms, DIM = 2)
         CALL COMP_MS(p,x,y,z,q,numpars)
         p%exist_ms=1

      END IF

      IF (p%num_children .GT. 0) THEN
          DO i=1,p%num_children
            CALL comp_ms_all(p%child(i)%p_to_tnode,0)
            counter = counter + 1
          END DO
      END IF
      print*, "counter:", counter
      RETURN

      END SUBROUTINE comp_ms_all      
      

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE COMP_MS(p,x,y,z,q,numpars)
      IMPLICIT NONE
C
C COMP_MS computes the moments for node P needed in the Taylor approximation
C
      INTEGER,INTENT(IN) :: numpars
      TYPE(tnode),POINTER :: p 
      INTEGER :: counter
      REAL(KIND=r8),DIMENSION(numpars),INTENT(IN) :: x,y,z,q
      REAL(KIND=r8), DIMENSION(:), ALLOCATABLE :: feature

C local variables

      INTEGER :: i,k1,k2,k3
      REAL(KIND=r8) :: dx,dy,dz,tx,ty,tz
     
      ALLOCATE(feature(torder))
      feature = 0.d0

      p%ms=0.0_r8
C      print*, "ibeg", p%ibeg
C      print*, "iend", p%iend
      DO i=p%ibeg,p%iend
         dx=x(i)-p%x_mid
         dy=y(i)-p%y_mid
         dz=z(i)-p%z_mid
         tz=1.0_r8
         DO k3=0,torder
            ty=1.0_r8
            DO k2=0,torder-k3
               tx=1.0_r8 
               DO k1=0,torder-k3-k2
                  p%ms(k1,k2,k3)=p%ms(k1,k2,k3)+q(i)*tx*ty*tz
                  tx=tx*dx
              END DO
              ty=ty*dy
            END DO
            tz=tz*dz
         END DO
      END DO   
      print*,"p%ms size", SIZE(p%ms)
C      print*,"feature size", SIZE(p%ms)
      
C      feature = RESHAPE(p%ms, (/1, torder/))

      RETURN

      END SUBROUTINE COMP_MS

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE COMPP_DIRECT(peng,ibeg,iend,x,y,z,q,kappa,numpars)
      IMPLICIT NONE
C
      INTEGER,INTENT(IN) :: ibeg,iend,numpars
      REAL(KIND=r8),DIMENSION(numpars),INTENT(IN) :: x,y,z,q
      REAL(KIND=r8),INTENT(IN) :: kappa
      REAL(KIND=r8),INTENT(OUT) :: peng

C local variables

      INTEGER :: j
      REAL(KIND=r8) :: dist2,dist,tx,ty,tz,pi
      
      pi=acos(-1.d0)
      peng=0.0_r8
C      print *,'tarpos',real(tarpos),real(kappa)
      DO j=ibeg,iend
         tx=x(j)-tarpos(1)
         ty=y(j)-tarpos(2)
         tz=z(j)-tarpos(3)
         dist2=tx*tx+ty*ty+tz*tz
         dist=SQRT(dist2)
C ############################################
C removal the singularities         
C      if (dist>1.d-6) then
C           peng=peng+q(j)*EXP(-kappa*dist)/dist/4/pi
            peng=peng+q(j)*EXP(-kappa*dist)/dist
C            print *,j,real(q(j)),real(exp(-kappa*dist)/dist),
C     &       real(dist),real(peng)
C      endif
C#############################################
      END DO   

      RETURN
      END SUBROUTINE COMPP_DIRECT

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE CLEANUP(p)
      IMPLICIT NONE
C
C CLEANUP deallocates allocated global variables and then
C calls recursive routine REMOVE_NODE to delete the tree.
C
      TYPE(tnode),POINTER :: p      

C local variables
  
      INTEGER :: err

      DEALLOCATE(cf, STAT=err)
      IF (err .NE. 0) THEN
         WRITE(6,*) 'Error deallocating Taylor variables! '
         STOP
      END IF

      DEALLOCATE(cf1,cf2,cf3,STAT=err)
      IF (err .NE. 0) THEN
         WRITE(6,*) 'Error deallocating Taylor variables! '
         STOP
      END IF

      DEALLOCATE(a,b,STAT=err)
      IF (err .NE. 0) THEN
         WRITE(6,*) 'Error deallocating Taylor variables! '
         STOP
      END IF      

      DEALLOCATE(xcopy,ycopy,zcopy,qcopy,orderarr,STAT=err)
      IF (err .NE. 0) THEN
         WRITE(6,*) 'Error deallocating copy variables! '
         STOP
      END IF  

      CALL REMOVE_NODE(p)
      DEALLOCATE(p, STAT=err)
      IF (err .NE. 0) THEN
         WRITE(6,*) 'Error deallocating root node! '
         STOP
      END IF 
      NULLIFY(p)         

      RETURN
      END SUBROUTINE CLEANUP

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      RECURSIVE SUBROUTINE REMOVE_NODE(p)
      IMPLICIT NONE
C
C REMOVE_NODE recursively removes each node from the
C tree and deallocates its memory for MS array if it
C exits.
C
      TYPE(tnode),POINTER :: p 

C local variables

      INTEGER :: i,err

      IF (p%exist_ms .EQ. 1) THEN
         DEALLOCATE(p%ms,STAT=err)
         IF (err .NE. 0) THEN
            WRITE(6,*) 'Error deallocating node MS! '
            STOP
         END IF               
      END IF

      IF (p%num_children .GT. 0) THEN
          DO i=1,p%num_children
            CALL REMOVE_NODE(p%child(i)%p_to_tnode)
            DEALLOCATE(p%child(i)%p_to_tnode,STAT=err)
            IF (err .NE. 0) THEN
               WRITE(6,*) 'Error deallocating node child! '
               STOP
            END IF                           
          END DO
      END IF 

      RETURN                
      END SUBROUTINE REMOVE_NODE      

      END MODULE treecode3d_procedures

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE TREECODE3D_YUKAWA(x,y,z,q,kappa,order,theta,
     &                             maxparnode,numpars,orderind,
     &                             tpoten)
      USE treecode3d_procedures
      IMPLICIT NONE
C
C TREECODE3D_YUKAWA approximates the electrostatic potential induced by
C charged particles undergoing screened Coulomb interactions (Yukawa) 
C interactions using a hierarchical oct-tree algorithm with body-cell 
C interactions approximated via Taylor expansions. 
C
C Input:
C -----
C
C X,Y,Z : (REAL one-dimensional arrays of length NUMPARS) Cartesian coordinates
C         of the particles 
C
C Q : (REAL one-dimensional array of length NUMPARS) Corresponding particle charge 
C
C KAPPA : (REAL) screened parameter
C
C ORDER :  (INTEGER) order of Taylor expansion used in the body-cell approximation
C
C THETA : (REAL) opening angle used in the BMAX MAC
C
C MAXPARNODE : (INTEGER) maximum number of particles allowed in a leaf.
C
C NUMPARS : (INTEGER) number of particles
C
C IFLAG   : (INTEGER) Flag determining what is to be computed:
C
C                IFLAG = 1  : potential energy
C                IFLAG = 2  : potential energy AND forces
C
C
C  **NOTE** : If you only need the potential energy set IFLAG=1. You MUST still 
C             pass the TFORCE array.  However, in this case it need only have 
C             one row, e.g.
C             
C
C Output:
C ------
C
C TPOTEN : (REAL one-dimensional array of length NUMPARS) ith entry of the arrray
C          is the approximate potential at the (reordered) ith particle
C
C
C ORDERIND : (INTEGER one-dimensional array of length NUMPARS) orderind(i) is new
C            position/index of the particle that was initially in the ith position
C            on entry to the subroutine.  The reordering results from building
C            the tree structure.  
C
C
      INTEGER,INTENT(IN) :: numpars,order,maxparnode
      REAL(KIND=r8),DIMENSION(numpars),INTENT(INOUT) :: x,y,z,q
      REAL(KIND=r8),DIMENSION(numpars),INTENT(INOUT) :: tpoten
      INTEGER,DIMENSION(numpars),INTENT(INOUT) :: orderind
      REAL(KIND=r8),INTENT(IN) :: kappa,theta

C local variables

      TYPE(tnode),POINTER :: troot
      INTEGER :: i,level,err
      REAL(KIND=r8), DIMENSION(6) :: xyzminmax

C variables needed for cpu time

      REAL(KIND=r8) :: totaltime,timebeg,timeend,timetree

C Call SETUP to allocate arrays for Taylor expansions
C and setup global variables. Also, copy variables into global copy arrays. 

      CALL SETUP(x,y,z,q,numpars,order,xyzminmax)

C nullify pointer to root of tree (TROOT) and create tree

      NULLIFY(troot)  

C creating tree

      level=0
      minlevel=50000
      maxlevel=0
      

      CALL CPU_TIME(timebeg)
      CALL CREATE_TREE(troot,1,numpars,x,y,z,q,maxparnode,
     &                 xyzminmax,level,numpars)
      CALL CPU_TIME(timeend)
      totaltime=timeend-timebeg
      WRITE(6,*) 'Time to create tree (secs):',totaltime 

C compute potential or potential 

      CALL CPU_TIME(timebeg)
         WRITE(6,*) ' '
         WRITE(6,*) 'Computing potential - treecode '
         CALL TREE_COMPP(troot,x,y,z,q,kappa,theta,tpoten,
     &        numpars)
      CALL CPU_TIME(timeend)
      timetree=timeend-timebeg
    
      WRITE(6,*) 'Treecode time (secs):',timetree
      WRITE(6,*) ' '
      WRITE(6,*) 'TOTAL time (tree creation + computation): ',
     &            totaltime + timetree
     
      DO i=1,numpars
         orderind(orderarr(i)) = i
      END DO

C Call CLEANUP to deallocate global variables and tree structure.

      WRITE(6,*) 'Deallocating tree structure'

      CALL CLEANUP(troot)

      END SUBROUTINE TREECODE3D_YUKAWA

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE PARTITION(a,b,c,q,indarr,ibeg,iend,val,midind,numpars)
      IMPLICIT NONE
C
C PARTITION determines the index MIDIND, after partitioning
C in place the  arrays A,B,C and Q,  such that 
C A(IBEG:MIDIND) <= VAL and  A(MIDIND+1:IEND) > VAL. 
C If on entry IBEG > IEND or  A(IBEG:IEND) > VAL then MIDIND
C is returned as IBEG-1. 
C 
      INTEGER,PARAMETER :: r8=SELECTED_REAL_KIND(12)
      INTEGER, INTENT(IN) :: numpars,ibeg,iend
      REAL(KIND=r8),DIMENSION(numpars),INTENT(INOUT) :: a,b,c,q
      INTEGER,DIMENSION(numpars),INTENT(INOUT) :: indarr
C     INTEGER,DIMENSION(1),INTENT(INOUT) :: indarr   
      INTEGER, INTENT(INOUT) :: midind   
      REAL(KIND=r8) val

C local variables

      REAL(KIND=r8) ta,tb,tc,tq
      INTEGER lower,upper,tind

      IF (ibeg .LT. iend) THEN

C temporarily store IBEG entries and set A(IBEG)=VAL for 
C the partitoning algorithm.  

         ta=a(ibeg)
         tb=b(ibeg)
         tc=c(ibeg)
         tq=q(ibeg)
         tind=indarr(ibeg)
         a(ibeg)=val 
         upper=ibeg
         lower=iend

         DO WHILE (upper .NE. lower)
            DO WHILE ((upper .LT. lower) .AND. (val .LT. a(lower)))
                  lower=lower-1
            END DO
            IF (upper .NE. lower) THEN
               a(upper)=a(lower)
               b(upper)=b(lower)
               c(upper)=c(lower)
               q(upper)=q(lower)
               indarr(upper)=indarr(lower)
            END IF
            DO WHILE ((upper .LT. lower) .AND. (val .GE. a(upper)))
                  upper=upper+1
            END DO
            IF (upper .NE. lower) THEN
               a(lower)=a(upper)
               b(lower)=b(upper)
               c(lower)=c(upper)
               q(lower)=q(upper)
               indarr(lower)=indarr(upper)
            END IF
         END DO
         midind=upper

C replace TA in position UPPER and change MIDIND if TA > VAL 

         IF (ta .GT. val) THEN
            midind=upper-1
         END IF
         a(upper)=ta
         b(upper)=tb
         c(upper)=tc
         q(upper)=tq
         indarr(upper)=tind

      ELSEIF (ibeg .EQ. iend) THEN
         IF (a(ibeg) .LE. val) THEN
            midind=ibeg
         ELSE
            midind=ibeg-1
         END IF
      ELSE
         midind=ibeg-1
      END IF

      RETURN
      END SUBROUTINE PARTITION

