MODULE treecode_procedures
IMPLICIT NONE

!-- define tree structure
TYPE tnode_pointer
      TYPE(tnode), POINTER :: p_to_tnode
END TYPE tnode_pointer
      
TYPE tnode
     INTEGER      :: node_idx
     INTEGER      :: numpar,ibeg,iend
     REAL*8       :: x_min,y_min,z_min
     REAL*8       :: x_max,y_max,z_max
     REAL*8       :: x_mid,y_mid,z_mid
     REAL*8       :: radius
     INTEGER      :: level,num_children,exist_ms
     REAL*8,DIMENSION(:,:,:),POINTER :: ms
     TYPE(tnode_pointer), DIMENSION(8) :: child
END TYPE tnode

INTEGER,ALLOCATABLE,DIMENSION(:)  :: orderarr
INTEGER icounter

CONTAINS
    !SETUP set the Cartesian coordinates of the smallest box containing the particles
    SUBROUTINE SETUP(x,y,z,q,numpars,xyzminmax)
    IMPLICIT NONE
    
    INTEGER i,err      
    INTEGER,INTENT(IN) :: numpars
    REAL*8,DIMENSION(numpars),INTENT(IN) :: x,y,z,q
    REAL*8,INTENT(INOUT),DIMENSION(6) :: xyzminmax

    !find bounds of Cartesian box enclosing the particles
    xyzminmax(1)=MINVAL(x(1:numpars))
    xyzminmax(2)=MAXVAL(x(1:numpars))
    xyzminmax(3)=MINVAL(y(1:numpars))
    xyzminmax(4)=MAXVAL(y(1:numpars))
    xyzminmax(5)=MINVAL(z(1:numpars))
    xyzminmax(6)=MAXVAL(z(1:numpars))

    ALLOCATE(orderarr(numpars),STAT=err)
    IF (err .NE. 0) THEN
        WRITE(6,*) 'Error allocating copy variables! '
        STOP
    END IF

    DO i=1,numpars
        orderarr(i)=i
    END DO  

    icounter=0

    END SUBROUTINE SETUP

    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    RECURSIVE SUBROUTINE CREATE_TREE(p,ibeg,iend,x,y,z,q,xyzmm,level,numpars,ef_p,ef_L)
    IMPLICIT NONE

    !CREATE_TREE recursively create the tree structure. Node P is
    !input, which contains particles indexed from IBEG to IEND. After
    !the node parameters are set subdivision occurs 
    !until the desired level is reached.
    !Real array XYZMM contains the min and max values of the coordinates
    !of the particle in P, thus defining the box.   

    TYPE(tnode),POINTER :: p
    INTEGER,INTENT(IN) :: ibeg,iend,level,numpars,ef_p,ef_L
    REAL*8,DIMENSION(numpars),INTENT(INOUT) :: x,y,z,q
    REAL*8,DIMENSION(6),INTENT(IN) :: xyzmm
          
    !local variables
    REAL*8 :: x_mid,y_mid,z_mid,xl,yl,zl,lmax,t1,t2,t3
    INTEGER, DIMENSION(8,2) :: ind
    REAL*8, DIMENSION(6,8) :: xyzmms
    INTEGER :: i,j,limin,limax,err,loclev,numposchild
    REAL*8, DIMENSION(6) ::  lxyzmm
         
    !allocate pointer 
    ALLOCATE(p,STAT=err)
    IF (err .NE. 0) THEN
        WRITE(6,*) 'Error allocating pointer! '
        STOP
    END IF

    !if (level==2) then
        icounter=icounter+1 !counter for number of clusters
    !end if
    
    !set node fields: number of particles, exist_ms and xyz bounds 
    p%numpar=iend-ibeg+1
    p%exist_ms=0

    p%x_min=xyzmm(1)
    p%x_max=xyzmm(2)
    p%y_min=xyzmm(3)
    p%y_max=xyzmm(4)
    p%z_min=xyzmm(5)
    p%z_max=xyzmm(6)        
    
    !compute aspect ratio
    xl=p%x_max-p%x_min
    yl=p%y_max-p%y_min
    zl=p%z_max-p%z_min
    lmax=MAX(xl,yl,zl)

    !midpoint coordinates , RADIUS and SQRADIUS 
    p%x_mid=(p%x_max+p%x_min)/2.d0
    p%y_mid=(p%y_max+p%y_min)/2.d0
    p%z_mid=(p%z_max+p%z_min)/2.d0
    t1=p%x_max-p%x_mid
    t2=p%y_max-p%y_mid
    t3=p%z_max-p%z_mid
    p%radius=SQRT(t1*t1+t2*t2+t3*t3)

    !set particle limits, tree level of node, and nullify children pointers
    p%ibeg=ibeg
    p%iend=iend
    p%level=level
    p%num_children=0

    IF (p%exist_ms .EQ. 0 .AND. level == level) THEN 
        !WG: level == level can be modified to output features at any level
        ALLOCATE(p%ms(0:ef_p,0:ef_p,0:ef_p),STAT=err)
        p%ms=0.d0 !this is needed to avoid random values assigned
        IF (err .NE. 0) THEN
            WRITE(6,*) 'Error allocating node MS! '
            STOP
        END IF
        CALL COMP_MS(p,x,y,z,q,numpars)
        p%exist_ms=1
        !print *,icounter, level, sum(p%ms(0:ef_p,0:ef_p,0:ef_p))
    END IF
    
    IF (p%level < ef_L) THEN
        DO i=1,8
            NULLIFY(p%child(i)%p_to_tnode)
        END DO

        ! set IND array to 0 and then call PARTITION routine. IND array holds indices
        ! of the eight new subregions. Also, setup XYZMMS array in case SHRINK=1
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

        CALL PARTITION_8(x,y,z,q,xyzmms,xl,yl,zl,lmax,numposchild, &
        x_mid,y_mid,z_mid,ind,numpars)

        ! create children if indicated and store info in parent
        loclev=level+1
        DO i=1,numposchild
            !IF (ind(i,1) .LE. ind(i,2)) THEN
                p%num_children=p%num_children+1
                lxyzmm=xyzmms(:,i)
                CALL CREATE_TREE(p%child(p%num_children)%p_to_tnode,ind(i,1),&
                ind(i,2),x,y,z,q,lxyzmm,loclev,numpars,ef_p,ef_L)
            !ELSE ! WG: zero particle in the cluster, commenting IF seems ok!
            !    print *,i,level,ind(i,1),ind(i,2)
            !END IF
        END DO
    END IF

    END SUBROUTINE CREATE_TREE


!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    SUBROUTINE PARTITION_8(x,y,z,q,xyzmms,xl,yl,zl,lmax,numposchild, &
        x_mid,y_mid,z_mid,ind,numpars)
    IMPLICIT NONE

    !PARTITION_8 determines the particle indices of the eight sub boxes
    !containing the particles after the box defined by particles I_BEG
    !to I_END is divided by its midpoints in each coordinate direction.
    !The determination of the indices is accomplished by the subroutine
    !PARTITION. A box is divided in a coordinate direction as long as the
    !resulting aspect ratio is not too large. This avoids the creation of
    !"narrow" boxes in which Talyor expansions may become inefficient.
    !On exit the INTEGER array IND (dimension 8 x 2) contains
    !the indice limits of each new box (node) and NUMPOSCHILD the number 
    !of possible children.  If IND(J,1) > IND(J,2) for a given J this indicates
    !that box J is empty.

    INTEGER, INTENT(IN) :: numpars
    REAL*8,DIMENSION(numpars),INTENT(INOUT) :: x,y,z,q
    INTEGER, DIMENSION(8,2),INTENT(INOUT) :: ind
    REAL*8,DIMENSION(6,8),INTENT(INOUT) :: xyzmms
    REAL*8, INTENT(IN) :: x_mid,y_mid,z_mid,xl,yl,zl,lmax
    INTEGER,INTENT(INOUT) :: numposchild

    !local variables
    INTEGER :: temp_ind,i
    REAL*8 :: critlen

    numposchild=1
    critlen=lmax/sqrt(2.d0)

    !IF (xl .GE. critlen) THEN !WG: criterion for treecode, commentted for uniform ML features
        CALL PARTITION(x,y,z,q,orderarr,ind(1,1),ind(1,2),x_mid,temp_ind,numpars)
        ind(2,1)=temp_ind+1
        ind(2,2)=ind(1,2)
        ind(1,2)=temp_ind
        xyzmms(:,2)=xyzmms(:,1)
        xyzmms(2,1)=x_mid
        xyzmms(1,2)=x_mid
        numposchild=2*numposchild
    !END IF 

    !IF (yl .GE. critlen) THEN
        DO i=1,numposchild
            CALL PARTITION(y,x,z,q,orderarr,ind(i,1),ind(i,2),y_mid,temp_ind,numpars)
            ind(numposchild+i,1)=temp_ind+1
            ind(numposchild+i,2)=ind(i,2)
            ind(i,2)=temp_ind
            xyzmms(:,numposchild+i)=xyzmms(:,i)
            xyzmms(4,i)=y_mid
            xyzmms(3,numposchild+i)=y_mid
        END DO
        numposchild=2*numposchild
    !END IF

    !IF (zl .GE. critlen) THEN
        DO i=1,numposchild
            CALL PARTITION(z,x,y,q,orderarr,ind(i,1),ind(i,2),z_mid,temp_ind,numpars)
            ind(numposchild+i,1)=temp_ind+1
            ind(numposchild+i,2)=ind(i,2)
            ind(i,2)=temp_ind
            xyzmms(:,numposchild+i)=xyzmms(:,i)
            xyzmms(6,i)=z_mid
            xyzmms(5,numposchild+i)=z_mid
        END DO
        numposchild=2*numposchild
    !END IF
      
    RETURN 
    END SUBROUTINE PARTITION_8


   !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    SUBROUTINE COMP_MS(p,x,y,z,q,numpars)
    USE COMDATA, only: ef_p,nMoMs,features  
    IMPLICIT NONE

    !COMP_MS computes the moments for node P needed in the Taylor approximation
    INTEGER,INTENT(IN) :: numpars
    TYPE(tnode),POINTER :: p 
    REAL*8,DIMENSION(numpars),INTENT(IN) :: x,y,z,q

    !local variables
    INTEGER :: i,k1,k2,k3
    REAL*8 :: dx,dy,dz,tx,ty,tz

    p%ms=0.d0
    DO i=p%ibeg,p%iend
        dx=x(i)-p%x_mid
        dy=y(i)-p%y_mid
        dz=z(i)-p%z_mid
        tz=1.d0
        DO k3=0,ef_p
            ty=1.d0
            DO k2=0,ef_p-k3
                tx=1.d0 
                DO k1=0,ef_p-k3-k2
                    p%ms(k1,k2,k3)=p%ms(k1,k2,k3)+q(i)*tx*ty*tz
                    tx=tx*dx
                END DO
                ty=ty*dy
            END DO
            tz=tz*dz
         END DO
    END DO   

    !counter the number of momemnts computed & store into feature vectors
    DO k3=0,ef_p
        DO k2=0,ef_p-k3 
            DO k1=0,ef_p-k3-k2
                nMoMs=nMoMs+1
                features(nMoMs)=p%ms(k1,k2,k3)
            END DO
        END DO
    END DO
    
    RETURN

    END SUBROUTINE COMP_MS

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    SUBROUTINE CLEANUP(p)
    USE COMDATA
    IMPLICIT NONE

    TYPE(tnode),POINTER :: p  
    ! local variables
    INTEGER :: err   

    CALL REMOVE_NODE(p)
    DEALLOCATE(p, STAT=err)
    IF (err .NE. 0) THEN
        WRITE(6,*) 'Error deallocating root node! '
        STOP
    END IF 
    NULLIFY(p)  

    DEALLOCATE(x,y,z,q,features,orderarr,STAT=err)   
    IF (err .NE. 0) THEN
        WRITE(*,*) 'Error deallocating variables!'
        STOP
    END IF       

    RETURN
    END SUBROUTINE CLEANUP

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    RECURSIVE SUBROUTINE REMOVE_NODE(p)
    IMPLICIT NONE

    !REMOVE_NODE recursively removes each node from the
    !tree and deallocates its memory for MS array if it
    !exits.

    TYPE(tnode),POINTER :: p 

    ! local variables

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

END MODULE treecode_procedures
!########################################################################
!########################################################################

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
SUBROUTINE PARTITION(a,b,c,q,indarr,ibeg,iend,val,midind,numpars)
IMPLICIT NONE
! PARTITION determines the index MIDIND, after partitioning
! in place the  arrays A,B,C and Q,  such that 
! A(IBEG:MIDIND) <= VAL and  A(MIDIND+1:IEND) > VAL. 
! If on entry IBEG > IEND or  A(IBEG:IEND) > VAL then MIDIND
! is returned as IBEG-1. 
! 
INTEGER, INTENT(IN) :: numpars,ibeg,iend
REAL*8,DIMENSION(numpars),INTENT(INOUT) :: a,b,c,q
INTEGER,DIMENSION(numpars),INTENT(INOUT) :: indarr
!INTEGER,DIMENSION(1),INTENT(INOUT) :: indarr   
INTEGER, INTENT(INOUT) :: midind   
REAL*8 val

!local variables

REAL*8 ta,tb,tc,tq
INTEGER lower,upper,tind

IF (ibeg .LT. iend) THEN

    !temporarily store IBEG entries and set A(IBEG)=VAL for 
    !the partitoning algorithm.  

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

    !replace TA in position UPPER and change MIDIND if TA > VAL 
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





