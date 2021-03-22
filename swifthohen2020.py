#!/usr/bin/python3
"""
# -------------------------------------------------------------------------- #
#                                  PREFACE                                   #
# -------------------------------------------------------------------------- #
#      FINITE DIFFERENCES METHOD FOR THE SWIFT-HOHENBERG EQUATION (SH3)      #
#           WITH STRICT IMPLEMENTATION OF THE LYAPUNOV FUNCTIONAL            #
#                                                                            #
# -------------------------------------------------------------------------- #
# The code was develop to study pattern formation using the evolution        #
# equation initially derived to describe Rayleigh-Benard convective patterns.#
#                                                                            #
# The major paper reference for this effort was considered:                  #
#                                                                            #
#    C.I. Christov and J. Pontes. Numerical scheme for swift-hohenberg       #
#    equation with strict implementation of lyapunov functional.             #
#    Mathematical and Computer Modelling, 35, 2002.                          #                                           #
#                                                                            #
# The governing SH3 equation reads:                                          #
#                                                                            #
#             u_t=epsilon*u-d*((k0^2+\nabla^2)^2)*u+g1*u^2-g*u^3             # 
#                                                                            #      
#                                                                            #                           
# This work contains a bidimensional approach,i.e: \nabla=u_xx+u_yy          #                                                                                                
#                                                                            #                          
# Random initial conditions and Generalized Dirichlet Boundary conditions    #                                                                                                    
# assumed:                                                                   #
# i,j=rows,columns                                                           #
#                                                                            #                           
#                                   u=du/dx=0                                #                                                                        
#                               --------------                               #                                                                         
#                               |            |                               #                                                                         
#                               |   random   |                               #                                                                         
#                (H)  u=du/dx=0 |   (t=0)    | u=du/dx=0                     #                                                                                   
#                               |            |                               #                                                                         
#                               |            |                               #                                                                         
#                               --------------                               #                                                                         
#                                  u=du/dx=0                                 #
#                                                                            #                                                                        
#                                     (L)                                    #
#                                                                            #                                                               
# The grid is structured, staggered (for boundary conditions) and uniform.   #                                                                                                                                                                                                  
#                                                                            #                           
# This is a semi-implicit second order scheme accordingly with Stabilizing   #                                                                                                  
# Correction and Crank-Nicolson schemes to preserve numerical stability.     #                                                                                             
# Therefore,iterations are needed to control the nonlinear terms.            #  
#                                                                            #  
# Typical parameters for Rayleigh-Benard convection pattern formation:       #
#                                                                            #  
#                                q0=3.1172                                   #  
#                                g=12.9                                      #  
#                                d=0.015                                     #                                                                                                     
#                                                                            #                                          
# -------------------------------------------------------------------------- #                                                                                                      
"""
__version__=2.0
__author__="""Daniel Lessa Coelho (danielcoelho.uerj@gmail.com)"""
# -------------------------------------------------------------------------- #
#-*-coding:utf-8-*-                                                          #
#                                                                            #
#                                 PACKAGES                                   #
#                                                                            #
# Loading only the necessary packages.                                       #
# -------------------------------------------------------------------------- #
# from glob import glob
import os
from scipy import sparse
import scipy as sp
from scipy import*
from scipy.sparse.linalg.dsolve import linsolve
import sys
import time
from datetime import datetime
SimStartsAt=datetime.now()
# -------------------------------------------------------------------------- #
#                      PROGRESS BAR FUNCTION DEFINITION                      #
#                                                                            #
# Allows visualization of the code evolution.                                #
# -------------------------------------------------------------------------- #
def progressbar(it,prefix="",size=60):
    count=len(it)
    def _show(_i):
        x=int(size*_i/count)
        sys.stdout.write(
            "%s[%s%s] %i/%i\r" % (prefix,"#"*x,"."*(size-x),_i+ni,
                                  n))
        sys.stdout.flush()
    _show(0)
    for i,item in enumerate(it):
        yield item
        _show(i+1)
    sys.stdout.write("\n")
    sys.stdout.flush()
# -------------------------------------------------------------------------- #
#                  SEARCHING LAST STORAGED TIME RESULTS                      #
#                                                                            #
# Searching and setting the initial condition from interrupted simulation.   #
# Uncomment below to apply the search.                                       #
# -------------------------------------------------------------------------- #
def SearchInitialCondition():
    folder='./L1NORM/';
    # folder='../Circular domain/L1norm/';
    InitialTime=0.0;dt=0.0;
    for i in os.listdir(folder):
        if i[:2]=='L1':                  # certify searching L1 files
            if i[-4:]=='.csv':           # certify searching .csv files
                if i[13]=='_':           # certify not picking the global L1 file
                    new_time=int(i[14:-4]);
                    if InitialTime<new_time:
                        InitialTime=new_time;
                else:
                    dt=float(i[10:-4]);#print(dt);
                    # print('Found "'+str(i)+'" -- OK!');
    # print('Last time found: '+str(InitialTime));
    un=sp.loadtxt('./RESULTS/u_result_'+str(dt)+
                  '_'+str(int(InitialTime))+
                  '.csv', delimiter=',');
    return dt,InitialTime,un
# -------------------------------------------------------------------------- #
#                       RANDOM INITIAL CONDITIONS                            #
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
def RandomInitialConditions(nx,ny):
  uamp=sp.sqrt(epsilon/g);
  print('Amplitude of the initial condition:'+str(uamp));
  dx=L/(nx); dy=dx; h=dx        # grid spacing (\Delta x=\Delta y=h) 
  x=sp.arange(0,L,dx);
  y=sp.arange(0,H,dy);
  X,Y=sp.meshgrid(x,y,sparse=True);
  def MMS(x,y):
      return uamp*cos((q0*(y)))+0*x
  un=MMS(X,Y)
  # un=sp.random.uniform(-uamp,uamp,(ny,nx))
  # un=sp.loadtxt('./RESULTS/u_result_'+str(dt)+'_'+
  #                 str(InitialTime)+'.csv',delimiter=',')
  for Z in (un,un):
      Z[0,:]=0
      Z[-1,:]=0
      Z[:,0]=0
      Z[:,-1]=0
      Z[1,:]=Z[0,:]
      Z[-2,:]=Z[-1,:]
      Z[:,1]=Z[:,0]
      Z[:,-2]=Z[:,-1]
  sp.savetxt('./RESULTS/u_result_'+str(dt)+'_'+str(int(0))+'.csv',un,
             delimiter=',')
  sp.savetxt('./RESULTS/u_result_'+str(dt)+'_'+str(InitialTime)+'.csv',un,
             delimiter=',')
  dx=L/(nx-2); dy=dx; h=dx        # grid spacing (\Delta x=\Delta y=h) 
  return un
# aqui precisa de um if pra acionar ou nao a geracao aleatoria daa condicao inicial:
# un=RandomInitialConditions(nx,ny);
# un=sp.loadtxt('./RESULTS/u_result_'+str(dt)+'_0.0.csv', delimiter=',')
# U0=un.copy()
# -------------------------------------------------------------------------- #
#                            FILING LOG BEFORE                               #
#                                                                            #
# Save a log file before simulation starts.                                  #
# -------------------------------------------------------------------------- #
# sys.stdout=open("LOGBEFORE.txt","w")
# -------------------------------------------------------------------------- #
#                           EQUATION PARAMETERS                              #
#                                                                            #
# Make folders to store RESULTS. If they don't exist, they're created.       #
# -------------------------------------------------------------------------- #
epsilon=.1;                      # bifurcation/control (forcing) parameter
d=1.0;                           # "diffusion coefficient"
g=1.0;                           # cubic nonlinearity coefficient
g1=0.0;                          # quadratic nonlinearity coefficient (hexs)
q0=1.0;                          # critical wavenumber
# -------------------------------------------------------------------------- #
#                  GENERALIZED DIRICHLET BOUNDARY CONDITIONS                 #
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
u_x0=0.;du_dx_x0=0.
u_xL=0.;du_dx_xL=0.
u_y0=0.;du_dy_y0=0.
u_yH=0.;du_dy_yH=0.
# -------------------------------------------------------------------------- #
#                         SIMULATION PARAMETERS (PAPER)                      #  
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
# dt=WantedTimeStep                         # time step
# InitialTime=0.                 # initial time
# FinalTime=1e6                  # final time
# n=int((FinalTime)/dt)          # number of steps
# # print(n)                     # show number of steps
# ni=int(InitialTime/dt)
# # res=16                       # mesh resolution: grid points per wavelength
# lambda0=(2*pi)/q0              # critical wavelength
# L=10.#ncx*lambda0              # domain length (x)
# H=5.#ncy*lambda0               # domain height (y)
# ncx=L/lambda0                  # number of wavelengths in x direction
# ncy=H/lambda0                  # number of wavelengths in y direction
# nx=82#ncx*res                  # total grid points in x direction
# ny=42#ncy*res                  # total grid points in y direction
# res=nx/ncx
# dx=L/(nx-2); dy=dx; h=dx       # grid spacing (\Delta x=\Delta y=h)                          
# IterationsLimit=1e6            # iterations limit (sufficiently large)
# Precision=1e00                 # convergence criteria  
# print('TOTAL TIME STEPS '+str('%0.0e')%(n)+' OK!')
# print('MESH '+str(nx)+' x '+str(ny)+' OK!')
# print('MESH SPACING dx = '+str(dx)+' dy = '+str(dy)+' OK!')
# -------------------------------------------------------------------------- #
#                          SIMULATION PARAMETERS                             #  
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
ncx=8                           # number of wavelengths in x direction
ncy=8                           # number of wavelengths in y direction
res=16                          # mesh resolution: grid points per wavelength
lambda0=(2*pi)/q0               # critical wavelength
L=ncx*lambda0                   # domain length (x)
H=ncy*lambda0                   # domain height (y)
nx=ncx*res                      # total grid points in x direction
ny=ncy*res                      # total grid points in y direction
dx=L/(nx-2); dy=dx; h=dx        # grid spacing (\Delta x=\Delta y=h)                        
IterationsLimit=1e6             # iterations limit (sufficiently large)
Precision=1e-08                 # convergence criteria  
L1cutoff=5e-7                   # L1 stop criteria
L1pointer=0                     # Inside loop pointer for L1 savings 
SaveL1each=1                    # Periodicity of L1 savings 
dt=0.5                          # default final time
# -------------------------------------------------------------------------- #
#                             STORING RESULTS                                #
#                                                                            #
# Make folders to store RESULTS. If they don't exist, they're created.       #
# -------------------------------------------------------------------------- #
# sys.stdout=open("LOGBEFORE.txt","w")
# os.mkdir('RESULTS')
# os.mkdir('L1norm')
# os.mkdir('addinfo')
# Those cannot be True at the same time.
# print('Simulation Time step? (dt)');
WantedTimeStep=0.5 #float(input());
print('Running the code for the first time? (y/n)');
run=input();
if run=='y' or run=='Y' or run=='yes':
  StartFromPrevious=False;
  StartFromPattern=True;
else:
  print('Continuing simulation...');
  StartFromPrevious=True;
  StartFromPattern=False;
dirName=['RESULTS','L1NORM','ADDINFO','LYAPFUNC']
for fol in range(4):
    folder=dirName[fol];
    if not os.path.exists(folder) and StartFromPrevious==False:
      RestoreLists="No"
      if fol==0:
        print('Starting from random initial conditions!');
      if folder=='RESULTS' and os.path.exists('RESULTS'):
        print("Directory RESULTS already exists");
      else:
        os.mkdir(folder);
        print("Directory "+str(folder)+" Created ");
      dt=WantedTimeStep;                         # time step
      InitialTime=0.                  # initial time
      if StartFromPattern==False or not os.path.exists(
              './RESULTS/u_result_'+str(dt)+'_0.csv'):
        un=RandomInitialConditions(nx,ny);U0=un.copy();

    if os.path.exists('RESULTS'):
      if StartFromPattern==True:
        RestoreLists="No"
        dt=WantedTimeStep;              # time step
        InitialTime=0.                  # initial time
        un=sp.loadtxt('./RESULTS/u_result_'+
                      str(dt)+'_0.0.csv',delimiter=',');
        un[0,:]=0;
        un[-1,:]=0;
        un[:,0]=0;
        un[:,-1]=0;
        un[1,:]=un[0,:];
        un[-2,:]=un[-1,:];
        un[:,1]=un[:,0];
        un[:,-2]=un[:,-1];
        sp.savetxt('./RESULTS/u_result_'+str(dt)+'_'+str(int(0))+'.csv',un,
                   delimiter=',')
        sp.savetxt('./RESULTS/u_result_'+str(dt)+'_'+str(InitialTime)+'.csv',
                   un,delimiter=',')
        U0=un.copy();
      if StartFromPrevious==True:
        RestoreLists="Yes";
        if fol==0:
          print('Starting from last stored time step!');
        dt,InitialTime,un=SearchInitialCondition();U0=un.copy();
        print("Directory "+str(folder)+" already exists");
# -------------------------------------------------------------------------- #
FinalTime=1e6                   # final time
n=int((FinalTime)/dt)           # number of steps
# print(n)                      # show number of steps
ni=int(InitialTime/dt)          # total number of steps until initial time
# -------------------------------------------------------------------------- #
#                   SCREEN SHOWING SIMULATION PARAMETERS                     #  
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
print('TOTAL TIME STEPS '+str('%0.0e')%(n)+' OK!')
print('MESH '+str(nx)+' x '+str(ny)+' OK!')
print('MESH SPACING dx = '+str(dx)+' dy = '+str(dy)+' OK!')
print(
'-------------------------- SIMULATION PARAMETERS --------------------------')
print('INITIAL TIME: '+str(InitialTime)+
      '  FINAL TIME: '+str('%0.1e')%(FinalTime)+'  '
      'dt: '+str(dt)+'  TOTAL TIME STEPS: '+str('%0.1e')%(n-ni));
print('L,H: '+str('%0.1e')%(L)+','+str('%0.1e')%(H)+'    NX,NY: '+str(nx)+
    ','+str(ny)+'    RESOLUTION: '+str('%0.1e')%(res));
print('WAVELENGTHS IN X,Y DIRECTIONS: '+
      str('%0.3f')%(ncx)+','+str('%0.3f')%(ncy));
print('CONVERGENCE PRECISION/ITERATIONS: '+str('%0.1e')%(Precision)+
      ','+str('%0.1e')%(IterationsLimit));
# -------------------------------------------------------------------------- #
#                    EQUATION PARAMETER: NONUNIFORM EPSILsON                 #                
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
epsilonmin=0.0
epsilonmax=0.2
maxgaussian=0.2
Deltaep=epsilonmax-epsilonmin
coefangular=Deltaep/L
def funcepsilon(x):
    return coefangular*x+epsilonmin
epsilonarray=[]
for j in range(nx):
    epsilonarray=sp.append(epsilonarray,funcepsilon((j-0.5)*dx))
# def gaussianepsilon(size,fwhm=nx,center=None):
#     """ Make a square gaussian kernel.
#     size is the length of a side of the square
#     fwhm is full-width-half-maximum,which
#     can be thought of as an effective radius.
#     """
#     x=sp.arange(0,size,1,float)
#     y=x[:,sp.newaxis]
#     R=1./size
    
#     if center is None:
#         x0=y0=size // 2
#     else:
#         x0=center[0]
#         y0=center[1]
#     return maxgaussian*sp.exp(-((x-x0)**2+(y-y0)**2)/R/fwhm**2)+epsilonmin
# epsilonmatrix=gaussianepsilon(nx)
# -------------------------------------------------------------------------- #
#                STRICT IMPLEMENTATION OF LYAPUNOV FUNCTIONAL                #
#                                                                            #
# Evaluating the lyapunov functional for the SH3.                            #
# -------------------------------------------------------------------------- #
def LyapunovFunctional(A):
  v=A.copy()
  # ncx=5;                  # number of wavelengths in x direction
  # ncy=5;                  # number of wavelengths in y direction
  # res=16;                 # mesh resolution: grid points per wavelength
  # L=ncx*lambda0;          # domain length (x)
  # H=ncy*lambda0;          # domain height (y)
  dx=L/(len(A[0,:])-2);dy=H/(len(A[:,0])-2);
  h=dx;nx1=len(A[0,:])-1;ny1=len(A[:,0])-1;
  EPSILON=0.0;Sv2=0.0;Sv3=0.0;Sv4=0.0;Sv_grad=0.0;Sv_d2=0.0;
  Iv2=0.0;Iv3=0.0;Iv4=0.0;Iv_grad=0.0;Iv_d2=0.0;
  for i in range(1,ny1):
    for j in range(1,nx1):
      EPSILON=epsilon;#epsilonarray[j];
      Sv2=.5*(-EPSILON+d*q0**4)*v[i,j]*v[i,j];
      Sv3=-(g1/3.)*v[i,j]*v[i,j]*v[i,j];
      Sv4=g*.25*v[i,j]*v[i,j]*v[i,j]*v[i,j];
      Sv_grad=(.5*d*q0**2/h**2)*(
            (v[i,j+1]-v[i,j])**2+
              (v[i,j]-v[i,j-1])**2+
              (v[i+1,j]-v[i,j])**2+
              (v[i,j]-v[i-1,j])**2
              );
      Sv_d2=(d*.5/h**4)*(
          v[i,j+1]+v[i,j-1]+
          v[i+1,j]+v[i-1,j]-4.*v[i,j]
          )**2;
      Iv2+=Sv2;
      Iv3+=Sv3;
      Iv4+=Sv4;
      Iv_grad+=Sv_grad;
      Iv_d2+=Sv_d2;
  LyapunovFunctionalValue=(Iv2+Iv3+Iv4-Iv_grad+Iv_d2);
  V=(ncx*res)**2-4*(ncx*res)+4;
  # print(EPSILON,0.3*V);
  # print('i = '+str(i)+' j = '+str(j))
  return LyapunovFunctionalValue
LyapFuncNew=LyapunovFunctional(un);
# -------------------------------------------------------------------------- #
#                           TIME DERIVATIVE OF PSI                           #
#                                                                            #
# Evaluating the time derivative of psi for comparison.                      #
# -------------------------------------------------------------------------- #
def PsiTimeDerivative(unew,uold):
  TotalSumddt=0.0;ERROR=0.0;N=nx*ny;NULL=0.0;
  for i in range(0,ny):
    for j in range(0,nx):
      Valuex=((unew[i,j]-uold[i,j])/dt)**2;
      error=-2*(unew[i,j]-uold[i,j]);
      TotalSumddt+=Valuex;
      ERROR+=error;
  # ERROR=ERROR+NULL*(-dt-N*(-h**2+dt**2));
  Integralddt2=-TotalSumddt;#*(1./dt)**2;
  return Integralddt2,ERROR
def PsiTimeDerivative2(unew,uold):
  f=unew.copy();
  fn=uold.copy();
  TotalSum=0.0
  for i in range(2,ny-2):
    for j in range(2,nx-2):
      f_xx=(1*f[i,j-1]-2*f[i,j+0]+1*f[i,j+1])/(h**2);
      f_yy=(1*f[i-1,j]-2*f[i+0,j]+1*f[i+1,j])/(h**2);
      f_xxxx=(1*f[i,j-2]-4*f[i,j-1]+6*f[i,j+0]-
            4*f[i,j+1]+1*f[i,j+2])/(h**4);
      f_yyyy=(1*f[i-2,j]-4*f[i-1,j]+6*f[i,j+0]-
            4*f[i+1,j]+1*f[i+2,j])/(h**4);
      f_xxyy=(f[i-1,j-1]-2*f[i,j-1]+f[i+1,j-1]-
                    2*f[i-1,j]+4*f[i,j]-2*f[i+1,j]+
                    f[i-1,j+1]-2*f[i,j+1]+f[i+1,j+1])/(h**4);
      Valuex=((epsilon-d*q0**4)*f[i,j]-(g*f[i,j]**3)-
            2*d*(q0**2)*(f_xx+f_yy)-d*(f_xxxx+f_yyyy)-
            2*d*(f_xxyy))**2;
      fn_xx=(1*fn[i,j-1]-2*fn[i,j+0]+1*fn[i,j+1])/(h**2);
      fn_yy=(1*fn[i-1,j]-2*fn[i+0,j]+1*fn[i+1,j])/(h**2);
      fn_xxxx=(1*fn[i,j-2]-4*fn[i,j-1]+6*fn[i,j+0]-
            4*fn[i,j+1]+1*fn[i,j+2])/(h**4);
      fn_yyyy=(1*fn[i-2,j]-4*fn[i-1,j]+6*fn[i,j+0]-
            4*fn[i+1,j]+1*fn[i+2,j])/(h**4);
      fn_xxyy=(fn[i-1,j-1]-2*fn[i,j-1]+fn[i+1,j-1]-
                    2*fn[i-1,j]+4*fn[i,j]-2*fn[i+1,j]+
                    fn[i-1,j+1]-2*fn[i,j+1]+fn[i+1,j+1])/(h**4);
      Valuexn=((epsilon-d*q0**4)*fn[i,j]-(g*fn[i,j]**3)-
            2*d*(q0**2)*(fn_xx+fn_yy)-d*(fn_xxxx+fn_yyyy)-
            2*d*(fn_xxyy))**2;
      TotalSum+=0.5*(Valuex+Valuexn);
  Integralddt2=-TotalSum;
  return Integralddt2
# -------------------------------------------------------------------------- #
#                                 CHECK
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
print(
'-------------------------------- CHECK ------------------------------------')
print('SIZE: '+str(len(un)))
print('PSI MAX. VALUE BEFORE: '+
      str(U0.max()))
print('PSI MIN. VALUE BEFORE: '+
      str(U0.min()))
# -------------------------------------------------------------------------- #
#                       CONSTANTS DEFINITIONS
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
A=d*q0**2/h**2
B=2*A
C=0.5*d*dt/h**4
D=4.*C
E=6.*C
mfig0=(1./dt)*sp.arange(0,11,1,dtype='int')
mfig1=(1./dt)*sp.arange(20,510,10,dtype='int')
mfig2=(1./dt)*sp.arange(750,5250,250,dtype='int')
mfig3=(1./dt)*sp.arange(5500,10500,500,dtype='int')
mfig4=(1./dt)*sp.arange(12500,FinalTime+dt,2500,dtype='int')
mfig=sp.append(mfig0,mfig1)
mfig=sp.append(mfig,mfig2)
mfig=sp.append(mfig,mfig3)
mfig=sp.append(mfig,mfig4)
fignum=0
unovo1=sp.zeros((ny,nx),dtype='float')
unovo2=sp.zeros((ny,nx),dtype='float')
bstep1=sp.zeros((nx),dtype='float')
bstep2=sp.zeros((ny),dtype='float')
o=0;
L1=1.;
Linf1=1e-14;
u=0;
if RestoreLists=="Yes":
  normL1=sp.loadtxt('./L1NORM/L1_result_'+str(dt)+
       '_'+str(int(InitialTime))+
       '.csv', delimiter=',')[0:int(ni/SaveL1each)+1];
  normL1TimeFile=sp.loadtxt('./L1NORM/normL1Time_dt='+str(dt)+
             '.csv', delimiter=',')[0:int(ni/SaveL1each)+1];
  ITETimeFile=sp.loadtxt('./ADDINFO/ITETime_dt='+str(dt)+
                   '.csv', delimiter=',')[0:int(ni)-1];
  umax=sp.loadtxt('./ADDINFO/MAX_result_'+str(dt)+
              '.csv', delimiter=',')[0:int(ni/SaveL1each)+1];
  NumberOfIterations=sp.loadtxt('./ADDINFO/ITE_result_'+str(dt)+
                            '.csv', delimiter=',')[0:int(ni)-1];
  LyapFunc=sp.loadtxt('./LYAPFUNC/LYAPFUNC_result_'+str(dt)+
                    '.csv', delimiter=',')[0:int(ni/SaveL1each)+1];
  DFunc=sp.loadtxt('./LYAPFUNC/DFUNC_result_'+str(dt)+
           '.csv', delimiter=',')[0:int(ni)-1];
else:
  normL1=[0];
  normL1TimeFile=[0];
  ITETimeFile=[];
  umax=[abs(un).max()];
  NumberOfIterations=[];
  LyapFunc=[LyapFuncNew];
  DFunc=[];
Dpsi=[];Dpsi_2=[];ERRORa=[];
# for filename in glob("L1_result_1.0.csv"):
#     os.remove("L1_result_1.0.csv")
# -------------------------------------------------------------------------- #
#                          PENTADIAGONAL MATRIX                              #
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
# Axis x-Const. Diagonals:
diag1x=C*sp.ones(nx)
diag2x=-D*sp.ones(nx)
diag4x=-D*sp.ones(nx)
diag5x=C*sp.ones(nx)
# Axis x-Neumann BC:
diag2x[0]=-1
diag4x[0],diag4x[1],diag4x[2]=1,1,0
diag5x[0],diag5x[1],diag5x[2],diag5x[3]=1,1,0,0
diag1x[nx-3],diag1x[nx-4]=0,0
diag2x[nx-2],diag2x[nx-3]=1,0
diag4x[nx-1]=-1
# Axis y-Const. Diagonals:
diag1y=C*sp.ones(ny)
diag2y=-D*sp.ones(ny)
diag4y=-D*sp.ones(ny)
diag5y=C*sp.ones(ny)
# Axis y-Neumann BC:
diag2y[0]=-1
diag4y[0],diag4y[1],diag4y[2]=1,1,0
diag5y[0],diag5y[1],diag5y[2],diag5y[3]=1,1,0,0
diag1y[ny-3],diag1y[ny-4]=0,0
diag2y[ny-2],diag2y[ny-3]=1,0
diag4y[ny-1]=-1
# -------------------------------------------------------------------------- #
#                           FILING LOG BEFORE                                #
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
# sys.stdout.close()
# -------------------------------------------------------------------------- #
#                      PROGRESS BAR/TEMPORAL LOOP                            #
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
print(
'---------------------------- LOOP-GDBC RAMPED -----------------------------')
LambdaY=sp.zeros((ny,nx),dtype='float')
# LambdaYmod=sp.zeros((ny,nx),dtype='float')
LambdaXun=sp.zeros((ny,nx),dtype='float')
LambdaYun=sp.zeros((ny,nx),dtype='float')
# LambdaYunmod=sp.zeros((ny,nx),dtype='float')
diag3x=sp.zeros((nx),dtype='float')
diag3x[0],diag3x[1]=1,1
diag3x[nx-1],diag3x[nx-2]=1,1
bstep1=sp.zeros((nx),dtype='float')
diag3y=sp.zeros((ny),dtype='float')
diag3y[0],diag3y[1]=1,1
diag3y[ny-1],diag3y[ny-2]=1,1
bstep2=sp.zeros((ny),dtype='float')
for o in progressbar(range(ni+1,n+1),"C",50):
    p=0
    unp=un.copy()
    L1pointer +=1
    while p<IterationsLimit:
        p +=1
# -------------------------------------------------------------------------- #
#                        FIRST HALF STEP EQUATION LOOP                       #
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
        for i in range(2,ny-2):  #epsilonmatrix[i,j] epsilonarray[j]
            for j in range(2,nx-2):
                diag3x[j]=1+E+0.25*d*dt*(q0**4)+0.125*g*dt*(unp[i,j]**2+
                                                            un[i,j]**2)
                fnmeio=0.5*epsilon*(unp[i,j]+un[i,j])+(
                            0.25*g1*(unp[i,j]+un[i,j])**2)-(
                            (d/h**4)*(
                            un[i-1,j-1]-2*un[i,j-1]+un[i+1,j-1]-
                            2*un[i-1,j]+4*un[i,j]-2*un[i+1,j]+
                            un[i-1,j+1]-2*un[i,j+1]+un[i+1,j+1]
                            ))-(
                            A*un[i-1,j]-B*un[i,j]+A*un[i+1,j]+
                            A*un[i,j-1]-B*un[i,j]+A*un[i,j+1]
                            )-(d/h**4)*(
                            unp[i-1,j-1]-2*unp[i,j-1]+unp[i+1,j-1]-
                            2*unp[i-1,j]+4*unp[i,j]-2*unp[i+1,j]+
                            unp[i-1,j+1]-2*unp[i,j+1]+unp[i+1,j+1]
                            )-(
                            A*unp[i-1,j]-B*unp[i,j]+A*unp[i+1,j]+
                            A*unp[i,j-1]-B*unp[i,j]+A*unp[i,j+1]
                            )
                LambdaY[i,j]=(-E-0.25*d*dt*(q0**4)-0.125*g*dt*(unp[i,j]**2+
                                                               un[i,j]**2))
                LambdaXun[i,j]=(-C*un[i,j-2]+D*un[i,j-1]+
                                LambdaY[i,j]*un[i,j]+D*un[i,j+1]-C*un[i,j+2])
                LambdaYun[i,j]=(-C*un[i-2,j]+D*un[i-1,j]+
                                LambdaY[i,j]*un[i,j]+D*un[i+1,j]-C*un[i+2,j])
                bstep1[j]=(un[i,j]+LambdaYun[i,j]+(LambdaXun[i,j]+
                           LambdaYun[i,j])+dt*fnmeio)
            data=sp.array([diag1x,diag2x,diag3x,diag4x,diag5x])
            offsets=sp.array([-2,-1,0,1,2])
            mtx=sparse.dia_matrix((data,offsets),shape=(nx,nx))
            bstep1[0]=2*u_x0
            bstep1[nx-1]=2*u_xL
            bstep1[1]=2*du_dx_x0
            bstep1[nx-2]=2*du_dx_xL
            mtx=mtx.tocsr()
            unovo1[i,:]=linsolve.spsolve(mtx,bstep1)
# -------------------------------------------------------------------------- #
#                       SECOND HALF STEP EQUATION LOOP                       #
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
        unovo2=unovo1
        for j in range(2,nx-2):
            for i in range(2,ny-2):
                diag3y[i]=1+E+0.25*d*dt*(q0**4)+0.125*g*dt*(unp[i,j]**2+
                                                            un[i,j]**2)
                bstep2[i]=unovo1[i,j]-LambdaYun[i,j]
            data=sp.array([diag1y,diag2y,diag3y,diag4y,diag5y])
            offsets=sp.array([-2,-1,0,1,2])
            mtx=sparse.dia_matrix((data,offsets),shape=(ny,ny))
            bstep2[0]=2*u_y0
            bstep2[ny-1]=2*u_yH
            bstep2[1]=2*du_dy_y0
            bstep2[ny-2]=2*du_dy_yH
            mtx=mtx.tocsr()
            unovo2[:,j]=linsolve.spsolve(mtx,bstep2)
        # ------------------------------------------------------------------ #
        #                      CONVERGENCE EVALUATION                        #
        #                                                                    #
        # Evaluate the relative rate of change between two successive        #
        # states each  time steps                                            #
        # ------------------------------------------------------------------ #
        for i in range(2,ny-2):
            for j in range(2,nx-2):
                Linf0=abs(unovo2[i,j]-unp[i,j])
                if Linf0>Linf1:
                    LinfNUM=Linf0.copy()
                else:
                    Linf1=Linf0.copy()
        Linf=LinfNUM/abs(unovo2.max())     
        # Linf=(abs((unovo2-unp).max()))/(abs(unovo2.max()))    
        if Linf<Precision or p==IterationsLimit:
            NumberOfIterations=sp.append(NumberOfIterations,p);
            ITETimeFile=sp.append(ITETimeFile,float(o*dt));
            sp.savetxt('./ADDINFO/ITETime_dt='+str(dt)+
                     '.csv',ITETimeFile,delimiter=',')
            sp.savetxt('./ADDINFO/ITE_result_'+str(dt)+'.csv',
                       NumberOfIterations,delimiter=',')
            break
        unp=unovo2.copy()
    # ---------------------------------------------------------------------- #
    #                       LYAPUNOV FUNCTIONAL CHECK                        #
    # ---------------------------------------------------------------------- #
    LyapFuncOld=LyapFuncNew.copy();
    LyapFuncNew=LyapunovFunctional(unovo2);
    dLyapunovdt=(LyapFuncNew-LyapFuncOld)/dt;
    # if dLyapunovdt>0:
    #   print('WARNING! Lyapunov increase, positive time derivative!');
    DFunc=sp.append(DFunc,dLyapunovdt);
    sp.savetxt('./LYAPFUNC/DFUNC_result_'+str(dt)+'.csv',DFunc,delimiter=',');
    # # DpsiValue=unovo2-un;
    # mIntddt2,ERROR=PsiTimeDerivative(unovo2,un);
    # Dpsi=sp.append(Dpsi,mIntddt2);
    # ERRORa=sp.append(ERRORa,ERROR);
    # sp.savetxt('./LYAPFUNC/DPSI_result_'+str(dt)+'.csv',Dpsi,delimiter=',');
    # sp.savetxt('./LYAPFUNC/ERROR_result_'+str(dt)+'.csv',ERRORa,delimiter=',');
    # mIntddt2_2=PsiTimeDerivative2(unovo2,un);
    # Dpsi_2=sp.append(Dpsi_2,mIntddt2_2);
    # sp.savetxt('./LYAPFUNC/DPSI2_result_'+str(dt)+'.csv',Dpsi_2,delimiter=',');
    # ---------------------------------------------------------------------- #
    #                          L1 NORM EVALUATION                            #
    #                                                                        #
    # Evaluate the relative rate of change between two successive            #
    # states each  time steps                                                #
    # ---------------------------------------------------------------------- #
    if L1pointer-SaveL1each>=0:
        L1pointer=L1pointer-SaveL1each
        L1=(1/dt)*(abs(unovo2-un).sum()/(abs(unovo2).sum()))
        normL1=sp.append(normL1,L1)
        normL1TimeFile=sp.append(normL1TimeFile,float(o*dt))
        sp.savetxt('./L1NORM/L1_result_'+str(dt)+'.csv',normL1,
                   delimiter=',')
        sp.savetxt('./L1NORM/normL1Time_dt='+str(dt)+'.csv',normL1TimeFile,
                   delimiter=',')
        sp.savetxt('./LYAPFUNC/LyapunovTime_dt='+str(dt)+'.csv',
                 normL1TimeFile,delimiter=',')
        umax=sp.append(umax,abs(unovo2).max())
        sp.savetxt('./ADDINFO/MAX_result_'+str(dt)+'.csv',umax,
                   delimiter=',')
        LyapFunc=sp.append(LyapFunc,LyapFuncNew)
        sp.savetxt('./LYAPFUNC/LYAPFUNC_result_'+str(dt)+'.csv',LyapFunc,
                   delimiter=',')
# -------------------------------------------------------------------------- #
#                           FILING RESULTS                                   #
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
    if o in mfig:
        sp.savetxt('./RESULTS/u_result_'+str(dt)+'_'+str(int(o*dt))+'.csv',
                   unovo2,delimiter=',')
        sp.savetxt('./L1NORM/L1_result_'+str(dt)+'_'+ str(int(o*dt))+'.csv',
                    normL1,delimiter=',')
        sp.savetxt('./ADDINFO/MAX_result_'+str(dt)+'_'+ str(int(o*dt))+'.csv',
                   umax,delimiter=',')
        sp.savetxt('./ADDINFO/ITE_result_'+str(dt)+'_'+ str(int(o*dt))+'.csv',
                   NumberOfIterations,delimiter=',')
    if L1<L1cutoff:
        print('-------------L1 BREAK!!!-------------')
        break
    un=unovo2.copy()
# -------------------------------------------------------------------------- #
#                            FILING LOG AFTER                                #
# -------------------------------------------------------------------------- #
sys.stdout=open("./LOGAFTER.txt","w")
# -------------------------------------------------------------------------- #
#                          SAVING FINAL RESULTS                              #
# -------------------------------------------------------------------------- #
sp.savetxt('./RESULTS/u_result_'+str(dt)+'_'+str(int(o*dt))+'.csv',
            unovo2,delimiter=',')
sp.savetxt('./L1NORM/L1_result_'+str(dt)+'_'+ str(int(o*dt))+'.csv',
            normL1,delimiter=',')
# -------------------------------------------------------------------------- #
#                        SHOW/SAVE SIMULATION LOG                            #
#                                                                            #
# Saving last file with the final informations about the simulation.         #
# -------------------------------------------------------------------------- #
print('---------------------- SIMULATION COMPLETED -------------------------')
print('Initial time: '+str(InitialTime)+'    final time: '+str(FinalTime)+
        '    dt: '+str(dt)+'    total time steps: '+str('%0.0e')%(n-ni))
print('L,H: '+str(L)+','+str(H)+'    nx,ny: '+str(nx)+
      ','+str(ny)+'    resolution: '+str(res))
print('wavelenghts in x,y direction: '+str(ncx)+','+str(ncy)+
  '    Convergence/Iterations: '+str(Precision)+','+str(IterationsLimit))
print('---------------------------- CHECK ----------------------------------')
print('SIZE: '+str(len(un)))
print('PSI MAX. VALUE BEFORE: '+
      str(U0.max()))
print('PSI MIN. VALUE BEFORE: '+
      str(U0.min()))
print('---------------------------- LOG ------------------------------------')
print('PSI MAX. VALUE AFTER: '+str(unovo2.max()))
print('PSI MIN. VALUE AFTER: '+str(unovo2.min()))
print('Finished in:')
print(datetime.now()-SimStartsAt)
# -------------------------------------------------------------------------- #
#                                FILING LOG                                  #
#                                                                            #
# Setting function's value and function's first derivative on the walls.     #
# These are also known as Dirichlet boundary conditions of first kind.       #
# -------------------------------------------------------------------------- #
sys.stdout.close()
