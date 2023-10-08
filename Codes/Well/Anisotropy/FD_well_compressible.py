from firedrake import *
import numpy as np
import math,sys,time
from matplotlib import pyplot as plt
from Limiter.flux_limiter_well import *

formulation = 'Upwind'
VertexLim = 'Kuzmin'



T = 15*24*3600            # final time
num_steps = 300    # number of time steps

dt = T / num_steps # time step size
#=====================================;
#  Create mesh and identify boundary  ;
#=====================================;
Length = 100
# Hight = 15
# nx = 20*3
nx = 40
# ny = 3*3
# mesh = RectangleMesh(nx, nx ,Length,Length,quadrilateral = True)
# mesh = RectangleMesh(nx, ny ,Length,Hight,quadrilateral = True)
# mesh = RectangleMesh(nx, ny ,Length,Hight,diagonal='crossed')
# mesh = RectangleMesh(nx, ny ,Length,Hight,diagonal='right')
# mesh = RectangleMesh(nx, nx ,Length,Length,diagonal='right')
mesh = RectangleMesh(nx, nx ,Length,Length,diagonal='crossed')
#==========================;
#  Define function spaces ;
#=========================;
pSpace = FunctionSpace(mesh,"DG" , 1)
sSpace = FunctionSpace(mesh,"DG" , 1)
wSpace = MixedFunctionSpace([pSpace,sSpace])
vSpace = VectorFunctionSpace(mesh,"DG" , 1)
u0 = Function(wSpace)
kSpace = FunctionSpace(mesh,"DG",0)
solMax = .85
solMin = .15
Vt = FunctionSpace(mesh, "HDiv Trace", 0)
w = TestFunction(Vt)
fluxes = Function(Vt)
with u0.dat.vec as solution_vec:
  DoF = solution_vec.getSize()
area_cell = Function(kSpace).interpolate(CellVolume(mesh)).vector().array()
#===================================;
#  Define trial and test functions  ;
#===================================;
(p_0,s_0) = TrialFunction(wSpace) # for linear solvers
# (p,s) = (u[0],u[1]) # for non-linear solvers
(z,v) = TestFunction(wSpace)
#=====================;
#  Define parameters  ;
#=====================;
x = SpatialCoordinate(mesh)
q_I = interpolate(conditional(And(And(x[0]< 12.5,x[0]>5),And(x[1]<12.5,x[1]>5)) , 0.0000175, 0.0 ), kSpace)
q_P = interpolate(conditional(And(And(x[0]< 95,x[0]>87.5),And(x[1]<95,x[1]>87.5)) , 0.0000175, 0.0 ), kSpace)

# for gravity field
# q_I = interpolate(conditional(And(And(x[0]< 12.5,x[0]>2.5),And(x[1]<12.5,x[1]>2.5)) , 0.00001, 0.0 ), kSpace)
# q_P = interpolate(conditional(And(And(x[0]< 97.5,x[0]>87.5),And(x[1]<97.5,x[1]>87.5)) , 0.00001, 0.0 ), kSpace)

# Isotropic K
# K = Constant(1e-12)
# Heterogen K
# k_subdomain = conditional(And(And(x>40,x<60),And(y>40,y<60)),1e-13,1e-12)
# layered anisotropy
theta_1 = pi/4.
theta_2 = 0.
theta_3 = pi/2.
theta_4 = pi/4.
def RMat(theta):
    return as_tensor([[cos(theta), -sin(theta)],[sin(theta), cos(theta)]])
k1 = Constant(2.25e-12)
k2 = Constant(2.25e-14)
D0 = as_tensor([[k1, 0],[0, k2]])

D1 = RMat(theta_1) * (D0 * RMat(theta_1).T)
D2 = RMat(theta_2) * (D0 * RMat(theta_2).T)
D3 = RMat(theta_3) * (D0 * RMat(theta_3).T)
D4 = RMat(theta_4) * (D0 * RMat(theta_4).T)
K = conditional(x[1] < -x[0]+50, D1,\
        conditional(x[1] < -x[0]+100, D2,\
        conditional(x[1] < -x[0]+150, D3, D4)))
# K_out = interpolate(K,kSpace)
# File('perm.pvd').write(K_out)

mu_w = Constant(5e-4)
mu_l = Constant(2e-3)

phi_0 = Constant(0.2)
rhow_0 = Constant(1000)
rhol_0 = Constant(850)

# Gravity
g = Constant((0,0))
# g = Constant((0,-9.8))
# g = Constant((9.8,0))

time = 0
p_an = Constant(1e-6)
# p_an = Constant(3e6)
# s_an = Function(sSpace).interpolate(\
        # conditional(x[0]<(100./nx),-(0.7/(100./nx))*x[0] + 0.85,0.15))
s_an = Constant(0.15)
p0 = interpolate( p_an, pSpace)
s0 = interpolate( s_an, sSpace)
v0 = interpolate( as_vector((0,0)),vSpace)

# def Pc(s,s0):
#     return conditional(gt(s0,0.05), 50 * s **(-0.5), 50* (1.5-10*s)*0.05**(-0.5))

s_rw = 0.15
s_ro = 0.15

def s_star(s):
    return (s-s_rw)/(1-s_rw-s_ro)

# def Pc(s):
#     return 10000 * (s_star(s)+0.05)**(-0.5)

def lmbda_w(s):
    # return  s_star(s)**(4) * K/mu_w
    return (s_star(s))*(s_star(s)) * 1./mu_w

def lmbda_l(s):
    # return (1-s_star(s))*(1-s_star(s))*(1-(s_star(s))**(2)) * K/mu_l 
    return (1-s_star(s))*(1-s_star(s)) * 1./mu_l

# def lmbda_t(s):
#     return lmbda_w(s) + lmbda_l(s)

Cr = Constant(6e-10) # This value should be taken (3~6e-6 PSI^-1) 4.35e-10 to 8.7e-10
# Cr = Constant(0.) 
Cw = Constant(1e-10)
Cl = Constant(1e-6)
def phi(p):
    return phi_0*(1+Cr*p)
    # return phi_0

def rhow(p):
    return rhow_0*(1+Cw*p)
    # return rhow_0

def rhol(p):
    return rhol_0*(1+Cl*p)
    # return rhol_0
def f_w(s):
    return lmbda_w(s)/(lmbda_w(s)+lmbda_l(s))

def f_o(s):
    return lmbda_l(s)/(lmbda_w(s)+lmbda_l(s))

# q_w = Constant(0)
# q_l = Constant(0)

#=================================;
#  Dirichlet boundary conditions  ;
#=================================;
# P_L = Constant(3e6)
# P_R = Constant(1e6)
# S_L = Constant(0.85)
bcs = []
#============================;
#   Define variational form  ;
#============================;
n = FacetNormal(mesh)
area = FacetArea(mesh)
h = CellSize(mesh)
# h = Constant(1/float(nx))
h_avg = (h('+')+h('-'))/2
# h_avg = Constant(1/float(nx))
SIG = 100
sigma = Constant(SIG)
sigma_bnd = Constant(10*SIG)
epsilon = Constant(1)
#====================;
#  March over time   ;
#====================;
du = TrialFunction(wSpace)
file_p = File("./Output/pres+FL+SL.pvd")
file_s = File("./Output/sat+FL+SL.pvd")
file_v = File("./Output/vel+FL+SL.pvd")
# file_error = File("./Output/error+FL+SL.pvd")


a_s0 = s_0*v*dx
L_s0 = s_an * v * dx

a_p0 = p_0*z*dx
L_p0 = p_an * z * dx

a_init = a_p0 + a_s0
L_init = L_p0 + L_s0
params = {'ksp_type': 'preonly', 'pc_type':'lu',"pc_factor_mat_solver_type": "mumps" }
A = assemble(a_init, bcs=bcs, mat_type='aij')
b = assemble(L_init)
solve(A,u0,b,solver_parameters=params)
pSol,sSol = u0.split()
p0.assign(pSol)
s0.assign(sSol)
sAvg = Function(kSpace).interpolate(s0)

s0.rename("Saturation","Saturation")
file_s.write(s0)

p0.rename("Pressure","Pressure")
file_p.write(p0)

v0.rename("Velocity","Velocity")
file_v.write(v0)
# non-linear problem
u = Function(wSpace)
u.assign(u0)
(p,s) = (u[0],u[1]) # for non-linear solvers

# Method I, Using NIPG symmetry term. this is not correct as our test function
# will be non-linear
a_p1 = (1./dt)*(phi(p) * rhol(p)*  (1-s) * z) * dx +\
    rhol(p) * lmbda_l(s) * inner(( K * grad(p) - K * rhol(p)*g),grad(z)) * dx -\
    inner(avg(lmbda_l(s)*rhol(p) * (K * grad(p) - K * rhol(p)*g)) , jump(z,n)) * dS +\
    epsilon * inner(avg(lmbda_l(s) * rhol(p) * (K * grad(z))) ,jump(p,n)) * dS +\
    sigma/h_avg * jump(p) * jump(z)  * dS

L_p1 = (1./dt) * phi(p0) * rhol(p0) * (1-s0) * z * dx +\
    (rhol(p0) * f_o(0.85) * q_I - rhol(p0) * f_o(s0) * q_P) * z * dx 


# Mehtod 2: Upwinding
a_p2 = (1./dt)*(phi(p) * rhol(p) * (1-s) * z) * dx +\
    rhol(p) * lmbda_l(s) * inner((K * grad(p) - K * rhol(p)*g),grad(z)) * dx -\
    conditional( gt(inner(avg(rhol(p0) *(K * grad(p0)- K * rhol(p0)*g)),n('+')),0) ,\
        inner(lmbda_l(s)('+') * avg(rhol(p) *(K * grad(p)- K * rhol(p)*g)) , jump(z,n)),\
        inner(lmbda_l(s)('-') * avg(rhol(p) *(K * grad(p)- K * rhol(p)*g)) , jump(z,n))\
                       ) * dS +\
    sigma/h_avg * jump(p) * jump(z)  * dS

L_p2 = (1./dt)*(phi(p0) * rhol(p0) * (1-s0) * z) * dx +\
    (rhol(p0) * f_o(0.85) * q_I - rhol(p0) * f_o(s0) * q_P) * z * dx

# # Upwinding using UFL condisional is tested and is correct check Upwind folder

a_s = (1./dt) * phi(p)* rhow(p) * s * v * dx  + \
    rhow(p) * lmbda_w(s) * inner(( K * grad(p) - K * rhow(p)*g) , grad(v)) * dx -\
    conditional( gt(inner(avg(rhow(p0) *(K * grad(p0)- K * rhow(p0)*g)),n('+')),0) ,\
        inner(lmbda_w(s)('+') * avg(rhow(p) *( K * grad(p)- K * rhow(p)*g)) , jump(v,n)),\
        inner(lmbda_w(s)('-') * avg(rhow(p) *( K * grad(p)- K * rhow(p)*g)) , jump(v,n))\
                       ) * dS +\
    sigma/h_avg * jump(s) * jump(v)  * dS

L_s = + (1./dt) * phi(p0)* rhow(p0) * s0 * v *  dx+\
    rhow(p0) *  (f_w(0.85) * q_I - f_w(s0)*q_P) * v * dx

if VertexLim == 'Kuzmin':
    # Make slope limiter
    limiter = VertexBasedLimiter(sSpace)
    limiter.apply(s0)
elif VertexLim == 'None':
    print('NO LIMITER Used')
else:
    sys.exit('***Not a valid limiter***\n'
          '****************************\n'
          '****************************\n'
          '****************************\n'
          '***************************')

if formulation == 'NIPG':
    F =  a_p1 - L_p1 + a_s - L_s
elif formulation == 'Upwind':
    F =  a_p2 - L_p2 + a_s - L_s
else:
    sys.exit('***%s does not exist *******\n'
          '****************************\n'
          '****************************\n'
          '****************************\n'
          '***************************'%formulation)


sAvg0 = sAvg
counter = 1
for nIter in range(num_steps):
    print ("=====================")
    print ('Time_step =%d'%counter)
    print ("=====================")
    sAvg0.assign(sAvg)
    #update time
    time += dt
    counter += 1
    # q_w.time = time
    # q_l.time = time

    # p_an.time=time
    # s_an.time=time
    J = derivative(F, u, du)
    #initial guess for solution is set to 1.0
    # u.assign(Constant(1))
    problem = NonlinearVariationalProblem(F,u,bcs,J)
    solver = NonlinearVariationalSolver(problem,solver_parameters=
                                            {
                                                #OUTER NEWTON SOLVER
                                            'snes_type': 'newtonls',
                                            'snes_atol': 1e-6,
                                            'snes_max_it': 200,
                                            'snes_monitor_short': None,
                                                # INNER LINEAR SOLVER
                                            # 'ksp_rtol': 1e-5,
                                            'ksp_max_it': 100,
                                            # Direct solver
                                            # 'ksp_type':'preonly',
                                            # 'mat_type': 'aij',
                                            # 'pc_type': 'lu',
                                            # "pc_factor_mat_solver_type": "mumps",
                                            # Iterative solvers
                                            # 'pc_type': 'hypre',
                                            # 'hypre_type': 'boomeramg',
                                            'ksp_type' : 'gmres',
                                            'mat_type': 'aij',
                                            'ksp_rtol' : 1e-8,
                                            'pc_type': 'lu',
                                            'ksp_gmres_restart': '100',
                                            # 'ksp_monitor':None
                                            })
    solver.solve()

    pSol,sSol = u.split()
    sAvg = Function(kSpace).interpolate(sSol)
    wells_avg = Function(kSpace).interpolate(rhow(p0)*( f_w(0.85) * q_I - f_w(s0)*q_P ))
    rPAvg_0 = Function(kSpace).interpolate(rhow(p0)*phi(p0))
    rhoAvg_0 = Function(kSpace).interpolate(rhow(p0))
    rPAvg = Function(kSpace).interpolate(rhow(pSol)*phi(pSol))


    F_flux = fluxes('+')*w('+')*dS + fluxes*w*ds - (\
        area * -1. *conditional(\
        gt(inner(avg(rhow(p0) *(K * grad(p0)- K * rhow(p0)*g)),n('+')),0.) ,\
        lmbda_w(sSol)('+') * inner(avg(rhow(pSol) *( K * grad(pSol)- K * rhow(pSol)*g)) ,n('+')) * w('+'),\
        lmbda_w(sSol)('-') * inner(avg(rhow(pSol) *( K * grad(pSol)- K * rhow(pSol)*g)) ,n('+')) * w('+')\
                       )* dS +\
    area * sigma/h_avg * jump(sSol) * w('+')  * dS
    )
    solve(F_flux == 0, fluxes)
    # -----------------------------------------;
    # calculate local mass balance in each cell:
    # -----------------------------------------;
    FLUX = get_flux(mesh,fluxes).apply()
    error =  rPAvg.vector().array() * sAvg.vector().array()\
            -(rPAvg_0.vector().array() * sAvg0.vector().array() -\
            dt/(area_cell) * FLUX.sum(axis=1) ) -\
            dt * wells_avg.vector().array()

    u_error = Function(kSpace)
    u_error.vector().set_local(error)
    error_plot = Function(kSpace).interpolate(u_error)
    # print('max error',error.max())
    # print('min error',error.min())

    # Flux limiter applied
    sSol,convergedIter = flux_limiter(mesh,s0,sSol,rPAvg_0,rPAvg,rhoAvg_0,q_I,q_P,fluxes,solMax,solMin,dt).apply()
    print('flux-limiter convergedIter = ',convergedIter)


    # Slope limiter applied
    limiter.apply(sSol)

    p0.assign(pSol)
    s0.assign(sSol)
    sAvg = Function(kSpace).interpolate(s0)
    v0 = Function(vSpace).interpolate(-1*K*lmbda_w(s0)*(grad(p0)-rhow_0*g))


    p0.rename("Pressure","Pressure")
    s0.rename("Saturation","Saturation")
    v0.rename("Velocity","Velocity")
    error_plot.rename("Error","error")

    # file_p.write(p0)
    # file_s.write(s0)
    # file_v.write(v0)
    # file_error.write(error_plot)

# print('dofs=', DoF)
