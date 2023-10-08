from firedrake import *
import numpy as np
import math,sys
from Limiter.flux_limiter import *
import time as tm
from scipy.io import loadmat

formulation = 'Upwind'
layer = '71' #  71 is a good choice as it has a channel crossing domain at y=x


save_itr = 2
T = 150*24*3600           # final time
num_steps = 900*save_itr    # number of time steps
dt = T / num_steps # time step size
#=====================================;
#  Create mesh and identify boundary  ;
#=====================================;
Length = 100
Hight = 100
nx = 90
ny = 60
# mesh = RectangleMesh(nx, nx ,Length,Length,quadrilateral = True)
mesh = RectangleMesh(nx, ny ,150,Hight,diagonal='right')
# mesh = RectangleMesh(nx, nx ,Length,Length,diagonal='crossed')
# mesh = RectangleMesh(nx, ny ,Length,Hight,quadrilateral = True)

# mesh = RectangleMesh(nx, ny ,150,Hight,diagonal = 'crossed')

# load the mesh generated with Gmsh
# mesh = Mesh('mesh/unstructured.msh')
# mesh = Mesh('mesh/crack.msh')
# File("mesh.pvd").write(mesh)
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
(z,v) = TestFunction(wSpace)
#=====================;
#  Define parameters  ;
#=====================;
x, y = SpatialCoordinate(mesh)
# Homogeneous
# K = Constant(1e-12)
# Non-homogen K
# k_subdomain = conditional(And(And(x>40,x<60),And(y>40,y<60)),1e-13,1e-12)
# K = interpolate(k_subdomain,kSpace)
# File('perm.pvd').write(K)
# SPE10 loaded from file
class MyExpression(Expression):
    def Perm(self,x_val,y_val):
        data = loadmat('./K_Input/all/Kx_%s.mat'%layer)
        data_m = data['Kx_layer']
        if x_val <25 or x_val >125:
            return 1e-11
        else:
            x_new , y_new = (x_val-25)*(60./Length) , y_val*(220./Length)
            return data_m[math.floor(x_new),math.floor(y_new)]

    def eval(self, value, x):
        value[0] = self.Perm(x[0],x[1])

    def value_shape(self):
        return ()

K = interpolate(MyExpression(),kSpace)
File('perm.pvd').write(K)
# Anisotropic K
# K = as_tensor(((3e-12, 0.0), (0.0, 1e-12)))

mu_w = Constant(5e-4)
mu_l = Constant(2e-3)

phi_0 = Constant(0.2)
rhow_0 = Constant(1000)
rhol_0 = Constant(850)
g = Constant((0,0))
# g = Constant((0,-9.8))
# g = Constant((0,9.8))


time = 0
x = SpatialCoordinate(mesh)
# p_an = Function(pSpace).interpolate(3e6*1-(2e6/100*x[0]))
p_an = Constant(1e-6)
# s_an = Function(sSpace).interpolate(\
#         conditional(x[0]<(100./nx),-(0.7/(100./nx))*x[0] + 0.85,0.15))
# s_an = Function(sSpace).interpolate(\
        # conditional(x[0]<(100./nx),0.85,conditional(\
            # x[0]<2*(100./nx),-(0.7/(100./nx))*(x[0]-(100./nx)) + 0.85,.15)\
            # ))
# s_an = Function(sSpace).interpolate(conditional(x[0]<(100./nx),0.85,0.15))
s_an = Constant(0.15)
p0 = interpolate( p_an, pSpace)
s0 = interpolate( s_an, sSpace)
v0 = interpolate( as_vector((0,0)),vSpace)


s_rw = 0.15
s_ro = 0.15

def s_star(s):
    return (s-s_rw)/(1-s_rw-s_ro)

def Pc(s):
    return 10000 * (s_star(s)+0.05)**(-0.5)

def lmbda_w(s):
    return (s_star(s))*(s_star(s)) * K/mu_w

def lmbda_l(s):
    return (1-s_star(s))*(1-s_star(s)) * K/mu_l

def lmbda_t(s):
    return lmbda_w(s) + lmbda_l(s)

Cr = Constant(6e-10) # This value should be taken (3~6e-6 PSI^-1) 4.35e-10 to 8.7e-10
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


q_w = Constant(0)
q_l = Constant(0)
#=================================;
#  Dirichlet boundary conditions  ;
#=================================;
P_L = Constant(3e6)
P_R = Constant(1e6)
S_L = Constant(0.85)
bcs = []
#============================;
#   Define variational form  ;
#============================;
n = FacetNormal(mesh)
area = FacetArea(mesh)
# if triangular mesh is used
h = CellSize(mesh)
h_avg = (h('+')+h('-'))/2
# if quadrilateral mesh is used
# h_avg = Constant(Length/float(nx))
# h = Constant(Length/float(nx))
SIG = 100
sigma = Constant(SIG)
sigma_bnd = Constant(10*SIG)
epsilon = Constant(1)
#====================;
#  March over time   ;
#====================;
du = TrialFunction(wSpace)
file_p = File("./Output/compressible/pres.pvd")
file_s = File("./Output/compressible/sat.pvd")
file_v = File("./Output/compressible/vel.pvd")


a_p0 = lmbda_t(s0) * inner(grad(p_0),grad(z)) * dx -\
    inner(avg(lmbda_t(s0) * grad(p_0)) , jump(z,n)) * dS -\
    lmbda_t(s0) * inner(grad(p_0) , n) * z * ds(1) -\
    lmbda_t(s0) * inner(grad(p_0) , n) * z * ds(2) +\
    epsilon * inner(avg(lmbda_t(s0) * grad(z)) , jump(p_0,n)) * dS +\
    epsilon * lmbda_t(s0) * inner(grad(z), n) * p_0 * ds(1) +\
    epsilon * lmbda_t(s0) * inner(grad(z), n) * p_0 * ds(2) +\
    sigma/h_avg * jump(p_0) * jump(z)  * dS +\
    sigma/h * p_0 * z * ds(1) +\
    sigma/h * p_0 * z * ds(2)


L_p0 = (q_l+q_w) * z * dx -\
    lmbda_l(s0) * inner(grad(Pc(s0)),grad(z)) * dx +\
    inner(avg(lmbda_l(s0) * grad(Pc(s0))) , jump(z,n)) * dS +\
    inner(lmbda_l(s0) * grad(Pc(s0)) , n ) * z * ds(1) -\
    epsilon * inner(avg(lmbda_l(s0) * grad(z)) , jump(Pc(s0),n))* dS -\
    epsilon * inner(lmbda_l(s0) * grad(z), n ) * Pc(s0) * ds(1) -\
    sigma/h_avg * jump(Pc(s0)) * jump(z)  * dS -\
    sigma/h * Pc(s0) * z  * ds(1) +\
    sigma/h * P_L * z * ds(1) +\
    sigma/h * P_R * z * ds(2) +\
    sigma/h * Pc(S_L) * z * ds(1) +\
    epsilon * inner(lmbda_t(s0) * grad(z) , n ) * P_R * ds(2) +\
    epsilon * inner(lmbda_t(s0) * grad(z) , n ) * P_L * ds(1) +\
    epsilon * inner(lmbda_l(s0) * grad(z), n) * Pc(S_L) * ds(1)


time = 0
a_s0 = s_0*v*dx
L_s0 = s_an * v * dx

a_init = a_p0 + a_s0
L_init = L_p0 + L_s0
#                             )
params = {'ksp_type': 'preonly', 'pc_type':'lu',"pc_factor_mat_solver_type": "mumps" }
A = assemble(a_init, bcs=bcs, mat_type='aij')
b = assemble(L_init)
solve(A,u0,b,solver_parameters=params)
#=====================;
# Saving stiffness Mat
#=====================;
pSol,sSol = u0.split()
# print(sSol.function_space())
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



# Mehtod 2: Upwinding
a_p2 = (1./dt)*(phi(p) * rhol(p) * (1-s) * z) * dx +\
    rhol(p) * lmbda_l(s) * inner((grad(p)-rhol(p)*g) ,grad(z)) * dx -\
    conditional( gt(inner(avg(rhol(p0) *(grad(p0)-rhol(p0)*g)),n('+')),0) ,\
        inner(lmbda_l(s)('+') * avg(rhol(p) *(grad(p)-rhol(p)*g)) , jump(z,n)),\
        inner(lmbda_l(s)('-') * avg(rhol(p) *(grad(p)-rhol(p)*g)) , jump(z,n))\
                       ) * dS -\
    inner(rhol(p)*lmbda_l(s) * (grad(p)-rhol(p)*g), n) * z * ds(1) -\
    inner(rhol(p)*lmbda_l(s) * (grad(p)-rhol(p)*g), n) * z * ds(2) +\
    sigma/h_avg * jump(p) * jump(z)  * dS +\
    sigma_bnd/h * p * z * ds(1) +\
    sigma_bnd/h * p * z * ds(2)
    # Another implementation on Dirichlet boundary terms but had no effect
    # -inner(rhol(P_L)*lmbda_l(S_L) * (grad(p) -rhol(p)*g), n) * z * ds(1) -\
    # inner(rhol(P_R)*lmbda_l(s) * (grad(p) -rhol(p)*g), n) * z * ds(2)

L_p2 = (1./dt)*(phi(p0) * rhol(p0) * (1-s0) * z) * dx +\
    rhol(p0) * q_l * z * dx +\
    sigma_bnd/h * P_L * z * ds(1) +\
    sigma_bnd/h * P_R * z * ds(2)

# # Upwinding using UFL condisional is tested and is correct check Upwind folder

a_s = (1./dt) * phi(p)* rhow(p) * s * v * dx  + \
    rhow(p) * lmbda_w(s) * inner((grad(p)-rhow(p)*g)  , grad(v)) * dx -\
    conditional( gt(inner(avg(rhow(p0) *(grad(p0)-rhow(p0)*g)),n('+')),0) ,\
        inner(lmbda_w(s)('+') * avg(rhow(p) *(grad(p)-rhow(p)*g)) , jump(v,n)),\
        inner(lmbda_w(s)('-') * avg(rhow(p) *(grad(p)-rhow(p)*g)) , jump(v,n))\
                       ) * dS -\
    lmbda_w(S_L)*rhow(p) * inner((grad(p)-rhow(p)*g),n) * v * ds(1) -\
    lmbda_w(s)*rhow(p) * inner((grad(p)-rhow(p)*g),n) * v * ds(2) +\
    sigma/h_avg * jump(s) * jump(v)  * dS +\
    sigma_bnd/h * s * v * ds(1) 

L_s = (1./dt) * phi(p0)* rhow(p0) * s0 * v * dx +\
    rhow(p0) * q_w * v * dx + \
    sigma_bnd/h * S_L * v * ds(1) 
    # sigma_bnd/h * P_L * v * ds(1) +\ # result in error on right BC
    # sigma_bnd/h * P_R * v * ds(2) # result in error on right BC

Slope_limiter = VertexBasedLimiter(sSpace)
Slope_limiter.apply(s0)

if formulation == 'Upwind':
    F =  a_p2 - L_p2 + a_s - L_s
else:
    sys.exit('***%s does not exist *******\n'
          '****************************\n'
          '****************************\n'
          '****************************\n'
          '***************************'%formulation)


sAvg0 = sAvg
counter = 1
print('Step \t solve \t FL(no_fluxCost)\t FL(all) \t SL \t convergedIter\n ')
for nIter in range(num_steps):
    print ('Time_Step =%d'%counter)
    sAvg0.assign(sAvg)
    #update time
    time += dt
    counter += 1

    solve_start = tm.time()
    q_w.time = time
    q_l.time = time

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
                                            'snes_rtol': 1e-5,
                                            'snes_max_it': 200,
                                            # "snes_linesearch_type": "basic",
                                            'snes_monitor': None,
                                            ## 'snes_view': None,
                                            # 'snes_converged_reason': None,
                                                # INNER LINEAR SOLVER
                                            'ksp_rtol': 1e-5,
                                            'ksp_max_it': 100,
                                            # Direct solver
                                            'ksp_type':'preonly',
                                            'mat_type': 'aij',
                                            'pc_type': 'lu',
                                            "pc_factor_mat_solver_type": "mumps",
                                            # Iterative solvers
                                            # 'pc_type': 'hypre',
                                            # 'hypre_type': 'boomeramg',
                                            # 'ksp_type' : 'fgmres',
                                            # 'ksp_gmres_restart': '100',
                                            # 'ksp_initial_guess_non_zero': True,
                                            # 'ksp_converged_reason': None,
                                            # 'ksp_monitor_true_residual': None,
                                            ## 'ksp_view':None,
                                            # 'ksp_monitor':None
                                            })
    solver.solve()

    pSol,sSol = u.split()
    solve_end = tm.time()

    sAvg = Function(kSpace).interpolate(sSol)

    # ---------------------;
    # flux-limiter applied ;
    # ---------------------;
    rPAvg_0 = Function(kSpace).interpolate(rhow(p0)*phi(p0))
    rPAvg = Function(kSpace).interpolate(rhow(pSol)*phi(pSol))

    FL_start = tm.time()
    # F_flux = fluxes('+')*w('+')*dS + fluxes*w*ds - (\
    #     (1./(0.2*1000)) * area * -1. *conditional( gt(inner(avg(rhow(p) *grad(p0)),n('+')),0.) ,\
    #     lmbda_w(sSol)('+') * inner(avg(rhow(p) *grad(p0)) ,n('+')) * w('+'),\
    #     lmbda_w(sSol)('-') * inner(avg(rhow(p) *grad(p0)) ,n('+')) * w('+')\
    #                    )* dS -\
    # (1./(0.2*1000)) * area * lmbda_w(S_L)*rhow(p) * inner(grad(pSol),n) * w * ds(1) -\
    # (1./(0.2*1000)) * area * lmbda_w(sSol)*rhow(p) * inner(grad(pSol),n) * w * ds(2) +\
    # (1./(0.2*1000)) * area * sigma/h_avg * jump(sSol) * w('+')  * dS +\
    # (1./(0.2*1000)) * area * sigma_bnd/h * sSol * w * ds(1) -\
    # (1./(0.2*1000)) * area * sigma_bnd/h * S_L * w * ds(1) 
    # )
    F_flux = fluxes('+')*w('+')*dS + fluxes*w*ds - (\
        area * -1. *conditional( gt(inner(avg(rhow(p0) *(grad(p0)-rhow(p0)*g)),n('+')),0.) ,\
        lmbda_w(sSol)('+') * inner(avg(rhow(p0) *(grad(p0)-rhow(p0)*g)) ,n('+')) * w('+'),\
        lmbda_w(sSol)('-') * inner(avg(rhow(p0) *(grad(p0)-rhow(p0)*g)) ,n('+')) * w('+')\
                       )* dS -\
    area * lmbda_w(S_L)*rhow(P_L) * inner((grad(pSol)-rhow(pSol)*g),n) * w * ds(1) -\
    area * lmbda_w(sSol)*rhow(P_R) * inner((grad(pSol)-rhow(pSol)*g),n) * w * ds(2) +\
    area * sigma/h_avg * jump(sSol) * w('+')  * dS +\
    area * sigma_bnd/h * sSol * w * ds(1) -\
    area * sigma_bnd/h * S_L * w * ds(1) 
    )

    solve(F_flux == 0, fluxes)
    # -----------------------------------------;
    # calculate local mass balance in each cell:
    # -----------------------------------------;
    # FLUX = get_flux(mesh,fluxes).apply()
    # print('FLUX',FLUX)
    # error = sAvg.vector().array() -( sAvg0.vector().array() -\
    #         dt/area_cell * FLUX.sum(axis=1) )
    
    # error =  rPAvg.vector().array() * sAvg.vector().array()\
    #         -(rPAvg_0.vector().array() * sAvg0.vector().array() -\
    #         dt/(area_cell) * FLUX.sum(axis=1) )
    # print('local mass balance',error)

    # # Applied flux limiter
    FL_solve_start = tm.time()
    sSol,convergedIter = flux_limiter(mesh,s0,sSol,fluxes,rPAvg_0,rPAvg,solMax,solMin,dt).apply()
    print("convergedIter",convergedIter)
    FL_end = tm.time()


    sAvg = Function(kSpace).interpolate(sSol)

    SL_start = tm.time()
    # Slope-limiter applied
    Slope_limiter.apply(sSol)
    SL_end = tm.time()

    p0.assign(pSol)
    s0.assign(sSol)
    sAvg = Function(kSpace).interpolate(s0)
    v0 = Function(vSpace).interpolate(-1*lmbda_w(s0)*(grad(p0)-rhow(p0)*g))

    print("min:%f,\t max:%f"%(s0.vector().array().min(),s0.vector().array().max()))
    p0.rename("Pressure","Pressure")
    s0.rename("Saturation","Saturation")
    v0.rename("Velocity","Velocity")
    if counter % (save_itr*10) == 0:
        file_p.write(p0)
        file_s.write(s0)
        file_v.write(v0)
    # print('%d \t %f \t %f \t %f \t %f \t %d '\
        # %((counter-1), (solve_end-solve_start),(FL_end-FL_solve_start),(FL_end-FL_start),(SL_end-SL_start),convergedIter) )



