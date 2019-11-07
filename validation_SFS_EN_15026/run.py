from __future__ import division, print_function

"""
Runs on Python 2.7. Should work on Python 3 as well, but that has not been tested.

2019

@author: Antti Mikkonen, a.mikkonen@iki.fi

"""

import fenics as fe
import scipy as sp
#from scipy import optimize
import os
import time
from matplotlib import pyplot as plt
from global_values import *
import solver
import csv
#import json
from scipy import interpolate


class Validation_SFS_EN_15026(object):

    def __init__(self):
        
        self.out_dir = "outData"
        
#        self.t_end  = 7*s2d
        self.t_end = 365*s2d        
        self.L     = 20
        self.gamma = 1.1
        self.dx0 = 0.1e-3    
    
        self.order_2nd = True
        self.dt = s2d/2#0.5*s2h
        self.time_gamma = 1.0
        self.max_dt = s2d/2
        
        self.max_steps = int(1.1*self.t_end / self.dt)
        
        # Initial conditions
#        self.initial_condition_type = "constant"
        self.T_init    = 20 + C2K    
        self.phi_init = 0.5
        
        # Boundary conditions
#        self.boundary_type_in = "dirichlet_constant"
        self.T_out       = 30 + C2K
        self.phi_out     = 0.95  
        
        
        self.T_in      = self.T_init
        self.phi_in    = self.phi_init  
        
#        self.boundary_type_out = "dirichlet_constant"
        
        
        # moisture locations
        self.prope_x_w = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
        # temperature locations
        self.prope_x_T = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        # times to log
        self.prope_times = sp.array([7*s2d, 30*s2d, 365*s2d])




    def make_mesh(self):
        x = [0, self.dx0]
        while x[-1] < self.L:
            x.append(x[-1]+self.gamma*(x[-1]-x[-2]))
        x[-1] = self.L        
        x = sp.array(x)
        
        self.n_total = len(x)
#        print(self.n)
        self.mesh = fe.IntervalMesh(self.n_total-1, 0, self.L)
        self.mesh.coordinates()[:] = sp.reshape(x,(self.n_total,1))

        self.v = fe.FunctionSpace(self.mesh, "CG", 1)
        self.v_materials = fe.FunctionSpace(self.mesh, 'DG', 1) 
        
        
        # Subdomains
        sub0 = fe.AutoSubDomain(lambda x: (x[0] <= self.L))
#        sub1 = fe.AutoSubDomain(lambda x: (x[0] >= L_1))    
        
        # Mark subdomains
        self.material_markers = fe.MeshFunction("size_t", self.mesh, 
                                           self.mesh.topology().dim(), 0)
        sub0.mark(self.material_markers, 0)
#        sub1.mark(material_markers, 1)
        
    def boundary_conditions(self):
        boundary_out = fe.CompiledSubDomain('on_boundary && near(x[0], 0)')
        boundary_in = fe.CompiledSubDomain('on_boundary && near(x[0], L)',
                                                      L=self.L)
       # Temperature boundaries
        bc_T_in  = fe.DirichletBC(self.v, fe.Constant(self.T_in), boundary_in)
        bc_T_s = fe.DirichletBC(self.v, fe.Constant(self.T_out), boundary_out)
        self.bc_T = [bc_T_in, bc_T_s]
        
        # Moisture boundaries
        bc_phi_in  = fe.DirichletBC(self.v, fe.Constant(self.phi_in), boundary_in)
        bc_phi_s = fe.DirichletBC(self.v, fe.Constant(self.phi_out), boundary_out)
        self.bc_phi = [bc_phi_in, bc_phi_s]

    def initial_conditions(self):
    
        # implicit
        self.T_old   = fe.Function(self.v)
        self.phi_old = fe.Function(self.v)
        
        T_in = self.T_in
        T_init = self.T_init
        phi_in = self.phi_in
        phi_init = self.phi_init
        
        class InitialConditionT(fe.Expression):
            def eval_cell(self, value, x, ufc_cell):
                if x[0] <= 1e-6:
                    value[0] = T_in
                else:
                    value[0] = T_init
        class InitialConditionPhi(fe.Expression):
            def eval_cell(self, value, x, ufc_cell):
                if x[0] <= 1e-6:
                    value[0] = phi_in
                else:
                    value[0] = phi_init            
        self.T_old.interpolate(InitialConditionT(element = self.v.ufl_element()))
        self.phi_old.interpolate(InitialConditionPhi(element = self.v.ufl_element()))
        
        
    def general_material_properties(self):    
        ############################################
        # Temperature based    
        ############################################        
        #Vinha PhD thesis. Eqs. 1.28 and 1.30 Eq. 1.29 page 27
        # T in Kelvins
        #verified with T=0C, 10C, 20C, 30C with
        #https://www.engineeringtoolbox.com/water-vapor-saturation-pressure-d_599.html?vA=-20&units=C#
        self.ps = fe.Expression("""610.5*pow(2.718,
                                        (17.269*Ti - 4717.03) / (Ti - 35.85))""",
                           Ti=self.T_old,
                           element = self.v.ufl_element())
#        self.ps = fe.Constant(3500)    
            
            
        # Petteri
        self.dps_dT = fe.Expression("""(610.5*pow(2.718,
                                             (17.269*Ti - 4717.03)/(Ti - 35.85))
                *(17.2672094763233/(Ti - 35.85) 
                - 0.999896315728952*(17.269*Ti - 4717.03)/pow(Ti - 35.85,2))
                )""",
                           Ti=self.T_old,
                           element = self.v.ufl_element())
#        self.dps_dT = fe.Constant(200)
    
    def material_properties_equations(self):

        # Dry material
        self.crho_m = 1.824e6

        ############################################
        # Moisture based 
        ############################################        
        self.w = fe.Expression("""146
                               /pow(1+
                                    pow(-8e-8*R_water*T_ref*rho_w*log(phi),1.6)
                                    ,0.375)""",
                           phi=self.phi_old,R_water=R_water, T_ref=T_ref,rho_w=rho_w,
                           element = self.v.ufl_element()) 
#        self.w = fe.Constant(0.20)
                               
        self.kT = fe.Expression("""(1.5 + 15.8/1000*w)""",
                           w=self.w,
                           element = self.v.ufl_element())
#        self.kT = fe.Constant(1.5)

        #  Vapour permeability of material
        self.delta_p = fe.Expression("""1/(R_water*T_ref)
                                *26.1e-6/200
                                *(1-w/146)/(0.503
                                *pow(1-w/146,2)
                                +0.497)""",
                           R_water=R_water, T_ref = T_ref,w=self.w,
                           element = self.v.ufl_element())
#        self.delta_p = fe.Constant(0.6e-12)
        
        self.Dw = fe.Expression("""-1*exp(- 39.2619
               + 0.0704*(w-73)
               - 1.7420e-4*pow(w-73,2)
               - 2.7953e-6*pow(w-73,3)
               - 1.1566e-7*pow(w-73,4)
               + 2.5969e-9*pow(w-73,5)
               )
               *
               (-a*c*d*pow(b/w, c)*pow(pow(b/w,c) - 1, d)/(w*(pow(b/w,c) - 1)))
               """,
               w=self.w, 
               a = 0.125e8, b = 146, c = 1.0/0.375, d = 0.625,
               element = self.v.ufl_element())
#        self.Dw = fe.Constant(2e-11)
        
        self.xi = fe.Expression("""-3.86765818700807e-10
            *pow(-R_water*T_ref*rho_w*log(phi),1.6)
            *pow(4.41513491667588e-12
                 *pow(-R_water*T_ref*rho_w*log(phi),1.6) + 1,
                 -1.375)
            /(phi*log(phi))
               """,
               R_water=R_water, T_ref = T_ref, rho_w=rho_w,
               phi=self.phi_old, 
               element = self.v.ufl_element())
#        self.xi = fe.Constant(48)
            

    def calc_lambda_t(self, w):
        return 1.5 + 15.8/1000*w
            
    
    def calc_Dw(self, w):
        x = (w-73)               
        K = sp.exp(- 39.2619
                   + 0.0704*x
                   - 1.7420e-4*x**2
                   - 2.7953e-6*x**3
                   - 1.1566e-7*x**4
                   + 2.5969e-9*x**5
                   )                        
    
        a = 0.125e8; b = 146; c = 1/0.375; d = 0.625
        # Petteri
        dp_s_over_dw = -(a*c*d*b**c) / (w**(c+1)) \
                       *((b/w)**c-1)**(d-1)
        # Sympy                   
    #    dp_s_over_dw = -a*c*d*(b/w)**c*((b/w)**c - 1)**d/(w*((b/w)**c - 1))
    
    
        return -K*dp_s_over_dw
    

    
    def calc_delta_p(self, w):
        """
        Vapour permeability of material
        """
        return 1/(R_water*T_ref)*26.1e-6/200*(1-w/146)/(0.503*(1-w/146)**2+0.497)
#        return M_w/(R_water*T_ref)*26.1e-6/200*(1-w/146)/(0.503*(1-w/146)**2+0.497)

    def calc_xi(self, phi):
        return (-3.86765818700807e-10
            *(-R_water*T_ref*rho_w*sp.log(phi))**1.6
            *(4.41513491667588e-12*(-R_water*T_ref*rho_w
                                    *sp.log(phi))**1.6 + 1)**(-1.375)
            /(phi*sp.log(phi))
            )
    def sorption_phi2w(self, phi):
        """Relative humidity - phi - to moisture content - w- 
        
        """
        return 146/(1+(-8e-8*R_water*T_ref*rho_w*sp.log(phi))**1.6)**0.375

    def material_properties_interpolation(self):

        # Dry material
        self.crho_m = 1.824e6

        with open("interpolator.C","r") as ifile:
            interpolator_code = ifile.read()

#        w_table = sp.array([40,60,80,100,120,140])
#        w_table = sp.array([40,140])
#        w_table = sp.linspace(30, 150, 100)    

        phi_table = sp.linspace(1e-10, 1-1e-10, 50)
        w_table = self.sorption_phi2w(phi_table)



        ############################################
        # Moisture based 
        ############################################        
        self.w = fe.Expression(cppcode=interpolator_code, 
                        element = self.v_materials.ufl_element())
        self.w.materials = self.material_markers
        self.w.x_k = fe.interpolate(self.phi_old, self.v_materials)
        for val, x in zip(self.sorption_phi2w(phi_table), phi_table):
            self.w.push_table(0, val, x)
            
#        self.w = fe.Expression(cppcode=interpolator_code, 
#                                element = self.v_materials.ufl_element())
#        self.w.materials = self.material_markers
#        self.w.x_k = fe.interpolate(self.phi_old, self.v_materials)
#        for val, x in zip(self.sorption_phi2w(phi_table), phi_table):
#            self.w.push_table(0, val, x)
        
        
        
#        self.w = fe.Expression("""146
#                               /pow(1+
#                                    pow(-8e-8*R_water*T_ref*rho_w*log(phi),1.6)
#                                    ,0.375)""",
#                           phi=self.phi_old,R_water=R_water, T_ref=T_ref,rho_w=rho_w,
#                           element = self.v_materials.ufl_element()) 
    
        ########################################
        # kT
        ########################################                               
        self.kT = fe.Expression(cppcode=interpolator_code, 
                                element = self.v_materials.ufl_element())
        self.kT.materials = self.material_markers
        self.kT.x_k = fe.interpolate(self.w, self.v_materials)
        for k, w in zip(self.calc_lambda_t(w_table), w_table):
            self.kT.push_table(0, k, w)
#        self.kT.print_table(0)

        ########################################
        # Vapour permeability of material
        ########################################                               
        self.delta_p = fe.Expression(cppcode=interpolator_code, 
                                element = self.v_materials.ufl_element())
        self.delta_p.materials = self.material_markers
        self.delta_p.x_k = fe.interpolate(self.w, self.v_materials)
        for val, x in zip(self.calc_delta_p(w_table), w_table):
            self.delta_p.push_table(0, val, x)
        
        ########################################
        # Dw
        ########################################
        self.Dw = fe.Expression(cppcode=interpolator_code, 
                        element = self.v_materials.ufl_element())
        self.Dw.materials = self.material_markers
        self.Dw.x_k = fe.interpolate(self.w, self.v_materials)
        for val, x in zip(self.calc_Dw(w_table), w_table):
            self.Dw.push_table(0, val, x)
        
        ########################################
        # xi
        ########################################
        self.xi = fe.Expression(cppcode=interpolator_code, 
                        element = self.v_materials.ufl_element())
        self.xi.materials = self.material_markers
        self.xi.x_k = fe.interpolate(self.phi_old, self.v_materials)
        for val, x in zip(self.calc_xi(phi_table), phi_table):
            self.xi.push_table(0, val, x)
        
#        self.xi = fe.Expression("""-3.86765818700807e-10
#            *pow(-R_water*T_ref*rho_w*log(phi),1.6)
#            *pow(4.41513491667588e-12
#                 *pow(-R_water*T_ref*rho_w*log(phi),1.6) + 1,
#                 -1.375)
#            /(phi*log(phi))
#               """,
#               R_water=R_water, T_ref = T_ref, rho_w=rho_w,
#               phi=self.phi_old, 
#               element = self.v_materials.ufl_element())
#        self.xi = fe.Constant(48)        

    def time_steps(self):
        self.dt_fe = fe.Constant(self.dt)


    def build_solver(self):
        self.solver = solver.RHTSolver(self.mesh, self.v,
                                       T_old=self.T_old, 
                                       phi_old=self.phi_old,
                                       bc_T=self.bc_T, 
                                       bc_phi=self.bc_phi,
                                       t_end=self.t_end,
                                       crho_m=self.crho_m,
                                       kT=self.kT, 
                                       w=self.w,
                                       dt_fe=self.dt_fe,
                                       delta_p=self.delta_p,
                                       dps_dT=self.dps_dT,
                                       ps=self.ps,
                                       Dw=self.Dw,
                                       xi=self.xi,
                                       max_steps=self.max_steps,
                                       time_gamma=self.time_gamma,
                                       max_dt=self.max_dt,
                                       # Propes
                                       prope_fields=["T", "w" ],
                                       prope_xs=[self.prope_x_T, self.prope_x_w],
                                       prope_times=self.prope_times,
                                       order_2nd=self.order_2nd,
                                       v_materials=self.v_materials,
                                       material_markers=self.material_markers,
                                       out_dir=self.out_dir,
                                       )
        return self.solver


def read_prope_file(path):    
    with open(path, "r") as ifile:
        with open(path, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            lines = [line for line in reader]

    x = [float(part) for part in lines[0][1:]]

    times = []
    vals = []
    for parts in lines[1:]:
        its = [float(part) for part in parts]
        vals.append(its[1:])
        times.append(its[0])    

    return sp.array(x), sp.array(times), sp.array(vals)


def validate_SFS_EN_15026():
    Tpath = os.path.join("outData", "prope_T.txt")
    wpath = os.path.join("outData", "prope_w.txt")
    xT, tT, T = read_prope_file(Tpath)
    xw, tw, w = read_prope_file(wpath)
     ##############################
    # Check with validation data
    ##############################
    
    print("7 days")
    print(w[0])
    print(T[0]-C2K)
    
    # 7th day moisture
    assert 50.2 <= w[0][0] <= 54.5 # Fails with linear interpolation
    assert 41.3 <= w[0][1] <= 45.6
    assert 40.8 <= w[0][2] <= 45.1
    assert 40.8 <= w[0][3] <= 45.1
    
    # 7th day temperature
    assert 26.4 <= T[0][0] - C2K <= 26.9
    assert 23.6 <= T[0][1] - C2K <= 24.1
    assert 21.7 <= T[0][2] - C2K <= 22.2
    assert 20.6 <= T[0][3] - C2K <= 21.1
    assert 20.0 <= T[0][4] - C2K <= 20.5
    assert 19.8 <= T[0][5] - C2K <= 20.4
    assert 19.8 <= T[0][6] - C2K <= 20.3
    assert 19.8 <= T[0][7] - C2K <= 20.3

    print("30 days")
    print(w[1])
    print(T[1]-C2K)

    # 30th day moisture
    assert 81.0 <= w[1][0] <= 85.3
    assert 51.1 <= w[1][1] <= 55.3
    assert 43.6 <= w[1][2] <= 47.9
    assert 41.5 <= w[1][3] <= 45.7
    assert 40.9 <= w[1][4] <= 45.2
    assert 40.8 <= w[1][5] <= 45.1
    assert 40.8 <= w[1][6] <= 45.1

    # 30th day moisture
    assert 28.1 <= T[1][0] - C2K <= 28.6
    assert 26.5 <= T[1][1] - C2K <= 27.0
    assert 25.0 <= T[1][2] - C2K <= 25.5
    assert 23.7 <= T[1][3] - C2K <= 24.3
    assert 22.7 <= T[1][4] - C2K <= 23.2
    assert 21.8 <= T[1][5] - C2K <= 22.3
    assert 20.7 <= T[1][6] - C2K <= 21.2
    assert 20.1 <= T[1][7] - C2K <= 20.6

    print("365 days")
    print(w[2])
    print(T[2]-C2K)

    # 365th day moisture
    assert 117.5 <= w[2][0] <= 121.8
    assert 104.4 <= w[2][1] <= 108.7
    assert 88.7 <= w[2][2] <= 93.0
    assert 75.6 <= w[2][3] <= 77.9
    assert 62.8 <= w[2][4] <= 67.1
    assert 55.7 <= w[2][5] <= 60.0
    assert 47.9 <= w[2][6] <= 52.2
    assert 44.1 <= w[2][7] <= 48.4

    # 365th day moisture
    assert 29.2 <= T[2][0] - C2K <= 29.8
    assert 28.8 <= T[2][1] - C2K <= 29.3
    assert 28.3 <= T[2][2] - C2K <= 28.8
    assert 27.8 <= T[2][3] - C2K <= 28.4
    assert 27.4 <= T[2][4] - C2K <= 27.9
    assert 26.9 <= T[2][5] - C2K <= 27.4
    assert 26.0 <= T[2][6] - C2K <= 26.6
    assert 25.2 <= T[2][7] - C2K <= 25.7


def read_1dmesh(path):
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for line in reader:
            parts = line
    x = [float(part[2:-1]) for part in parts]
    return sp.array(x)

def read_1dfield(path):
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    
        times = []
        vals = []
        for parts in reader:
#            its = [float(part[2:-1]) for part in parts]
            its = [float(part) for part in parts]
            vals.append(its[1:])
            times.append(its[0])
    
    
    return sp.array(times), sp.array(vals)



def post_case(path):
    mesh = read_1dmesh(os.path.join(path, "mesh.txt"))
    times, Ts = read_1dfield(os.path.join(path, "T.txt"))
    times, phis = read_1dfield(os.path.join(path, "phi.txt"))    

    fig, axes = plt.subplots(2)
    
    # T
    ax = axes[0]
    for k,tim in enumerate(times):
        ax.plot(mesh, Ts[k]-C2K)
    ax.set_xlabel(r"$x$ $(m)$")
    ax.set_ylabel(r"$T$ $(^\circ C)$")
    ax.set_xlim(0,5)    

    # phi
    ax = axes[1]
    for k,tim in enumerate(times):
        ax.plot(mesh, phis[k]*100)
    ax.set_xlabel(r"$x$ $(m)$")
    ax.set_ylabel(r"$\phi$ $(\%)$")    
    ax.set_xlim(0,0.1)

    for ax in axes.ravel():
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(path, "results.pdf"))

def main_SFS_EN_15026():
    
    
    # Pre
    case = Validation_SFS_EN_15026()
    case.make_mesh()
    case.boundary_conditions()
    case.initial_conditions()
    case.general_material_properties()
    case.material_properties_equations()
#    case.material_properties_interpolation()
    case.time_steps()
    
    # Solver
    solver = case.build_solver()
    solver.solve()

def run_SFS_EN_15026():
    main_SFS_EN_15026()
    validate_SFS_EN_15026()
    post_case("outData")
    
class DiffusionCase(object):

    def __init__(self, measurement):
        
        self.measurement = measurement
        self.out_dir = os.path.join("diffusionCases", self.measurement.id)
        
        self.t_end = measurement.measurement_time[-1] #10# 7*s2d#
        self.order_2nd = False
        self.dt = s2h#60*s2min # 10# 
        self.time_gamma = 1.0
        self.max_dt = self.dt#60
        
        self.mesh_fact = 1
        
        
        self.max_steps = int(1.1*self.t_end / self.dt)
        
        # moisture locations
        offset = 7e-3
        # Inside on the left at x=0
#        self.prope_x = sp.array([measurement.Ls[:2].sum()+offset, 
#                                 measurement.Ls[:3].sum()-offset])
        self.prope_x = sp.array([measurement.Ls[0]+measurement.Ls[1]+offset, 
                                 measurement.Ls[0]+measurement.Ls[1]+measurement.Ls[2]-offset])
#        print(self.prope_x)
    
    def make_mesh(self):
#        print(self.measurement.Ls)        
        
        xs = sp.array([0, 
                       self.measurement.Ls[0], 
              self.measurement.Ls[:2].sum(),
             self.measurement.Ls[:3].sum(),
             self.measurement.Ls[:4].sum()])        
#        print(xs)
        self.xs = xs

        # x0        
        dx0 = 3e-3/self.mesh_fact
        x0 = list(sp.linspace(xs[0], xs[1], sp.ceil((xs[1] - xs[0])/dx0)))
#        print(x0, len(x0))        
        
        # x1, single cell
#        dx1 = 3e-3
#        x1 = sp.array([])
        
        # x2
        dx2 = 4e-3/self.mesh_fact
        x2 = list(sp.linspace(xs[2], xs[3], sp.ceil((xs[3] - xs[2])/dx2)))
#        print(x2, len(x2))        
        
        # x3
        dx3 = 3e-3/self.mesh_fact
        x3 = list(sp.linspace(xs[3]+dx2, xs[4], sp.ceil((xs[4] - xs[3])/dx3)))
#        print(x3, len(x3))        
        
        
        xpoints = x0+x2+x3
#        print(xpoints, len(xpoints))
        self.mesh = fe.IntervalMesh(len(xpoints)-1, xs[0], xs[-1])
        self.mesh.coordinates()[:] = sp.reshape(xpoints,(len(xpoints),1))

        self.v = fe.FunctionSpace(self.mesh, "CG", 1)
        self.v_materials = fe.FunctionSpace(self.mesh, 'DG', 1) 
        
        # Subdomains
        sub0 = fe.AutoSubDomain(lambda x: (x[0] <= xs[1]))
        sub1 = fe.AutoSubDomain(lambda x: (x[0] >= xs[1] and x[0] <= xs[2]))    
        sub2 = fe.AutoSubDomain(lambda x: (x[0] >= xs[2] and x[0] <= xs[3]))
        sub3 = fe.AutoSubDomain(lambda x: (x[0] >= xs[3]))

        # Mark subdomains
        self.material_markers = fe.MeshFunction("size_t", self.mesh, 
                                           self.mesh.topology().dim(), 0)
        sub0.mark(self.material_markers, 0)
        sub1.mark(self.material_markers, 1)
        sub2.mark(self.material_markers, 2)
        sub3.mark(self.material_markers, 3)
   
#        print(self.material_markers.array())
    
    def make_mesh_hack(self):
#        print(self.measurement.Ls)        
        

        xs = sp.array([0, 
                       self.measurement.Ls[0], 
              self.measurement.Ls[:2].sum(),
             self.measurement.Ls[:3].sum(),
             self.measurement.Ls[:4].sum()])        
#        print(xs)
        self.xs = xs


        ###############################################
        # ORG
        ###############################################


        # x0        
        dx0 = 3e-3/self.mesh_fact
        x0 = list(sp.linspace(xs[0], xs[1], sp.ceil((xs[1] - xs[0])/dx0)))
#        print(x0, len(x0))        
        
        # x1, 
        dx1 = self.measurement.Ls[1] / 5
        x1 = list(sp.linspace(xs[1]+dx1, xs[2], sp.ceil(self.measurement.Ls[1]/dx1)))
        
        # x2
        dx2 = 4e-3/self.mesh_fact
        x2 = list(sp.linspace(xs[2]+dx2, xs[3], sp.ceil((xs[3] - xs[2])/dx2)))
#        print(x2, len(x2))        
        
        # x3
        dx3 = 3e-3/self.mesh_fact
        x3 = list(sp.linspace(xs[3]+dx2, xs[4], sp.ceil((xs[4] - xs[3])/dx3)))
#        print(x3, len(x3))        
#        
#        
        xpoints = x0+x1+x2+x3
##        print(xpoints, len(xpoints))
#        self.mesh = fe.IntervalMesh(len(xpoints)-1, xs[0], xs[-1])
#        self.mesh.coordinates()[:] = sp.reshape(xpoints,(len(xpoints),1))

    

        self.mesh = fe.IntervalMesh(len(xpoints)-1, xs[0], xs[-1])
        self.mesh.coordinates()[:] = sp.reshape(xpoints,(len(xpoints),1))


    
        ###############################################
        # REST
        ###############################################








        self.v = fe.FunctionSpace(self.mesh, "CG", 2)
        self.v_materials = fe.FunctionSpace(self.mesh, 'DG', 1) 
        
        # Subdomains
        sub0 = fe.AutoSubDomain(lambda x: (x[0] <= xs[1]))
        sub1 = fe.AutoSubDomain(lambda x: (x[0] >= xs[1] and x[0] <= xs[2]))    
        sub2 = fe.AutoSubDomain(lambda x: (x[0] >= xs[2] and x[0] <= xs[3]))
        sub3 = fe.AutoSubDomain(lambda x: (x[0] >= xs[3]))

        # Mark subdomains
        self.material_markers = fe.MeshFunction("size_t", self.mesh, 
                                           self.mesh.topology().dim(), 0)
        sub0.mark(self.material_markers, 0)
        sub1.mark(self.material_markers, 1)
        sub2.mark(self.material_markers, 2)
        sub3.mark(self.material_markers, 3)
        
        
        # 0, sisalevy ok, eniten eroa
        # 1, hoyrynsulku ok
        # 2, insulation ok
        # 3, tuulensuoja ok
   
        for ind in self.material_markers.array():
            print(ind)
    
    
#        print(self.material_markers.array())
    
    def boundary_conditions(self):
        boundary_in = fe.CompiledSubDomain('on_boundary && near(x[0], 0)')
        boundary_out = fe.CompiledSubDomain('on_boundary && near(x[0], L)',
                                            L=self.measurement.Ls.sum())

        class TimeInterpolation(fe.Expression):
            def eval(self, value, x):
                """x: spatial point, value[0]: function value."""
                value[0] = self.interpolator(self.t)

                                            
        #########################################
        # T_in
        #########################################
        self.bc_T_in = TimeInterpolation(element=self.v.ufl_element())
        self.bc_T_in.interpolator = interpolate.interp1d(
                                        self.measurement.measurement_time,
                                        self.measurement.T_in
                                        )
        self.bc_T_in.t = 0.0
        DirichletBC_T_in  = fe.DirichletBC(self.v, self.bc_T_in, boundary_in)
        #########################################
        # T_out
        #########################################
        self.bc_T_out = TimeInterpolation(element=self.v.ufl_element())
        self.bc_T_out.interpolator = interpolate.interp1d(
                                        self.measurement.measurement_time,
                                        self.measurement.T_out
                                        )
        self.bc_T_out.t = 0.0
        DirichletBC_T_out  = fe.DirichletBC(self.v, self.bc_T_out, boundary_out)
        #########################################
        # phi_in
        #########################################
        self.bc_phi_in = TimeInterpolation(element=self.v.ufl_element())
        self.bc_phi_in.interpolator = interpolate.interp1d(
                                        self.measurement.measurement_time,
                                        self.measurement.RH_in
                                        )
        self.bc_phi_in.t = 0.0
        DirichletBC_phi_in  = fe.DirichletBC(self.v, self.bc_phi_in, boundary_in)
        #########################################
        # phi_out
        #########################################
        self.bc_phi_out = TimeInterpolation(element=self.v.ufl_element())
        self.bc_phi_out.interpolator = interpolate.interp1d(
                                        self.measurement.measurement_time,
                                        self.measurement.RH_out
                                        )
        self.bc_phi_out.t = 0.0
        DirichletBC_phi_out  = fe.DirichletBC(self.v, self.bc_phi_out, boundary_out)
            
        self.bc_T = [DirichletBC_T_in, DirichletBC_T_out]
        self.bc_phi = [DirichletBC_phi_in, DirichletBC_phi_out]
   
    def initial_conditions(self):
    
        # Initial conditions
        T_init    = (self.measurement.T_S[0] + self.measurement.T_U[0]) / 2
        phi_init = (self.measurement.RH_S[0] + self.measurement.RH_U[0]) / 2    
#        print(phi_init)
        assert(T_init>0 and T_init<100+C2K)
        assert(phi_init>=0 and phi_init<=1)
    
        # implicit
        self.T_old   = fe.Function(self.v)
        self.phi_old = fe.Function(self.v)
        
        self.T_old.interpolate(fe.Constant(T_init))
        self.phi_old.interpolate(fe.Constant(phi_init))
           
    def general_material_properties(self):    
        ############################################
        # Temperature based    
        ############################################        
        #Vinha PhD thesis. Eqs. 1.28 and 1.30 Eq. 1.29 page 27
        # T in Kelvins
        #verified with T=0C, 10C, 20C, 30C with
        #https://www.engineeringtoolbox.com/water-vapor-saturation-pressure-d_599.html?vA=-20&units=C#
        # 610.5*2.718^((17.269*T - 4717.03) / (T - 35.85))  
        # 610.5*2.718**((17.269*T - 4717.03) / (T - 35.85))  
        self.ps = fe.Expression("""610.5*pow(2.718,
                                        (17.269*Ti - 4717.03) / (Ti - 35.85))""",
                           Ti=self.T_old,
                           element = self.v.ufl_element())
            
            
        # Petteri
        #610.5*2.718^((17.269*T - 4717.03)/(T - 35.85)) *(17.2672094763233/(T - 35.85) - 0.999896315728952*(17.269*T - 4717.03)/T - 35.85^(2))
        
        # Sympy
        #610.5*2.718**((17.269*T - 4717.03)/(T - 35.85))*(17.2672094763233/(T - 35.85) - 0.999896315728952*(17.269*T - 4717.03)/(T - 35.85)**2)
        
#        self.dps_dT = fe.Expression("""(610.5*pow(2.718,((17.269*Ti - 4717.03)/(Ti - 35.85)))
#                *(17.2672094763233/(Ti - 35.85) - 0.999896315728952
#                *(17.269*Ti - 4717.03)
#                /pow((Ti - 35.85),2) ))""",
#                           Ti=self.T_old,
#                           element = self.v.ufl_element())
        
        self.dps_dT = fe.Expression("""(610.5*pow(2.718,
                                             (17.269*Ti - 4717.03)/(Ti - 35.85))
                *(17.2672094763233/(Ti - 35.85) 
                - 0.999896315728952*(17.269*Ti - 4717.03)/pow(Ti - 35.85,2))
                )""",
                           Ti=self.T_old,
                           element = self.v.ufl_element())

    def material_properties_equations(self):

        # Dry material
        self.crho_m = 1.824e6

        ############################################
        # Moisture based 
        ############################################        
        self.w = fe.Expression("""146
                               /pow(1+
                                    pow(-8e-8*R_water*T_ref*rho_w*log(phi),1.6)
                                    ,0.375)""",
                           phi=self.phi_old,R_water=R_water, T_ref=T_ref,rho_w=rho_w,
                           element = self.v.ufl_element()) 
#        self.w = fe.Constant(0.20)
                               
        self.kT = fe.Expression("""(1.5 + 15.8/1000*w)""",
                           w=self.w,
                           element = self.v.ufl_element())
#        self.kT = fe.Constant(1.5)

        #  Vapour permeability of material
        self.delta_p = fe.Expression("""1/(R_water*T_ref)
                                *26.1e-6/200
                                *(1-w/146)/(0.503
                                *pow(1-w/146,2)
                                +0.497)""",
                           R_water=R_water, T_ref = T_ref,w=self.w,
                           element = self.v.ufl_element())
#        self.delta_p = fe.Constant(0.6e-12)
        
        self.Dw = fe.Expression("""-1*exp(- 39.2619
               + 0.0704*(w-73)
               - 1.7420e-4*pow(w-73,2)
               - 2.7953e-6*pow(w-73,3)
               - 1.1566e-7*pow(w-73,4)
               + 2.5969e-9*pow(w-73,5)
               )
               *
               (-a*c*d*pow(b/w, c)*pow(pow(b/w,c) - 1, d)/(w*(pow(b/w,c) - 1)))
               """,
               w=self.w, 
               a = 0.125e8, b = 146, c = 1.0/0.375, d = 0.625,
               element = self.v.ufl_element())
#        self.Dw = fe.Constant(2e-11)
        
        self.xi = fe.Expression("""-3.86765818700807e-10
            *pow(-R_water*T_ref*rho_w*log(phi),1.6)
            *pow(4.41513491667588e-12
                 *pow(-R_water*T_ref*rho_w*log(phi),1.6) + 1,
                 -1.375)
            /(phi*log(phi))
               """,
               R_water=R_water, T_ref = T_ref, rho_w=rho_w,
               phi=self.phi_old, 
               element = self.v.ufl_element())
#        self.xi = fe.Constant(48)


    def material_properties_interpolation(self):
 
        with open("interpolator.C","r") as ifile:
            interpolator_code = ifile.read()


        # Dry material
#        self.crho_m = fe.Expression(cppcode=interpolator_code, 
#                        element = self.v_materials.ufl_element())
#        self.crho_m.materials = self.material_markers
#        # Useless
#        self.crho_m.x_k = fe.interpolate(self.phi_old, self.v_materials)
#
#        for matIndex, material in enumerate(self.measurement.materials):
#            self.crho_m.push_table(matIndex, material.c * material.rho, 0.5)
##            self.crho_m.push_table(matIndex, 280*1500, 0.5)
        
#        self.crho_m = fe.Constant(280*1500)


        code = '''
class MyFunc : public Expression
{
public:

  std::shared_ptr<MeshFunction<std::size_t> > cell_data;

  MyFunc() : Expression()
  {
  }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    assert(cell_data);
    const Cell cell(*cell_data->mesh(), c.index);
    switch ((*cell_data)[cell.index()])
    {
    case 0:
      values[0] = 574*1100;
      break;
    case 1:
      values[0] = 980*2300;
      break;
    case 2:
      values[0] = 22*850;//280*1500;//
      break;
    case 3:
      values[0] = 280*1500;
      break;  
    //default:
     // values[0] = 0.0;
    }
  }
};'''

        #element = self.v_materials.ufl_element()
        self.crho_m = fe.Expression(code, degree=0)
        self.crho_m.cell_data = self.material_markers
        
        
        

 

        ############################################
        # Moisture based 
        ############################################        
        self.w = fe.Expression(cppcode=interpolator_code, 
                        element = self.v_materials.ufl_element())
        self.w.materials = self.material_markers
        self.w.x_k = fe.interpolate(self.phi_old, self.v_materials)

#        fig, axes = plt.subplots(4)
#        RHfit = sp.linspace(0,1,1000)
#        wfits = []
        for matIndex, material in enumerate(self.measurement.materials):
            if hasattr(material, 'w_sorption_RH'):
                
                # Fit
#                zfit = sp.polyfit(material.w_sorption_RH[0:-1], material.w_sorption[0:-1], 4)
#                polyfucn = sp.poly1d(zfit)
#                wfit = polyfucn(RHfit)
#                wfits.append(wfit)
                
                # Plot
#                ax = axes[matIndex]    
#                ax.plot(material.w_sorption_RH, material.w_sorption, "d")
#                ax.plot(RHfit, wfit, "-")
#                ax.set_ylim(material.w_sorption[0], material.w_sorption[-2])
                
#                # Push fit
#                for val, x in zip(wfit, RHfit):
#                    self.w.push_table(matIndex, val, x)

                
                
                for val, x in zip(material.w_sorption, material.w_sorption_RH):
                    self.w.push_table(matIndex, val, x)
                
            else:
                self.w.push_table(matIndex, material.w_sorption, 0.5)
#                wfits.append([material.w_sorption]*len(RHfit))

#        for ax in axes.ravel():
##            ax.set_ylim(0,None)
#            ax.set_xlim(0,1)
#        fig.tight_layout()
            

#            print()
#            print("w", matIndex)
#            self.w.print_table(matIndex)
                
#        self.w = fe.Constant(40)
#        self.w = fe.Expression("""146
#                               /pow(1+
#                                    pow(-8e-8*R_water*T_ref*rho_w*log(phi),1.6)
#                                    ,0.375)""",
#                           phi=self.phi_old,R_water=R_water, T_ref=T_ref,rho_w=rho_w,
#                           element = self.v.ufl_element()) 




        ########################################
        # kT
        ########################################                               
#        self.kT = fe.Expression(cppcode=interpolator_code, 
#                        element = self.v_materials.ufl_element())
#        self.kT.materials = self.material_markers
#        self.kT.x_k = fe.interpolate(self.phi_old, self.v_materials)
#        for matIndex, material in enumerate(self.measurement.materials):
#            if hasattr(material, 'kt_RH'):
#                for val, x in zip(material.kt, material.kt_RH):
#                    self.kT.push_table(matIndex, val, x)
#            else:
#                self.kT.push_table(matIndex, material.kt, 0.5)
#                
##            print()
##            print("kT", matIndex)
##            self.kT.print_table(matIndex)
##        self.kT = fe.Constant(0.05)
##        self.kT = fe.Expression("""(1.5 + 15.8/1000*w)""",
##                   w=self.w,
##                   element = self.v.ufl_element())   
        
        
        
        
                kcode = '''
class MyFunc : public Expression
{
public:

  std::shared_ptr<MeshFunction<std::size_t> > cell_data;

  MyFunc() : Expression()
  {
  }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    assert(cell_data);
    const Cell cell(*cell_data->mesh(), c.index);
    switch ((*cell_data)[cell.index()])
    {
    case 0:
      values[0] = 0.19;
      break;
    case 1:
      values[0] = 0.4;
      break;
    case 2:
      values[0] = 0.0351;
      break;
    case 3:
      values[0] = 0.0517;
      break;  
    //default:
     // values[0] = 0.0;
    }
  }
};'''

        #element = self.v_materials.ufl_element()
        self.kT = fe.Expression(kcode, degree=1)
        self.kT.cell_data = self.material_markers
    
        ########################################
        # Vapour permeability of material
        ########################################       
        self.delta_p = fe.Expression(cppcode=interpolator_code, 
                        element = self.v_materials.ufl_element())
        self.delta_p.materials = self.material_markers
        self.delta_p.x_k = fe.interpolate(self.phi_old, self.v_materials)
        for matIndex, material in enumerate(self.measurement.materials):
            if hasattr(material, 'delta_p_RH'):
                for val, x in zip(material.delta_p, material.delta_p_RH):
                    self.delta_p.push_table(matIndex, val, x)
            else:
                self.delta_p.push_table(matIndex, material.delta_p, 0.5)
#            print()
#            print("delta_p", matIndex)
#            self.delta_p.print_table(matIndex)
#        self.delta_p = fe.Constant(3e-6)
#        self.delta_p = fe.Expression("""1/(R_water*T_ref)
#                                *26.1e-6/200
#                                *(1-w/146)/(0.503
#                                *pow(1-w/146,2)
#                                +0.497)""",
#                           R_water=R_water, T_ref = T_ref,w=self.w,
#                           element = self.v.ufl_element())
        
        
        ########################################
        # Dw
        ########################################
        self.Dw = fe.Expression(cppcode=interpolator_code, 
                        element = self.v_materials.ufl_element())
        self.Dw.materials = self.material_markers
        self.Dw.x_k = fe.interpolate(self.phi_old, self.v_materials)
        for matIndex, material in enumerate(self.measurement.materials):
            if hasattr(material, 'Dw_RH'):
                for val, x in zip(material.Dw, material.Dw_RH):
                    self.Dw.push_table(matIndex, val, x)
            else:
                self.Dw.push_table(matIndex, material.Dw, 0.5)
                
#            print()
#            print("Dw", matIndex)
#            self.Dw.print_table(matIndex)

#        self.Dw = fe.Constant(0)
#        self.Dw = fe.Expression("""-1*exp(- 39.2619
#               + 0.0704*(w-73)
#               - 1.7420e-4*pow(w-73,2)
#               - 2.7953e-6*pow(w-73,3)
#               - 1.1566e-7*pow(w-73,4)
#               + 2.5969e-9*pow(w-73,5)
#               )
#               *
#               (-a*c*d*pow(b/w, c)*pow(pow(b/w,c) - 1, d)/(w*(pow(b/w,c) - 1)))
#               """,
#               w=self.w, 
#               a = 0.125e8, b = 146, c = 1.0/0.375, d = 0.625,
#               element = self.v.ufl_element())

        ########################################
        # xi
        ########################################
        self.xi = fe.Expression(cppcode=interpolator_code, 
                        element = self.v_materials.ufl_element())
        self.xi.materials = self.material_markers
        self.xi.x_k = fe.interpolate(self.phi_old, self.v_materials)
        
        
#        fig, axes = plt.subplots(4)
        for matIndex, material in enumerate(self.measurement.materials):
                        
            if hasattr(material, 'w_sorption_RH'):
                
                RHs_org = material.w_sorption_RH
                ws = material.w_sorption                

#                RHs_org = RHfit
#                ws = wfits[matIndex]
                
                xis = []
                RHs = []
                for k in range(len(ws)-1):
                    dw  = ws[k+1] - ws[k]
                    dRH = RHs_org[k+1] - RHs_org[k]
                    xis.append(dw/dRH)
                    RHs.append(RHs_org[k]+0.5*dRH)
                
                
                # Plot
#                ax = axes[matIndex]    
#                ax.plot(RHs, xis, "-")
                
                
                
                for val, x in zip(xis, RHs):
                    self.xi.push_table(matIndex, val, x)
            else:
                self.xi.push_table(matIndex, 0, 0.5)

#            print()
#            print("xi", matIndex)
#            self.xi.print_table(matIndex)


#        self.xi = fe.Expression("""-3.86765818700807e-10
#            *pow(-R_water*T_ref*rho_w*log(phi),1.6)
#            *pow(4.41513491667588e-12
#                 *pow(-R_water*T_ref*rho_w*log(phi),1.6) + 1,
#                 -1.375)
#            /(phi*log(phi))
#               """,
#               R_water=R_water, T_ref = T_ref, rho_w=rho_w,
#               phi=self.phi_old, 
#               element = self.v.ufl_element())
#        self.xi = fe.Constant(0)


    def material_properties_hack(self):
        
 
        with open("interpolator_x.C","r") as ifile:
            interpolator_code = ifile.read()

        # Dry material
#        self.crho_m = fe.Expression(cppcode=interpolator_code, 
#                        element = self.v_materials.ufl_element())
#        self.crho_m.materials = self.material_markers
#        # Useless
#        self.crho_m.x_k = fe.interpolate(self.phi_old, self.v_materials)
#
#        for matIndex, material in enumerate(self.measurement.materials):
#            self.crho_m.push_table(matIndex, material.c * material.rho, 0.5)
##            self.crho_m.push_table(matIndex, 280*1500, 0.5)
        
#        self.crho_m = fe.Constant(280*1500)

#
#        code = '''
#class MyFunc : public Expression
#{
#public:
#
#  std::shared_ptr<MeshFunction<std::size_t> > cell_data;
#
#  MyFunc() : Expression()
#  {
#  }
#
#  void eval(Array<double>& values, const Array<double>& x,
#            const ufc::cell& c) const
#  {
#    assert(cell_data);
#    const Cell cell(*cell_data->mesh(), c.index);
#    switch ((*cell_data)[cell.index()])
#    {
#    case 0:
#      values[0] = 574*1100;
#      break;
#    case 1:
#      values[0] = 980*2300;
#      break;
#    case 2:
#      values[0] = 22*850;//280*1500;//
#      break;
#    case 3:
#      values[0] = 280*1500;
#      break;  
#    //default:
#     // values[0] = 0.0;
#    }
#  }
#};'''
#
#        #element = self.v_materials.ufl_element()
#        self.crho_m = fe.Expression(code, degree=0)
#        self.crho_m.cell_data = self.material_markers
        
        
        
        code = '''
class MyFunc : public Expression
{
public:

  //std::shared_ptr<MeshFunction<std::size_t> > cell_data;

  float x0 = 0.0;  
  float x1 = 0.0;
  float x2 = 0.0;

  MyFunc() : Expression()
  {
  }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    assert(cell_data);
    
    // mat 0
    if (x[0] <= x0) {
        values[0] = 574*1100;
    // mat 1
    } else if (x[0] <= x1) {
        values[0] = 980*2300;    
    // mat 2
    } else if (x[0] <= x2) {
        values[0] = 22*850;    
    // mat 3
    } else {
        values[0] = 280*1500;
    }
  }
};'''

        #element = self.v_materials.ufl_element()
        self.crho_m = fe.Expression(code, degree=0)
        self.crho_m.x0 = self.measurement.Ls[:1].sum()
        self.crho_m.x1 = self.measurement.Ls[:2].sum()
        self.crho_m.x2 = self.measurement.Ls[:3].sum()
        
    



        ############################################
        # Moisture based 
        ############################################        
        self.w = fe.Expression(cppcode=interpolator_code, 
                               element = self.v_materials.ufl_element())
#        self.w.materials = self.material_markers
        self.w.x0 = self.measurement.Ls[:1].sum()
        self.w.x1 = self.measurement.Ls[:2].sum()
        self.w.x2 = self.measurement.Ls[:3].sum()
        
        self.w.x_k = fe.interpolate(self.phi_old, self.v_materials)

#        fig, axes = plt.subplots(4)
#        RHfit = sp.linspace(0,1,10000)
#        wfits = []
        for matIndex, material in enumerate(self.measurement.materials):
            if hasattr(material, 'w_sorption_RH'):
                
#                # Fit
#                zfit = sp.polyfit(material.w_sorption_RH[0:-1], material.w_sorption[0:-1], 4)
#                polyfucn = sp.poly1d(zfit)
#                wfit = polyfucn(RHfit)
#                wfits.append(wfit)
#            
#                wmax = material.w_sorption[-2]
##                print(wmax)
#                def fit_func(phi, a,b,c):
#                    return wmax / (1+(a*sp.log(phi))**b)**c
#                
#                
##                p0=[      [-8e-8*462*293.15*1000, 1.6,0.3],
##                    [1,1,1],
##                    [-19.3092323179 ,1.35936769118 ,0.51126125371],
##                    
##                       [-500,1.35936769118 ,0.51126125371]
##                       ]
#
#                popt, pcov = optimize.curve_fit(fit_func, 
#                                                material.w_sorption_RH[:-1], 
#                                                material.w_sorption[:-1],
##                                                p0[matIndex]
#                                                [-8e-8*462*293.15*1000,
#                                                 1.6,0.3]
#                                                
#                                                )
##                print(*popt)
#                wfit=fit_func(RHfit, *popt)
#
#                
#                # Plot
#                ax = axes[matIndex]    
#                ax.plot(material.w_sorption_RH, material.w_sorption, "d")
#                ax.plot(RHfit, wfit, "-")
#                ax.set_ylim(material.w_sorption[0], material.w_sorption[-2])
#                
#                # Push fit
#                for val, x in zip(wfit, RHfit):
#                    self.w.push_table(matIndex, val, x)

                
                
                for val, x in zip(material.w_sorption, material.w_sorption_RH):
                    self.w.push_table(matIndex, val, x)
                
            else:
                self.w.push_table(matIndex, material.w_sorption, 0.5)
#                wfits.append([material.w_sorption]*len(RHfit))
#
#            print()
#            print("w", matIndex)
#            self.w.print_table(matIndex)


#        self.w = fe.Constant(40)


        #146/(1+(-8e-8*R_water*T_ref*rho_w*log(phi))^1.6)^0.375
        #146/(1+(-8e-8*462*293.15*1000*log(phi))^1.6)^0.375
#        self.w = fe.Expression("""146
#                               /pow(1+
#                                    pow(-8e-8*R_water*T_ref*rho_w*log(phi),1.6)
#                                    ,0.375)""",
#                           phi=self.phi_old,R_water=R_water, T_ref=T_ref,rho_w=rho_w,
#                           element = self.v.ufl_element()) 

        ########################################
        # xi
        ########################################
        
        
        self.xi = fe.diff(self.w, self.phi_old)

        self.xi = fe.Expression(cppcode=interpolator_code, 
                        element = self.v_materials.ufl_element())
#        self.xi.materials = self.material_markers
        self.xi.x0 = self.measurement.Ls[:1].sum()
        self.xi.x1 = self.measurement.Ls[:2].sum()
        self.xi.x2 = self.measurement.Ls[:3].sum()
        
        self.xi.x_k = fe.interpolate(self.phi_old, self.v_materials)
        
        
#        fig, axes = plt.subplots(4)
        for matIndex, material in enumerate(self.measurement.materials):
                        
            if hasattr(material, 'w_sorption_RH'):
                
                RHs_org = material.w_sorption_RH
                ws = material.w_sorption                

#                RHs_org = RHfit
#                ws = wfits[matIndex]
                
                xis = []
                RHs = []
                for k in range(len(ws)-1):
                    dw  = ws[k+1] - ws[k]
                    dRH = RHs_org[k+1] - RHs_org[k]
                    xis.append(dw/dRH)
                    RHs.append(RHs_org[k]+0.5*dRH)
                
                
                # Plot
#                ax = axes[matIndex]    
#                ax.plot(RHs, xis, "-")
                
                
                
                for val, x in zip(xis, RHs):
                    self.xi.push_table(matIndex, val, x)
            else:
                self.xi.push_table(matIndex, 0, 0.5)


#        self.xi = fe.Expression("""-3.86765818700807e-10
#            *pow(-R_water*T_ref*rho_w*log(phi),1.6)
#            *pow(4.41513491667588e-12
#                 *pow(-R_water*T_ref*rho_w*log(phi),1.6) + 1,
#                 -1.375)
#            /(phi*log(phi))
#               """,
#               R_water=R_water, T_ref = T_ref, rho_w=rho_w,
#               phi=self.phi_old, 
#               element = self.v.ufl_element())



        ########################################
        # kT
        ########################################                               
        self.kT = fe.Expression(cppcode=interpolator_code, 
                        element = self.v_materials.ufl_element())
#        self.kT.materials = self.material_markers
        self.kT.x0 = self.measurement.Ls[:1].sum()
        self.kT.x1 = self.measurement.Ls[:2].sum()
        self.kT.x2 = self.measurement.Ls[:3].sum()
        
        self.kT.x_k = fe.interpolate(self.phi_old, self.v_materials)
        for matIndex, material in enumerate(self.measurement.materials):
            if hasattr(material, 'kt_RH'):
                for val, x in zip(material.kt, material.kt_RH):
                    self.kT.push_table(matIndex, val, x)
            else:
                self.kT.push_table(matIndex, material.kt, 0.5)
                
#            print()
#            print("kT", matIndex)
#            self.kT.print_table(matIndex)
    
        ########################################
        # Vapour permeability of material
        ########################################       
        self.delta_p = fe.Expression(cppcode=interpolator_code, 
                        element = self.v_materials.ufl_element())
        self.delta_p.materials = self.material_markers
        self.delta_p.x_k = fe.interpolate(self.phi_old, self.v_materials)
        for matIndex, material in enumerate(self.measurement.materials):
            if hasattr(material, 'delta_p_RH'):
                for val, x in zip(material.delta_p, material.delta_p_RH):
                    self.delta_p.push_table(matIndex, val, x)
            else:
                self.delta_p.push_table(matIndex, material.delta_p, 0.5)
        
        ########################################
        # Dw
        ########################################
        self.Dw = fe.Expression(cppcode=interpolator_code, 
                        element = self.v_materials.ufl_element())
#        self.Dw.materials = self.material_markers
        self.Dw.x0 = self.measurement.Ls[:1].sum()
        self.Dw.x1 = self.measurement.Ls[:2].sum()
        self.Dw.x2 = self.measurement.Ls[:3].sum()
        
        self.Dw.x_k = fe.interpolate(self.phi_old, self.v_materials)
        for matIndex, material in enumerate(self.measurement.materials):
            if hasattr(material, 'Dw_RH'):
                for val, x in zip(material.Dw, material.Dw_RH):
                    self.Dw.push_table(matIndex, val, x)
            else:
                self.Dw.push_table(matIndex, material.Dw, 0.5)


#        self.Dw = fe.Constant(0)

        

    def time_steps(self):
        self.dt_fe = fe.Constant(self.dt)
   
   
    def build_solver(self):
        self.solver = solver.RHTSolver_new(self.mesh, self.v,
                                       T_old=self.T_old, 
                                       phi_old=self.phi_old,
                                       bc_T=self.bc_T, 
                                       bc_phi=self.bc_phi,
                                       t_end=self.t_end,
                                       crho_m=self.crho_m,
                                       kT=self.kT, 
                                       w=self.w,
                                       dt_fe=self.dt_fe,
                                       delta_p=self.delta_p,
                                       dps_dT=self.dps_dT,
                                       ps=self.ps,
                                       Dw=self.Dw,
                                       xi=self.xi,
                                       max_steps=self.max_steps,
                                       time_gamma=self.time_gamma,
                                       max_dt=self.max_dt,
                                       # Propes
                                       prope_fields=["T",  "phi" ],
#                                       prope_fields=["T"],
                                       prope_xs=self.prope_x,
                                       order_2nd=self.order_2nd,
                                       v_materials=self.v_materials,
                                       material_markers=self.material_markers,
                                       out_dir=self.out_dir,
                                       bc_T_in=self.bc_T_in, 
                                       bc_T_out=self.bc_T_out, 
                                       bc_phi_in=self.bc_phi_in, 
                                       bc_phi_out=self.bc_phi_out
                                       )
        return self.solver   
#def read_prope_file_new(path):    
#    with open(path, "r") as csvfile:
#        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#        lines = [line for line in reader]
#
##    x = [float(part) for part in lines[0][1:]]
#
#    times = []
#    vals = []
#    for parts in lines[0:]:
#        its = [float(part) for part in parts]
#        vals.append(its[1:])
#        times.append(its[0])    
#
#    return sp.array(times), sp.array(vals)
#    
#  
#def read_measurement(measurement_data_root, case_ids):
#    
#    measurements = []
#    for case_id in case_ids:
#        measurement_data_path = os.path.join(measurement_data_root, case_id)
#        with open(measurement_data_path, "r") as ifile:
#            measurement = Dict(json.load(ifile))
#            measurement.materials = [Dict(material) for material in measurement.materials]
#        measurements.append(measurement)    
#    return measurements
#
#
#def post_diffusion(rootpath):
#    measurement_data_root = os.path.join(os.path.pardir, "pureDiffusion","dataIn", "jsonCases")
#    case_ids = ["1a"]
##    case_ids = ["1a",
##                "1b","2a","2b","3a","3b","4a","4b","5a","5b","6a","6b",
##                "7a","7b","8a","8b","9b","9Xb","10b","10Xb","11b","11Xb",
##                "12b","12Xb","13b","14b","15b","16b","17b","18b","19b",
##                "20b","21b","22b","23b","24b","25b","26b","27b","28b",
##                "33b","34b","35b","36b","37b","38b","39b","40b"]
#
#    measurements = read_measurement(measurement_data_root, case_ids)
#        
#
#
#    for kcase, case_id in enumerate(case_ids):
#        print(case_id)
#        
#        measurement = measurements[kcase]
#        
#        
#        path = os.path.join(rootpath,case_id)
#        
##        path =  os.path.join(os.path.pardir,case_id)
#                             
#        
#    
#    
#    #    print(measurement.Ls)
#        change_x = sp.array([ 
#                           measurement.Ls[0], 
#                           measurement.Ls[:2].sum(),
#                           measurement.Ls[:3].sum(),
#                 ])   
#        
#        # 0.013   0.0002  0.173   0.025
#    #    change_x2 = [0.013, 
#    #                 0.013+0.0002,
#    #                 0.013  + 0.0002+  0.173]
#    
#        change_x2 = [ 0.0202 , 0.1792]
#        
#        mesh = read_1dmesh(os.path.join(path, "mesh.txt"))
#        times, Ts = read_1dfield(os.path.join(path, "T.txt"))
#        times, phis = read_1dfield(os.path.join(path, "phi.txt"))    
#    
#    
#        # FIELDS
#        fig, axes = plt.subplots(2)
#        
#        # T
#        ax = axes[0]
#        for k,tim in enumerate(times):
#            if tim < times[-1]:
#                continue
#            
#            
#    #        print(Ts[k]-C2K)
#            ax.plot(mesh, Ts[k]-C2K)
#        ax.set_xlabel(r"$x$ $(m)$")
#        ax.set_ylabel(r"$T$ $(^\circ C)$")
#        
#    
#        # phi
#        ax = axes[1]
#        for k,tim in enumerate(times):
#            if tim < times[-1]:
#                continue
#    #        print(phis[k]*100)
#            ax.plot(mesh, phis[k]*100)
#        ax.set_xlabel(r"$x$ $(m)$")
#        ax.set_ylabel(r"$\phi$ $(\%)$")    
#    
#        for ax in axes.ravel():
#            ax.set_xlim(mesh.min(),mesh.max())    
#            ax.grid(True)
#            for x in change_x:
#                ax.axvline(x)
#            for x in change_x2:
#                ax.axvline(x, linestyle="--")
#    
#    
#        fig.tight_layout()
#        fig.savefig(os.path.join(path, "fields.pdf"))    
#        
#        # Propes
#        
#        times_prope, Ts_prope = read_prope_file_new(os.path.join(path,"prope_T.txt"))
#        times_prope, phis_prope = read_prope_file_new(os.path.join(path,"prope_phi.txt"))
#        
#        
#        
#        
#        fig, axes = plt.subplots(2)
#        
#        
#        
#        # T
#        ax = axes[0]
#        
#        
#    #    ax.plot(measurement.wufi_time/s2d, measurement.T_S_wufi-C2K, ":") 
#    #    ax.plot(measurement.wufi_time/s2d, measurement.T_U_wufi-C2K, ":") 
#    
#        ax.plot(measurement.prope_time/s2d, measurement.T_S-C2K, label="S") 
#        ax.plot(measurement.prope_time/s2d, measurement.T_U-C2K, label="U") 
#    
#        
#        ax.plot(times_prope/s2d, Ts_prope[:,0]-C2K, "--", label="S")
#        ax.plot(times_prope/s2d, Ts_prope[:,1]-C2K, "--", label="U")
#        ax.set_xlabel(r"$t$ $(d)$")
#        ax.set_ylabel(r"$T$ $(^\circ C)$")
#    
#        # phi
#        ax = axes[1]
#    #    ax.plot(measurement.wufi_time/s2d, measurement.RH_S_wufi*100, ":") 
#    #    ax.plot(measurement.wufi_time/s2d, measurement.RH_U_wufi*100, ":")
#        ax.plot(measurement.prope_time/s2d, measurement.RH_S*100, label="S") 
#        ax.plot(measurement.prope_time/s2d, measurement.RH_U*100, label="U")
#        
#        ax.plot(times_prope/s2d, phis_prope[:,0]*100, "--", label="S")
#        ax.plot(times_prope/s2d, phis_prope[:,1]*100, "--", label="U")
#        ax.set_xlabel(r"$t$ $(d)$")
#        ax.set_ylabel(r"$\phi$ $(\%)$")    
#    
#        for ax in axes.ravel():
#            ax.grid(True)
#            ax.set_xlim((times_prope/s2d).min(),(times_prope/s2d).max())
#            ax.legend(frameon=False)
#        fig.tight_layout()
#        fig.savefig(os.path.join(path, "propes.pdf"))
#        
#        
#        
#        ##########################################
#        # Boundary
#        ##############################################
#        
#    #    fig, axes = plt.subplots(2)
#    #    # T
#    #    ax = axes[0]
#    #    ax.plot(measurement.measurement_time/s2d, measurement.T_in-C2K, label="in") 
#    #    ax.plot(measurement.measurement_time/s2d, measurement.T_out-C2K, label="out") 
#    #
#    #    
#    #    ax.plot(times/s2d, Ts[:,0]-C2K, "--", label="in")
#    #    ax.plot(times/s2d, Ts[:,-1]-C2K, "--", label="out")
#    #    ax.set_xlabel(r"$t$ $(d)$")
#    #    ax.set_ylabel(r"$T$ $(^\circ C)$")
#    #
#    #    # phi
#    #    ax = axes[1]
#    ##    ax.plot(measurement.wufi_time/s2d, measurement.RH_S_wufi*100) 
#    ##    ax.plot(measurement.wufi_time/s2d, measurement.RH_U_wufi*100)
#    #    ax.plot(measurement.measurement_time/s2d, measurement.RH_in*100, label="in") 
#    #    ax.plot(measurement.measurement_time/s2d, measurement.RH_out*100, label="out")
#    #    
#    #    ax.plot(times/s2d, phis[:,0]*100, "--", label="in")
#    #    ax.plot(times/s2d, phis[:,-1]*100, "--", label="out")
#    #    ax.set_xlabel(r"$t$ $(d)$")
#    #    ax.set_ylabel(r"$\phi$ $(\%)$")    
#    #
#    #    for ax in axes.ravel():
#    #        ax.grid(True)
#    #        ax.set_xlim((times_prope/s2d).min(),(times_prope/s2d).max())
#    #        ax.legend(frameon=False)
#    #    fig.tight_layout()
#    #    fig.savefig(os.path.join(path, "boundary.pdf"))
#    #    
#        
#        # MESH
#    #    fig, ax = plt.subplots(1)
#    #    for x in mesh:
#    #        ax.axvline(x)
#    #    ax.set_ylim(0,mesh.max()/10 )
#    #    ax.set_xlim(mesh.min(), mesh.max())
#    #    fig.tight_layout()
#    #    fig.savefig(os.path.join(path, "mesh.pdf"))
#    #    
#    #    
#    #    fig, ax = plt.subplots(1)
#    #    ax.plot(mesh)
#    ##    ax.set_ylim(0,mesh.max()/10 )
#    ##    ax.set_xlim(mesh.min(), mesh.max())
#    #    fig.tight_layout()
#    #    fig.savefig(os.path.join(path, "mesh2.pdf"))
#    
#
#
#def read_comsol(path):
#    times_h = []
#    vals = []
#    with open(path, "r") as ifile:
#        for line in ifile.readlines()[5:]:
##            print(line)
#            parts = line.split()
#            times_h.append(float(parts[0]))
#            vals.append(float(parts[1]))
#
#    return sp.array(times_h)*60**2, sp.array(vals)
#
#
#def  verify_diffusion():
#    
#    
#    measurement_data_root = os.path.join(os.path.pardir, "pureDiffusion","dataIn", "jsonCases")
#    case_ids = ["1a"]
#    measurements = read_measurement(measurement_data_root, case_ids)
#    measurement = measurements[0]
#        
#
#
#    for kcase, case_id in enumerate(case_ids):
#        print(case_id)
#        
#        measurement = measurements[kcase]
#    
#    
#    path = "/home/ami/projects/2019_moistureTransport/src/diffusionCases/1a"
#    comsol_root = "/media/ami/laja/projects/2019_moistureTransport/pureDiffusion/calculations/temp/1a"
#    # Propes
#    
#    times_prope, Ts_prope = read_prope_file_new("diffusionCases/1a/prope_T.txt")
#    times_prope, phis_prope = read_prope_file_new("diffusionCases/1a/prope_phi.txt")
#    
#
#    # VAARIN PAIN NIMET
#    times_comsol, T_U_comsol = read_comsol(os.path.join(comsol_root, "comsolOut","T_S.txt"))
#    times_comsol, T_S_comsol = read_comsol(os.path.join(comsol_root, "comsolOut","T_U.txt"))
#    
#    times_comsol, phi_U_comsol = read_comsol(os.path.join(comsol_root, "comsolOut","RH_S.txt"))
#    times_comsol, phi_S_comsol = read_comsol(os.path.join(comsol_root, "comsolOut","RH_U.txt"))
#
#    mask = times_comsol < times_prope[-1]
#    times_comsol = times_comsol[mask]
#    T_S_comsol = T_S_comsol[mask]
#    T_U_comsol = T_U_comsol[mask]
#    
#    fig, axes = plt.subplots(2)
#    
#    
#    
#    # T
#    ax = axes[0]
#    
#    
##    ax.plot(measurement.wufi_time/s2d, measurement.T_S_wufi-C2K) 
##    ax.plot(measurement.wufi_time/s2d, measurement.T_U_wufi-C2K) 
#
##    ax.plot(measurement.prope_time/s2d, measurement.T_S-C2K, label="S") 
##    ax.plot(measurement.prope_time/s2d, measurement.T_U-C2K, label="U") 
#
#        
#    ax.plot(times_comsol/s2d, T_S_comsol, "-", label="S")
#    ax.plot(times_comsol/s2d, T_U_comsol, "-", label="U")
#    
#
#
#    ax.plot(times_prope/s2d, Ts_prope[:,0]-C2K, "--", label="S")
#    ax.plot(times_prope/s2d, Ts_prope[:,1]-C2K, "--", label="U")
#    ax.set_xlabel(r"$t$ $(d)$")
#    ax.set_ylabel(r"$T$ $(^\circ C)$")
#
#    # phi
#    ax = axes[1]
##    ax.plot(measurement.wufi_time/s2d, measurement.RH_S_wufi*100) 
##    ax.plot(measurement.wufi_time/s2d, measurement.RH_U_wufi*100)
##    ax.plot(measurement.prope_time/s2d, measurement.RH_S*100, label="S") 
##    ax.plot(measurement.prope_time/s2d, measurement.RH_U*100, label="U")
#    
#    
#    ax.plot(times_comsol/s2d, phi_S_comsol*100, "-", label="S")
#    ax.plot(times_comsol/s2d, phi_U_comsol*100, "-", label="U")
#
#
#    ax.plot(times_prope/s2d, phis_prope[:,0]*100, "--", label="S")
#    ax.plot(times_prope/s2d, phis_prope[:,1]*100, "--", label="U")
#    ax.set_xlabel(r"$t$ $(d)$")
#    ax.set_ylabel(r"$\phi$ $(\%)$")    
#
#    for ax in axes.ravel():
#        ax.grid(True)
#        ax.set_xlim((times_prope/s2d).min(),(times_prope/s2d).max())
#        ax.legend(frameon=False)
#    fig.tight_layout()
#    fig.savefig(os.path.join(path, "propes.pdf"))
    
################################################################################
if __name__ == "__main__":
    start = time.time()
    print("START")
    run_SFS_EN_15026()
    print("END %.4f s" % (time.time()-start))
