from __future__ import division, print_function

"""
First attempt on writing a HAM solver.

Runs on Python 2.7

Spring 2019

@author: Antti Mikkonen, a.mikkonen@iki.fi
      
"""

import fenics as fe
import scipy as sp
import time
import os
import csv
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from global_values import * 
import shutil

#CRITICAL  = 50, // errors that may lead to data corruption and suchlike
#ERROR     = 40, // things that go boom
#WARNING   = 30, // things that may go boom later
#INFO      = 20, // information of general interest
#PROGRESS  = 16, // what's happening (broadly)
#TRACE     = 13, // what's happening (in detail)
#DBG       = 10  // sundry
fe.set_log_level(30)    
#fe.set_log_level(20)    
    
class RHTSolver(object):
    def __init__(self, mesh, V, T_old, bc_T, bc_phi, t_end, phi_old, crho_m, kT, w, dt_fe, delta_p, 
                 dps_dT, ps, Dw, xi, max_steps, time_gamma, max_dt,
                 prope_fields,prope_xs,prope_times, order_2nd, v_materials,
                 material_markers, out_dir):
        self.max_steps = max_steps
        self.t_end = t_end
        self.bc_T = bc_T
        self.bc_phi = bc_phi
        self.order_2nd = order_2nd
        
        self.T_old = T_old
        self.phi_old = phi_old

        self.w = w
        self.Dw = Dw
        self.kT = kT
        self.delta_p = delta_p
        self.xi = xi
        self.time_gamma = time_gamma        
        
        self.dt_fe = dt_fe        
        
        self.v_materials = v_materials
        
        self.material_markers = material_markers
        
        self.mesh = mesh
        self.out_dir = out_dir  #os.path.join("out_data")
    
        self.clean_out_data()
        self.write_mesh(os.path.join(self.out_dir, "mesh.txt"))
    
        self.max_dt = max_dt
    
    
        self.prope_times=prope_times
        self.prope_times_k = 0
    
        self.prope_fields = prope_fields
        self.prope_xs = prope_xs
    
    
        ##########################################################################
        # IMPLICIT FUNCTIONS
        ##########################################################################
        
        # HEAT SOLVER
        T = fe.TrialFunction(V)
        v_T = fe.TestFunction(V)

        # Moisture
        phi   = fe.TrialFunction(V)
        v_phi = fe.TestFunction(V)

        ###########################################################################
        # EXPLICIT FUNCTIONS
        ###########################################################################

        # Short names
        alpha_t = kT / (crho_m + c_w*w)
        beta_t  = fe.Constant(h_e) / (crho_m + c_w*w)

        ###########################################################################
        # GOVERNING EQUATIONS
        ###########################################################################
    
        if not order_2nd:
    
            # Heat Transfer, 1st order
            self.a_T = (T*v_T 
                   + self.dt_fe*alpha_t * fe.dot( fe.grad(T), fe.grad(v_T) )
                   #- dt_fe*beta_t*delta_p*phi_old*dp_s_dT * fe.dot( fe.grad(T), fe.grad(v_T) )# Using T
                   )*fe.dx
            self.L_T = (T_old*v_T
                   + self.dt_fe*beta_t*delta_p  #TODO! tarkasta merkki!
        #             * (fe.dot( fe.grad(p_v), fe.grad(v_T)))  # Sung p_v
                     * (phi_old*dps_dT * fe.dot( fe.grad(T_old), fe.grad(v_T) ) # Using T_old
                        + ps * fe.dot( fe.grad(phi_old), fe.grad(v_T) ))
                   )*fe.dx     
            T = fe.Function(V)
            self.T = T
            self.T_old = T_old
        
            # Moisture
            self.a_phi = (xi*phi*v_phi 
                   + self.dt_fe*Dw*xi * fe.dot( fe.grad(phi),   fe.grad(v_phi) )
                   + self.dt_fe*delta_p
                       *(phi*dps_dT * fe.dot( fe.grad(T), fe.grad(v_phi) ) 
                          + ps * fe.dot( fe.grad(phi), fe.grad(v_phi) ) 
                         )
                   )*fe.dx
            self.L_phi = (xi*phi_old*v_phi)*fe.dx 
            phi   = fe.Function(V)    
            self.phi = phi
            self.phi_old = phi_old
    
        else:
            T_old2 = fe.Function(V)
            T_old2.assign(T_old)
            self.T_old2 = T_old2
        
            # Heat Transfer, 2nd order
            self.a_T = (3*T*v_T 
                   + 2*self.dt_fe*alpha_t * fe.dot( fe.grad(T), fe.grad(v_T) )
                   #- dt_fe*beta_t*delta_p*phi_old*dp_s_dT * fe.dot( fe.grad(T), fe.grad(v_T) )# Using T
                   )*fe.dx
            self.L_T = (4*T_old*v_T - T_old2*v_T
                   + 2*self.dt_fe*beta_t*delta_p  #TODO! tarkasta merkki!
        #             * (fe.dot( fe.grad(p_v), fe.grad(v_T)))  # Sung p_v
                     * (phi_old*dps_dT * fe.dot( fe.grad(T_old), fe.grad(v_T) ) # Using T_old
                        + ps * fe.dot( fe.grad(phi_old), fe.grad(v_T) ))
                   )*fe.dx     
            T = fe.Function(V)
            self.T = T
    
    
            phi_old2 = fe.Function(V)
            phi_old2.assign(phi_old)
            self.phi_old2 = phi_old2
    
            # Moisture
            self.a_phi = (3*xi*phi*v_phi 
                   + 2*self.dt_fe*Dw*xi * fe.dot( fe.grad(phi),   fe.grad(v_phi) )
                   + 2*self.dt_fe*delta_p
                       *(phi*dps_dT * fe.dot( fe.grad(T), fe.grad(v_phi) ) 
                          + ps * fe.dot( fe.grad(phi), fe.grad(v_phi) ) 
                         )
                   )*fe.dx
            self.L_phi = (4*xi*phi_old*v_phi - xi*phi_old2*v_phi)*fe.dx 
    #        L_phi = (4*xi*phi_old*v_phi - xi_old*phi_old2*v_phi)*fe.dx 
            phi   = fe.Function(V)
            self.phi = phi

    def solve(self):
        ###########################################################################
        # SOLVER LOOP
        ###########################################################################

        step = 0
        self.t = 0
        ts = [self.t]
        
        
        
        prev_time_for_prope_time = None
        
        while step < self.max_steps:
            
            # Check last step
            if self.t + float(self.dt_fe) > self.t_end:
                self.dt_fe = fe.Constant(self.t_end-self.t)
            # Check prope time
            elif self.t + float(self.dt_fe) > self.prope_times[self.prope_times_k]:
                prev_time_for_prope_time = float(self.dt_fe)
                self.dt_fe = fe.Constant(self.prope_times[self.prope_times_k]-self.t)
                self.prope_times_k += 1
            
            # Progress time    
            step += 1
            self.t += float(self.dt_fe)
            ts.append(self.t)
            
            
            ##############################
            # Solver
            ##############################
        
            # Solve heat and moisture
            fe.solve(self.a_T == self.L_T, self.T, self.bc_T) 
            fe.solve(self.a_phi == self.L_phi, self.phi, self.bc_phi) 
            
            # Assing
            if self.order_2nd:
                self.T_old2.assign(self.T_old)
                self.phi_old2.assign(self.phi_old)
            
            # Update solved fields
            self.T_old.assign(self.T)
            self.phi_old.assign(self.phi)
            
            # Update material properties
            phi_old_int = fe.interpolate(self.phi_old, self.v_materials)
            w_int       = fe.interpolate(self.w, self.v_materials)
            
            self.w.x_k  = phi_old_int
            self.kT.x_k      = w_int
            self.delta_p.x_k = w_int
            self.Dw.x_k      = w_int
            self.xi.x_k      = phi_old_int
        
            ##############################
            # Post
            ##############################
            if sp.isclose(self.t, self.prope_times[self.prope_times_k]):
                self.prope()
                print("step=%i progress=%.2f t=%.2f (d) dt=%.2f (h)" % (step, self.t/self.t_end, self.t/s2d, float(self.dt_fe)/s2h) )
                self.save_time()
            
            elif step % 100 == 0:
                print("step=%i progress=%.2f t=%.2f (d) dt=%.2f (h)" % (step, self.t/self.t_end, self.t/s2d, float(self.dt_fe)/s2h) )
                self.save_time()
            
            
            
            
            ##############################
            # Next step
            ##############################
            
            # Check end
            if sp.isclose(self.t, self.t_end) or self.t > self.t_end:
                break
    
            # Increase timestep
            
            
            
            if prev_time_for_prope_time:
                self.dt_fe = fe.Constant(prev_time_for_prope_time)    
                prev_time_for_prope_time = None
            
            
            self.dt_fe *= self.time_gamma
            if float(self.dt_fe) > self.max_dt:
                self.dt_fe = fe.Constant(self.max_dt)
        
        
    def solve_new(self):
        ###########################################################################
        # SOLVER LOOP
        ###########################################################################

        step = 0
        self.t = 0
        ts = [self.t]
        
        
        
        prev_time_for_prope_time = None
        
        while step < self.max_steps:
            
            # Check last step
            if self.t + float(self.dt_fe) > self.t_end:
                self.dt_fe = fe.Constant(self.t_end-self.t)
            # Check prope time
            elif self.t + float(self.dt_fe) > self.prope_times[self.prope_times_k]:
                prev_time_for_prope_time = float(self.dt_fe)
                self.dt_fe = fe.Constant(self.prope_times[self.prope_times_k]-self.t)
                self.prope_times_k += 1
            
            # Progress time    
            step += 1
            self.t += float(self.dt_fe)
            ts.append(self.t)
            
            
            ##############################
            # Solver
            ##############################
        
            # Solve heat and moisture
            fe.solve(self.a_T == self.L_T, self.T, self.bc_T) 
            fe.solve(self.a_phi == self.L_phi, self.phi, self.bc_phi) 
            
            # Assing
            if self.order_2nd:
                self.T_old2.assign(self.T_old)
                self.phi_old2.assign(self.phi_old)
            
            # Update solved fields
            self.T_old.assign(self.T)
            self.phi_old.assign(self.phi)
            
            # Update material properties
            phi_old_int = fe.interpolate(self.phi_old, self.v_materials)
#            w_int       = fe.interpolate(self.w, self.v_materials)
            
            self.w.x_k       = phi_old_int
            self.kT.x_k      = phi_old_int
            self.delta_p.x_k = phi_old_int
            self.Dw.x_k      = phi_old_int
            self.xi.x_k      = phi_old_int
        
            ##############################
            # Post
            ##############################
            if sp.isclose(self.t, self.prope_times[self.prope_times_k]):
                self.prope()
                print("step=%i progress=%.2f t=%.2f (d) dt=%.2f (h)" % (step, self.t/self.t_end, self.t/s2d, float(self.dt_fe)/s2h) )
                self.save_time()
            
            elif step % 100 == 0:
                print("step=%i progress=%.2f t=%.2f (d) dt=%.2f (h)" % (step, self.t/self.t_end, self.t/s2d, float(self.dt_fe)/s2h) )
                self.save_time()
            
            
            
            
            ##############################
            # Next step
            ##############################
            
            # Check end
            if sp.isclose(self.t, self.t_end) or self.t > self.t_end:
                break
    
            # Increase timestep
            
            
            
            if prev_time_for_prope_time:
                self.dt_fe = fe.Constant(prev_time_for_prope_time)    
                prev_time_for_prope_time = None
            
            
            self.dt_fe *= self.time_gamma
            if float(self.dt_fe) > self.max_dt:
                self.dt_fe = fe.Constant(self.max_dt)    
            
    
    def prope(self):
        
        # HACK!
        centers = (self.mesh.coordinates()[:-1] + self.mesh.coordinates()[1:])/2
        
        
#        idx = []
#        for vertex in vertices(mesh):
#            if vertex_func[vertex]:
#                idx.append(vertex.index())
#        print('idx: ', idx)
#        
#        ss_coord_array = mesh.coordinates()[idx]
#        print('coord_array: ', ss_coord_array)
        
        
        
#        print(len(self.material_markers.array()))
#        print(len(self.mesh.coordinates()[:]))
#        
#        print(len(centers))
#        print(self.mesh.num_cells())
#       
        
        
        for field_name, xs in zip(self.prope_fields, self.prope_xs):
            # Chose field
            if field_name == "T":
                field = self.T
            elif field_name == "w":
                field = self.w
            else:
                raise ValueError("Unkown field name", field_name)

            prope_vals = []
            for x in xs:
                if field_name in ["T"]:
                    prope_vals.append(field(x))    
                elif field_name in ["w"]:
                    phi = self.phi_old(x)
                    # HACK! TODO! INDEX
                    nearest = sp.absolute(centers - x).argmin()
                    marker = self.material_markers.array()[nearest]
                    prope_vals.append(field(x))
#                    prope_vals.append(field.interpolate_linear(marker,phi))
                else:
                    raise ValueError("Unkown field name", field_name)
    
                
#            print(self.t, prope_vals)
            with open(os.path.join(self.out_dir, "prope_"+field_name+".txt"), "a") as ofile:
                writer = csv.writer(ofile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                if self.prope_times_k == 0:
                    writer.writerow(["(s,m)"]+xs)    

                writer.writerow([self.t]+prope_vals)
            

    
    def write_mesh(self, path):
        
        with open(path, "w") as ofile:
            writer = csv.writer(ofile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(self.mesh.coordinates()[:])
    
    
    def write_field(self, field, tim, path):
        
        #vals = field.vector().get_local()[self.mesh.get_vertex_values()]
        vals = [field(x) for x in self.mesh.coordinates()[:]]
        
        
        with open(path, "a") as ofile:
            writer = csv.writer(ofile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([tim]+list(vals))
        
    def clean_out_data(self):
        shutil.rmtree(self.out_dir)
        os.mkdir(self.out_dir)
        
    
    
    def save_time(self):
        self.write_field(self.T, self.t, os.path.join(self.out_dir, "T.txt"))
        self.write_field(self.phi, self.t, os.path.join(self.out_dir, "phi.txt"))



