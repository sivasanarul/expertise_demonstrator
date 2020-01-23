

"""
DMUMPS test routine.

Run as:

    mpirun -np 2 python dsimpletest.py

The solution should be [ 1. 2. 3. 4. 5.].
"""
import logging
import sys, os
import numpy as np
import mumps

import matplotlib.pyplot as plt
import amfe
from pyfeti import utils
from amfe.contact import jenkins, Nonlinear_force_assembler, Create_node2node_force_object
import time
import scipy.sparse as sparse
import scipy
import scipy.sparse as sp
from scipy.optimize import minimize, root
from contpy import optimize as copt, frequency
import numdifftools as nd
from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        logging.info('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

def components2list(component_dict,dimension=3):
    K_list = []
    M_list = []
    f_list = []
    for domain_id, param_dict in component_dict.items():

        globals().update(param_dict)
        #print(param_dict)
        #print(locals())
        m = mesh 
        my_comp = amfe.MechanicalSystem()
        my_comp.set_mesh_obj(m)
        my_comp.set_domain(domain_tag,material)

        if dimension==3:

            if external_force_direction == 0:
                direction = np.array([1.,0.,0.])
            elif external_force_direction == 1:
                direction = np.array([0.,1.,0.])
            else:
                direction = np.array([0.,0.,1.])
                
        elif dimension==2:
            if external_force_direction == 0:
                direction = np.array([1.,0.])
            elif external_force_direction == 1:
                direction = np.array([0.,1.])
            

        my_comp.apply_neumann_boundaries(external_force_tag,force_value,direction)
        print('Number of nodes is equal to %i' %my_comp.mesh_class.no_of_nodes)

        K, f_ = my_comp.assembly_class.assemble_k_and_f()
        _, fext = my_comp.assembly_class.assemble_k_and_f_neumann()
        #M = my_comp.assembly_class.assemble_m()
        M = 0.0*K

        try:
            connectivity = []
            for _,item in m.el_df.iloc[:, m.node_idx:].iterrows():
                connectivity.append(list(item.dropna().astype(dtype='int64')))
            m.el_df['connectivity'] = connectivity
        except:
            pass

        id_matrix = my_comp.assembly_class.id_matrix
        id_map_df = dict2dfmap(id_matrix)
        s = create_selection_operator(id_map_df,m.el_df)

        from pyfeti.src.linalg import Matrix
        K1 = Matrix(K,key_dict=s.selection_dict)
        M1 = Matrix(M,key_dict=s.selection_dict)


        # applying Dirichlet B.C.
        K1.eliminate_by_identity(Dirichlet_tag,1.0E15)
        M1.eliminate_by_identity(Dirichlet_tag,0.0)

        K_list.append(K1.data)
        M_list.append(M1.data)
        f_list.append(fext)
        
    return K_list, M_list, f_list
    
def list2global(K_list, M_list, f_list, alpha=1.0E-3, beta=1.0E-7):
   
    K_global = sparse.block_diag(K_list)
    M_global = sparse.block_diag(M_list)
    
    C_global = alpha*K_global + beta*M_global
    f_global = np.concatenate(f_list)
    #f_global/=np.linalg.norm(f_global)

    return K_global.tocsc(),M_global.tocsc(),C_global.tocsc(),f_global


def create_map_local_domain_dofs_dimension(component_dict,dimension=3):
    map_local_domain_dofs_dimension = {}
    for domain_id, param_dict in component_dict.items():
        m_ = param_dict['mesh']
        map_local_domain_dofs_dimension[domain_id] = m_.no_of_nodes*dimension
    return map_local_domain_dofs_dimension

class SplitOperator():
    def __init__(self,map_local_domain_dofs_dimension):
        self.map_local_domain_dofs_dimension = map_local_domain_dofs_dimension
        
    def LinearOperator(self,u):
        u_list = []
        idx = 0
        for key, item in self.map_local_domain_dofs_dimension.items():
            try:
                u_list.append(u[idx:idx+item])
            except:
                u_list.append(u[idx:])
            idx = item
        return u_list
        
class Nonlinear_Force():
    def __init__(self,Fnl_obj_list):
        self.Fnl_obj_list = Fnl_obj_list
        
    def compute_force(self,u,X0=None):
        Fnl_obj_list = self.Fnl_obj_list
        for fnl_obj_item in Fnl_obj_list:
            try:
                output += fnl_obj_item.compute(u,X0)
            except:
                output = fnl_obj_item.compute(u,X0)

        return output
        
    def compute_jacobian(self,u,X0=None):
        Fnl_obj_list = self.Fnl_obj_list
        for fnl_obj_item in Fnl_obj_list:
            try:
                output += fnl_obj_item.Jacobian(u,X0)
            except:
                output = fnl_obj_item.Jacobian(u,X0)

        return output
    
    
def force_in_time(fext,time_points,omega):
    
    f_list = []
    for i in range(time_points):
        a = rate*np.sin(2.0*np.pi*i/time_points)
        f1 = a*fext
        f_list.append(f1)

    return np.array(f_list).T


def main(x_divisions=100,y_divisions=20):
    # geometric properties
    dimension= 2
    width= 5.0
    heigh = 1.0
    init_gap_x = width - 0.2*width
    init_gap_y = -heigh - 0.01*heigh


    # Create the MUMPS context and set the array and right hand side
    ctx = mumps.DMumpsContext(sym=0, par=1)

    if ctx.myid == 0:
        #-------------------------------------------------------------------------------------------------------------------------
        # Creating mesh for 2 bodies
        d1= utils.DomainCreator(width=width, heigh=heigh, 
                                x_divisions=x_divisions, y_divisions=y_divisions, 
                                domain_id=1, start_x=0.0, start_y=0.0)

        mesh_file_1 = 'domain_1.msh'
        d1.save_gmsh_file(mesh_file_1)
        m = amfe.Mesh()
        try:
            m.import_msh(mesh_file_1)
        except:
            logging.error('Could not read the mesh.')
            exit()

        m1 = m

        rho = 7.85E-9 # ton/mm
        E = 2.10E5 # MPa = N/mm2
        my_material_template = amfe.KirchhoffMaterial(E=E, nu=0.3, rho=rho, plane_stress=False)

        component_dict = {1:{'mesh' : m1, 
                            'domain_tag':3,
                            'external_force_tag':2,
                            'external_force_direction':1,
                            'force_value':1.0,
                            'Dirichlet_tag':1,
                            'material' : my_material_template}
                        }

        K_list, M_list, f_list =components2list(component_dict,dimension=2)

        K_global,M_global,C_global,f_global =  list2global(K_list, M_list, f_list,alpha = 1.0E-2, beta = 1.0E-4)



        # Set up the test problem:
        n = K_global.shape[0]
        K_global = K_global.tocoo()
        irn = K_global.row + 1
        jcn = K_global.col + 1
        a = K_global.data
        b = f_global

        logging.info('Total dofs = %i' %(f_global.shape[0]))

        irn.astype('i')
        jcn.astype('i')
        a.astype('d') 
        b.astype('d')


    
    if ctx.myid == 0:
        ctx.set_shape(n)
        ctx.set_centralized_assembled(irn, jcn, a)
        x = b.copy()
        ctx.set_rhs(x)

    ctx.set_silent() # Turn off verbose output

    @timing
    def call_solver():
        ctx.run(job=6) # Analysis + Factorization + Solve

    call_solver()

    if ctx.myid == 0:
        logging.info("Solution x_max = %f, x_min = %f." %(x.max(),x.min(),))

    ctx.destroy() # Free memory

if __name__ == "__main__":

    
    parameter_list = sys.argv
    
    try:
        for param in parameter_list[1:]:
            print(param)

        x_divisions = int(parameter_list [1])
        y_divisions = int(parameter_list [2])
        nproc = int(parameter_list [3])
    except:
        x_divisions =  500
        y_divisions = 100
        nproc = 1

    logging.basicConfig(filename='log_x%i_y%i_np%i.log' %(x_divisions,y_divisions,nproc), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Testing x=%i, y=%i, processor=%i' %(x_divisions,y_divisions,nproc))
    main(x_divisions,y_divisions)

