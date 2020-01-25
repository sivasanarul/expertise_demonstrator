import numpy as np
import matplotlib.pyplot as plt
import amfe
from pyfeti import utils
from amfe.contact import jenkins, Nonlinear_force_assembler, Create_node2node_force_object
import time
import scipy.sparse as sparse
import scipy
from scipy.optimize import minimize, root
from contpy import optimize as copt, frequency, operators
import numdifftools as nd
import os

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


def list2global(K_list, M_list, f_list, alpha=1.0E-3, beta=1.0E-7):
    
        K_global = sparse.block_diag(K_list)
        M_global = sparse.block_diag(M_list)
        
        C_global = alpha*K_global + beta*M_global
        f_global = np.concatenate(f_list)
        f_global/=np.linalg.norm(f_global)

        return K_global.tocsc(),M_global.tocsc(),C_global.tocsc(),f_global

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


def preprocessing(variables_filename = 'preprocessing_variables.pkl'):

    #-------------------------------------------------------------------------------------------------------------------------
    #                                    PRE-PROCESING
    #-------------------------------------------------------------------------------------------------------------------------

    # READING MESH

    try:
        m = utils.load_object(mesh_file_pkl,tries=1,sleep_delay=1)
        if m is None:
            raise ValueError('No mesh')

    except:
        m = amfe.Mesh()
        m.import_inp(mesh_file_1)
        utils.save_object(m,mesh_file_pkl)

    #-------------------------------------------------------------------------------------------------------------------------
    # setting and single domain
    m.split_in_groups()
    for component_string in component_string_list:
        m.change_tag_in_eldf('phys_group',component_string,tag_domain)

    print(m.get_phys_group_types())
    #-------------------------------------------------------------------------------------------------------------------------
    # Defining contact pairs
    contact_pairs = {}
    for i,j in contact_pairs_tuple: 
        contact_pairs[i,j] = contact_string_func(i,j)

    contact_dict = {}
    for key, (contact, target) in contact_pairs.items():
        # contact pair IJ
        di = m.get_submesh('phys_group',contact)
        dj = m.get_submesh('phys_group',target)

        contactij = amfe.contact.Contact(di,dj,tol_radius=tol_radius)
        print('Contact (%i,%i), : Number of contact pairs = %i' %(key[0],key[1],len(contactij.contact_elem_dict)))
        contact_dict[key] = contactij


    #-------------------------------------------------------------------------------------------------------------------------
    # Creating Nonlinear nodal force based on Jenkins elements
    map_local_domain_dofs_dimension = {1:m.no_of_nodes*dimension}
    elem_properties = {'ro':ro,'N0':N0,'k':k, 'mu':mu}
    Fnl_obj_list = []
    for bodies_contact_id, contact_ij in contact_dict.items():
        create_obj_1 = Create_node2node_force_object(contact_ij,(1,1),elem_type,elem_properties,dimension,map_local_domain_dofs_dimension)
        Fnl_obj_1 = create_obj_1.assemble_nonlinear_force()
        shift_dict = create_obj_1.shift_dict
        Fnl_obj_list.append(Fnl_obj_1)

    

    #-------------------------------------------------------------------------------
    # Creating  mechanical component   
    # 
    from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager
    mesh_dict = {1:{'m':m, 'domain_tag':tag_domain,'external_force_tag':external_force_tag,
                    'external_force_direction':external_force_direction,'force_value':force_multiplier ,'Dirichlet_tag':Dirichlet_tag}}


    return locals()


def main(**kwargs):
    #-------------------------------------------------------------------------------------------------------------------------
    #                                                       HBM   Simulation
    #-------------------------------------------------------------------------------------------------------------------------
    globals().update(kwargs)
    print('Starting HBM simulation')
    #-----------------------------------------------------------------------------------
    #HBM setup
    print('time_points = %i' %time_points)
    rate = force_multiplier 
    ndofs = K_global.shape[0]
    def force_in_time(fext,time_points,omega):
        
        f_list = []
        for i in range(time_points):
            a = rate*np.sin(2.0*np.pi*i/time_points)
            f1 = a*fext
            f_list.append(f1)

        return np.array(f_list).T

    I_base = np.eye(f_global.shape[0])
    I_list = [I_base]
    I_list.extend((nH-1)*[0.*I_base])
    I_aug = np.vstack(I_list)
    force_global_in_time = force_in_time(f_global,time_points,omega)
    force_mode_global_in_time = force_in_time(f_global,time_points,omega)
    time_axis = list(range(time_points))

    #------------------------------------------------------------------------------------
    # HBM operator, AFT and Nonlinear forces

    Q = frequency.assemble_hbm_operator(ndofs,number_of_harm=nH ,n_points=time_points) # bases of truncaded Fourier
    nonlinear_force_obj = Nonlinear_Force(Fnl_obj_list)
    AFT  = operators.Nonlinear_Force_AFT(Q,nonlinear_force_obj)

    force_global_ = Q.H.dot(force_global_in_time)
    force_mode_global_ = Q.H.dot(force_mode_global_in_time)
    force_in_time_calc = Q.dot(force_global_)

    Z = lambda w : frequency.create_Z_matrix(K_global,C_global,M_global,f0= w/(2.0*np.pi),nH=nH, static=False)
    Zw = Z(omega)

    Zw_real = copt.complex_matrix_to_real_block(Zw)
    Zw_real = Zw_real.tocsc() 


    @timing
    def AFT_compute_force_and_jac(u_):
        return AFT.compute_force_and_jac(u_)

    @timing
    def f_real_and_J(x):
        u_ = copt.real_array_to_complex(x)
        
        fnl_complex_eval, Jnl_eval_1, Jnl_eval_conj_1 = AFT_compute_force_and_jac(u_)
        J_block_real = copt.complex_matrix_to_real_block(Jnl_eval_1, Jnl_eval_conj_1)

        J = Zw_real - J_block_real
        R = Zw.dot(u_) - force_global_ - fnl_complex_eval
        R_real = copt.complex_array_to_real(R)
        return R_real, J

    def compute_solution(u__inital_real,maxiter=200, tol=1.0E-6, verbose=True):
    
        sol2 = copt.Newton(f_real_and_J,u__inital_real,jac=True,maxiter=maxiter, tol=tol, verbose=verbose)
        u_sol =   copt.real_array_to_complex(sol2.x)
        str_results = 'test' 
        print('Total dof = %i' %len(sol2.x))
    
        if not sol2.success:
            print('did not converge')
            
        return u_sol

    # compute soluton
    Zw_inv = sparse.linalg.splu(Zw)
    u__inital = Zw_inv.solve(force_global_)
    u__inital_real = copt.complex_array_to_real(u__inital)
    u_sol = compute_solution(u__inital_real,maxiter=newton_max_int,tol=newton_tol,verbose=True)


    # Post-processing
    u_sol_real = copt.complex_array_to_real(u_sol)
    Rcalc, Jcalc = f_real_and_J(copt.complex_array_to_real(u_sol))
    Rnorm_target = np.linalg.norm(Rcalc)
    fnl_sol, Jnl_eval_1, Jnl_eval_conj_1 = AFT_compute_force_and_jac(u_sol)
    u_sol_time = Q.dot(u_sol)
    fnl_sol_time = Q.dot(fnl_sol)
    solution_dict = {'u_sol':u_sol,'fnl_sol':fnl_sol,'u_sol_time':u_sol_time,'fnl_sol_time': fnl_sol_time, 
                    'force_global_in_time':force_global_in_time, 'shift_dict':shift_dict,'dimension':dimension,
                    'ndofs':ndofs,'time_points':time_points}
    utils.save_object(solution_dict,save_solution_obj_file)


if __name__ == "__main__":

    import logging
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    omega0 = 10.0
    delta_omega = 50.0
    mpi_case = comm.rank

    logging.basicConfig(filename='multiple_freq_id_%i.log' %mpi_case,level=logging.INFO)

    # Newton Parameters
    newton_max_int = 4
    newton_tol = 1.0E-7

    # Solution File to be saved
    save_solution_obj_file = 'simple_parametric_bladed_disk_freq_id_%i.pkl' %mpi_case
    #save_mesh_file = 'mesh_simple_parametric_bladed_disk_v1.pkl' 

    # HBM parameters 
    nH = 1 # number of harmonics
    omega = omega0 + mpi_case*delta_omega # frequency in rad s
    time_points = nH*25

    logging.info('HBM simulation for frequency = %f [rad/s]' %omega)
    # Linear Elastic Material Properties
    rho = 7.85E3 # ton/mm
    E = 2.10E11 # MPa = N/m2
    nu = 0.3

    # Jenkins element properties
    dimension= 3 # dimention of the problem
    ro=1.0E4 # Normal Contact stiffness
    N0=0.0E0 # Normal Preload
    k= 1.0E2 # Tangent Contact Stiffness 
    mu= 0.2 # Friction coef 
    elem_type = 'jenkins' # type of the contact implemented in amfe.contact module

    #external force paramenters and Boundary Conditions
    force_multiplier = 1.0E6
    external_force_tag = 'NEUMANN_RIGHT_1_ELSET'
    external_force_direction = 0
    Dirichlet_tag = 'DIRICHLET_ELSET'
    rate = 5.00E2 #

    # Mesh file 
    mesh_file_1 = os.path.join('meshes','simple_parametric_bladed_disk_9_sectors_131976_nodes_83305_elem_tet4.inp')
    mesh_file_pkl = os.path.join('meshes','simple_parametric_bladed_disk_9_sectors_131976_nodes_83305_elem_tet4.pkl')

    # tags for assembling K and M and contact pairs
    tag_domain = 'PART'
    component_string_list = ['PART_2_1_1_SOLID_ELSET',
                            'PART_2_1_2_SOLID_ELSET',
                            'PART_2_1_3_SOLID_ELSET',
                            'PART_2_1_4_SOLID_ELSET',
                            'PART_2_1_5_SOLID_ELSET',
                            'PART_2_1_6_SOLID_ELSET',
                            'PART_2_1_7_SOLID_ELSET',
                            'PART_2_1_8_SOLID_ELSET',
                            'PART_2_1_9_SOLID_ELSET']


    tol_radius = 0.01 # tolerance for the contact pair distance
    contact_string_func = lambda i,j : ('B%i_%i%i_ELSET' %(i,i,j),'B%i_%i%i_ELSET' %(j,j,i)) 

    contact_pairs_tuple = ((1,2),
                        (2,3),
                        (3,4),
                        (4,5),
                        (5,6),
                        (6,7),
                        (7,8),
                        (8,9),
                        (9,1))


    

    #plotting
    plot_geometry = False

    variables_dict = preprocessing()
    
    var_dict = {}
    var_list = ['K_global','M_global','C_global','f_global']
    for var_name in var_list:
        var_dict[var_name] = utils.load_object(var_name + '.pkl')

    variables_dict.update(var_dict)

    main(**variables_dict)
        
