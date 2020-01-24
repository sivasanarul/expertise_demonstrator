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


# Solution File to be saved
save_solution_obj_file = 'simple_parametric_bladed_disk_v8.pkl'
save_mesh_file = 'mesh_simple_parametric_bladed_disk_v1.pkl' 

# HBM parameters 
nH = 1 # number of harmonics
omega = 0.5 # frequency in rad s
time_points = nH*25

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
width = 0.200

#-------------------------------------------------------------------------------------------------------------------------
#                                                       HBM   Simulation
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


if plot_geometry:
    ax1 = amfe.plot3Dmesh(m)
    
    ax1.set_xlim((-1.1*width,1.1*width))
    ax1.set_ylim((-1.1*width,1.1*width))
    ax1.set_zlim((-1.1*width,1.1*width))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.show()


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

#-------------------------------------------------------------------------------
# Creating  mechanical component   
# 
from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager
mesh_dict = {1:{'m':m, 'domain_tag':tag_domain,'external_force_tag':external_force_tag,
                'external_force_direction':external_force_direction,'force_value':force_multiplier ,'Dirichlet_tag':Dirichlet_tag}}

# save mesh_dict obj
utils.save_object(mesh_dict,save_mesh_file)

my_material = amfe.KirchhoffMaterial(E=E, nu=nu, rho=rho, plane_stress=False)
K_list = []
M_list = []
f_list = []
for domain_id,param_dict in mesh_dict.items():
    
    locals().update(param_dict)

    my_comp = amfe.MechanicalSystem()
    my_comp.set_mesh_obj(m)
    my_comp.set_domain(domain_tag,my_material)
    

    if external_force_direction == 0:
        direction = np.array([1.,0.,0.])
    elif external_force_direction == 1:
        direction = np.array([0.,1.,0.])
    else:
        direction = np.array([0.,0.,1.])

    my_comp.apply_neumann_boundaries(external_force_tag,force_value,direction)
    print('Number of nodes is equal to %i' %my_comp.mesh_class.no_of_nodes)

    K, f_ = my_comp.assembly_class.assemble_k_and_f()
    _, fext = my_comp.assembly_class.assemble_k_and_f_neumann()
    M = my_comp.assembly_class.assemble_m()

    
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
    
def list2global(K_list, M_list, f_list, alpha=1.0E-3, beta=1.0E-7):
   
    K_global = sparse.block_diag(K_list)
    M_global = sparse.block_diag(M_list)
    
    C_global = alpha*K_global + beta*M_global
    f_global = np.concatenate(f_list)
    f_global/=np.linalg.norm(f_global)

    return K_global.tocsc(),M_global.tocsc(),C_global.tocsc(),f_global

K_global,M_global,C_global,f_global =  list2global(K_list, M_list, f_list,alpha = 1.0E-2, beta = 1.0E-4)

# solving Eigenvalue problem for problem change of basis
K_global_inv = sparse.linalg.splu(K_global)
D = sparse.linalg.LinearOperator(shape=K_global.shape, matvec = lambda x : K_global_inv.solve(M_global.dot(x)))

#saving case variables
utils.save_object(globals(),'case_variables.pkl')







'''
#mesh_list = [m]
ax2.set_xlim((-1.1*width,1.1*width))
ax2.set_ylim((-1.1*width,1.1*width))
ax2.set_zlim((-1.1*width,1.1*width))
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.show()


'''

'''
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


force_ref = np.empty((ndofs,time_points))
fnl = lambda U : _fnl(U)[0]


Fnl_ = lambda u_ : Q.H.dot(fnl(Q.dot(u_)))
R_ = lambda u_ : Zw.dot(u_) - force_global_ - Fnl_(u_) 
Zw_inv = sparse.linalg.splu(Zw)
u__inital = Zw_inv.solve(force_global_)

u__inital_real = 0.0*copt.complex_array_to_real(u__inital)
func_real = lambda x : copt.func_wrapper(R_,x)
JR = nd.Jacobian(Fnl_)
Fnl_conj = lambda u : Fnl_(u).conj()
JR_conj = nd.Jacobian(Fnl_conj)
Fnl_real = lambda x : copt.func_wrapper(Fnl_,x)
JR_real = nd.Jacobian(Fnl_real)
Fl = lambda u : Zw.dot(u)
Fl_real = lambda x : copt.func_wrapper(Fl,x)
JFl_real = nd.Jacobian(Fl_real)


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



@timing
def compute_solution(u__inital_real):
    

    sol2 = copt.LevenbergMarquardt(f_real_and_J,u__inital_real,method=None,jac=True,maxiter=200)
    u_sol =   copt.real_array_to_complex(sol2.x)
    str_results = 'test' 
    print('Total dof = %i' %len(sol2.x))
   
    if not sol2.success:
        print('did not converge')
        
    return u_sol


# compute soluton
Zw_inv = sparse.linalg.splu(Zw)
u__inital = Zw_inv.solve(force_global_)
u__inital_real = 0.0*copt.complex_array_to_real(u__inital)
u_sol = compute_solution(u__inital_real)


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
'''