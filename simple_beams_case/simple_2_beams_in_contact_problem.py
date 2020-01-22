
import numpy as np
import matplotlib.pyplot as plt
import amfe
from pyfeti import utils
from amfe.contact import jenkins, Nonlinear_force_assembler, Create_node2node_force_object
import time
import scipy.sparse as sparse
import scipy
#import sparse as sp 
from scipy.optimize import minimize, root
from contpy import optimize as copt, frequency
import numdifftools as nd
from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager, save_object, load_object
from contpy import optimize as copt, frequency, operators

#------------------------------------------------------------------------------------------
#            Function declarations
#------------------------------------------------------------------------------------------


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

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
        
    return K_list, M_list, f_list
    
def list2global(K_list, M_list, f_list, alpha=1.0E-3, beta=1.0E-7):
   
    K_global = sparse.block_diag(K_list)
    M_global = sparse.block_diag(M_list)
    
    C_global = alpha*K_global + beta*M_global
    f_global = np.concatenate(f_list)
    f_global/=np.linalg.norm(f_global)

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
        a = force_amplitude*np.sin(2.0*np.pi*i/time_points)
        f1 = a*fext
        f_list.append(f1)

    return np.array(f_list).T


#------------------------------------------------------------------------------------------
#            Case setup
#------------------------------------------------------------------------------------------

# geometric properties
dimension= 2
width= 5.0
heigh = 1.0
init_gap_x = width - 0.2*width
init_gap_y = -heigh - 0.01*heigh

#mesh properties
x_divisions,y_divisions= 21,7

# Tags and setup
domain_tag = 3
external_force_tag = 2
external_force_direction = 1
Dirichlet_tag_body_1 = 1
Dirichlet_tag_body_2 = 2

# Linear Elastic Properties
rho = 7.85E-9 # ton/mm
E = 2.10E5 # MPa = N/mm2
nu = 0.3 # Poisson ratio
my_material_template = amfe.KirchhoffMaterial(E=E, nu=nu, rho=rho, plane_stress=False)

# Damping paramters
alpha = 1.0E-2
beta = 1.0E-4


# contact properties
tol_radius = 5.0E-1
contact_12_tag = 4
contact_21_tag = 5

# Jenkins elem paramenters
ro= 1.0E5 # Normal Contact stiffness
N0=0.0E0 # Normal Preload
k= 0*1.0E5 # Tangent Contact Stiffness 
mu= 0.2 # Friction coef 

# HBM parameters
nH = 3
omega = 1.0
time_points = nH*25
force_amplitude = 1.0E2			# force amplitude



# plot geoemtry
plot_geometry = False

#-------------------------------------------------------------------------------------------------------------------------
#     HBM Simulation
#-------------------------------------------------------------------------------------------------------------------------
# Creating mesh for 2 bodies
d1= utils.DomainCreator(width=width, heigh=heigh, 
                         x_divisions=x_divisions, y_divisions=y_divisions, 
                         domain_id=1, start_x=0.0, start_y=0.0)

mesh_file_1 = 'domain_1.msh'
d1.save_gmsh_file(mesh_file_1)
m = amfe.Mesh()
m.import_msh(mesh_file_1)


m1 = m.translation(np.array([0., 0.]))
m2 = m.translation(np.array([init_gap_x,init_gap_y]))


if plot_geometry:
    ax1 = amfe.plot2Dmesh(m1)
    amfe.plot2Dmesh(m2,ax=ax1)
    ax1.set_xlim([-0.5 ,2.2*width])
    ax1.set_ylim([-1.1*width,1.1*width])
    plt.show()


# Contact setup
m1.split_in_groups()
m2.split_in_groups()

# contact pair 12
d1 = m1.get_submesh('phys_group',contact_12_tag)
d2 = m2.get_submesh('phys_group',contact_21_tag)

contact12 = amfe.contact.Contact(d1,d2,tol_radius=tol_radius)
print('Number of contact pairs = %i' %len(contact12.contact_elem_dict))


contact_dict = {'12' : {'contact' : contact12, 
                        'contact_pair_id' : (1,2),
                        'elem_type' : 'jenkins' , 
                        'elem_properties' : {'ro':ro,'N0':N0,'k':k, 'mu':mu}}}

component_dict = {1:{'mesh' : m1, 
                     'domain_tag':domain_tag,
                     'external_force_tag':external_force_tag,
                     'external_force_direction':external_force_direction,
                     'force_value':1.0,
                     'Dirichlet_tag':Dirichlet_tag_body_1,
                     'material' : my_material_template},
                   2:{'mesh': m2, 
                      'domain_tag':domain_tag,
                      'external_force_tag':external_force_tag,
                      'external_force_direction':external_force_direction,
                      'force_value':0,
                       'Dirichlet_tag':Dirichlet_tag_body_2,
                       'material' : my_material_template}}

map_local_domain_dofs_dimension = create_map_local_domain_dofs_dimension(component_dict,dimension=dimension)
SO = SplitOperator(map_local_domain_dofs_dimension)

# Assembling linear properties
K_list, M_list, f_list =components2list(component_dict,dimension=2)
K_global,M_global,C_global,f_global =  list2global(K_list, M_list, f_list,alpha = alpha, beta = beta)


# Assembling nonlinear properties
Fnl_obj_list = []
for contact_key,contact_item in contact_dict.items():
    bodies_contact_id = contact_item['contact_pair_id']
    contact_12 = contact_item['contact']
    elem_type = contact_item['elem_type']
    elem_properties = contact_item['elem_properties']
    create_obj_1 = Create_node2node_force_object(contact12,bodies_contact_id,elem_type,elem_properties,dimension,map_local_domain_dofs_dimension)
    Fnl_obj_list.append(create_obj_1.assemble_nonlinear_force())


ndofs = K_global.shape[0]
Q = frequency.assemble_hbm_operator(ndofs,number_of_harm=nH ,n_points=time_points) # bases of truncaded Fourier
nonlinear_force_obj = Nonlinear_Force(Fnl_obj_list)
AFT  = operators.Nonlinear_Force_AFT(Q,nonlinear_force_obj)
Z = lambda w : frequency.create_Z_matrix(K_global,C_global,M_global,f0= w/(2.0*np.pi),nH=nH, static=False)
Zw = Z(omega)
Zw_real = copt.complex_matrix_to_real_block(Zw)
force_global_in_time = force_in_time(f_global,time_points,omega)
force_global_ = Q.H.dot(force_global_in_time)
#-------------------------------------------------------------------------------------------------

def Residual_and_Jac_in_real_block(u_real):
    u_ = copt.real_array_to_complex(u_real)
    fnl_complex_eval, Jnl_eval_1, Jnl_eval_conj_1 = AFT.compute_force_and_jac(u_)
    J_block_real = copt.complex_matrix_to_real_block(Jnl_eval_1, Jnl_eval_conj_1)
    J = Zw_real - J_block_real
    R = Zw.dot(u_) - force_global_ - fnl_complex_eval
    R_real = copt.complex_array_to_real(R)
    return R_real, J



Zw_inv = sparse.linalg.splu(Zw)
u__initial = Zw_inv.solve(force_global_)
u__inital_real = copt.complex_array_to_real(u__initial)
sol2 = copt.LevenbergMarquardt(Residual_and_Jac_in_real_block,0*u__inital_real,jac=True,maxiter=100)
#sol3 = copt.Newton(Residual_and_Jac_in_real_block,sol2.x,jac=True,maxiter=10)
u_sol =   copt.real_array_to_complex(sol2.x)
u_sol_time = Q.dot(u_sol)


mesh_list = [m1,m2]

save_object(mesh_list,'mesh_list.pkl')
save_object(u_sol_time,'u_sol_time.pkl')
save_object(component_dict,'component_dict.pkl')
save_object(width,'width.pkl')
save_object(SO,'SO.pkl')
save_object(time_points,'time_points.pkl')

