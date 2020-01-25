# coding: utf-8

# # My HBM code
#
# $Ax=b$

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
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


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))

        return ret

    return wrap
#

# geometric properties
dimension = 2
width = 5.0
heigh = 1.0
init_gap_x = width - 0.2 * width
init_gap_y = -heigh

nH = 1
omega = 1.0
time_points = nH * 100
rate = 5.0E2

ro = 0* 1.0E3
N0 = 0* 1.0E5
k = 0* 1.0E4
mu = 0.3

# mesh properties
# x_divisions,y_divisions= 51,21
x_divisions, y_divisions = 21, 6

# -------------------------------------------------------------------------------------------------------------------------
# Creating mesh for 2 bodies
d1 = utils.DomainCreator(width=width, heigh=heigh,
                         x_divisions=x_divisions, y_divisions=y_divisions,
                         domain_id=1, start_x=0.0, start_y=0.0)

mesh_file_1 = 'domain_1.msh'
d1.save_gmsh_file(mesh_file_1)
m = amfe.Mesh()
m.import_msh(mesh_file_1)

m1 = m.translation(np.array([0., 0.]))
m2 = m.translation(np.array([init_gap_x, init_gap_y]))

#plot geometry
'''
ax1 = amfe.plot2Dmesh(m1)
amfe.plot2Dmesh(m2,ax=ax1)
ax1.set_xlim([-0.5 ,2.2*width])
ax1.set_ylim([-1.1*width,1.1*width])
#ax1.legend('off')
plt.show()
'''


# Defining contact pairs
m1.split_in_groups()
m2.split_in_groups()

tol_radius = 5.0E-5
contact_12_tag = 4
contact_21_tag = 5

# contact pair 12
d1 = m1.get_submesh('phys_group', contact_12_tag)
d2 = m2.get_submesh('phys_group', contact_21_tag)

contact12 = amfe.contact.Contact(d1, d2, tol_radius=tol_radius)
print('Number of contact pairs = %i' % len(contact12.contact_elem_dict))

# In[ ]:


rho = 7.85E-9  # ton/mm
E = 2.10E5  # MPa = N/mm2
my_material_template = amfe.KirchhoffMaterial(E=E, nu=0.3, rho=rho, plane_stress=False)

component_dict = {1: {'mesh': m1,
                      'domain_tag': 3,
                      'external_force_tag': 2,
                      'external_force_direction': 0,
                      'force_value': 1.0,
                      'Dirichlet_tag': 1,
                      'material': my_material_template},
                  2: {'mesh': m2,
                      'domain_tag': 3,
                      'external_force_tag': 1,
                      'external_force_direction': 0,
                      'force_value': 0,
                      'Dirichlet_tag': 2,
                      'material': my_material_template}}

# ro=1.0E7

contact_dict = {'12': {'contact': contact12,
                       'contact_pair_id': (1, 2),
                       'elem_type': 'jenkins',
                       'elem_properties': {'ro': ro, 'N0': N0, 'k': k, 'mu': mu}}}

# In[ ]:

from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager


def components2list(component_dict, dimension=3):
    K_list = []
    M_list = []
    f_list = []
    for domain_id, param_dict in component_dict.items():

        globals().update(param_dict)
        # print(param_dict)
        # print(locals())
        m = mesh
        my_comp = amfe.MechanicalSystem()
        my_comp.set_mesh_obj(m)
        my_comp.set_domain(domain_tag, material)

        if dimension == 3:

            if external_force_direction == 0:
                direction = np.array([1., 0., 0.])
            elif external_force_direction == 1:
                direction = np.array([0., 1., 0.])
            else:
                direction = np.array([0., 0., 1.])

        elif dimension == 2:
            if external_force_direction == 0:
                direction = np.array([1., 0.])
            elif external_force_direction == 1:
                direction = np.array([0., 1.])

        my_comp.apply_neumann_boundaries(external_force_tag, force_value, direction)
        print('Number of nodes is equal to %i' % my_comp.mesh_class.no_of_nodes)

        K, f_ = my_comp.assembly_class.assemble_k_and_f()
        _, fext = my_comp.assembly_class.assemble_k_and_f_neumann()
        M = my_comp.assembly_class.assemble_m()

        try:
            connectivity = []
            for _, item in m.el_df.iloc[:, m.node_idx:].iterrows():
                connectivity.append(list(item.dropna().astype(dtype='int64')))
            m.el_df['connectivity'] = connectivity
        except:
            pass

        id_matrix = my_comp.assembly_class.id_matrix
        id_map_df = dict2dfmap(id_matrix)
        s = create_selection_operator(id_map_df, m.el_df)
        dof_manager_obj = DofManager(m.el_df,id_map_df)

        from pyfeti.src.linalg import Matrix
        K1 = Matrix(K, key_dict=s.selection_dict)
        M1 = Matrix(M, key_dict=s.selection_dict)

        # applying Dirichlet B.C.
        K1.eliminate_by_identity(Dirichlet_tag, 1.0E15)
        M1.eliminate_by_identity(Dirichlet_tag, 0.0)

        K_list.append(K1.data)
        M_list.append(M1.data)
        f_list.append(fext)

    return K_list, M_list, f_list, s, dof_manager_obj


def list2global(K_list, M_list, f_list, alpha=1.0E-3, beta=1.0E-7):
    K_global = sparse.block_diag(K_list)
    M_global = sparse.block_diag(M_list)

    C_global = alpha * K_global + beta * M_global
    f_global = np.concatenate(f_list)
    f_global /= np.linalg.norm(f_global)

    return K_global.tocsc(), M_global.tocsc(), C_global.tocsc(), f_global


def create_map_local_domain_dofs_dimension(component_dict, dimension=3):
    map_local_domain_dofs_dimension = {}
    for domain_id, param_dict in component_dict.items():
        m_ = param_dict['mesh']
        map_local_domain_dofs_dimension[domain_id] = m_.no_of_nodes * dimension
    return map_local_domain_dofs_dimension






class SplitOperator():
    def __init__(self, map_local_domain_dofs_dimension):
        self.map_local_domain_dofs_dimension = map_local_domain_dofs_dimension

    def LinearOperator(self, u):
        u_list = []
        idx = 0
        for key, item in self.map_local_domain_dofs_dimension.items():
            try:
                u_list.append(u[idx:idx + item])
            except:
                u_list.append(u[idx:])
            idx = item
        return u_list


class Nonlinear_Force():
    def __init__(self, Fnl_obj_list):
        self.Fnl_obj_list = Fnl_obj_list

    def compute_force(self, u, X0=None):
        Fnl_obj_list = self.Fnl_obj_list
        for fnl_obj_item in Fnl_obj_list:
            try:
                output += fnl_obj_item.compute(u, X0)
            except:
                output = fnl_obj_item.compute(u, X0)

        return output

    def compute_jacobian(self, u, X0=None):
        Fnl_obj_list = self.Fnl_obj_list
        for fnl_obj_item in Fnl_obj_list:
            try:
                output += fnl_obj_item.Jacobian(u, X0)
            except:
                output = fnl_obj_item.Jacobian(u, X0)

        return output


def force_in_time(fext, time_points, omega):
    f_list = []
    for i in range(time_points):
        a = rate * np.sin(2.0 * np.pi * i / time_points)
        f1 = a * fext
        f_list.append(f1)

    return np.array(f_list).T


# In[ ]:


K_list, M_list, f_list, s, dof_manager_obj = components2list(component_dict, dimension=2)

K_global, M_global, C_global, f_global = list2global(K_list, M_list, f_list, alpha=1.0E-2, beta=1.0E-4)




# solving Eigenvalue problem for problem change of basis
K_global_inv = sparse.linalg.splu(K_global)
D = sparse.linalg.LinearOperator(shape=K_global.shape, matvec=lambda x: K_global_inv.solve(M_global.dot(x)))

# modes = K_global.shape[0] - 2
modes = 37
val, Phi = sparse.linalg.eigs(D, k=modes)
Phi = Phi.real
normalizeit = False
if normalizeit:
    for i in range(modes):
        vi = Phi[:, i]
        Phi[:, i] = vi / np.linalg.norm(vi)



val_full = np.sqrt(1/val)/(2*np.pi)

# so now i reduce!
contact_nodes_HCB = []
contact_nodes_HCB.extend(contact12.master_nodes)
contact_dof_HCB = dof_manager_obj.get_dofs_from_node_list(contact_nodes_HCB, direction ='xy')

get_global_id_from_local = lambda local_id : m.no_of_nodes + local_id
slave_nodes_global = list(map(get_global_id_from_local,contact12.slaves_nodes))
slave_nodes_2_dof_map = lambda node_id : [2*node_id, 2*node_id + 1]

all_slave_nodes_2_dof =  []
for node_id in slave_nodes_global:
    all_slave_nodes_2_dof.extend(slave_nodes_2_dof_map(node_id))

contact_dof_HCB.extend(all_slave_nodes_2_dof)



Neuman_dof_HCB =  list(s.selection_dict[2])
master_dof_HCB = []
master_dof_HCB.extend(contact_dof_HCB)
#master_dof_HCB.extend(Neuman_dof_HCB)
master_dof_HCB = list(set(master_dof_HCB))

ndof = K_global.shape[1]
all_dof = list(range(ndof))
slave_dof_HCB = list(set(all_dof) - set(master_dof_HCB))

map_local_domain_dofs_dimension = create_map_local_domain_dofs_dimension(component_dict, dimension=2)



'''
master_dofs = np.array(list(s.selection_dict[2]))
slave_dofs = np.array(list(s.get_complementary_set([2])))
'''
from amfe.mechanical_system import CraigBamptonComponent

HCB_object = CraigBamptonComponent()
T, T_local, P, K_local, M_local = HCB_object.compute_Craig_Bampton_Red(M_global, K_global,
                                                                       master_dofs = np.array(master_dof_HCB),
                                                                       slave_dofs = np.array(slave_dof_HCB), no_of_modes=90)

T = np.array(T)
new_modes = 50
#T = T_local
Kcraig = T.T@K_global@T
Mcraig = T.T@M_global@T
K_craig_inv = sparse.linalg.splu(Kcraig)
Dcraig = sparse.linalg.LinearOperator(shape=Kcraig.shape, matvec = lambda x : K_craig_inv.solve((Mcraig.dot(x)).T))
val_craig, Phi_craig = sparse.linalg.eigs(Dcraig,k=new_modes)

val_craig_Hz = np.sqrt(1/val_craig)/(2*np.pi)

q = Phi_craig.T
Phi_craig_exp = T.dot(Phi_craig)
mode_id = 0
plt.plot(Phi[:,mode_id],Phi_craig_exp[:,mode_id],"o")

SO = SplitOperator(map_local_domain_dofs_dimension)
u_array = np.array(Phi_craig_exp[:,mode_id])
u_array /= np.linalg.norm(u_array)
u_array_full = np.array(Phi[:,mode_id+1])
u_array_full /= np.linalg.norm(u_array_full)
u_list = SO.LinearOperator(u_array)
u_list_full = SO.LinearOperator(u_array_full)

'''
amfe.plotDeformQuadMesh(m1.connectivity,m1.nodes,u_list[0].flatten(),factor=10)
amfe.plotDeformQuadMesh(m1.connectivity,m1.nodes,-u_list_full[0].flatten(),factor=10)
amfe.plotDeformQuadMesh(m2.connectivity,m2.nodes,u_list[1].flatten(),factor=10)
amfe.plotDeformQuadMesh(m2.connectivity,m2.nodes,-u_list_full[1].flatten(),factor=10)
'''

mesh_list = [m1, m2]

Fnl_obj_list = []
for contact_key, contact_item in contact_dict.items():
    bodies_contact_id = contact_item['contact_pair_id']
    contact_12 = contact_item['contact']
    elem_type = contact_item['elem_type']
    elem_properties = contact_item['elem_properties']
    create_obj_1 = Create_node2node_force_object(contact12, bodies_contact_id, elem_type, elem_properties, dimension,
                                                 map_local_domain_dofs_dimension)
    Fnl_obj_list.append(create_obj_1.assemble_nonlinear_force())

# In[ ]:


from contpy import optimize as copt, frequency, operators



ndofs = K_global.shape[0]
Q = frequency.assemble_hbm_operator(ndofs, number_of_harm=nH, n_points=time_points)  # bases of truncaded Fourier
nonlinear_force_obj = Nonlinear_Force(Fnl_obj_list)
AFT = operators.Nonlinear_Force_AFT(Q, nonlinear_force_obj)
Z = lambda w: frequency.create_Z_matrix(K_global, C_global, M_global, f0=w / (2.0 * np.pi), nH=nH, static=False)
Zw = Z(omega)

T_inv = np.linalg.pinv(T)

Zw_HCB = T_inv @ Zw @ T
Zw_HCB_real = copt.complex_matrix_to_real_block(Zw_HCB)
Zw_real = copt.complex_matrix_to_real_block(Zw)
force_global_in_time = force_in_time(f_global, time_points, omega)
force_global_ = Q.H.dot(force_global_in_time)


# In[ ]:


def Residual_and_Jac_in_real_block(u_real):
    u_ = copt.real_array_to_complex(u_real)
    fnl_complex_eval, Jnl_eval_1, Jnl_eval_conj_1 = AFT.compute_force_and_jac(u_)
    J_block_real = copt.complex_matrix_to_real_block(Jnl_eval_1, Jnl_eval_conj_1)
    J = Zw_real - J_block_real
    R = Zw.dot(u_) - force_global_ - fnl_complex_eval
    R_real = copt.complex_array_to_real(R)
    return R_real, J


def Residual_and_Jac_in_real_block_HCB(q_real):
    q_ = copt.real_array_to_complex(q_real)
    u_ = T.dot(q_)
    fnl_complex_eval, Jnl_eval_1, Jnl_eval_conj_1 = AFT.compute_force_and_jac(u_)
    V = T
    J_block_real = copt.complex_matrix_to_real_block(T_inv @ Jnl_eval_1 @ V, T_inv @ Jnl_eval_conj_1 @ V)
    J = Zw_HCB_real - J_block_real
    R = Zw.dot(u_) - force_global_ - fnl_complex_eval
    VTR = T_inv.dot(R)
    R_real = copt.complex_array_to_real(VTR)
    return R_real #, J


Zw_inv = sparse.linalg.splu(Zw)
u__initial = Zw_inv.solve(force_global_)
u__inital_real = copt.complex_array_to_real(u__initial)
# solving the system without any reduction
# sol2 = copt.LevenbergMarquardt(Residual_and_Jac_in_real_block,0.0*u__inital_real,method=None,jac=True,maxiter=200)
sol2 = copt.Newton(Residual_and_Jac_in_real_block, u__inital_real, method=None, jac=True, maxiter=200)
u_sol = copt.real_array_to_complex(sol2.x)
u_sol_time = Q.dot(u_sol)

print('no. of iterations for sol2 %i' % sol2.nfev)


q__initial = T_inv.dot(u__initial)
q__initial_real = np.array(copt.complex_array_to_real(q__initial)).flatten()

jacR = nd.Jacobian(Residual_and_Jac_in_real_block_HCB)

#sol2 = copt.Newton(Residual_and_Jac_in_real_block_HCB, 0.0 * q__initial_real, method=None, jac=True, maxiter=200)
sol2 = copt.LevenbergMarquardt(Residual_and_Jac_in_real_block_HCB, q__initial_real, method=None, jac=jacR , maxiter=200)
q_sol = copt.real_array_to_complex(sol2.x)
u_sol_HCB = T.dot(q_sol)
u_sol_time_HCB = Q.dot(u_sol_HCB)




