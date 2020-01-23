
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
from functions import *
from matplotlib import animation, rc
from contpy import optimize as copt, frequency, operators

# geometric properties
dimension= 2
width= 100.0
heigh = 5.0
init_gap_x = width - 0.2*width
init_gap_y = -heigh - 0.01*heigh


#mesh properties
#x_divisions,y_divisions= 51,21
x_divisions,y_divisions= 21,6


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


# Defining contact pairs
m1.split_in_groups()
m2.split_in_groups()


tol_radius = 5.0E-1
contact_12_tag = 4
contact_21_tag = 5

# contact pair 12
d1 = m1.get_submesh('phys_group',contact_12_tag)
d2 = m2.get_submesh('phys_group',contact_21_tag)

contact12 = amfe.contact.Contact(d1,d2,tol_radius=tol_radius)
print('Number of contact pairs = %i' %len(contact12.contact_elem_dict))


rho = 7.85E-6 # ton/mm
E = 2.10E3 # MPa = N/mm2
my_material_template = amfe.KirchhoffMaterial(E=E, nu=0.3, rho=rho, plane_stress=False)

component_dict = {1:{'mesh' : m1, 
                     'domain_tag':3,
                     'external_force_tag':2,
                     'external_force_direction':1,
                     'force_value':1.0,
                     'Dirichlet_tag':1,
                     'material' : my_material_template},
                   2:{'mesh': m2, 
                      'domain_tag':3,
                      'external_force_tag':1,
                      'external_force_direction':0,
                      'force_value':0,
                       'Dirichlet_tag':2,
                       'material' : my_material_template}}


#ro=1.0E7
ro=1.E2
N0=0.0E0
k= 1.0E1
mu= 0.3
contact_dict = {'12' : {'contact' : contact12, 
                        'contact_pair_id' : (1,2),
                        'elem_type' : 'jenkins' , 
                        'elem_properties' : {'ro':ro,'N0':N0,'k':k, 'mu':mu}}}



# creating linear matrices
K_list, M_list, f_list =components2list(component_dict,dimension=2)
alpha, beta= 1.0E-1, 1.0E-4
K_global,M_global,C_global,f_global =  list2global(K_list, M_list, f_list,alpha = alpha, beta = beta)


# solving Eigenvalue problem for problem change of basis
K_global_inv = sparse.linalg.splu(K_global)
D = sparse.linalg.LinearOperator(shape=K_global.shape, matvec = lambda x : K_global_inv.solve(M_global.dot(x)))

#modes = K_global.shape[0] - 2
modes = 40
val, Phi = sparse.linalg.eigs(D,k=modes)
print('Eigenfrequencies in rad/s', np.sqrt(1.0/val) )
Phi = Phi.real
normalizeit = True
if normalizeit:
    for i in range(modes):
        vi = Phi[:,i]
        Phi[:,i] = vi/np.linalg.norm(vi)  

map_local_domain_dofs_dimension = create_map_local_domain_dofs_dimension(component_dict,dimension=2)
SO = SplitOperator(map_local_domain_dofs_dimension)
u_compont_list = SO.LinearOperator(Phi[:,5])


time_points = 100
frame_list = list(map(lambda x : 2.0*np.sin(2.0*np.pi*x/100),range(time_points)))

u_list = []
for i in frame_list:
    u_list.append(i*np.array(u_compont_list).T)

u_list = np.array(u_list).T

anim_obj = HBM_animation(component_dict,u_list,width,time_points,factor=100.0,color_id=2)
#anim_obj.show()
#anim_obj.save()


mesh_list = [m1,m2]
Fnl_obj_list = []
for contact_key,contact_item in contact_dict.items():
    bodies_contact_id = contact_item['contact_pair_id']
    contact_12 = contact_item['contact']
    elem_type = contact_item['elem_type']
    elem_properties = contact_item['elem_properties']
    create_obj_1 = Create_node2node_force_object(contact12,bodies_contact_id,elem_type,elem_properties,dimension,map_local_domain_dofs_dimension)
    Fnl_obj_list.append(create_obj_1.assemble_nonlinear_force())
    




def solve_HBM(u__initial,omega,nH,time_points,rate):
    
    def Residual_and_Jac_mode(q_real):


        q_complex = copt.real_array_to_complex(q_real)
        u_ = V.dot(q_complex)
        fnl_complex_eval, Jnl_eval_1, Jnl_eval_conj_1 = AFT.compute_force_and_jac(u_)
        J_block_real = copt.complex_matrix_to_real_block(V.T@Jnl_eval_1@V, V.T@Jnl_eval_conj_1@V)

        Zw_mode = V.T@Zw@V
        Zw_mode_real = copt.complex_matrix_to_real_block(Zw_mode)
        J = Zw_mode_real  - J_block_real
        R = Zw.dot(u_) - force_global_ - fnl_complex_eval
        VTR = V.T.dot(R)
        R_real = copt.complex_array_to_real(VTR)
        return R_real, J
    
    Z = lambda w : frequency.create_Z_matrix(K_global,C_global,M_global,f0= w/(2.0*np.pi),nH=nH, static=False)
    Zw = Z(omega)
    Zw_real = copt.complex_matrix_to_real_block(Zw)
    force_global_in_time = force_in_time(f_global,time_points,rate,omega)
    force_global_ = Q.H.dot(force_global_in_time)


    #if u__initial is None:
    Zw_inv = sparse.linalg.splu(Zw)
    u__initial = Zw_inv.solve(force_global_)
    
    u__inital_real = copt.complex_array_to_real(u__initial)
    u__inital_real = 0.0*u__inital_real




    q_complex_1 = V.T.dot( u__initial)
    q_real_init_1 = copt.complex_array_to_real(q_complex_1 )
    sol2_q = copt.Newton(Residual_and_Jac_mode,q_real_init_1,method=None,jac=True,maxiter=200,tol=1.E-8)

    q_sol_complex = copt.real_array_to_complex(sol2_q.x)
    u_sol_mode = V.dot(q_sol_complex)
    u_sol_time_ = Q.dot(u_sol_mode)

    return u_sol_mode



nH = 1
omega = 1.0
time_points = nH*25
rate = 1.0E2

# select the modal basis
n_modes = 2
V_ = Phi[:,0:n_modes]
V= sparse.block_diag(([V_]*nH))

ndofs = K_global.shape[0]
Q = frequency.assemble_hbm_operator(ndofs,number_of_harm=nH ,n_points=time_points) # bases of truncaded Fourier
nonlinear_force_obj = Nonlinear_Force(Fnl_obj_list)
AFT  = operators.Nonlinear_Force_AFT(Q,nonlinear_force_obj)

natural_continuation = True
if natural_continuation:
    w_list = np.linspace(10,1000,60)
    u_list_freq = []
    u__initial = None
    for wi in w_list:
        print('Computing the solution for omega = %f' %wi)
        u_i = solve_HBM(u__initial,wi,nH,time_points,rate)
        u_list_freq.append(u_i)
        u__initial = u_i


    plt.figure()
    plt.plot(w_list, np.abs(np.array(u_list_freq)),'--o')
    plt.savefig('natural_continuation_algorithm.png')


utils.save_object(u_list_freq,'u_list_freq.pkl')
utils.save_object(w_list,'w_list.pkl')

def augmented_residual(q_real,w):
    Z = lambda w : frequency.create_Z_matrix(K_global,C_global,M_global,f0= w/(2.0*np.pi),nH=nH, static=False)
    Zw = Z(w)
    Zw_real = copt.complex_matrix_to_real_block(Zw)
    Zw_mode = V.T@Zw@V
    Zw_mode_real = copt.complex_matrix_to_real_block(Zw_mode)
    force_global_in_time = force_in_time(f_global,time_points,rate,w)
    force_global_ = Q.H.dot(force_global_in_time)
    q_complex = copt.real_array_to_complex(q_real)
    u_ = V.dot(q_complex)
    fnl_complex_eval, Jnl_eval_1, Jnl_eval_conj_1 = AFT.compute_force_and_jac(u_)
    J_block_real = copt.complex_matrix_to_real_block(V.T@Jnl_eval_1@V, V.T@Jnl_eval_conj_1@V)
    J = Zw_mode_real  - J_block_real
    R = Zw.dot(u_) - force_global_ - fnl_complex_eval
    VTR = V.T.dot(R)
    R_real = copt.complex_array_to_real(VTR)
    return R_real, J



class DynamicSystem():
    def __init__(self,func, tol=1.0E-12):
        self.tol  = tol
        self.func = func
        self.x = None
        self.residual = None
        self.jacobian = None

    def call_residual(self,x,w):
        self.x = x
        self.residual , self.jacobian = self.func(x,w)
        return self.residual

    def call_jacobian(self,x,w):
        if self.x is None:
            self.residual , self.jacobian = self.func(x,w)

        elif np.linalg.norm(x - self.x) > self.tol or self.jacobian is None:
            self.residual , self.jacobian = self.func(x,w)

        return self.jacobian.A


dRw = lambda x,w  : np.array([(copt.complex_matrix_to_real_block(-2.0*w*V.T@sparse.block_diag(([M_global]*nH))@V + 1J*V.T@sparse.block_diag(([C_global]*nH))@V)).dot(x)])
dynamic_obj = DynamicSystem(augmented_residual)
R = lambda x,w : dynamic_obj.call_residual(x, w)
JRx = lambda w : lambda x :  dynamic_obj.call_jacobian(x, w)
JRw = lambda x : lambda w :  dRw(x,w)


w_range = (10,500)
w0 = 50
Rq = lambda x : R(x,w0)
Jw0 = JRx(w0)
JacRq = lambda x : Jw0(x) 
q_init = np.zeros(2*V.shape[1])
dynamic_obj.call_jacobian(q_init, w0)
sol_init = copt.Newton(Rq,q_init,maxiter=200,jac=JacRq)
q0 = sol_init.x
r = R(q0,w0)
jrx = JRx(w0)(q0)
jrp = JRw(q0)(w0)

x_sol, p_sol, info_dict = copt.continuation(R,x0=q0,jacx=JRx,jacp=JRw,
                                                p_range=w_range,p0=w0,max_dp=10.0,step=+10.0,max_int=400,correction_method='matcont')


u_sol = []
for x_sol_row in x_sol.T:
    u_sol.append(V@copt.real_array_to_complex(x_sol_row))

plt.figure()
plt.plot(p_sol, np.abs(np.array(u_sol)),'--o')
plt.savefig('continuation_algorithm.png')



utils.save_object(info_dict,'info_dict.pkl')
utils.save_object(p_sol,'p_sol.pkl')
utils.save_object(x_sol,'x_sol.pkl')

