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


results_folder = '/home/mlid150/Documents/demo_salomon/nonlinear_case4'
filename_map = lambda mpi_case : os.path.join(results_folder,'simple_parametric_bladed_disk_freq_id_%i.pkl' %mpi_case)

max_mpi = 120
omega0 = 30.0
delta_omega = 80.0


def get_case(mpi_case):
    omega = omega0 + mpi_case*delta_omega # frequency in rad s
    case_dict = utils.load_object(filename_map(mpi_case),tries=1,sleep_delay=0.001)
    return case_dict, omega



u_sol_list = []
omega_list = []
u_max_list = []
for mpi_case in range(max_mpi):

    case_dict, omega = get_case(mpi_case)
    if case_dict is not None:
        u_sol = case_dict['u_sol']
        u_max_list.extend([u_sol.max()])
        u_sol_list.append(u_sol)
        omega_list.extend([omega])
        print("Mpi case = %i, omega = %f" %(mpi_case,omega))
 
U_array = np.array(u_sol_list)
plt.plot(omega_list,np.abs(U_array[:,100]),'--o')
plt.show()

U_array = np.array(u_sol_list)
plt.plot(omega_list,np.abs(U_array[:,1000]),'--o')
plt.show()

U_array = np.array(u_sol_list)
plt.plot(omega_list,np.abs(u_max_list),'--o')
plt.show()

x=1