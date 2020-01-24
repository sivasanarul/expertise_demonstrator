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
import os

# ------- variables -------
plot_geometry =  True
plot_deformed_geometry =  True


#-----------


C_dict = utils.load_object('C_global.pkl')
f_dict = utils.load_object('f_global.pkl')
K_dict = utils.load_object('K_global.pkl')
M_dict = utils.load_object('M_global.pkl')

preprocessing_variables_dict = utils.load_object('preprocessing_variables.pkl')

K_global_inv = sparse.linalg.splu(K_dict)
D = sparse.linalg.LinearOperator(shape=K_dict.shape, matvec = lambda x : K_global_inv.solve(M_dict.dot(x)))

#modes = K_global.shape[0] - 2
modes = 20
val, Phi = sparse.linalg.eigs(D,k=modes)
Phi = Phi.real

#amfe.plotDeformQuadMesh()

# ---------------------- PLOT files ----------------------
# Mesh file
mesh_file_1 = os.path.join('meshes','simple_parametric_bladed_disk_9_sectors_131976_nodes_83305_elem_tet4.inp')
mesh_file_pkl = os.path.join('meshes','simple_parametric_bladed_disk_9_sectors_131976_nodes_83305_elem_tet4.pkl')

m = utils.load_object(mesh_file_pkl,tries=1,sleep_delay=1)


if plot_geometry:

    plot_object= amfe.Plot3DMesh(m)
    plot_object.set_displacement(Phi[:,9])
    plot_object.show(factor = 1000)
    ax1 = plot_object.ax
    width = 0.200
    ax1.set_xlim((-1.1 * width, 1.1 * width))
    ax1.set_ylim((-1.1 * width, 1.1 * width))
    ax1.set_zlim((-1.1 * width, 1.1 * width))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.show()

print("all is well")

