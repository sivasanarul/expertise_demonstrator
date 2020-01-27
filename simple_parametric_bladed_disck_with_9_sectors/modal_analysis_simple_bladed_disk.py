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

K_global = utils.load_object('K_global.pkl')
M_global = utils.load_object('M_global.pkl')

print('Dfos = %i' %K_global.shape[0])
K_global_inv = sparse.linalg.splu(K_global)

D = sparse.linalg.LinearOperator(shape= K_global.shape, matvec = lambda x : K_global_inv.solve(M_global.dot(x)))


num_modes = 20
val, Phi = sparse.linalg.eigs(D,k=num_modes)
#val, Phi = sparse.linalg.eigs(K_global, k=num_modes, M=M_global, which='SM')


omega = np.sqrt(1./val)
freq = omega/(2.0*np.pi)

print(freq)


save_modal_force = False
if save_modal_force:
    modal_participation = 1./num_modes
    f  = np.sum(Phi,axis=1)
    f /= np.linalg.norm(f)
    utils.save_object(f,'force_modes.pkl')


