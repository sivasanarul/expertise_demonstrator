

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


variable_dict = utils.load_object('case_variables.pkl')
globals().update(variable_dict)



#modes = K_global.shape[0] - 2
modes = 1000
val, Phi = sparse.linalg.eigs(variable_dict.D,k=modes)
Phi = Phi.real

normalizeit = False
if normalizeit:
    for i in range(modes):
        vi = Phi[:,i]
        Phi[:,i] = vi/np.linalg.norm(vi)


