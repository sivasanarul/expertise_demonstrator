import numpy as np
import mumps
from mpi4py import MPI
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


# In[ ]:


# geometric properties
dimension = 2
width = 5.0
heigh = 1.0
init_gap_x = width - 0.2 * width
init_gap_y = -heigh

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
rho = 7.85E-9  # ton/mm
E = 2.10E5  # MPa = N/mm2
my_material_template = amfe.KirchhoffMaterial(E=E, nu=0.3, rho=rho, plane_stress=False)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print ("rank:"+ str(rank))

# Set up the test problem:
n = 5
irn = np.array([1,2,4,5,2,1,5,3,2,3,1,3], dtype='i')
jcn = np.array([2,3,3,5,1,1,2,4,5,2,3,3], dtype='i')
a = np.array([3.0,-3.0,2.0,1.0,3.0,2.0,4.0,2.0,6.0,-1.0,4.0,1.0], dtype='d')

b = np.array([20.0,24.0,9.0,6.0,13.0], dtype='d')


# Create the MUMPS context and set the array and right hand side
ctx = mumps.DMumpsContext(sym=0, par=1)
if ctx.myid == 0:
    ctx.set_shape(5)
    ctx.set_centralized_assembled(irn, jcn, a)
    x = b.copy()
    ctx.set_rhs(x)

ctx.set_silent() # Turn off verbose output

ctx.run(job=6) # Analysis + Factorization + Solve

if ctx.myid == 0:
    print("Solution is %s." % (x,))

ctx.destroy() # Free memory
