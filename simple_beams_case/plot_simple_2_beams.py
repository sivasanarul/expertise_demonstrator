
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
from matplotlib import animation, rc

mesh_list = load_object('mesh_list.pkl')
u_sol_time = load_object('u_sol_time.pkl')
component_dict = load_object('component_dict.pkl')
width = load_object('width.pkl')
SO = load_object('SO.pkl')
time_points = load_object('time_points.pkl')

u_list = SO.LinearOperator(u_sol_time)
fig2, ax2 = plt.subplots(1,1)


factor=6.0
def update_HBM(j,factor=4.0):
    i=0
    ax2.clear()
    for key, mesh_dict_ in component_dict.items():
        me = mesh_dict_['mesh']
        p1, _ = amfe.plotDeformQuadMesh(me.connectivity,me.nodes,u_list[i].T[j],factor=factor,ax=ax2,color_id=2)
        i+=1

    ax2.set_xlim([0,2.2*width])
    ax2.set_ylim([-1.1*width,1.1*width])
    ax2.legend('off')
    
    
#update(10)
anim = animation.FuncAnimation(fig2, update_HBM,
                               frames=range(time_points), interval=1)

plt.show()          

x=1