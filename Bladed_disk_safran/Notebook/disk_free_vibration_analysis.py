#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os 

import scipy.sparse as sparse
import scipy.linalg as linalg
import numpy as np
import collections
import copy

import amfe
from amfe.utils.utils import OrderedSet
from amfe.cyclic.cyclic import SelectionOperator, apply_cyclic_symmetry, get_dofs, rotate_u, set_cyclic_modes_to_component
from amfe.cyclic.cyclic import create_voigt_rotation_matrix, assemble_cyclic_modes
from amfe.linalg.arnoldi import arnoldi_iteration, inverse_arnoldi_iteration, general_inverse_arnoldi_iteration, generalized_arnoldi_iteration
from amfe.linalg.arnoldi import lanczos, is_eigvec, nullspace, LinearSys, power_iteration, ProjLinearSys, compute_modes
from amfe.units import convert



# In[2]:


mesh_folder = '/home/mlid150/Documents/Demonstrator/Safran Demonstrator/Meshes'
msh_name = r'3D_safran_tet4_disk_1661_nodes.inp'
msh_file = os.path.join(mesh_folder,msh_name)
m = amfe.Mesh()
m.import_inp(msh_file,1000)


# In[3]:


from pyfeti import utils
utils.save_object(m,'disk.pkl')


# In[4]:


ax1 = amfe.plot3Dmesh(m)
#pltmesh.show(plot_nodes=False)
vlim=[0,175]
ax1.set_xlim(vlim)
ax1.set_ylim(vlim)
ax1.set_zlim(vlim)
#pltmesh.set_equal_axis_lim(vlim)


# In[5]:


print(m.get_elem_types())
print(m.get_phys_group_types())


# In[6]:


map_string = lambda  i : 'DISC_1_%i_SOLID_ELSET' %i

# rename solid components

for i in range(1,7):
    m.change_tag_in_eldf('phys_group',map_string(i),'DISK')


# In[7]:


print(m.get_phys_group_types())


# In[8]:


m.split_in_groups()  
disk = m.get_submesh('phys_group','DISK')


# In[9]:


ax2 = amfe.plot3D_submesh(disk,plot_nodes=False)
vlim=[-50,50] 
ax2.set_xlim(vlim)
ax2.set_ylim(vlim)
ax2.set_zlim(0,100)


# In[10]:


# creating a mechanical component
my_comp = amfe.MechanicalSystem()
my_comp.set_mesh_obj(m)
rho = 7.85E-9 # ton/mm3
E = 2.10E5 # MPa = N/mm2
my_material = amfe.KirchhoffMaterial(E=E, nu=0.3, rho=rho, plane_stress=False)
my_comp.set_domain('DISK',my_material)
print('Number of nodes is equal to %i' %my_comp.mesh_class.no_of_nodes)


# In[11]:


K, f = my_comp.assembly_class.assemble_k_and_f()


# In[12]:


M = my_comp.assembly_class.assemble_m()


# In[19]:


from scipy import sparse


lu = sparse.linalg.splu(K.tocsc())
D = sparse.linalg.LinearOperator(shape=K.shape,matvec=lambda x :lu.solve(M.dot(x)))
eigval, eigvec = sparse.linalg.eigsh(D,k=20)


# In[20]:


omega = 1.0/np.sqrt(eigval.real)
print(omega)

f = omega/(2.0*np.pi)


# In[21]:


np.sort(eigval.real)


# In[22]:


print(f)


# In[23]:


ansys_freq = np.array([0.,0., 0., 9.8464E-4,1.8173E-3,3.0027E-3,8005.3,
                      10360,11601,15739,21363,24524,24739,27430,30435,31976,37227,46724,47497,48321])


# In[24]:


plt.figure()
plt.plot(ansys_freq,f,'o')
plt.xlabel('Ansys frequency [rad]')
plt.ylabel('AMFE frequency [rad]')
plt.show()


# In[ ]:




