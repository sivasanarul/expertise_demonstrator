from pyfeti.src.utils import DomainCreator, DiskCreator, DiskCreator_3D, PrismaCreator, dict2dfmap,  create_selection_operator
import amfe
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import copy




width,heigh,thickness = 100,2,1
creator_obj  = PrismaCreator(width,heigh,thickness,
                             x_divisions=11,y_divisions=7,z_divisions=4)

mesh_path = 'prisma.msh'
creator_obj.save_gmsh_file(mesh_path)
m = amfe.Mesh()
m.import_msh(mesh_path)

m4 = copy.deepcopy(m)



my_material = amfe.KirchhoffMaterial(E=210.0E9, nu=0.3, rho=7.86E3, plane_stress=False)

my_system = amfe.MechanicalSystem()
my_system.set_mesh_obj(m4)
my_system.set_domain(3,my_material)
#my_system.apply_neumann_boundaries(2,10e5,np.array([0.,1.0 ]))
#_, fext = my_system1.assembly_class.assemble_k_and_f_neumann()
K, _ = my_system.assembly_class.assemble_k_and_f()
M = my_system.assembly_class.assemble_m()
Dirichlet_tag =1

id_matrix = my_system.assembly_class.id_matrix
id_map_df = dict2dfmap(id_matrix)

connectivity = []
for _, item in m4.el_df.iloc[:, m4.node_idx:].iterrows():
    connectivity.append(list(item.dropna().astype(dtype='int64')))
m4.el_df['connectivity'] = connectivity

s = create_selection_operator(id_map_df, m4.el_df)

from pyfeti.src.linalg import Matrix

K1 = Matrix(K, key_dict=s.selection_dict)
M1 = Matrix(M, key_dict=s.selection_dict)

# applying Dirichlet B.C.
Dirichlet_tag = 2210
K1.eliminate_by_identity(Dirichlet_tag, 1.0E15)
M1.eliminate_by_identity(Dirichlet_tag, 0.0)

K1 = K1.data
M1 = M1.data

K_global_inv = sparse.linalg.splu(K1)
D = sparse.linalg.LinearOperator(shape=K1.shape, matvec = lambda x : K_global_inv.solve(M1.dot(x)))

#modes = K_global.shape[0] - 2
modes = 20
val, Phi = sparse.linalg.eigs(D,k=modes)


#plot_object = amfe.Plot3DMesh(m4,displacement_id=9)
#plot_object.show(factor=100)
#plt.show()
'''
for key in m4.groups:
    submesh = m4.groups[key]
    elem_list_type = submesh.get_element_type_list()
'''
#plt.figure()
plot_object = amfe.Plot3DMesh(m)

ax1 =plot_object.ax
mult = 1.2
ax1.set_xlim([-mult,mult*width])
ax1.set_ylim([-mult,mult*heigh])
ax1.set_zlim([-mult,mult*thickness])

ax1.set_xlabel('Width [m]')
ax1.set_ylabel('Heigh [m]')
ax1.set_zlabel('Thickness [m]')
plot_object.set_displacement(Phi)
#plot_object.pre_proc()
amfe.plot_deform_3D_mesh(m,Phi[:,8],factor=1)
#plot_object.show(factor=1000,displacement_id=9)
plt.show()

print("all is well")