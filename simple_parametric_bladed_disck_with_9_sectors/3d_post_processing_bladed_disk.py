import numpy as np
import matplotlib.pyplot as plt
import amfe
from pyfeti import utils
from amfe.contact import jenkins, Nonlinear_force_assembler, Create_node2node_force_object
import time
import scipy.sparse as sparse
import sparse as sp
from scipy.optimize import minimize, root
from contpy import optimize as copt, frequency
import numdifftools as nd
import os

results_folder = '/home/mlid150/Documents/demo_salomon/nonlinear_case3'
mpi_case = 99
filename_map = lambda mpi_case : os.path.join(results_folder,'simple_parametric_bladed_disk_freq_id_%i.pkl' %mpi_case)


#solution_file = 'simple_parametric_bladed_disk_v8.pkl'
solution_file =  filename_map(mpi_case)
mesh_dict_file = 'mesh_simple_parametric_bladed_disk_v1.pkl'

solution_dict = utils.load_object(solution_file)
mesh_dict = utils.load_object(mesh_dict_file)
locals().update(solution_dict)
bodies_id_list = list(shift_dict.keys())
bodies_id_list.sort()

if False:
    plt.plot(u_sol)
    plt.show()

    plt.plot(time_axis,u_sol_time.T,'o')
    plt.show()

contact_u = []
contact_f = []
force_1_list = []
force_2_list = []
force_1_external = []
force_reaction_2 = []
node_id_reaction = 700
node_direction_reaction = 1
node_id_reaction_2 = node_id_reaction
node_direction_reaction_2 = 1
time_space = 5
dof_id_contact = node_id_reaction*dimension + node_direction_reaction
dof_id_contact_2 = node_id_reaction*dimension + node_direction_reaction_2

u_dict = {}
f_dict = {}
fnl_dict = {}
def post_processing(u_sol_time,f_sol_time,force_global_in_time,body_id_i=1, body_id_j=2):

    for i in range(u_sol_time.shape[1]):
        u_global = u_sol_time[:,i]      
        if i>1:
            x0 = u_sol_time[:,i-1]
        else:
            x0 = 0.0*u_global

        fnl_ = -f_sol_time[:,i]
    
        #Fnl_obj.update_alpha()
        f = force_global_in_time[:,i]
        
        domains_id_list = list(shift_dict.keys())
        domains_id_list.sort()
        for key in domains_id_list:
            if (key+1) in shift_dict:
                range_index = range(shift_dict[key],shift_dict[key+1])
            else:
                range_index = range(shift_dict[key],ndofs)

            try:
                    u_dict[key].append(u_global[range_index])
                    fnl_dict[key].append(fnl_[range_index])
                    f_dict[key].append(f[range_index])
                
            except:
                
                u_dict[key] = [u_global[range_index]]
                fnl_dict[key] = [fnl_[range_index]]
                f_dict[key] = [f[range_index]]

        
        
        f1 = fnl_dict[body_id_i][i]
        f2  = fnl_dict[body_id_j][i]
        u1 = u_dict[body_id_i][i]
        u2 = u_dict[body_id_j][i]
        force_1_list.append(f1)
        force_2_list.append(f2)

        f_max = f2[dof_id_contact_2]
        force_reaction_2.append(f_max)    
        
        u_max = u2[dof_id_contact]
        f_max = f2[dof_id_contact]

        contact_u.append(u_max)
        contact_f.append(f_max)


    return u_dict, fnl_dict, f_dict
      

def amplitude(i):
    a = rate*np.sin(2.0*np.pi*i/time_points)
    return a

def update(i,body_id=1):

        ax2.clear()
        factor = 10.0
        collections = []
        for body_key in bodies_id_list:
            plt_obj_dict[body_key].show(displacement_id=i,collections=collections,factor=factor)
            collections = plt_obj_dict[body_key].ax.collections
            if body_key == body_id:
                amfe.plot3Dnode_id(plt_obj_dict[body_key].updated_mesh_obj,node_id=node_id_reaction,ax=ax2,plot_nodeid=True,fonte=15,color='black')
            
                
        #coord1 = m1.nodes + factor*u1_list[i].reshape(m1.nodes.shape)
        #kwargs = {'head_width':0.05, 'head_length':0.05,'width':0.02,'fc':'k', 'ec':'k'}
        #amfe.plot_force(force_1_list[i], coord1 , factor=force_scalling , ax=ax2, dim=dimension,**kwargs)

        #kwargs_f = {'head_width':0.05, 'head_length':0.05,'width':0.02,'fc':'r', 'ec':'r'}
        #amfe.plot_force(force_1_external[i], coord1 , factor=1.E-5, ax=ax2, dim=dimension,**kwargs_f)

        
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)
        ax2.set_zlim(z_lim)
        ax2.set_xlabel('x [mm]')
        ax2.set_ylabel('y [mm]')
        ax2.set_zlabel('z [mm]')
        ax2.set_title('Bladed-Disk. Amplification factor = %i.' %int(factor))
        ax2.set_xticks([x_lim[0],0.,x_lim[1]])
        ax2.set_yticks([y_lim[0],0.,y_lim[1]])
        ax2.set_zticks([z_lim[0],0.,z_lim[1]])

        ax3.plot(i,amplitude(i),'ko')
        ax3.set_ylim([-1.2*fmax,fmax*1.2])
        ax3.set_xlim([0,time_points])
        ax3.set_xlabel('time point [s]')
        ax3.set_ylabel('Force [N]')
        ax3.set_title('Applied Force in time.')


        ax4.plot(contact_u[i],contact_f[i],'bo')
        var1 = max_contact_u*scale_mult
        ax4.set_xlim([-var1,var1])
        ax4.set_ylim([-max_contact_force*scale_mult ,max_contact_force*scale_mult])
        ax4.set_xlabel('Displacement Y [mm]')
        ax4.set_ylabel('Force [N]')
        ax4.set_title('Reaction Force Y-axis at node %i.' %node_id_reaction)
        ax4.set_xticks([-var1,0.0,var1])

        ax5.plot(i, force_reaction_2[i],'ro')
        ax5.set_ylim([-scale_mult*(max_reaction_force),scale_mult*(max_reaction_force)])
        ax5.set_xlim([0,time_points])
        ax5.set_xlabel('time point [s]')
        ax5.set_ylabel('Force [N]')
        ax5.set_title('Reaction Force Y-axis at node %i.' %node_id_reaction)

        #print(i)
        return None


u1_list = []
u2_list = []
fmax = force_global_in_time.max()
rate = fmax
u_dict, fnl_dict, f_dict = post_processing(u_sol_time,fnl_sol_time,force_global_in_time,body_id_i=1, body_id_j=1)



from matplotlib.animation import FuncAnimation
from matplotlib import animation
max_contact_force = max(np.abs(np.array(contact_f)))
max_contact_u = max(np.abs(np.array(contact_u)))
scale_mult = 1.1
max_reaction_force = max(np.abs(np.array(force_reaction_2)))
x_min,x_max = -0.20,0.20
y_min,y_max = -0.20,0.20

#update(0)
fig1 = plt.figure(figsize=(12,12))
ax2 = fig1.add_subplot(2, 2, 1, projection='3d')
ax3 = fig1.add_subplot(2, 2, 2)
ax4 = fig1.add_subplot(2, 2, 3)
ax5 = fig1.add_subplot(2, 2, 4)
width = 0.2
x_lim = (-1.1*width,1.1*width)
y_lim = (-1.1*width,1.1*width)
z_lim = (-1.1*width,1.1*width)
plt_obj_dict = {}
color_dict = {1:'b',2:'y',3:'r'}
alpha_dict = {1:0.5,2:0.5,3:0.5}
for key, mesh_dict_ in mesh_dict.items():
    m = mesh_dict_['m']
    plt_obj_dict[key] = amfe.Plot3DMesh(m,displacement_list=u_dict[key],ax=ax2,plot_nodes=False,alpha=alpha_dict[key],color=color_dict[key])

#update(1)
ani = FuncAnimation(fig1, update, frames=np.arange(0,time_points,time_space), interval=2)  



def save_animation(version_id=2):
    # Set up formatting for the movie files
    Writer =animation.writers['imagemagick']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1000)
    str_res_fig = 'simple_parametric_bladed_disk_9_sectors_nonlinear_case3_mpi_case_%i' %version_id
    git_name = str_res_fig + '.gif'
    ani.save( git_name , writer=writer, dpi = 100 )

save_gif = True
if save_gif:
    save_animation(mpi_case)



