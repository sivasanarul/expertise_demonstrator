
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
from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager
from matplotlib import animation, rc

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

def components2list(component_dict,dimension=3):
    K_list = []
    M_list = []
    f_list = []
    for domain_id, param_dict in component_dict.items():

        globals().update(param_dict)
        #print(param_dict)
        #print(locals())
        m = mesh 
        my_comp = amfe.MechanicalSystem()
        my_comp.set_mesh_obj(m)
        my_comp.set_domain(domain_tag,material)

        if dimension==3:

            if external_force_direction == 0:
                direction = np.array([1.,0.,0.])
            elif external_force_direction == 1:
                direction = np.array([0.,1.,0.])
            else:
                direction = np.array([0.,0.,1.])
                
        elif dimension==2:
            if external_force_direction == 0:
                direction = np.array([1.,0.])
            elif external_force_direction == 1:
                direction = np.array([0.,1.])
            

        my_comp.apply_neumann_boundaries(external_force_tag,force_value,direction)
        print('Number of nodes is equal to %i' %my_comp.mesh_class.no_of_nodes)

        K, f_ = my_comp.assembly_class.assemble_k_and_f()
        _, fext = my_comp.assembly_class.assemble_k_and_f_neumann()
        M = my_comp.assembly_class.assemble_m()


        try:
            connectivity = []
            for _,item in m.el_df.iloc[:, m.node_idx:].iterrows():
                connectivity.append(list(item.dropna().astype(dtype='int64')))
            m.el_df['connectivity'] = connectivity
        except:
            pass

        id_matrix = my_comp.assembly_class.id_matrix
        id_map_df = dict2dfmap(id_matrix)
        s = create_selection_operator(id_map_df,m.el_df)

        from pyfeti.src.linalg import Matrix
        K1 = Matrix(K,key_dict=s.selection_dict)
        M1 = Matrix(M,key_dict=s.selection_dict)


        # applying Dirichlet B.C.
        K1.eliminate_by_identity(Dirichlet_tag,K1.data.diagonal().max())
        M1.eliminate_by_identity(Dirichlet_tag,0.0)

        K_list.append(K1.data)
        M_list.append(M1.data)
        f_list.append(fext)
        
    return K_list, M_list, f_list
    
def list2global(K_list, M_list, f_list, alpha=1.0E-3, beta=1.0E-7):
   
    K_global = sparse.block_diag(K_list)
    M_global = sparse.block_diag(M_list)
    
    C_global = beta*K_global + alpha*M_global
    f_global = np.concatenate(f_list)
    f_global/=np.linalg.norm(f_global)

    return K_global.tocsc(),M_global.tocsc(),C_global.tocsc(),f_global


def create_map_local_domain_dofs_dimension(component_dict,dimension=3):
    map_local_domain_dofs_dimension = {}
    for domain_id, param_dict in component_dict.items():
        m_ = param_dict['mesh']
        map_local_domain_dofs_dimension[domain_id] = m_.no_of_nodes*dimension
    return map_local_domain_dofs_dimension

class SplitOperator():
    def __init__(self,map_local_domain_dofs_dimension):
        self.map_local_domain_dofs_dimension = map_local_domain_dofs_dimension
        
    def LinearOperator(self,u):
        u_list = []
        idx = 0
        for key, item in self.map_local_domain_dofs_dimension.items():
            try:
                u_list.append(u[idx:idx+item])
            except:
                u_list.append(u[idx:])
            idx = item
        return u_list
        
class Nonlinear_Force():
    def __init__(self,Fnl_obj_list):
        self.Fnl_obj_list = Fnl_obj_list
        
    def compute_force(self,u,X0=None):
        Fnl_obj_list = self.Fnl_obj_list
        for fnl_obj_item in Fnl_obj_list:
            try:
                output += fnl_obj_item.compute(u,X0)
            except:
                output = fnl_obj_item.compute(u,X0)

        return output
        
    def compute_jacobian(self,u,X0=None):
        Fnl_obj_list = self.Fnl_obj_list
        for fnl_obj_item in Fnl_obj_list:
            try:
                output += fnl_obj_item.Jacobian(u,X0)
            except:
                output = fnl_obj_item.Jacobian(u,X0)

        return output
    
    
def force_in_time(fext,time_points,rate,omega):
    
    f_list = []
    for i in range(time_points):
        a = rate*np.sin(2.0*np.pi*i/time_points)
        f1 = a*fext
        f_list.append(f1)

    return np.array(f_list).T


class LM_Krylov(copt.LinearSolver):
    def __init__(self,x0,tol=1.0E-6,maxiter=50,verbose=False):
        self.tol = tol
        self.maxiter = maxiter
        self.verbose = verbose
        self.x0 = x0
        self._counter = 0
        self.max_reuse_precond = 10
        self.M = None
        self.info = 0
        super().__init__(**self.__dict__)
        
    def solve(self,A,b):
        
        if self._counter ==0:
            if self.verbose:
                print('Computing exact preconditioner')
            LU = sparse.linalg.splu(A)
            self.M = sparse.linalg.LinearOperator(shape=A.shape,matvec = lambda x : LU.solve(x))
            self._counter +=1
        elif self._counter==self.max_reuse_precond:
            self._counter = 0
        else:
            self._counter +=1
            
        x, self.info = sparse.linalg.gmres(A, b, x0=self.x0, tol=self.tol,maxiter=self.maxiter,M=self.M,  restart=self.max_reuse_precond)
        
        if self.verbose:
            print('GMRes info - %i' %self.info)
            
        if self.info!=0:
            self._counter = 0
            x = self.solve(A,b)
            
        return x
        
    def update(self,xn):
        if self.info==0:
            self.x0 = xn
        #self.maxiter += 2
        
   

class LM_CG_Krylov(copt.LinearSolver):
    def __init__(self,x0,tol=1.0E-6,maxiter=50,verbose=False):
        self.tol = tol
        self.maxiter = maxiter
        self.verbose = verbose
        self.x0 = x0
        self._counter = 0
        self.max_reuse_precond = 10
        self.M = None
        self.info = 0
        super().__init__(**self.__dict__)
        
    def solve(self,A,b):
        
        if self._counter ==0:
            if self.verbose:
                print('Computing exact preconditioner')
            LU = sparse.linalg.splu(A)
            self.M = sparse.linalg.LinearOperator(shape=A.shape,matvec = lambda x : LU.solve(x))
            self._counter +=1
        elif self._counter==self.max_reuse_precond:
            self._counter = 0
        else:
            self._counter +=1
            
        x, self.info = sparse.linalg.cg(A, b, x0=self.x0, tol=self.tol,maxiter=self.maxiter,M=self.M)
        
        if self.verbose:
            print('GMRes info - %i' %self.info)
            
        if self.info!=0:
            self._counter = 0
            x = self.solve(A,b)
            
        return x
        
    def update(self,xn):
        if self.info==0:
            self.x0 = xn




class HBM_animation():

    def __init__(self,component_dict,u_list,width,time_points,factor=1.0,color_id=2):
        
        fig, ax = plt.subplots(1,1)

        self.fig = fig
        self.ax = ax
        self.color_id = color_id
        self.factor = factor
        self.component_dict = component_dict
        self.u_list = u_list
        self.width = width
        self.time_points = time_points
        self.anim = None

    def _update_HBM(self,j,factor=None):

        if factor is None:
            factor = self.factor
        
        ax2 = self.ax
        component_dict = self.component_dict
        u_list = self.u_list
        width = self.width
        i=0
        ax2.clear()
        for key, mesh_dict_ in component_dict.items():
            me = mesh_dict_['mesh']
            p1, _ = amfe.plotDeformQuadMesh(me.connectivity,me.nodes,u_list[i].T[j],factor=factor,ax=ax2,color_id=self.color_id)
            i+=1

        ax2.set_xlim([0,2.2*width])
        ax2.set_ylim([-1.1*width,1.1*width])
        ax2.legend('off')
        ax2.set_title('Deformed Mesh without reduction')  

    def __animate(self):
        self.anim = animation.FuncAnimation(self.fig, self._update_HBM,
                                frames=range(self.time_points), interval=1)

        return self.anim 

    def show(self):
        
        anim = self.__animate()
        plt.show()

        return self.fig, self.ax

    def save(self,gif_name='HBM_animation.gif'):

        Writer =animation.writers['imagemagick']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1000)
        
        anim = self.__animate()

        anim.save( gif_name , writer=writer, dpi = 100 )