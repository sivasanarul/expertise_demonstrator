import os

sizes = (250,300)

parallel_list = (1,2,4,6,8)

for size in sizes:
    for np in parallel_list: 
        commnd = 'mpirun -np %i  python test_with_variable_matrix_dist.py %i %i %i' %(np,size*5,size,np)
        os.system(commnd)