import matplotlib.pyplot as plt
import numpy as np

np_proc_list = [1,2,4,6,8,10]
dof_list = np.array([100_000,400_000, 625_000, 900_000, 1_225_000, 1_600_000])
results = {100_000 : {1:11722.369, 2:16090.621 ,4:16513.632  ,6:10286.244 , 8:9199.382 , 10:7897.653 } ,
           400_000 : {1:23459.407, 2:38013.666 , 4:41908.204 , 6:39795.026 ,8:29261.376 ,10:29795.982},
           625_000 : {1:25035.515, 2:40832.020 , 4: 48223.548, 6:47308.792 ,8:43684.345 ,10:55819.776},
           900_000 : {1:29625.043, 2:31897.013 , 4: 75459.363, 6:91868.153 ,8:81330.205 ,10:62500.740},
           1_225_000 : {1:40845.603, 2:37096.359 , 4: 73202.935, 6:98156.912 ,8:76825.816 ,10:74824.878},
           1_600_000 : {1:59784.237, 2:54870.672 , 4: 115622.056, 6:128815.684 ,8:119130.908 ,10:110959.052}}

A = np.zeros((len(np_proc_list),len(dof_list)))
i=0
j=0
for i,dofs in enumerate(dof_list):
    for j,nproc in enumerate(np_proc_list):
        A[i,j] = results[dofs][nproc]
        

plt.figure()
plt.plot(dof_list/1E3,A,'o-')
plt.xlabel('Dofs x 1000')
plt.ylabel('Time [ms]')
plt.legend(np_proc_list)
plt.show()


plt.figure()
plt.plot(np_proc_list,A.T,'o-')
plt.xlabel('Number of Processors')
plt.ylabel('Time [ms]')
plt.legend(dof_list)
plt.show()

plt.figure()
plt.plot(dof_list/1E3,A.T[0,:],'o-')
plt.xlabel('Dofs x 1000')
plt.ylabel('Time [ms]')
plt.legend(['1 MPI'])
plt.show()