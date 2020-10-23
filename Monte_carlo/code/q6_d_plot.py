import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

on_policy_q=np.load('q6d_on_policy_q_value.npy')

v = np.zeros((11,11))
for i in range(11):
    for j in range(11):
       v[i,j]=np.max(on_policy_q[i,j,:])

print(v)
np.save('6d_on_policy_value',v)

def surface_plot(matrix,**kwargs):
    (x,y) = np.meshgrid(np.arange(matrix.shape[0]),np.arange(matrix.shape[1]))
    #print((x,y))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(x,10-y,matrix,**kwargs)
    return (fig,ax,surf)



(fig,ax,surf) = surface_plot(v,cmap=cm.coolwarm)
fig.colorbar(surf)
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_zlabel('value')
ax.set_title('on policy max q_value')
#plt.savefig('q6d_value_on_policy')
plt.show()