import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from gym.utils import seeding


v_record= np.load('jack_1.npy')
v_record = v_record[:,11:21,:]
def surface_plot(matrix,**kwargs):
    (x,y) = np.meshgrid(np.arange(matrix.shape[0])+1,np.arange(matrix.shape[1])+12)
    #print((x,y))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(x,y,matrix,**kwargs)
    return (fig,ax,surf)


(fig,ax,surf) = surface_plot(v_record[0],cmap=cm.coolwarm)
fig.colorbar(surf)
ax.set_ylabel('player sum')
ax.set_xlabel('dealer showing')
ax.set_zlabel('value')
ax.set_title('10,000 episodes No usable ace')

(fig1,ax1,surf) = surface_plot(v_record[1],cmap=cm.coolwarm)
fig1.colorbar(surf)
ax1.set_ylabel('player sum')
ax1.set_xlabel('dealer showing')
ax1.set_zlabel('value')
ax1.set_title('10,000 episodes, Usable ace')
#print(v_record,v_record.shape)
#plt.savefig('1_usable')
plt.show()
