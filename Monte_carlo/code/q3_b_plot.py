import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def surface_plot(matrix,**kwargs):
    (x,y) = np.meshgrid(np.arange(matrix.shape[0])+1,np.arange(matrix.shape[1])+12)
    #print((x,y))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(x,y,matrix,**kwargs)
    return (fig,ax,surf)

best_policy = np.load("jack_2.npy")
print(best_policy.shape)
best_policy = best_policy[:,11:21,:]

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
ax1.imshow(best_policy[0],interpolation='none')
ax1.imshow(best_policy[0],interpolation='none',extent=[1,10,21,11])
ax1.set_ylabel('player sum')
ax1.set_xlabel('dealer showing')
ax1.set_title('No Usable ace')

ax2.imshow(best_policy[1],interpolation='none',extent=[1,10,21,11])
ax2.set_ylabel('player sum')
ax2.set_xlabel('dealer showing')
ax2.set_title('Usable ace')
#plt.savefig('q3_b_best_policy')
plt.show()

q_value=np.load('jack_2_value.npy')
q_use = q_value[1]
q_unsuse = q_value[0]
v_use = np.max(q_use,axis=2)[11:21,:]
print(v_use.shape)
v_unuse=np.max(q_unsuse,axis=2)[11:21,:]
print(v_use.shape)

(fig2,ax,surf) = surface_plot(v_unuse,cmap=cm.coolwarm)
fig2.colorbar(surf)
ax.set_ylabel('player sum')
ax.set_xlabel('dealer showing')
ax.set_zlabel('value')
ax.set_title('V*, No usable ace')
#plt.savefig('q4_b_value1')

(fig1,ax1,surf) = surface_plot(v_use,cmap=cm.coolwarm)
fig1.colorbar(surf)
ax1.set_ylabel('player sum')
ax1.set_xlabel('dealer showing')
ax1.set_zlabel('value')
ax1.set_title('V*, Usable ace')
#plt.savefig('q4_b_value2')
plt.show()