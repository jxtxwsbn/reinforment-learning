import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

with open('episode.p','rb') as f:
    data = pickle.load(f)

#q_value = np.load('q_value.npy')
greedy_policy = np.load('greedy_policy_q6.npy')
#for k,v in data.items():
#    print(k)
#print(data[999])
print(greedy_policy)


q_target = np.zeros((11,11,4))
c_matrix = np.zeros((11,11,4))



for i in range(10000):
    episode = data[i]

    G =0
    w =1
    #print('episode{}'.format(i))

    for j in range(len(episode)//4):
        #print(len((episode)))
        id = j+1
        reward = episode[-id * 4 + 2]
        action_index = int(episode[-id * 4 + 1])
        position = episode[-id * 4]
        prob = episode[-id * 4+3]
        (x, y) = position
        (nx, ny) = (int(10 - y), int(x))
        G = 0.99 * G + reward
        #print('location1',w,c_matrix[nx,ny,action_index])
        c_matrix[nx,ny,action_index] = w+c_matrix[nx,ny,action_index]
        #print('location2',w,c_matrix[nx,ny,action_index])
        q_target[nx,ny,action_index]+=(w/c_matrix[nx,ny,action_index])*(G-q_target[nx,ny,action_index])
        if greedy_policy[nx,ny]==action_index:
            w = w * (1 / prob)
        else: w = 0*w
        #print('location3',w)
        if w==0:
            break

np.save('q_target_off_6c',q_target)
v = np.zeros((q_target.shape[0],q_target.shape[1]))
for i in range(q_target.shape[0]):
    for j in range(q_target.shape[1]):
       v[i,j]=np.max(q_target[i,j,:])
np.save('6c_off_value',v)

print(v)

def surface_plot(matrix,**kwargs):
    (x,y) = np.meshgrid(np.arange(matrix.shape[0]),np.arange(matrix.shape[1]))
    #print((x,y))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(x,10-y,matrix,**kwargs)
    return (fig,ax,surf)



(fig,ax,surf) = surface_plot(v,cmap=cm.coolwarm)
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_zlabel('value')
ax.set_title('off policy max q_value')
fig.colorbar(surf)
#plt.savefig('q6c_value_off')
plt.show()