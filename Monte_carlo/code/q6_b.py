import numpy as np
import pickle
from matplotlib.table import Table
import matplotlib.pyplot as plt
policy = np.load('policy_q6a.npy')
q_value = np.load('q_value_q6a.npy')




def rooms():
    return np.array([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    ])
map = rooms()
map[0,10]=3

greedy_policy = np.zeros((11,11))
#value = np.zeros((11,11))
print(greedy_policy.shape)
for i in range(q_value.shape[0]):
    for j in range(q_value.shape[1]):
        best = np.argmax(q_value[i,j,:])
        greedy_policy[i,j]=best

greedy_policy[map==1]=-np.inf
print(greedy_policy)

np.save('greedy_policy_q6',greedy_policy)
def draw_policy(policy):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    num_rows, num_cols = (11,11)
    per_height, per_width = 1.0 / num_rows, 1.0 / num_cols

    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            val=policy[i,j]
            print(i,j,val)
            if val==2:
                val='up'
            if  val==3:
                val='down'
            if  val==0:
                val='right'
            if  val==1:
                val='left'
            if val==-np.inf:
                val='wall'
            tb.add_cell(i, j, text=val,
                        width=per_width, height=per_height,
                        loc='center')
        tb.set_fontsize(20)
        ax.add_table(tb)
draw_policy(greedy_policy)
#plt.savefig('q6_greedy_policy')
plt.show()