import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

world_size = 5
action = np.array([[0, 1],
                    [0,-1],
                    [-1, 0],
                    [1, 0]])
#probability = 0.25


def draw_grid_world(value_array):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    num_rows, num_cols = value_array.shape
    per_height, per_width = 1.0 / num_rows, 1.0 / num_cols

    for (i, j), val in np.ndenumerate(value_array):

        tb.add_cell(i, j, text=val,
                    width=per_width, height=per_height,
                    loc='center')

    tb.set_fontsize(28)
    ax.add_table(tb)

def draw_policy(dic):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    num_rows, num_cols = (5,5)
    per_height, per_width = 1.0 / num_rows, 1.0 / num_cols

    for (i, j), val in dic.items():
        #print(i,j,val)
        if np.array_equal(np.array([-1,0]),val):
            val='up'
        if np.array_equal(np.array([1,0]),val):
            val='down'
        if np.array_equal(np.array([0,1]),val):
            val='right'
        if np.array_equal(np.array([0,-1]),val):
            val='left'
        tb.add_cell(i, j, text=val,
                    width=per_width, height=per_height,
                    loc='center')
        tb.set_fontsize(20)
        ax.add_table(tb)


def take_action(state,action):
    #take the state and action as input and output the next state and reward
    A = np.array([0,1])
    B = np.array([0,3])
    A_ = np.array([4,1])
    B_ = np.array([2,3])

    next_state = state + action
    if np.array_equal(state,A):
        reward = 10
        next_state = A_

        return next_state,reward

    if np.array_equal(state,B):

        return B_,5

    if np.max(next_state) > 4 or np.min(next_state)<0:
        next_state = state
        reward = -1
        return next_state,reward

    else:
        return next_state,0


def estimate(gamma=0.9,diff=1e-3,):
    probability=np.zeros((5,5,4))+0.25
    state_value = np.zeros((world_size,world_size))
    #print(state_value,state_value.shape)
    iteration_time = 0
    dic = {}

    while True:
        old_state_value = state_value.copy()

        for i in range(world_size):
            for j in range(world_size):
                iteration_value = 0
                for k,a in enumerate(action):
                    state = np.array([i,j])
                    (next_x,next_y),reward = take_action(state,a)
                    iteration_value += probability[i,j,k] * (reward + gamma * state_value[next_x, next_y])
                state_value[i,j] = iteration_value

        difference = abs(state_value-old_state_value).max()
        if difference<diff:
            probability_old = probability.copy()
            probability = np.zeros_like(probability)
            for m in range(world_size):
                for n in range(world_size):
                    value_list = np.zeros((4,1))
                    for k,a in enumerate(action):
                        state = np.array([m,n])
                        (next_x, next_y), reward = take_action(state, a)
                        value = reward + gamma * state_value[next_x, next_y]
                        value_list[k] = value
                    fb = np.argmax(value_list)
                    probability[m,n,fb]=1
                    dic[m,n]=action[fb]
            if np.array_equal(probability_old,probability):
                    break
        iteration_time += 1

    return state_value, iteration_time,dic
value, iteration_times,dic = estimate()
print(value)
draw_grid_world(np.round(value, decimals=3))
plt.savefig('q5_c1')
plt.show()
draw_policy(dic)
plt.savefig('q5_c2')
plt.show()


