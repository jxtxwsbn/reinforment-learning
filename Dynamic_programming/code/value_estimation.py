import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

world_size = 5
action = np.array([[0, -1],
                    [0, 1],
                    [-1, 0],
                    [1, 0]])
probability = 0.25


def draw_grid_world(value_array):
    fig, ax = plt.subplots(1, 1)
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

    state_value = np.zeros((world_size,world_size))
    #print(state_value,state_value.shape)
    iteration_time = 0

    while True:
        old_state_value = state_value.copy()

        for i in range(world_size):
            for j in range(world_size):
                iteration_value = 0
                for a in action:
                    state = np.array([i,j])
                    #print(state)
                    (next_x,next_y),reward = take_action(state,a)
                    #print(next_x,next_y)
                    iteration_value += probability * (reward + gamma * state_value[next_x, next_y])
                state_value[i,j] = iteration_value

        difference = abs(state_value-old_state_value).max()

        if difference<diff:
            break
        iteration_time += 1

    return state_value, iteration_time
value, iteration_times = estimate()
print('value for the policy:','\n',value)
draw_grid_world(np.round(value, decimals=3))
plt.savefig("q5_a")
plt.show()


