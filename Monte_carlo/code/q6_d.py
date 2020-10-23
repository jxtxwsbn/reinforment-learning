import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
greedy_policy = np.load('greedy_policy_q6.npy')
print(greedy_policy)

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
print(map.shape)
goal_postion = (10,10)
initial_position=(0,0)
(gx,gy)=goal_postion
(ngx,ngy)=(10-gy,gx)
map[ngx,ngy]=2
print(map)

def movement(action):
    #the dymanic of the environment
    rand1 = random.uniform(0,1)
    #print(rand1)
    change =None
    if action=='stay':
        change = (0,0)

    if action == 0:#'r'
        if rand1 < 0.8:
            change = (1, 0)
        elif 0.8 < rand1 < 0.9:
            change = (0, 1)
        else:
            change = (0, -1)

    if action ==1: # 'l'
        if rand1 < 0.8:
            change = (-1, 0)
        elif 0.8 < rand1 < 0.9:
            change = (0, 1)
        else:
            change = (0, -1)

    if action ==2:# 'up'
        if rand1 < 0.8:
            change = (0, 1)
        elif 0.8 < rand1 < 0.9:
            change = (1, 0)
        else:
            change = (-1, 0)

    if action ==3:# 'd'
        if rand1 < 0.8:
            change = (0, -1)
        elif 0.8 < rand1 < 0.9:
            change = (1, 0)
        else:
            change = (-1, 0)

    return change

def env_feedback(position,action):
    done = False
    (x,y) = position
    (vx,vy)=action
    (x1,y1)=(x+vx,y+vy)
    (nx1,ny1)=(int(10-y1),int(x1))
    reward =0
    if 10>=nx1>=0 and 10>=ny1>=0 and map[nx1,ny1]!=1:
        next_pos = (x1,y1)
        if map[nx1,ny1] == 2:
            reward=1
            done=True
    else:
        next_pos = position
    return reward,next_pos,done


def get_first_index(state,record):
    for i in range(len(record)):
        if isinstance(record[i],tuple):
            if state == record[i]:
                return i
        else:
            continue


q_value = np.zeros((11,11,4))
#q_value[map==1,:]=-np.inf

action_all = [(1,0),(-1,0),(0,1),(0,-1)]
n_matrix = np.zeros((11,11,4))

episode_num =0
#episode_length =[]
policy = greedy_policy.copy()
#dic ={}
Return=[]
while episode_num<10000:
    position = (0,0)
    step =0
    record=[]
    for i_step in range(459):
        record.append(position)
        (x,y)=position
        (nx,ny)=(int(10-y),int(x))
        action_index = int(policy[nx,ny])

        action = movement(action_index)
        reward, next_pos, done = env_feedback(position,action)
        record.append(action_index)
        record.append(reward)
        position = next_pos
        step +=1
        if done:
            break
        #if step > 459:
            #break
    print(len(record)//3,record[-1])
    #print(step)
    G=0
    total_state = len(record)//3
    for i in range(total_state):
        id = i+1
        reward = record[-id*3+2]
        action_index = int(record[-id*3+1])
        position = record[-id*3]
        (x,y)=position
        (nx,ny)=(int(10-y),int(x))
        G = 0.99*G + reward
        first_index=get_first_index(position,record)//3

        position_index = total_state-i-1

        if position_index==first_index:
            n_matrix[nx,ny,action_index] +=1
            q_value[nx,ny,action_index] += (1/ n_matrix[nx,ny,action_index])*(G-q_value[nx,ny,action_index])

    Return.append(G)
    episode_num +=1
plt.plot(Return,color="b", linestyle="-", linewidth=1)
plt.show()
print(q_value)
np.save('q6d_on_policy_q_value',q_value)