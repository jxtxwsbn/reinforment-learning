import numpy as np
import random
import matplotlib.pyplot as plt
import json
import pickle
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

def break_tie(q_value):
    max_value = max(q_value)
    best_mask = (q_value==max_value)
    best_indexs =[]
    for f in range(len(best_mask)):
        if best_mask[f]*1==1:
            best_indexs.append(f)
    #print(best_indexs)
    #best = random.choice(best_indexs)
    return best_indexs


goal_postion = (10,10)
initial_position=(0,0)
(gx,gy)=goal_postion
(ngx,ngy)=(10-gy,gx)
map[ngx,ngy]=2
print(map)

def movement(action):
    #the dymanic of the environment
    rand1 = random.uniform(0,1)
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

def action_from_policy(position,policy,actions):
    (x,y) = position
    (nx,ny)=(10-y,x)
    action_index = np.random.choice(4, p=policy[nx, ny, :])
    #action_index = action_index[0]
    action = actions[action_index]
    return action,action_index
def get_pro(position,action_index,policy):
    (x, y) = position
    (nx, ny) = (10 - y, x)
    prob = policy[nx,ny,action_index]
    return prob
q_value_initial = np.zeros((11,11,4))
action_all = [(1,0),(-1,0),(0,1),(0,-1)]
initial_policy = np.zeros((11,11,4))+0.1
#initial_policy[map==1,:]
initial_policy[:,:,0]+=0.3
initial_policy[:,:,2]+=0.3

ten_runs =np.zeros((10,10000))

for run_num in range(1):
    print('run+++++')
    Return = []
    episode_num = 0
    episode_length = []
    policy=initial_policy.copy()
    q_value = q_value_initial.copy()
    n_matrix = np.zeros((11, 11, 4)).copy()
    #print(policy[:,:,0])
    dic = {}
    while episode_num<10000:
        position = (0,0)
        record =[]
        step=0
        for i_step in range(459):
            record.append(position)
            action,action_index = action_from_policy(position,policy,actions=action_all)
            action = movement(action_index)
            probablity = get_pro(position,action_index,policy)
            reward, next_pos, done = env_feedback(position,action)
            record.append(action_index)
            record.append(reward)
            record.append(probablity)
            position = next_pos
            step +=1
            if done:
                break

        episode_length.append(step)
        #print(len(record)//3,record[-1])
        G=0
        total_state = len(record)//4
        #print('record/3',len(record)/3)
        for i in range(total_state):
            id = i+1
            reward = record[-id*4+2]
            action_index = int(record[-id*4+1])
            position = record[-id*4]
            (x,y)=position
            (nx,ny)=(int(10-y),int(x))
            G = 0.99*G + reward
            first_index =get_first_index(position,record)//4
            position_index = total_state-i-1

            #print('posotion_index',position_index)
            #print('first index',first_index)
            if position_index==first_index:
                n_matrix[nx,ny,action_index]+=1
                q_value[nx,ny,action_index]+=(1/ n_matrix[nx,ny,action_index])*(G-q_value[nx,ny,action_index])
                best_indexs = break_tie(q_value[nx,ny,:])
                #print('best index',best_indexs)
                p = 0.1/4
                for k in range(4):
                    policy[nx,ny,k]=p
                for j in range(len(best_indexs)):
                    best = best_indexs[j]
                    policy[nx,ny,best]+=0.9/(len(best_indexs))
        Return.append(G)
        dic[episode_num]=record
        episode_num+=1


print(len(Return))
plt.plot(Return,color="b", linestyle="-", linewidth=1)
plt.show()


#print(dic)

with open('episode.p','wb') as f:
    pickle.dump(dic,f)
np.save('policy_q6a',policy)
np.save('q_value_q6a',q_value)


