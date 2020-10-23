import numpy as np
import random
import matplotlib.pyplot as plt
with open('data/q_star.npy', 'rb') as f:
    data= np.load(f)

print(data,data.mean())

def action2reward(a,q_star):
    #take action a 0:9, return the reward for the action
    a =int(a)
    return np.random.normal(q_star[a],1)

best_action= np.argmax(data)
print('best action',best_action)



epsilon1=0.1
epsilon2=0.01
total_step=1000
num_run = 2000
action_record=np.zeros((num_run,total_step))
reward_record = np.zeros((num_run,total_step))

for j in range(num_run):
    n_a = np.zeros(10)
    q = np.zeros(10)
    for i in  range(total_step):
        #print(i)
        rand1= random.uniform(0,1)
        if rand1<0:
            a = random.randint(0,9)
        else:
            a = np.argmax(q)
        #print(a)
        r = action2reward(a,q_star=data)
        n_a[a] += 1
        q[a] = q[a] + (1/n_a[a])*(r-q[a])
        action_record[j,i]=a
        reward_record[j,i]=r
'''
#print(q,n_a)
with open('greedy_reward_record.npy', 'wb') as file:
    np.save(file, reward_record)
with open('greedy_action_record.npy', 'wb') as file:
    np.save(file, action_record)
'''
