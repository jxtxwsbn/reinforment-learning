import numpy as np
import random
import matplotlib.pyplot as plt
with open('data/q_star.npy', 'rb') as f:
    q_star= np.load(f)

print(q_star,q_star.mean())

def action2reward(a,q_star):
    #take action a 0:9, return the reward for the action
    a =int(a)
    return np.random.normal(q_star[a],1)



total_step=1000
num_run = 2000
action_record=np.zeros((num_run,total_step))
reward_record = np.zeros((num_run,total_step))

for j in range(num_run):
    n_a = np.zeros(10)
    q = np.zeros(10)
    for i in range(10):
        a =i
        r = action2reward(a, q_star)
        n_a[a] += 1
        q[a] = q[a] + (1 / n_a[a]) * (r - q[a])
        action_record[j, i] = a
        reward_record[j, i] = r
    #print('n_a',n_a)

    for i in range(10,total_step):
        a = np.argmax(q+2*np.sqrt(np.divide(np.log(i),n_a)))
        #print(a)
        r = action2reward(a,q_star)
        n_a[a] += 1
        q[a] = q[a] + (1/n_a[a])*(r-q[a])
        action_record[j,i]=a
        reward_record[j,i]=r
#print(action_record)
'''
with open('reward_record_ucb.npy', 'wb') as file:
    np.save(file, reward_record)
with open('action_record_ucb.npy', 'wb') as file:
    np.save(file, action_record)
'''
