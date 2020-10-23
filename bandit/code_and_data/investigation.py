import numpy as np
import random
import matplotlib.pyplot as plt
'''
q_star_4 = np.random.normal(4,1,10)
with open('q_star_4.npy', 'wb') as file:
    np.save(file, q_star_4)
print(q_star_4,q_star_4.mean(),np.argmax(q_star_4))

'''

def get_cdf(density):
    cdf = np.zeros(len(density))
    for i in range(len(density)):
        for j in range(i+1):
            cdf[i] += density[j]
    return cdf

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x/np.sum(exp_x)
    return softmax_x

def sample_action(p):
    rand = np.random.uniform(0, 1)
    for i in range(len(p)):
        if p[i]>rand:
            action = i
            break
    return action



with open('data/q_star_4.npy', 'rb') as f:
    q_star= np.load(f)

print(q_star,q_star.mean())

def action2reward(a,q_star):
    #take action a 0:9, return the reward for the action
    a =int(a)
    return np.random.normal(q_star[a],1)

best_action= np.argmax(q_star)
print('best action',best_action)


epsilon1=0.1
total_step=1000
num_run = 2000
action_record=np.zeros((num_run,total_step))
reward_record = np.zeros((num_run,total_step))

for j in range(num_run):
    #n_a = np.zeros(10)
    h = np.zeros(10)
    r_average =0
    for i in range(total_step):
        density = softmax(h)
        #print(density)
        #cdf = get_cdf(density)
        #print(cdf)
        #a = sample_action(cdf)
        a= int(np.random.choice(np.linspace(0,9,10),p=density))
        r = action2reward(a,q_star)
        for k in range(len(h)):
            if k==a:
                h[k] = h[k] + 0.4*(r-r_average)*(1-density[k])
            else:
                h[k] = h[k] - 0.4*(r-r_average)*density[k]
        n = i+1
        #r_average = r_average + (1/n)*(r-r_average)

        action_record[j,i]=a
        reward_record[j,i]=r

print(density)

with open('04_reward_record.npy', 'wb') as file:
    np.save(file, reward_record)
with open('04_action_record.npy', 'wb') as file:
    np.save(file, action_record)
