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
total_step=10000
num_run = 2000

increment = np.zeros((total_step,10))
for i in range(10):
    increment[:,i]=np.random.normal(0,0.01,total_step)


action_record=np.zeros((num_run,total_step))
reward_record = np.zeros((num_run,total_step))
action_record_average=np.zeros((num_run,total_step))
reward_record_average = np.zeros((num_run,total_step))
best_reward_record = np.zeros((num_run,total_step))
best_action_record = np.zeros((num_run,total_step))

for j in range(num_run):
    q_star = np.zeros(10)
    n_a = np.zeros(10)
    q = np.zeros(10)
    q_c=np.zeros(10)
    for i in  range(total_step):
        #print(i)
        rand1= random.uniform(0,1)
        if rand1<0.1:
            a = random.randint(0,9)
        else:
            a = np.argmax(q)
        #print(q_star,'\n')
        r = action2reward(a,q_star)
        #=======================#
        #        average        #
        n_a[a] += 1
        q[a] = q[a] + (1/n_a[a])*(r-q[a])
        #        average        #
        # ======================#

        #=======================#
        rand = random.uniform(0, 1)
        if rand < 0.1:
            a_c = random.randint(0, 9)
        else:
            a_c = np.argmax(q_c)
        #print(q_star, '\n')
        r_c = action2reward(a_c, q_star)
        #   constant stepsize   #
        q_c[a_c] = q_c[a_c] + 0.1*(r_c-q_c[a_c])
        # =======================#
        action_record_average[j,i]=a
        reward_record_average[j,i]=r
        action_record[j,i]=a_c
        reward_record[j,i]=r_c
        best_reward_record[j,i]=np.max(q_star)
        best_action = np.argmax(q_star)
        best_action_record[j,i]=best_action
        q_star = q_star + increment[i]

#print(reward_record,best_reward_record)
print(q,q_c,q_star)

'''
with open('nonstationary_reward_record_a.npy', 'wb') as file:
    np.save(file, reward_record_average)
with open('nonstationary_action_record_a.npy', 'wb') as file:
    np.save(file, action_record_average)
with open('nonstationary_reward_record_c.npy', 'wb') as file:
    np.save(file, reward_record)
with open('nonstationary_action_record_c.npy', 'wb') as file:
    np.save(file, action_record)
with open('best_qstar_record.npy','wb') as file:
    np.save(file,best_reward_record)
with open('best_action_record.npy','wb') as file:
    np.save(file,best_action_record)
'''
