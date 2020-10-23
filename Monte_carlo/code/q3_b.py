import gym
from gym import envs
import numpy as np
import random


env = gym.make('Blackjack-v0')
#print(env.action_space)
#print(env.observation_space)
q = np.zeros((2,21,10,2)) #usable ace or not * players'sum * dealers' show card
n = np.zeros((2,21,10,2))
policy = np.random.choice(2,(2,21,10))
#policy = np.zeros((2,10,10))
for i_episode in range(4000000):
    trajetory =[]
    observation = env.reset()
    #print('start_state',observation)
    start_action = np.random.choice(2,1)
    start_action = start_action[0]
    #print('start action',start_action)
    trajetory.append(observation)
    trajetory.append(start_action)

    (index1, index2, index3) = observation
    index3 = index3 * 1

    observation, reward, done, info = env.step(start_action)
    #trajetory.append(observation)
    trajetory.append(reward)
    #print(trajetory,done)
    if done:
        G = reward
        n[index3, index1 - 1, index2 - 1, start_action] += 1
        q[index3,index1 - 1,index2 - 1, start_action] += (1/n[index3,index1-1,index2-1,start_action])*(G-q[index3, index1-1, index2-1,start_action])
        policy[index3,index1 - 1,index2 - 1] = np.argmax(q[index3,index1 - 1,index2 - 1,:])
        continue
    else:
        for t in range(30):
            trajetory.append(observation)
            (index1, index2, index3) = observation
            index3 = index3*1
            action =  int(policy[index3,index1 - 1,index2 - 1])
            #print(action)
            observation, reward, done, info = env.step(action)
            trajetory.append(action)
            trajetory.append(reward)

            if done:
                #trajetory=trajetory[:-1]
                num_state =len(trajetory)//3
                #print('trajectory',trajetory,len(trajetory),len(trajetory)//3)
                G=0
                for i in range(num_state):
                    (index1, index2, index3) = trajetory[-3 * (i + 1)]
                    index3 = index3 * 1
                    #print(trajetory[-3 * (i + 1)])
                    r = trajetory[-3 * (i + 1) + 2]
                    #print(r)
                    action = trajetory[-3 * (i + 1) + 1]
                    #print(action)
                    G = 1 * G + r
                    #print(G)
                    n[index3, index1 - 1, index2 - 1,action] += 1
                    q[index3, index1 - 1, index2 - 1,action] += (1 / n[index3, index1 - 1, index2 - 1,action]) * (G - q[index3, index1 - 1, index2 - 1,action])
                    policy[index3, index1 - 1, index2 - 1] = np.argmax(q[index3, index1 - 1, index2 - 1, :])
                #print("Episode finished after {} timesteps".format(t + 1))
                break
np.save('jack_2', policy)
np.save('jack_2_value',q)
env.close()
