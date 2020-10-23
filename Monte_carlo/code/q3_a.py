import gym
from gym import envs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import argparse

env = gym.make('Blackjack-v0')
#print(env.action_space)
#print(env.observation_space)
v = np.zeros((2,21,10)) #usable ace or not * players'sum * dealers' show card
n = np.zeros((2,21,10))

#===========change the episode number here===================#
for i_episode in range(10000):
    trajetory =[]
    observation = env.reset()
    #print(observation)
    for t in range(22):
        trajetory.append(observation)
        index1= observation[0]
        index2 = observation[1]
        index3 = observation[2]*1
        #print(index3)
        if index1<20:
            action =1
        else:
            action=0
        observation, reward, done, info = env.step(action)
        trajetory.append(action)
        trajetory.append(reward)

        if done:
            num_state =len(trajetory)//3
            G=0
            for i in range(num_state):
                (index1,index2,index3) = trajetory[-3*(i+1)]
                index3=index3*1
                r = trajetory[-3*(i+1)+2]
                #print(r)
                G = trajetory[-1]
                #print(G)
                n[index3,index1-1,index2-1] += 1
                v[index3, index1-1, index2-1] += (1/n[index3,index1-1,index2-1])*(G-v[index3, index1-1, index2-1])
            #print("Episode finished after {} timesteps".format(t+1))
            break

np.save('jack_1.npy', v)
env.close()