import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open('data/epi01_reward_record.npy', 'rb') as f:
    reward_record= np.load(f)
with open('data/epi01_action_record.npy', 'rb') as f:
    action_record= np.load(f)

with open('data/reward_record_ucb.npy', 'rb') as f:
    reward_record_ucb= np.load(f)
with open('data/action_record_ucb.npy', 'rb') as f:
    action_record_ucb= np.load(f)


with open('data/q_star.npy', 'rb') as f:
    q_star= np.load(f)



average = reward_record.mean(axis=0)
average_ucb = reward_record_ucb.mean(axis=0)


std_error = reward_record.std(axis=0)/np.sqrt(reward_record.shape[0])
std_error_ucb = reward_record_ucb.std(axis=0)/np.sqrt(reward_record_ucb.shape[0])


best_action = np.asarray(action_record==np.argmax(q_star))
best_action = best_action.sum(axis=0)/best_action.shape[0]

best_action_ucb = np.asarray(action_record_ucb==np.argmax(q_star))
best_action_ucb = best_action_ucb.sum(axis=0)/best_action_ucb.shape[0]


fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(10,5))

action1,=ax1.plot(best_action*100,color='r')
action2,= ax1.plot(best_action_ucb*100,color='b')


ax1.legend([action1,action2],['epsilon=0.1','ucb c=2'])#bbox_to_anchor=(0.2,0.2),loc='upper left'
ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
ax1.set_xticks(np.arange(0,1001,250))
ax1.set_yticks(np.linspace(0, 100 ,11))
ax1.set_title('optimal action')
ax1.set_xlabel('steps')
ax1.set_xlim(-10,1000)

color=np.random.rand(3)
#print(color)
line1, =ax2.plot(average,color='r')
ax2.fill_between(np.arange(0,1000,1),average-1.98*std_error,average+1.98*std_error,alpha=0.5,facecolor='g')
line2, =ax2.plot([-10,1000],[np.max(q_star),np.max(q_star)],color=color,linewidth=1,linestyle='--')

line3,= ax2.plot(average_ucb,color='b')
ax2.fill_between(np.arange(0,1000,1),average_ucb-1.98*std_error_ucb,average_ucb+1.98*std_error_ucb,alpha=0.5,facecolor='g')



ax2.set_yticks(np.linspace(0,2,5))
ax2.set_title('average reward')
ax2.set_xlabel('steps')
ax2.set_xlim(-10,1000)
ax2.legend([line1,line3,line2],['epsilon=0.1','ucb c=2','the best reward=1.67'])#,bbox_to_anchor=(0.,0.2),loc='upper left'
#plt.savefig('ucb.png')
plt.show()


#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
#ax.set_xticks(np.arange(1, len(labels) + 1))
