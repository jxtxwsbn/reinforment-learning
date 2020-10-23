import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open('data/epi01_reward_record.npy', 'rb') as f:
    reward_record= np.load(f)

with open('data/epi01_action_record.npy', 'rb') as f:
    action_record= np.load(f)

with open('data/epi001_reward_record.npy', 'rb') as f:
    reward_record_001= np.load(f)

with open('data/epi001_action_record.npy', 'rb') as f:
    action_record_001= np.load(f)

with open('data/greedy_reward_record.npy', 'rb') as f:
    reward_record_000= np.load(f)

with open('data/greedy_action_record.npy', 'rb') as f:
    action_record_000= np.load(f)


with open('data/q_star.npy', 'rb') as f:
    q_star= np.load(f)


average = reward_record.mean(axis=0)
average_001 = reward_record_001.mean(axis=0)
average_000 = reward_record_000.mean(axis=0)


std_error = reward_record.std(axis=0)/np.sqrt(reward_record.shape[0])
std_error_001 = reward_record_001.std(axis=0)/np.sqrt(reward_record_001.shape[0])
std_error_000 = reward_record_000.std(axis=0)/np.sqrt(reward_record_000.shape[0])

best_action = np.asarray(action_record==np.argmax(q_star))
best_action = best_action.sum(axis=0)/best_action.shape[0]

best_action_001 = np.asarray(action_record_001==np.argmax(q_star))
best_action_001 = best_action_001.sum(axis=0)/best_action_001.shape[0]

best_action_000 = np.asarray(action_record_000==np.argmax(q_star))
best_action_000 = best_action_000.sum(axis=0)/best_action_000.shape[0]


fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(10,5))

action1,=ax1.plot(best_action*100,color='r')
action2,= ax1.plot(best_action_001*100,color='b')
action3,= ax1.plot(best_action_000*100,color='g')

ax1.legend([action1,action2,action3],['epsilon=0.1','epsilon=0.01','greedy'])#bbox_to_anchor=(0.2,0.2),loc='upper left'
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

line3,= ax2.plot(average_001,color='b')
ax2.fill_between(np.arange(0,1000,1),average_001-1.98*std_error_001,average_001+1.98*std_error_001,alpha=0.5,facecolor='g')

line4,= ax2.plot(average_000,color='g')
ax2.fill_between(np.arange(0,1000,1),average_000-1.98*std_error_000,average_000+1.98*std_error_000,alpha=0.5,facecolor='g')


ax2.set_yticks(np.linspace(0,2,5))
ax2.set_title('average reward')
ax2.set_xlabel('steps')
ax2.set_xlim(-10,1000)
ax2.legend([line1,line3,line4,line2],['epsilon=0.1','epsilon=0.01','greedy','the best reward=1.67'])#,bbox_to_anchor=(0.,0.2),loc='upper left'
plt.show()
