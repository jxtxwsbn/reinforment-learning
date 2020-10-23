import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open('data/nonstationary_reward_record_a.npy', 'rb') as f:
    reward_record_a= np.load(f)
with open('data/nonstationary_action_record_a.npy', 'rb') as f:
    action_record_a= np.load(f)

with open('data/nonstationary_reward_record_c.npy', 'rb') as f:
    reward_record_c= np.load(f)
with open('data/nonstationary_action_record_c.npy', 'rb') as f:
    action_record_c= np.load(f)

with open('data/best_qstar_record.npy', 'rb') as f:
    best_reward_record= np.load(f)
with open('data/best_action_record.npy', 'rb') as f:
    best_action_record= np.load(f)


average_a = reward_record_a.mean(axis=0)
average_c = reward_record_c.mean(axis=0)
average_b = best_reward_record.mean(axis=0)


std_error_a = reward_record_a.std(axis=0)/np.sqrt(reward_record_a.shape[0])
std_error_c = reward_record_c.std(axis=0)/np.sqrt(reward_record_c.shape[0])


best_action_a = np.asarray(action_record_a==best_action_record)
best_action_a = best_action_a.sum(axis=0)/best_action_a.shape[0]

best_action_c = np.asarray(action_record_c==best_action_record)
best_action_c = best_action_c.sum(axis=0)/best_action_c.shape[0]


#========================================================#
fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(10,5))

action1,=ax1.plot(best_action_a*100,color='r')
action2,= ax1.plot(best_action_c*100,color='b')

ax1.legend([action1,action2],['sample-average','constant step size'])#bbox_to_anchor=(0.2,0.2),loc='upper left'
ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
#ax1.set_xticks(np.arange(0,action_record_a.shape[1],1000))
ax1.set_yticks(np.linspace(0, 100 ,11))
ax1.set_title('optimal action')
ax1.set_xlabel('steps')
ax1.set_xlim(-10,action_record_a.shape[1])

color=np.random.rand(3)
#print(color)
line1, =ax2.plot(average_a,color='r',linewidth=0.3)
ax2.fill_between(np.arange(0,action_record_a.shape[1],1),average_a-1.98*std_error_a,average_a+1.98*std_error_a,alpha=0.9,facecolor='g')
line2, =ax2.plot(average_b,color=color,linewidth=1,linestyle='--')

line3,= ax2.plot(average_c,color='b',linewidth=0.3)
ax2.fill_between(np.arange(0,action_record_a.shape[1],1),average_c-1.98*std_error_c,average_c+1.98*std_error_c,alpha=0.9,facecolor='g')



#ax2.set_yticks(np.linspace(0,2,5))
ax2.set_title('average reward')
ax2.set_xlabel('steps')
ax2.set_xlim(-10,action_record_a.shape[1])
ax2.legend([line1,line3,line2],['sample-average','constant stepsize','the best reward'])#,bbox_to_anchor=(0.,0.2),loc='upper left'
#plt.savefig('nonstationary.png')
plt.show()
