import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open('data/epi01_reward_record.npy', 'rb') as f:
    reward_record= np.load(f)
with open('data/epi01_action_record.npy', 'rb') as f:
    action_record= np.load(f)

with open('data/q5e0_reward_record.npy', 'rb') as f:
    reward_record_50= np.load(f)
with open('data/q5e0_action_record.npy', 'rb') as f:
    action_record_50= np.load(f)

with open('data/q5e01_reward_record.npy', 'rb') as f:
    reward_record_501= np.load(f)
with open('data/q5e01_action_record.npy', 'rb') as f:
    action_record_501= np.load(f)

with open('data/greedy_reward_record.npy', 'rb') as f:
    reward_record_000= np.load(f)
with open('data/greedy_action_record.npy', 'rb') as f:
    action_record_000= np.load(f)



with open('data/q_star.npy', 'rb') as f:
    q_star= np.load(f)



average = reward_record.mean(axis=0)
average_000 = reward_record_000.mean(axis=0)
average_50 = reward_record_50.mean(axis=0)
average_501 = reward_record_501.mean(axis=0)


std_error = reward_record.std(axis=0)/np.sqrt(reward_record.shape[0])
std_error_000 = reward_record_000.std(axis=0)/np.sqrt(reward_record_000.shape[0])
std_error_50 = reward_record_50.std(axis=0)/np.sqrt(reward_record_50.shape[0])
std_error_501 = reward_record_501.std(axis=0)/np.sqrt(reward_record_501.shape[0])



best_action = np.asarray(action_record==np.argmax(q_star))
best_action = best_action.sum(axis=0)/best_action.shape[0]

best_action_50 = np.asarray(action_record_50==np.argmax(q_star))
best_action_50 = best_action_50.sum(axis=0)/best_action_50.shape[0]

best_action_501 = np.asarray(action_record_501==np.argmax(q_star))
best_action_501 = best_action_501.sum(axis=0)/best_action_501.shape[0]

best_action_000 = np.asarray(action_record_000==np.argmax(q_star))
best_action_000 = best_action_000.sum(axis=0)/best_action_000.shape[0]



fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(10,5))

action1,=ax1.plot(best_action*100,color='r')
action2,= ax1.plot(best_action_50*100,color='b')
action4,= ax1.plot(best_action_501*100,color='y')
action3,= ax1.plot(best_action_000*100,color='g')

ax1.legend([action1,action2,action4,action3],['Q1=0,epsilon=0.1','Q1=5,epsilon=0','Q1=5,epsilon=0.1','greedy Q1=0,epsilon=0'])#bbox_to_anchor=(0.2,0.2),loc='upper left'
ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
ax1.set_xticks(np.arange(0,1001,250))
ax1.set_yticks(np.linspace(0, 100 ,11))
ax1.set_title('optimal action')
ax1.set_xlabel('steps')
ax1.set_xlim(-10,1000)

color=np.random.rand(3)
line1, =ax2.plot(average,color='r')#q=0,e=01
ax2.fill_between(np.arange(0,1000,1),average-1.98*std_error,average+1.98*std_error,alpha=0.5,facecolor='g')
line2, =ax2.plot([-10,1000],[np.max(q_star),np.max(q_star)],color=color,linewidth=1,linestyle='--')

line3,= ax2.plot(average_50,color='b')#q=5,e=0
ax2.fill_between(np.arange(0,1000,1),average_50-1.98*std_error_50,average_50+1.98*std_error_50,alpha=0.5,facecolor='g')

line4,= ax2.plot(average_000,color='g')#q=0,e=0
ax2.fill_between(np.arange(0,1000,1),average_000-1.98*std_error_000,average_000+1.98*std_error_000,alpha=0.5,facecolor='g')

line5,= ax2.plot(average_501,color='y')#q=5,e=01
ax2.fill_between(np.arange(0,1000,1),average_501-1.98*std_error_501,average_501+1.98*std_error_501,alpha=0.5,facecolor='g')

ax2.set_yticks(np.linspace(0,2,5))
ax2.set_title('average reward')
ax2.set_xlabel('steps')
ax2.set_xlim(-10,1000)
ax2.legend([line1,line3,line5,line4,line2],['Q1=0,epsilon=0.1','Q1=5,epsilon=0','Q1=5,epsilon=0.1','greedy Q1=0,epsilon=0','the best reward=1.67'])#,bbox_to_anchor=(0.,0.2),loc='upper left'
#plt.savefig('oi.png')
plt.show()
