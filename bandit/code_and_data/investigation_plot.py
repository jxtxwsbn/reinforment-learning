import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


with open('data/01_b_reward_record.npy', 'rb') as f2:
    reward_record_01b= np.load(f2)
with open('data/01_b_action_record.npy', 'rb') as f2:
    action_record_01b= np.load(f2)

with open('data/01_reward_record.npy', 'rb') as file:
    reward_record_01= np.load(file)
with open('data/01_action_record.npy', 'rb') as file:
    action_record_01= np.load(file)

with open('data/04_b_reward_record.npy', 'rb') as f:
    reward_record_04b= np.load(f)
with open('04_b_action_record.npy', 'rb') as f:
    action_record_04b= np.load(f)

with open('data/04_reward_record.npy', 'rb') as file:
    reward_record_04= np.load(file)
with open('data/04_action_record.npy', 'rb') as file:
    action_record_04= np.load(file)




with open('data/q_star_4.npy', 'rb') as f:
    q_star= np.load(f)

print(q_star,np.argmax(q_star))



average_01b = reward_record_01b.mean(axis=0)
average_01 = reward_record_01.mean(axis=0)

average_04b = reward_record_04b.mean(axis=0)
average_04 = reward_record_04.mean(axis=0)



std_error_01b = reward_record_01b.std(axis=0)/np.sqrt(reward_record_01b.shape[0])
std_error_01 = reward_record_01.std(axis=0)/np.sqrt(reward_record_01.shape[0])
std_error_04b = reward_record_04b.std(axis=0)/np.sqrt(reward_record_04b.shape[0])
std_error_04 = reward_record_04.std(axis=0)/np.sqrt(reward_record_04.shape[0])




best_action_01b = np.asarray(action_record_01b==np.argmax(q_star))
best_action_01b = best_action_01b.sum(axis=0)/best_action_01b.shape[0]

best_action_01 = np.asarray(action_record_01==np.argmax(q_star))
best_action_01 = best_action_01.sum(axis=0)/best_action_01.shape[0]

best_action_04b = np.asarray(action_record_04b==np.argmax(q_star))
best_action_04b = best_action_04b.sum(axis=0)/best_action_04b.shape[0]

best_action_04 = np.asarray(action_record_04==np.argmax(q_star))
best_action_04 = best_action_04.sum(axis=0)/best_action_04.shape[0]


fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(10,5))


action2,= ax1.plot(best_action_01b*100,color='b')
action3,= ax1.plot(best_action_01*100,color='y')
action4,= ax1.plot(best_action_04b*100,color='r')
action5,= ax1.plot(best_action_04*100,color='g')


ax1.legend([action2,action3,action4,action5],['with baseline a=0.1','without baseline a=0.1','with baseline a=0.4','without baseline a=0.4'])#bbox_to_anchor=(0.2,0.2),loc='upper left'
ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
ax1.set_xticks(np.arange(0,1001,250))
ax1.set_yticks(np.linspace(0, 100 ,11))
ax1.set_title('optimal action')
ax1.set_xlabel('steps')
ax1.set_xlim(-10,1000)

color=np.random.rand(3)

line2, =ax2.plot([-10,1000],[np.max(q_star),np.max(q_star)],color=color,linewidth=1,linestyle='--')

line3,= ax2.plot(average_01b,color='b')
ax2.fill_between(np.arange(0,1000,1),average_01b-1.98*std_error_01b,average_01b+1.98*std_error_01b,alpha=0.5,facecolor='g')

line4,= ax2.plot(average_01,color='y')
ax2.fill_between(np.arange(0,1000,1),average_01-1.98*std_error_01,average_01+1.98*std_error_01,alpha=0.5,facecolor='g')

line5,= ax2.plot(average_04b,color='r')
ax2.fill_between(np.arange(0,1000,1),average_04b-1.98*std_error_04b,average_04b+1.98*std_error_04b,alpha=0.5,facecolor='g')

line6,= ax2.plot(average_04,color='g')
ax2.fill_between(np.arange(0,1000,1),average_04-1.98*std_error_04,average_04+1.98*std_error_04,alpha=0.5,facecolor='g')

#ax2.set_yticks(np.linspace(0,2,5))
ax2.set_title('average reward')
ax2.set_xlabel('steps')
ax2.set_xlim(-10,1000)
ax2.legend([line3,line4,line5,line6,line2],['with baseline a=0.1','without basline a=0.1','with baseline a=0.4','without basline a=0.4','the best reward=5.89'])#,bbox_to_anchor=(0.,0.2),loc='upper left'
plt.savefig('gradient.png')
plt.show()
