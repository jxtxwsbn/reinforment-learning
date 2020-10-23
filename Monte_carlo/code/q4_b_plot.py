import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open('ten_runs_01_new.npy', 'rb') as f:
    reward_record= np.load(f)
with open('ten_runs_001.npy', 'rb') as f:
    reward_record_01= np.load(f)
with open('ten_runs_00.npy', 'rb') as f:
    reward_record_0= np.load(f)

average = reward_record.mean(axis=0)
average_01 = reward_record_01.mean(axis=0)
average_0 = reward_record_0.mean(axis=0)

std_error = reward_record.std(axis=0)/np.sqrt(reward_record.shape[0])
std_error_01 = reward_record_01.std(axis=0)/np.sqrt(reward_record_01.shape[0])
std_error_0 = reward_record_0.std(axis=0)/np.sqrt(reward_record_0.shape[0])

best_return = np.max(reward_record)

print(best_return)

fig,ax2 = plt.subplots(ncols=1,nrows=1,figsize=(5,5))

color=np.random.rand(3)
line1, =ax2.plot(average,color='r')
ax2.fill_between(np.arange(0,10000,1),average-1.98*std_error,average+1.98*std_error,alpha=0.5,facecolor='g')
line2, =ax2.plot([-10,10000],[best_return,best_return],color=color,linewidth=1,linestyle='--')
line3, =ax2.plot(average_01,color='b')
ax2.fill_between(np.arange(0,10000,1),average_01-1.98*std_error_01,average_01+1.98*std_error_01,alpha=0.5,facecolor='g')
line4, =ax2.plot(average_0,color='y')
ax2.fill_between(np.arange(0,10000,1),average_0-1.98*std_error_0,average_0+1.98*std_error_0,alpha=0.5,facecolor='g')



ax2.set_yticks(np.linspace(0,1,5))
ax2.set_title('10 runs goal=(10,10)')
ax2.set_xlabel('episode number')
ax2.set_ylabel('discounted return')
ax2.set_xlim(-10,10000)
ax2.legend([line1,line2,line3,line4],['epsilon=0.1','the best return','epsilon=0.01','epsilon=0'])#,bbox_to_anchor=(0.,0.2),loc='upper left'
#plt.savefig('4b.png')
plt.show()
