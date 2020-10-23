import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open('ten_runs_01_new.npy', 'rb') as f:
    reward_record= np.load(f)

average = reward_record.mean(axis=0)

std_error = reward_record.std(axis=0)/np.sqrt(reward_record.shape[0])

best_return = np.max(reward_record)

print(best_return)

fig,ax2 = plt.subplots(ncols=1,nrows=1,figsize=(5,5))

color=np.random.rand(3)
line1, =ax2.plot(average,color='r')#q=0,e=01
ax2.fill_between(np.arange(0,10000,1),average-1.98*std_error,average+1.98*std_error,alpha=0.5,facecolor='g')
line2, =ax2.plot([-10,10000],[best_return,best_return],color=color,linewidth=1,linestyle='--')

ax2.set_yticks(np.linspace(0,1,5))
ax2.set_title('10 runs episolon=0.1,goal=(10,10)')
ax2.set_xlabel('episode numner')
ax2.set_ylabel('discounted return')
ax2.set_xlim(-10,10000)
ax2.legend([line1,line2],['epsilon=0.1','the best return'])#,bbox_to_anchor=(0.,0.2),loc='upper left'
#plt.savefig('4a1.png')
plt.show()
