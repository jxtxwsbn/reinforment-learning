import numpy as np
import matplotlib.pyplot as plt

with open('data/testbed.npy', 'rb') as f:
    data= np.load(f)

fig,ax= plt.subplots()
ax.violinplot(data.transpose((1,0)),showmeans=True)
ax.get_xaxis().set_tick_params(direction='out')
#ax.xaxis.set_ticks_position('bottom')
labels=[i+1 for i in range(10)]
ax.set_xticks(np.arange(1, len(labels) + 1))
#ax.set_yticks
#ax.set_xticklabels(labels)
ax.set_xlabel('Action')
ax.set_ylabel('Reward distribution')
#plt.savefig('violin_testbed')
plt.show()
