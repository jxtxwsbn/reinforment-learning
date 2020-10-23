import numpy as np


q_star = np.random.normal(0,1,10)

print(q_star)
'''
with open('q_star.npy', 'wb') as f:
    np.save(f, q_star)

testbed_data = np.zeros((10,10000))
for i in range(10):
    testbed_data[i]=np.random.normal(q_star[i],1,10000)

with open('testbed.npy', 'wb') as file:
    np.save(file, testbed_data)
'''


