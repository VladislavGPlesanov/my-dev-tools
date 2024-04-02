import numpy as np
from collections import deque

def to_numpy(list_in):

    return np.asarray(list_in, dtype=np.uint32)

###################

mylist = []

dq = deque()

for i in range(0,10):
    mylist.append(i)
    print("mylist={}".format(len(mylist)))
    #dq.extend(to_numpy(mylist))
    dq.append(to_numpy(mylist))
    print(dq)
    print("deque={}".format(len(dq)))
    #print(dq.popleft())
    #print("deque={}".format(len(dq)))
    #print(dq)








