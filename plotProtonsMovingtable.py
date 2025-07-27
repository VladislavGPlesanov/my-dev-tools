import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import numpy.ma as ma

picname = sys.argv[1]

filepath = '/home/vlad/IonTestBeams/ParticleSeparation/raw_data/'
files = ['data_250514_141251.csv',  
         'data_250514_142103.csv',
         'data_250514_142339.csv',
         'data_250514_142535.csv',
         'data_250514_142702.csv',
         'data_250514_142852.csv']

tablePosiitons = [285, 274, 264, 254, 244, 234]


#plt.figure()
fig = plt.figure(figsize=(8,8))
gs = GridSpec(6,1,hspace=0.1)

axes = [fig.add_subplot(gs[i,0]) for i in range(6)]

cnt=0
for file, ax in zip(files,axes):
    f = open(filepath+file, 'r')
    
    intLong = []
    
    for line in f:
    
        if('timestamp' in line):
            continue
    
        words = line.split(',')
    
        nonempty = np.nonzero(np.array(words))
    
        #print(nonempty)
        #print(nonempty[0][1])
    
        idx_time = nonempty[0][0]
        idx_long = nonempty[0][1]
        idx_short = nonempty[0][2]
    
        t = float(words[idx_time])
        ilong = float(words[idx_long])
        ishort = float(words[idx_short])

        if ((ilong-ishort)/ilong > 0):
            intLong.append(ilong)

    #plt.hist(intLong, bins=100, histtype='step',label=f'x={tablePosiitons[cnt]}')
    ax.hist(intLong, bins=100, histtype='step',label=f'x={tablePosiitons[cnt]}')
    ax.set_xlim([0,25000]) 
    #ax.set_ylim([0,5e5])
    #ax.set_yscale('log')
    if(cnt<5):
        ax.tick_params(axis='x',labelbottom=False)
    ax.legend()
    
    cnt+=1

plt.tight_layout()
plt.xlabel('long')
#plt.ylabel('N counts')
#plt.legend()

#plt.title('peakProgression')
plt.savefig(f'peakProgression-{picname}.png')


