import numpy as np
import sys
import matplotlib.pyplot as plt

infiles = sys.argv[1:]

nfiles = len(infiles)
#print(infiles)
print(nfiles)

xpos = []
ypos = []
gains = []

skip_these = ['FEC', 'Board', 'Chip']

matrix_hits = np.zeros((256,256),dtype=np.uint16)
temp_mat = np.zeros((256,256),dtype=np.uint16)

#eventlist = [np.zeros((256,256),dtype=np.uint16) for i in range(10)]

#print(f"eventlist is {eventlist} with {len(eventlist)}")

randarr = np.random.randint(0,10000,size=10)

print(randarr)
#exit(0)

devx = []
devy = []

ifile = 0 
for file in infiles:
    f = open(file)
    nlines = 0
    #ifile_x, ifile_y, ifile_q = [], [], []
    ifile_x, ifile_y = [], []
    for line in f:
        words = line.split()
        if(words[0] in skip_these):
            continue
        else:      
            x = int(words[0])
            y = int(words[1])
            q = int(words[2])
            xpos.append(x)
            ypos.append(y)
            gains.append(q)
            if(nlines // 100):
                print(f'{x} : {y} = {q}')
            if(ifile == 3426):
                matrix_hits[x,y] += q
            if(ifile in randarr):
                temp_mat[x, y] += q
            
            ifile_x.append(x)
            ifile_y.append(y)
            #ifile_q.append(q)

        nlines+=1
    # ------------------------------------
    if(np.count_nonzero(temp_mat)>0):
        fig, ax = plt.subplots()
        cax = fig.add_axes([0.86,0.1,0.05,0.8])
        ms = ax.matshow(matrix_hits, cmap='plasma')
        fig.colorbar(ms,cax=cax,orientation='vertical')
        plt.plot()
        fig.savefig(f"simulated-gain-vsxy-position-{ifile}.png")
        ##
        temp_mat = np.zeros((256,256),dtype=np.uint16)
    # ------------------------------------    
    devx.append(np.std(np.array(ifile_x)))
    devy.append(np.std(np.array(ifile_y)))
    # ------------------------------------    
    ifile += 1

# ---- xy pos in scatter ----
plt.figure(0)
plt.scatter(xpos,ypos)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,256)
plt.ylim(0,256)
plt.savefig('ebala-10k-xy.png')

# ---- gain in scatter ---- 
plt.figure(1)
plt.hist(gains, 100, range=(0,50000), histtype='step', facecolor='r')
plt.savefig('ebala-10k-gains.png')
# -------------------------------
plt.figure(2)
plt.hist(devx, 100, range=(0,100), histtype='step', facecolor='g')
plt.savefig('ebala-devx.png')

plt.figure(3)
plt.hist(devy, 100, range=(0,100), histtype='step', facecolor='b')
plt.savefig('ebala-devy.png')

## ---- matrix 2d hist ----
#fig, ax = plt.subplots()
#cax = fig.add_axes([0.86,0.1,0.05,0.8])
#ms = ax.matshow(matrix_hits, cmap='plasma')
#fig.colorbar(ms,cax=cax,orientation='vertical')
#plt.plot()
#fig.savefig("simulated-gain-vsxy-position-3426.png")
#



