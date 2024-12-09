import sys
import glob
import tables as tb
import numpy as np
import matplotlib.pyplot as plt

###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)

location = sys.argv[1]
picname = sys.argv[2]

dir_files = glob.glob(location+"*.h5")

for i in dir_files:
    print(i)
inputlist = sorted(dir_files)

matrices = []

for file in inputlist:

    imatrix = np.zeros((256,256), dtype=np.uint16)
    cnt=0
    with tb.open_file(file, 'r') as f:
       
        print(f"Reading {f}")
        groups = f.walk_groups('/')
        grouplist = []
        for gr in groups:
            #print(f'found {gr}')
            grouplist.append(gr)
        main_group = str(grouplist[len(grouplist)-1])
        #print(f"last entry in walk_groups = \n{main_group}")
        grouplist = None 

        basewords = main_group.split('(')
        #print(basewords)

        base_group_name = basewords[0][:-1]+'/'

        x = f.get_node(base_group_name+"x")[:]
        y = f.get_node(base_group_name+"y")[:]

        for xpos, ypos in zip(x,y):
            np.add.at(imatrix, (xpos,ypos), 1)
            progress(len(x),cnt)
        matrices.append(imatrix)
        imatrix = None
        cnt+=1
         

combinedMat = np.sum(matrices, axis=0)

fig, ax = plt.subplots(figsize=(12,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
#ms = ax.matshow(combinedMat, cmap='hot')
ms = ax.matshow(combinedMat.T, cmap='viridis')
fig.colorbar(ms,cax=cax,orientation='vertical', label="occupancy")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.xaxis.set_label_position('top') 
ax.invert_yaxis()

plt.savefig(f"matrix-Accumulated-{picname}.png", dpi=400)

       
