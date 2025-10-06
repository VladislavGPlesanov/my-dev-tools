import sys
import glob
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from time import sleep
import matplotlib.patches as patch

def getBaseGroupName(file):

    print(f"Reading {f}")
    groups = file.walk_groups('/')
    grouplist = []
    for gr in groups:
        #print(f'found {gr}')
        grouplist.append(gr)
    main_group = str(grouplist[len(grouplist)-1])
    print(f"last entry in walk_groups = \n{main_group}")
    grouplist = None 
    
    basewords = main_group.split('(')
    print(basewords)
    
    base_group_name = basewords[0][:-1]+'/'

    return base_group_name

###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

fbefore = sys.argv[1]
fafter = sys.argv[2]
picname = sys.argv[3]

mat_before = np.zeros((256,256),dtype=np.uint16)
mat_after = np.zeros((256,256),dtype=np.uint16)

with tb.open_file(fbefore) as f:

    #print(f"Reading {f}")
    #groups = f.walk_groups('/')
    #grouplist = []
    #for gr in groups:
    #    print(f'found {gr}')
    #    grouplist.append(gr)
    #main_group = str(grouplist[len(grouplist)-1])
    #print(f"last entry in walk_groups = \n{main_group}")
    #grouplist = None 
    #
    #basewords = main_group.split('(')
    #print(basewords)
    #
    #base_group_name = basewords[0][:-1]+'/'
    bgname = getBaseGroupName(f)

    cluster_x = f.get_node(bgname+"x")
    cluster_y = f.get_node(bgname+"y")
    ToT = f.get_node(bgname+"ToT")

    for xpos, ypos, tot in zip(cluster_x, cluster_y, ToT):
        np.add.at(mat_before, (xpos,ypos), 1)            

print(f"Processed FILE before: {fbefore}")

with tb.open_file(fafter) as f:

    bgname = getBaseGroupName(f)

    cluster_x = f.get_node(bgname+"x")
    cluster_y = f.get_node(bgname+"y")
    ToT = f.get_node(bgname+"ToT")

    for xpos, ypos, tot in zip(cluster_x, cluster_y, ToT):
        np.add.at(mat_after, (xpos,ypos), 1)            


print(f"Processed FILE after: {fafter}")

mat_difference = mat_before - mat_after

norm_before_mat = mat_before / np.sum(mat_before)
norm_after_mat = mat_after / np.sum(mat_after)

print("PLOTTING")

fig, ax = plt.subplots(figsize=(12,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
ms = ax.matshow(norm_before_mat, cmap='gist_earth_r')
fig.colorbar(ms,cax=cax,orientation='vertical', label="occupancy")
ax.set_title("Matrix Before irradiation")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.xaxis.set_label_position('top') 
ax.invert_yaxis()
plt.savefig(f"DESY-P09-FE55-matrix-before-{picname}.png", dpi=400)
ms = None
plt.close()

fig, ax = plt.subplots(figsize=(12,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
ms = ax.matshow(norm_after_mat, cmap='gist_earth_r')
fig.colorbar(ms,cax=cax,orientation='vertical', label="occupancy")
ax.set_title("Matrix After irradiation")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.xaxis.set_label_position('top') 
ax.invert_yaxis()
plt.savefig(f"DESY-P09-FE55-matrix-after-{picname}.png", dpi=400)
ms = None
plt.close()





