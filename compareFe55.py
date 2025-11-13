import sys
import glob
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lmfit import Model
from time import sleep
import matplotlib.patches as patch

def initMatrix(dtype):

    return np.zeros((256,256),dtype=dtype)

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

def plotGain(datalist, picname):

    fig, ax = plt.subplots(1,2,figsize=(12,6))

    cnt=0
    print("Plotting TOT/Pixel")
    for dl in datalist:

        #print("A")
        #unique_tot = np.sort(np.unique(dl[0]))
        #print("B")
        #bin_edges = np.concatenate((unique_tot-0.5, [unique_tot[-1]+0.5]))
        #print("C")
        #hist, bins = np.histogram(dl[0], bins=bin_edges, density=True)
        #print("D")
        #bin_centers = (bins[:-1] + bins[1:])/2
        #print("E")

        minbin, maxbin = -0.5, 512.5        
        nbins = 512
        hist, bins = np.histogram(dl[0], bins=nbins, range=(minbin,maxbin), density=True)

        #ax[cnt].hist(dl[0],bins=bin_edges, density=True)
        ax[cnt].hist(bins[:-1], weights=hist, bins=nbins, range=(minbin,maxbin), density=True)
        ax[cnt].set_title("Uncalibrated Charge per Pixel "+dl[1])
        ax[cnt].set_xlabel("ToT,[N(CLK cycles)]")
        ax[cnt].set_ylabel(r"$N_{entries}$"+"/"+r"$\sum{ToT}$")
        ax[cnt].set_yscale('log')
        ax[cnt].grid(which='major')
        cnt+=1
        print(f"PLOT[{cnt}] done")

    plt.tight_layout()
    plt.savefig(f"ChargePerPixel-{picname}.png", dpi=400)
    print(f"Saved: ChargePerPixel-{picname}")

def plotMatrix(matrix, labels, picname, auxpicname):

    fig, ax = plt.subplots(figsize=(12,8))
    cax = fig.add_axes([0.86,0.1,0.05,0.8])
    ms = None
    if("diff" in picname):
        ms = ax.matshow(matrix.T, cmap=use_cmap, norm=LogNorm(vmin=1,vmax=np.nanmax(matrix)))
    else:
        ms = ax.matshow(matrix.T, cmap=use_cmap)
    if(len(labels)>3):
        fig.colorbar(ms,cax=cax,orientation='vertical', label=labels[3])
    else:
        fig.colorbar(ms,cax=cax,orientation='vertical', label="Occupancy")
    ax.set_title(labels[2])
    ax.set_ylabel(labels[1])
    ax.set_xlabel(labels[0])
    ax.xaxis.set_label_position('top') 
    ax.invert_yaxis()
    plt.savefig(f"{picname}-{auxpicname}.png", dpi=400)
    ms = None
    plt.close()

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

tot_mat_bef = initMatrix(np.uint16) 
tot_mat_aft = initMatrix(np.uint16) 

TOT_before, TOT_after = None, None

mask_matrix = None
maskfile = '/home/vlad/Timepix3/masks/W15-G6_mask_2023-07-07_19-07-00.h5'
with tb.open_file(maskfile) as f:

    mask_matrix = f.root.mask_matrix[:].T


with tb.open_file(fbefore) as f:

    bgname = getBaseGroupName(f)

    cluster_x = f.get_node(bgname+"x")
    cluster_y = f.get_node(bgname+"y")
    ToT = f.get_node(bgname+"ToT")

    for xpos, ypos, tot in zip(cluster_x, cluster_y, ToT):
        np.add.at(mat_before, (xpos,ypos), 1)            
        np.add.at(tot_mat_bef, (xpos,ypos), tot)            

    TOT_before = np.concatenate(ToT)

print(f"Processed FILE before: {fbefore}")

with tb.open_file(fafter) as f:

    bgname = getBaseGroupName(f)

    cluster_x = f.get_node(bgname+"x")
    cluster_y = f.get_node(bgname+"y")
    ToT = f.get_node(bgname+"ToT")

    for xpos, ypos, tot in zip(cluster_x, cluster_y, ToT):
        np.add.at(mat_after, (xpos,ypos), 1)            
        np.add.at(tot_mat_aft, (xpos,ypos), tot)            

    TOT_after = np.concatenate(ToT)

print(f"Processed FILE after: {fafter}")

datalist = [[TOT_before,"Before Irrad."],[TOT_after,"After Irrad."]]

plotGain(datalist, picname)

masked_positions = mask_matrix.astype(bool)
dead_before_mask = mat_before == 0
dead_after_mask  = mat_after == 0

# Exclude masked pixels from all further comparisons
dead_before_mask = dead_before_mask & (~masked_positions)
dead_after_mask  = dead_after_mask & (~masked_positions)

# Identify status changes
newly_dead_mask  = (~dead_before_mask) & dead_after_mask   # alive → dead
revived_mask     = dead_before_mask & (~dead_after_mask)   # dead → alive
still_dead_mask  = dead_before_mask & dead_after_mask       # always dead

# Extract coordinates
new_y, new_x = np.where(newly_dead_mask)
rev_y, rev_x = np.where(revived_mask)
still_y, still_x = np.where(still_dead_mask)

#print(f"Before we have:\n{ndead_bef} pixels dead\nAfter we have:\n{ndead_aft} pixels dead")

mat_difference = mat_before - mat_after

norm_before_mat = mat_before / np.sum(mat_before)
norm_after_mat = mat_after / np.sum(mat_after)

diff_tot_mat = tot_mat_bef - tot_mat_aft

norm_tot_bef = tot_mat_bef / np.sum(tot_mat_bef)
norm_tot_aft = tot_mat_aft / np.sum(tot_mat_aft)

use_cmap = 'jet'
#use_cmap = 'gist_earth_r'
print("PLOTTING")

plotMatrix(
    norm_before_mat, 
    ["x", "y", "Matrix Before Irradiation"],
    "DESY-P09-FE55-matrix-before",
    picname
)

plotMatrix(
    norm_after_mat, 
    ["x", "y", "Matrix After Irradiation"],
    "DESY-P09-FE55-matrix-after",
    picname
)

plotMatrix(
    mat_difference, 
    ["x", "y", "Response Difference (ocucpancy)"],
    "DESY-P09-FE55-matrix-diff",
    picname
)

plotMatrix(
    norm_tot_bef, 
    ["x", "y", "Matrix Before Irradiation (TOT)", "Norm. ToT"],
    "DESY-P09-FE55-matrix-TOT-before",
    picname
)

plotMatrix(
    norm_tot_aft, 
    ["x", "y", "Matrix After Irradiation (TOT)", "Norm ToT"],
    "DESY-P09-FE55-matrix-TOT-after",
    picname
)

plotMatrix(
    diff_tot_mat, 
    ["x", "y", "Response Difference (TOT)", "Norm ToT"],
    "DESY-P09-FE55-matrix-TOT-diff",
    picname
)

fig, ax = plt.subplots(figsize=(12,8))
#cax = fig.add_axes([0.86,0.1,0.05,0.8])
#ms = ax.matshow(mat_before.T, cmap=use_cmap, norm=LogNorm(vmin=1,vmax=np.nanmax(mat_before)))
#ax.scatter(bdead_x, bdead_y, marker='*', c='r', label="Old dead pixels")
#ax.scatter(dead_x, dead_y, marker='*', c='b', label="Newly dead pixels")

ax.scatter(still_x, still_y, s=10, c='gray', label='Always Dead', alpha=0.6)
ax.scatter(new_x, new_y, marker='*', c='red', label='Newly Dead', s=30)
ax.scatter(rev_x, rev_y, marker='o', facecolors='none', edgecolors='lime', label='Revived', s=30)

#fig.colorbar(ms,cax=cax,orientation='vertical')
ax.set_title("Matrix - Dead Pixel Location")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.xaxis.set_label_position('top') 
ax.invert_yaxis()
plt.legend()
plt.savefig(f"Dead-Pixels-{picname}.png", dpi=400)
#ms = None
plt.close()




