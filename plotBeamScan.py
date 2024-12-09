import sys
import glob
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
from lmfit import Model
import matplotlib.patches as mpatches


###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

def gauss(x, A, mu, sigma):

    return A*np.exp(-((x-mu)**2)/(2*sigma**2))

def getMaxBin(numbers):

    maxcounts = 0
    maxbin = 0
    cnt = 0
    for n in numbers:
        if(n>maxcounts):
            maxcounts = n
            maxbin = cnt
            cnt+=1
        else:
            cnt+=1

    return maxbin, maxcounts

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)


def getBaseName(f):

    print(f"Reading {f}")
    groups = f.walk_groups('/')
    grouplist = []
    for gr in groups:
        grouplist.append(gr)
    main_group = str(grouplist[len(grouplist)-1])
    grouplist = None 

    basewords = main_group.split('(')

    base_group_name = basewords[0][:-1]+'/'

    return base_group_name

def getBeamCenter(matrix):
    
    x_coords, y_coords = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), indexing='ij')

    # Total sum of the matrix (normalization factor)
    total_weight = matrix.sum()
    
    # Weighted means (center of mass)
    mean_x = (matrix * x_coords).sum() / total_weight
    mean_y = (matrix * y_coords).sum() / total_weight
    
    # Weighted standard deviations
    std_x = np.sqrt((matrix * (x_coords - mean_x)**2).sum() / total_weight)
    std_y = np.sqrt((matrix * (y_coords - mean_y)**2).sum() / total_weight)

    return mean_x, mean_y, std_x, std_y


########## main starts here ############    
file = sys.argv[1]
picname = sys.argv[2]

matrix = np.zeros((256,256),dtype=np.uint16)

with tb.open_file(file, 'r') as f:

    gname = getBaseName(f)

    #Tot = f.get_node(gname+"ToT")
    x = f.get_node(gname+"x")
    y = f.get_node(gname+"y")

    nclusters = len(x)
    cnt = 0 
    for xpos, ypos in zip(x,y):
        np.add.at(matrix, (xpos,ypos), 1)
        progress(nclusters,cnt)
        cnt+=1

f.close()

###########################################################

fig, ax = plt.subplots(figsize=(12,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
ms = ax.matshow(matrix.T, cmap='viridis')

#ellipse = Ellipse(
#    xy=(peak_x, peak_y),  # Note: matplotlib uses (y, x) for coordinates in images
#    width=std_y,        # 2 standard deviations in y
#    height=std_x,       # 2 standard deviations in x
#    edgecolor='blue',
#    facecolor='none',
#    lw=0.5
#)
#ax.add_patch(ellipse)
#
#peak_x, peak_y , stdevX, stdevY = get BeamCenter(matrix)
## Annotate the peak
#ax.plot(peak_x, peak_y, 'ro--', label='Peak')

fig.colorbar(ms,cax=cax,orientation='vertical', label="occupancy")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.xaxis.set_label_position('top') 
ax.invert_yaxis()

## Create a proxy artist for the legend
#legend_patch = mpatches.Patch(color='blue', label=f"{picname}")  # Adjust color if needed
#ax.legend(handles=[legend_patch])

#plt.legend()

plt.savefig(f"matrix-spotScan-{picname}.png", dpi=400)





