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

def plotGain(data, picname):

    fig, ax = plt.subplots(figsize=(12,6))

    print("Plotting TOT/Pixel")
    minbin, maxbin = -0.5, 512.5        
    nbins = 512
    hist, bins = np.histogram(data, bins=nbins, range=(minbin,maxbin), density=True)

    ax.hist(bins[:-1], weights=hist, bins=nbins, range=(minbin,maxbin), density=True)
    ax.set_title("Uncalibrated Charge per Pixel")
    ax.set_xlabel("ToT,[N(CLK cycles)]")
    ax.set_ylabel(r"$N_{entries}$"+"/"+r"$\sum{ToT}$")
    ax.set_yscale('log')
    ax.grid(which='major')

    plt.tight_layout()
    plt.savefig(f"ChargePerPixel-{picname}.png", dpi=400)
    print(f"Saved: ChargePerPixel-{picname}")

###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

picname = sys.argv[1]
files = sys.argv[2:]

print(files)
TOTlist = []
#matrix_list = []
ratenamelist = []
sumTOTlist = []

for file in files:

    tmpTOT = []

    words = file.split("-")
    
    rate = words[6]

    with tb.open_file(file) as f:
    
        bgname = getBaseGroupName(f)
    
        #cluster_x = f.get_node(bgname+"x")
        #cluster_y = f.get_node(bgname+"y")
        ToT = f.get_node(bgname+"ToT")
        sumTOT = f.get_node(bgname+"sumTot")
    
        #for xpos, ypos, tot in zip(cluster_x, cluster_y, ToT):
        #    np.add.at(tmp_matrix, (xpos,ypos), tot)            
    
        tmpTOT= np.concatenate(ToT)
        sumTOTlist.append([np.array(sumTOT),rate])

        #cluster_x, cluster_y, ToT = None, None, None
        ToT = None
        bgname = None

    print(f"Processed FILE before: {file}")
    #matrix_list.append(tmp_matrix)
    
    plotGain(tmpTOT, rate+"-"+picname)
    print("Gain plotted!")
    tmpTOT = None

print("=================CHINAZES=======================")

nbins = 100
minbin, maxbin = 0,20000

plt.figure(figsize=(10,8))
#plt.hist([],weights=[], bins=cnbins, range=(pminbin, pmaxbin), color='white', label=r"$\mathrm{V}_{\mathrm{Grid}}:$")
for data in sumTOTlist:
    counts, bin_edges = np.histogram(data[0], bins=nbins, range=(minbin,maxbin), density=True)
    weights = counts/sum(counts)
    plt.hist(bin_edges[:-1], weights=weights, bins=nbins, range=(minbin, maxbin), align='left', histtype='stepfilled', alpha=0.2, label=f"{data[1]}")
    
plt.title(r"$\Sigma$ (TOT) per event")
plt.xlabel(r"$\Sigma$ (TOT)")
plt.ylabel(r"$N_{events}$")
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(f"sumTOT-rateScan-Combined-{picname}.png")










