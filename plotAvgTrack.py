import sys
import glob
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
from lmfit import Model

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

def getTrackPos(matrix):

    print("\nmaking avg list\n")
    ypos_avg, xbins = [], []

    for i in range(256):

        tmp_list = matrix[:,i].tolist()

        if sum(tmp_list)<50:
            ypos_avg.append(0)
            xbins.append(i)
            tmp_list = None
            continue

        nbins=255

        edges = [b for b in range(256) ]

        peakbin, maxbin_cnt = getMaxBin(tmp_list)
        bin_centers = []
        for e in edges:
            bin_centers.append(e+0.5)

        model = Model(gauss)
        pars = model.make_params(A=maxbin_cnt, mu=peakbin, sigma=np.std(tmp_list))
        pars['A'].min = maxbin_cnt*0.8
        pars['A'].max = maxbin_cnt*1.2

        pars['mu'].min = peakbin*0.6
        pars['mu'].max = peakbin*1.4
 
        result = model.fit(tmp_list[:-1], pars, x=bin_centers[:-1])

        mu = result.params["mu"].value

        #if(i%25==0):
        #    #print(f"----------- Plotting column {i} ---------")
        #    plt.figure(figsize=(16,8))
        #    plt.hist(np.asarray(edges), weights=np.asarray(tmp_list), bins=nbins, range=(0,256), align='left', histtype='stepfilled', facecolor='b')
        #    plt.plot(edges[:-1], result.best_fit, '--r')
        #    plt.savefig(f"matrix-column-testfit-{i}.png")

        ypos_avg.append(mu)
        xbins.append(i)
        #if(i%25==0):
        #    print(f"column {i}: tmp_list={len(tmp_list)}, setting, med={ypos_avg[i]}")
        tmp_list = None
        #time.sleep(0.5)
        progress(256,i)

    return xbins, ypos_avg

picname = sys.argv[2]
location = sys.argv[1]

dir_files = glob.glob(location+"*.h5")
inputlist = sorted(dir_files)

tracks_x, tracks_y = [], []
Vlist = []

for file in inputlist:

    matrix = np.zeros((256,256),dtype=np.uint16)

    Vanode = file[-6:]
    Vanode = int(Vanode[:-3])

    with tb.open_file(file, 'r') as f:
    
        gname = getBaseName(f)
    
        x = f.get_node(gname+"x")
        y = f.get_node(gname+"y")
    
        nclusters = len(x)
        cnt = 0 
        for xpos, ypos in zip(x,y):
            np.add.at(matrix, (xpos,ypos), 1)
            progress(nclusters,cnt)
            cnt+=1

    f.close()
    ix, iy = getTrackPos(matrix.T)
    matrix = np.zeros((256,256), dtype=np.uint16)
    tracks_x.append(ix)
    tracks_y.append(iy)

    Vlist.append(Vanode)

plt.figure(figsize=(16,8))
ith = 0
for i, j in zip(tracks_x, tracks_y):
    plt.scatter(i,j, label=f"Vanode={Vlist[ith]}")
    ith+=1
plt.plot([10,245],[50,58], color='m', linestyle='--', label="Beam movement")
#plt.hlines(40,10,235, colors='m', linestyles='--', label="beam movement")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cu Tracks @ 8.05 keV")
plt.xlim(0,256)
plt.ylim(0,100)
plt.grid(True)
plt.legend()
plt.savefig(f"CuTracks-{picname}.png")






                  
