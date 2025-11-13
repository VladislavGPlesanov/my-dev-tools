import numpy as np
import numpy.ma as ma
import sys
import os
import glob
import tables as tb
import matplotlib
matplotlib.use("Agg")  # non-GUI backend (renders to files only)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from MyPlotter import myUtils  
from lmfit import Model
from matplotlib import cm

from sklearn.cluster import DBSCAN

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

def getFrameTime(SR,ST):

    return 256**SR * 46 * ST / 4e7

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)

def countIons(counts,mean):

    return np.round(float(counts)/mean, 2)

#####################################################
picname = sys.argv[1]
SR = sys.argv[2]
ST = sys.argv[3]
filedirs = sys.argv[4:]

outdir = f"NITRO-{picname}/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

########### some global constants ###############

t_dead = 0.048 # dead time in full matrix mode
avg_hits = 689.59
avg_tot = 786183.57

cut_hits = 30000

glob_ions_nhits = 0
glob_ions_TOT = 0

#################################################
for idirectory in filedirs:

    inputlist = glob.glob(idirectory+"*.txt")

    nfiles = len(inputlist)

    print(f"FOUND: {nfiles} (frames) in directory: {idirectory}") 

    for file in inputlist:

        tmp_array = np.loadtxt(file, dtype=int)

        nhits = np.count_nonzero(tmp_array)
        sumTOT = np.sum(tmp_array)
        
        tmp_array = None

        glob_ions_nhits += countIons(nhits, avg_hits)
        glob_ions_TOT += countIons(sumTOT, avg_tot)

        if(nhits>=cut_hits):
            continue


print(f"Counted:\nN_ions(hits) = {glob_ions_nhits:.2f}\nN_ions(TOT) = {glob_ions_TOT:.2f}")

 
        







