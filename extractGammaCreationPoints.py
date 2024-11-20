import numpy as np
import matplotlib.pyplot as plt
import sys
from MyPlotter import myPlotter

infile = sys.argv[1] 

conpoints = []
primEl = []
angles = []
seeds = []

nbins = 51
minbin = 0.0
maxbin = 2.0

f = open(infile)
cnt = 0 
for line in f:

    entries = line.split('\t')
    angles.append(float(entries[2]))
    conpoints.append(float(entries[3]))
    primEl.append(int(entries[4]))
    seeds.append(int(entries[5]))

    cnt+=1

mp = myPlotter()
mp.simpleHist(primEl, 
              51, 
               0,
             500, 
             ["Primary Ionization Stats","N_prim_electrons", "c_clusters"], 
             "PrimaryIonization", 
             ylog=False)

mp.simpleHist(angles, 51, 0.0, 6.3, ["Photon Angles","Angles[rad]","N"], "PhotonInitAngles", ylog=False)
mp.simpleHist(conpoints, 51, -0.1, 2.1, ["Conversion points","Cylinder Depth","N"], "ConvPoints", ylog=False)
mp.simpleHist(seeds, 51,min(seeds)*0.9, max(seeds)*1.1, ["seeds","seed nr","N"], "DegradSeeds", ylog=False)

