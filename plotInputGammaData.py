import numpy as np
import matplotlib.pyplot as plt
import sys


infile = sys.argv[1] 
picname = sys.argv[2]

nbins = 51
minbin = 0.0
maxbin = 6.28

angles = []

f = open(infile)
cnt = 0 
for line in f:

    entries = line.split('\t')
    if(cnt%1000==0):
        print(entries)
   
    angles.append(float(entries[2]))
    cnt+=1


plt.figure(figsize=(8,8))
plt.hist(angles, nbins, range=(minbin,maxbin), histtype='step', facecolor='b')
plt.xlabel("Polarization angle, [rad]")
plt.ylabel("n simulated \u03b3")
plt.plot()
plt.savefig(f"simGammaInputAngles-{picname}.png")


