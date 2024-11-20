import numpy as np
import sys
import matplotlib.pyplot as plt


#energy = float(sys.argv[1]) 
files = sys.argv[1:] 

distances = []
fracLost = []
fracLost_7p5k = []
fracLost_8k = []

#plt.figure(figsize=(16,9))
plt.figure()

for file in files:

    f = open(file)

    cnt = 0
    for line in f:

        if(cnt==0):
            words = line.split(' ')
            path = words[3].split("=")     
            cm = float(path[1])
            
            print(f"words={words}\n<path={path}>\n<cm={cm}>\n-------------------------------------")

            distances.append(cm)
            cnt+=1
            continue
        elif(cnt==1):
            cnt+=1
            continue
        else:
            words = line.split()
            Egamma = float(words[0])
            frac = 1.0-float(words[1])
            if(Egamma == 7000.0):
                fracLost.append(frac)
            if(Egamma == 7500.0):
                fracLost_7p5k.append(frac)
            if(Egamma == 8000.0):
                fracLost_8k.append(frac)

        cnt+=1

plt.scatter(distances,fracLost,marker='o',color='red',label="7 keV")
plt.scatter(distances,fracLost_7p5k,marker='o',color='orange',label="7.5 keV")
plt.scatter(distances,fracLost_8k,marker='o',color='yellow',label="8 keV")
plt.xscale('log')
plt.title("x-ray absoprtion with distance")
plt.xlabel("Distance in air, [cm]")
plt.ylabel("Fraction of xrays lost")
plt.legend(loc='upper left')
plt.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--', linewidth=0.25)
plt.ylim(-0.05,1.05)
plt.savefig(f"FractionOfXraysLostInAir.png")

