import numpy as np
import matplotlib.pyplot as plt
from MyPlotter import myUtils
import sys 

picname = sys.argv[1]
infiles = sys.argv[2:]

print(picname)
print(infiles)

filelabels = []
data = []
g1, g2 = [], []
headers = None

NobleGas = None

mu = myUtils()

for fname in infiles:

    cleanName = mu.removePath(fname)

    #Efficiencies_Ne_CO2_3cm_779torr_500Vcm
    cut_idx = cleanName.find("_")
    if(NobleGas is None):
        NobleGas = cleanName.split("_")[1]

    filelabels.append(cleanName[cut_idx+1:])


for file in infiles:

    idata = []
    
    f = open(file,'r')
    
    for line in f:

        words = line.split(',')
        print(f"words={words}")
        if(len(words)<2):
            continue
        if("Eff" in line):
            if(headers is None):
                headers = words
                continue
            else:
                continue

        eff = float(words[2])

        if(len(g1)<5):
            r1 = float(words[0])
            g1.append(r1)
        
        idata.append(eff)

    data.append(idata)

print(len(g1))
print(len(data[0]))


plt.figure(figsize=(8,8))
for efficiency, label in zip(data,filelabels):
    plt.scatter(g1,efficiency, s=40, marker="*", label=label)
plt.xlabel(f"Ratio of {NobleGas}, [%]")
plt.ylabel("X-ray absorption efficiency, [%]")
plt.legend()
plt.grid()
plt.savefig(f"RatioVsAbsEff-{picname}.png")
plt.close()



