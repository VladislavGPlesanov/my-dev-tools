import numpy as np
import sys
import matplotlib.pyplot as plt
from MyPlotter import myUtils as mu

plotname = sys.argv[1]
files = sys.argv[2:]
print(plotname)
print(f"\033[1;34;40m {files} \033[0m")

efields = []
tdiffs = []
ldiffs = []
eVels = []
labels = []

EFIELD = np.linspace(100,2000,19)

pltTitle = ""

cnt = 0
for file in files:

    tmp_EF, tmp_tdiff, tmp_ldiff, tmp_evel = [],[],[],[]

    ilabel = None 

    f= open(file,'r')
    for line in f:
        words = None
        if("#" in line):
            if(ilabel is None):
                ilabel=line[1:].split("@")[0]
                ilabel+=", "+line[1:].split("@")[1].split(",")[1]
            else:
                continue
        else:
            words = line.split(',')
            tmp_EF.append(float(words[1]))
            tmp_tdiff.append(float(words[2])) #
            tmp_ldiff.append(float(words[3])) 
            #eVel.append(float(words[3])) # cm/ns - default from MAGBOLTZ
            tmp_evel.append(float(words[4])*1000) # cm/us 
        cnt+=1
    
    efields.append(tmp_EF)
    tdiffs.append(tmp_tdiff)
    ldiffs.append(tmp_ldiff)
    eVels.append(tmp_evel)
    labels.append(ilabel)

    tmp_EF, tmp_tdiff, tmp_ldiff, tmp_evel = None,None,None,None
    ilabel = None

plt.figure(figsize=(10,8))

markers = ['*','o','s','x','+','v','^','d','1','2','3','4','h']
#for ef,td,lab in zip(efields,tdiffs,labels):
imar = 0
for td,lab in zip(tdiffs,labels):
    #plt.scatter(ef, td, s=20, marker='o',label=lab)
    if("CO2" in lab):
        plt.scatter(EFIELD, td, s=20, c='m', marker=markers[imar],label=lab)
    elif("Isobutane" in lab):
        plt.scatter(EFIELD, td, s=20, c='y', marker=markers[imar],label=lab)
    else:
        plt.scatter(EFIELD, td, s=20, c='b', marker=markers[imar],label=lab)

    imar+=1   

plt.xlabel(r'$E_{Drift}$, [V/cm]')
plt.ylabel(r'Transversal Diffusion, $\mu\mathrm{m}$/$\sqrt{\mathrm{cm}}$')
plt.title(f'Transversal Diffusion')
plt.legend(loc='center right')
plt.xlim([0,2400])
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
plt.savefig("DiffusionCombined-"+plotname+".png")
plt.savefig("DiffusionCombined-"+plotname+".pdf")

#Atmp = [1.025, 1, 1.5]
#Print("test\n")
#For i in atmp:
#    print(mu.atmToTorr(i))


