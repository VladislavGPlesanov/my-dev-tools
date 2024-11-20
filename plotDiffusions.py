import numpy as np
import sys
import matplotlib.pyplot as plt

infile = sys.argv[1]
#columns = list(sys.argv[2].split(',')) #this shluld be comma sep. list
#colnames = list(sys.argv[3].split(',')) # description of x labels
plotname = sys.argv[2]

#data = [ [] for i in range(len(columns))]
efield = []
tdiff = []
ldiff = []
eVel = []

#print(f'created list of lists:{len(data)}') 
#print(f"going to plot data of columns {columns}")

f = open(infile)

skip = ['gases', 'ratios', 'Efield']
pltTitle = ""

mtp = 1000;

startE=100.0
endE=1600.0
div = 200 # points
#Efields = np.linspace(100,1600,101)
Efields = np.linspace(100,1600,11)
print(Efields)

cnt = 0
for line in f:
    words = None
    if(cnt<2):
        words = line.split(':')
    else:
        words = line.split(',')
    if(cnt<20):
        print(words)
    if(words[0] in skip):
        print(f"in skip part - {words}")
        if(words[0]=='gases'):
            pltTitle+=line
        if(words[0]=='ratios'):
            pltTitle+=f'({words[0]})'
        if(cnt<20):
            print(pltTitle)
    else:
        print(f"filling lists from {words}")
        efield.append(float(words[0]))
        tdiff.append(float(words[1])) #
        ldiff.append(float(words[2])) 
        #eVel.append(float(words[3])) # cm/ns - default from MAGBOLTZ
        eVel.append(float(words[3])*1000) # cm/us 
        if(cnt<20):
            print(len(efield))
    cnt+=1

print(f"got : {len(efield)} Efield values")
print(f"got : {len(tdiff)} tdiff values")
print(f"got : {len(ldiff)} ldiff values")
print(f"got : {len(eVel)} eVel values")



plt.figure(figsize=(10,10))
print("tryina' plot {} vs {}".format(len(efield), len(tdiff)))
#plt.scatter(efield, tdiff, s=20,c='g',marker='o',linestyle='dashed')
plt.scatter(Efields, tdiff, s=20,c='g',marker='o',label="Trans.Diff.")
#plt.scatter(Efields, ldiff, s=20,c='b',marker='o',label="Long.Diff.")
plt.xlabel(f'E, [V/cm]')
plt.title(f'diffusion parameters: {pltTitle}')
plt.ylabel('um/sqrt[cm]')
#plt.xlim(startE-10,endE+10)
#plt.xlim(10,1200)
#plt.xscale('log')
plt.legend()
plt.grid(which='major', color='grey', linestyle='-',linewidth=1)
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.75)
#plt.grid(True)
plt.savefig("DiffusionCombined-"+plotname+".png")

##-------------------------------------------------------------
#plt.figure(figsize=(10,10))
#print("tryina' plot {} vs {}".format(len(efield), len(tdiff)))
##plt.scatter(efield, tdiff, s=20,c='g',marker='o',linestyle='dashed')
#plt.scatter(Efields, tdiff, s=20,c='g',marker='o',linestyle='dashed')
#plt.xlabel(f'E, [V/cm]')
#plt.title(f'Transverse diffusion: {pltTitle}')
#plt.ylabel('um/sqrt[cm]')
#plt.savefig("DiffusionT-"+plotname+".png")
##-------------------------------------------------------------
#plt.figure(figsize=(10,10))
#print("tryina' plot {} vs {}".format(len(efield), len(tdiff)))
##plt.scatter(efield, ldiff, s=20,c='b',marker='o',linestyle='dashed')
#plt.scatter(Efields, ldiff, s=20,c='b',marker='o',linestyle='dashed')
#plt.xlabel(f'E, [V/cm]')
#plt.ylabel('um/sqrt[cm]')
#plt.title(f'Longitudinal diffusion: {pltTitle}')
#plt.savefig("DiffusionL-"+plotname+".png")
#-------------------------------------------------------------
plt.figure(figsize=(10,10))
print("tryina' plot {} vs {}".format(len(efield), len(tdiff)))
#plt.scatter(efield, eVel, s=20,c='r',marker='o',linestyle='dotted')
#plt.scatter(Efields, eVel, s=20,c='r',marker='o',linestyle='dotted')
plt.scatter(Efields, eVel, s=20,c='r',marker='o',linestyle='dotted')
plt.xlabel('E/p, [V/cm/atm]')
plt.title(f'Electron velocity: {pltTitle}')
plt.ylabel('cm/us')
plt.xscale('log')
plt.xlim(10,1500)
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
plt.savefig("e_velocity-"+plotname+".png")





