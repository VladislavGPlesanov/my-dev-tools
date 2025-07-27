import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
from MyPlotter import myPlotter

def findClosest(number, listA, listB):

    if(number in listA):
        return listB[listA.index(number)]
    else:
      #this_num = 0
      #this_i = 0
      for i in range(len(listA)):
        if(number > listA[i] and number < listA[i+1]):
            #this_num = np.mean([i, listA[cnt+1]])
            #this_i = i
            return np.mean([ listB[i], listB[i+1]])
        #cnt+=1
      return np.mean([ listB[listA.index[this_i]], listB[listA.index[this_i+1]]])

#########################################################################################

kaptonFile = 'scintiData/proton_range_kapton.dat'
macroFile = 'scintiData/proton_range_POLYCARBONATE_macrolon.dat'

f = open(kaptonFile,'r')

protonEnergy, dedx = [],[] 

for line in f:

    if('#' in line):
        continue

    words = line.split(' ')
    protonEnergy.append(float(words[0]))
    dedx.append(float(words[1]))

    line = None
    words = None

f.close()
f = None

kapton_density = 1.42 # g/cm^3
E_proton_cyclo = 13.609 # MeV
cycloEnergyLoss = findClosest(13.609, protonEnergy, dedx)*kapton_density

d_kapton = 50 # um

lossInKaptonTape = cycloEnergyLoss/1e4 * d_kapton

#mp = myPlotter()
#plt.figure(figsize=(8,8))
#plt.scatter(protonEnergy,dedx,s=10,c='b',marker='.')
#plt.xlabel("Proton Energy, [MeV]")
#plt.ylabel(r"Energy Loss, -$\frac{\mathrm{dE}}{\mathrm{dx}}$, [$\frac{\mathrm{MeV}\cdot\mathrm{cm^2}}{\mathrm{g}}$]")
#plt.xscale('log')
#plt.yscale('log')
#plt.savefig("proton-dedx-kapton.png")

plt.figure(figsize=(8,8))
plt.scatter(protonEnergy,np.array(dedx)*kapton_density,s=10,c='b',marker='.')
plt.vlines(E_proton_cyclo, 0, 1e3, colors='red', linestyles='--', label=f"Proton Energy @ {E_proton_cyclo:.3f} [MeV]")
plt.hlines(cycloEnergyLoss, 0, 1e4, colors='green', linestyles='--', label=f"dE/dx={cycloEnergyLoss:.3f}"+r"[$\frac{\mathrm{MeV}}{\mathrm{cm}}$]")
plt.scatter([],[], c='w', label=f"Loss in tape: {lossInKaptonTape:.3f} [MeV]")
plt.title("Energy loss of protons in Kapton")
plt.xlabel("Proton Energy, [MeV]")
plt.ylabel(r"Energy loss [MeV/$\mathrm{cm}^{-1}$]")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("proton-dedx-MeV-per-cm-kapton.png")
plt.close()

f = open(macroFile,'r')

pEnergy, dedx_mac, range_mac = [],[], []

macro_density = 1.2 # g/cm^3

for line in f:

    if("#" in line):
        continue
    
    words = line.split(' ')

    pEnergy.append(float(words[0]))
    dedx_mac.append(float(words[1]))
    range_mac.append(float(words[2]))

    line, words = None, None

f.close()
f = None

pLossInMacro = findClosest(E_proton_cyclo,pEnergy,dedx_mac)*macro_density

prange_mac = findClosest(E_proton_cyclo,pEnergy,range_mac)/macro_density

dc_wall_thick = 10 #mm

plt.figure(figsize=(8,8))
plt.scatter(pEnergy, np.array(dedx_mac)*macro_density, s=10, c='b',marker='.')
plt.vlines(E_proton_cyclo,0,pLossInMacro, colors='red', linestyles='--', label=r"$E_{p}$="+f"{E_proton_cyclo:.3f} [MeV]")
plt.hlines(pLossInMacro,0, E_proton_cyclo, colors='green', linestyles='--', label=r"$dE/dx$="+f"{pLossInMacro:.3f} [MeV/cm]")
plt.scatter([],[], c='w', label=r"CSDA range $\approx$"+f"{prange_mac:.3f} [cm]")
#plt.scatter([],[], c='w', label=f"Loss in tape: {lossInKaptonTape:.3f} [MeV]")
plt.xlabel("Proton Energy, [MeV]")
plt.ylabel(r"Energy loss [MeV/$\mathrm{cm}^{-1}$]")
plt.title("Energy loss of protons in Polycarbonate(macrolone)")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig("proton-dedx-MeV-per-cm-macrolone.png")









