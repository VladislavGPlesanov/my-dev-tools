import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import skewnorm

###############################################################
def getData(path):

    data = []

    f = open(path,'r')

    for line in f:
       
       if("Energy" in line):
           continue
       
       data.append(float(line))

    return np.array(data, dtype=float)

#############################################################
def getEnergy(number, elist):

    print(f"Checking {number}")

    if(number < min(elist) or number > max(elist)):
        print(f"Energy [{number}] outside the scope of the list")
        return -1
    else:
        for i in range(len(elist)):
            if(number > elist[i] and number < elist[i+1]):
                 if(abs(number - elist[i]) <= abs(number - elist[i+1])):
                    return elist[i]
                 else:
                    return elist[i+1]
    return 0

def findClosest(number, listA, listB):                                                                                                                                                                  
                                                                                                  
     if(number in listA):
         print(f"{number} in listA at index {listA.index(number)}")  
         return listB[listA.index(number)]
     else:                                                                                        
       this_num = 0
       this_i = 0                                                                                 
       cnt=0
       for i in range(len(listA)):
         if(number > listA[i] and number < listA[i+1]):                                     
             this_num = np.mean([i, listA[cnt+1]])                                                
             this_i = i
             print(f"{number} is between {listB[i]} and listB{[i+1]}")                                                      
             return np.mean([ listB[i], listB[i+1]])
         else:
             print(f"EBALA, {number}, listA[0]={listA[0]}, listA[n]={listA[len(listA)-1]}")
         cnt+=1                                                                                   

def calcIntensity(I0, att_coeff, distance):                                                                                                                                                             
    
    return I0*np.exp(-(att_coeff)*distance)                                                       
        
def makeAvgBin(listA):                                                                            
    
    tmp = []
    for i in range(len(listA)):
        if(i==0):
            continue                                                                              
        tmp.append(np.mean([listA[i],listA[i-1]]))                                                
    return tmp    
                                                                                                  
#################################################

mainpath = '/home/vlad/readoutSW/tools/ProtonsKaptonSimulationSpectra/'
fprim_prot = 'build1_Primaries_Espectrum_crossingAl.csv'
fsec_elec = 'Secondaries_Espectrum_detside_e-.csv'
fsec_posi = 'Secondaries_Espectrum_detside_e+.csv'
fsec_gamma = 'Secondaries_Espectrum_detside_gamma.csv'
fsec_neut = 'Secondaries_Espectrum_detside_neutron.csv'

#nbins = 60
nbins = 30

f = open(mainpath+fprim_prot,'r')

protons = getData(mainpath+fprim_prot)
electrons = getData(mainpath+fsec_elec)
positrons = getData(mainpath+fsec_posi)
gammas = getData(mainpath+fsec_gamma)
neutrons = getData(mainpath+fsec_neut)

prot_counts, edges = np.histogram(protons, bins=nbins) 
elec_cts, _ = np.histogram(electrons, bins=nbins) 
posi_cts, _ = np.histogram(positrons, bins=nbins) 
gamma_cts, _ = np.histogram(gammas, bins=nbins) 
neut_cts, _ = np.histogram(neutrons, bins=nbins) 

Nprotons = len(protons)
Nelec    = len(electrons)
Nposi    = len(positrons)
Ngamma   = len(gammas)
Nneut    = len(neutrons)

tot_counts = prot_counts + elec_cts + posi_cts + gamma_cts + neut_cts

# accounting for asorption coefficient of gammas in ArCO2 80:20
tfile_arco2 = 'scintiData/XrayTrans-ArCO2-8020-1p42cm.dat'
xrayE, muRho = [], []

tf = open(tfile_arco2,'r')

for line in tf:

    if('#' in line):
        continue
        
    words = line.split(' ')
    print(words)
    #words = line.split()
    xrayE.append(float(words[4]))
    muRho.append(float(words[10]))
    words = None
    line = None

tf.close()

bin_centers = 0.5 * (edges[:-1]+edges[1:])
bin_width = np.diff(edges)
# putting calculated attenuation coefs for ArCO2 80:20 

arco2_E = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]
arco2_murho = [0.19425495200000004, 0.141300866, 0.121093862, 0.101000028, 0.08932717080000001, 0.081112613, 0.0748118364, 0.0655185788, 0.058832721799999994, 0.052589040000000004, 0.047925688200000004, 0.0414998296, 0.034246177600000004, 0.030322452600000004, 0.027931784400000003, 0.0263941882, 0.0246020552, 0.0237146282, 0.0229839324, 0.023057215000000002]

cts_in_anode = []
#    n_cts, energy
for icount, edge in zip(gamma_cts, edges):
    print(f"Using edge={edge}, icount={icount}")
    thisEnergy = getEnergy(edge, arco2_E) 
    attCoeff = arco2_murho[arco2_E.index(thisEnergy)]
    print(f'attCoeff={attCoeff}')
    att = calcIntensity(1,attCoeff, 3.25)# 4cm from side wall to center of the anode. from wall to anode cutout 4-0.75=3.25
    cts_in_anode.append(int(np.ceil(icount*(1-att))))

# for ArCO2 above 11keV 1,42mm of gas >90%transparent and 99%  transparent above 24.7keV
# so using a factor of 0.1 for gamma counts (as worst case scenatio with most counts in the first bin)
# 

cts_in_anode_red = []
for i in cts_in_anode:
    cts_in_anode_red.append(int(np.ceil(i*0.1)))

plt.figure()
plt.hist(edges[:-1], weights=gamma_cts, bins=nbins, range=(0,15), label=f"Total flux, Simulation")
plt.hist(edges[:-1], weights=cts_in_anode, bins=nbins, range=(0,15), label=f"Flux reaching anode zone [{sum(cts_in_anode)/sum(gamma_cts)*100:.2f}%]")
plt.hist(edges[:-1], weights=cts_in_anode_red, bins=nbins, range=(0,15), label=f"Converted flux (1.42mm above chip) [{sum(cts_in_anode_red)/sum(gamma_cts)*100:.2f}%]")
plt.title('gamma energy spectrum for 13,609 MeV beam (after kapton)')
plt.xlabel(r'$E_{\gamma}$, [MeV]')
plt.ylabel('CTS')
#plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig("gamma-spectrum-kapton-run.png")
plt.close()

print(f'{Nprotons}, {Nelec}, {Nposi}, {Ngamma}, {Nneut}')

sn_ratios_per_energy = prot_counts/(elec_cts+posi_cts+gamma_cts+neut_cts)

plt.figure()
plt.scatter(bin_centers,sn_ratios_per_energy,marker='+',color='black')
plt.title("SN ratio for energy dist histogram")
plt.xlabel('Energy, [MeV]')
plt.ylabel('S/N ratio')
plt.yscale('log')
plt.grid(True)
plt.savefig("SN_ratio_protonsAfterKapton.png")
plt.close()
######################################################################

alpha, loc, scale = skewnorm.fit(prot_counts)
print(f"Fitting skewed gauss to proton data: a={alpha}, loc={loc}, scale={scale}")
xpts = np.linspace(0,12,1000)
#prot_pdf = skewnorm.pdf(xpts, alpha, loc, scale)
prot_pdf = skewnorm.pdf(xpts, -1, 9, 2)



cdf = np.cumsum(prot_counts)
cdf = cdf / cdf[-1]

n_samples = 10000
rand_probes = np.random.rand(n_samples)

samples = np.interp(rand_probes, cdf, bin_centers)

plt.figure(figsize=(10,10))
#plt.hist(bin_centers, weights=prot_counts, bins=nbins, range=(0,12), alpha=0.25, label='Original dist.')
plt.hist(samples, bins=nbins, density=True, alpha=0.5, label='Sampled dist')
#plt.plot(xpts, prot_pdf, 'r--', label="Skewed Gauss")
plt.title(r'Proton Energy spectrum after crossing 50$\mu$m of Kapton at $E_{beam}=13.609\,$MeV')
plt.xlabel(r'$E_{p^{+}}$, [Mev]')
plt.ylabel(r'Rate per primary $p^{+}$')
#plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig("protonSpectrum-afterkapton.png")
plt.close()

########################################################################
print(len(sn_ratios_per_energy))
print(sn_ratios_per_energy)

#  plotting stacked histogram

bottom = np.zeros_like(prot_counts)
plt.figure(figsize=(10,8))

plt.bar(bin_centers, prot_counts, width=bin_width, bottom = bottom, label=r'$p^{+}$'+f', {Nprotons/Nprotons}', color = 'red')
bottom+=prot_counts

plt.bar(bin_centers, elec_cts, width=bin_width, bottom = bottom, label=r'$e^{-}$'+f', {Nelec/Nprotons:.2f}'+r'$\cdot\phi_{p}$', color = 'blue')
bottom+=elec_cts

plt.bar(bin_centers, posi_cts, width=bin_width, bottom = bottom, label=r'$e^{+}$'+f', {Nposi/Nprotons:.2f}'+r'$\cdot\phi_{p}$', color = 'lightseagreen')
bottom+=posi_cts

plt.bar(bin_centers, cts_in_anode_red, width=bin_width, bottom = bottom, label=r'Converted above anode $\gamma$'+f', {sum(cts_in_anode_red)/Nprotons:.2f}'+r'$\cdot\phi_{p}$', color = 'gold')
bottom+=cts_in_anode_red
#plt.bar(bin_centers, gamma_cts, width=bin_width, bottom = bottom, label=r'$\gamma$'+f', {Ngamma/Nprotons:.2f}'+r'$\cdot\phi_{p}$', color = 'gold')
#bottom+=gamma_cts

plt.bar(bin_centers, neut_cts, width=bin_width, bottom = bottom, label=r'$n^{0}$'+f', {Nneut/Nprotons:.2f}'+r'$\cdot\phi_{p}$', color = 'deeppink')
bottom+=neut_cts

# overline
#plt.step(edges[:-1], tot_counts, where='post', color='black', linewidth=2, label='Total counts from simulation')

plt.title('Combioned Spectra for Protons traversing Kapton foil')
plt.xlabel('Particle energy, [MeV]')
plt.ylabel('CTS')
#plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("CombinedSpectrum-ProtonsThroughKapton.png")

print('--- FINITO! ---')
