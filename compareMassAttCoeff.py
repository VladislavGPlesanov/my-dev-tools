import numpy as np
import sys
import matplotlib.pyplot as plt

def getCompoundAttCoeff(weights, mus):

    coeff = 0
    for iw, imu in zip(weights, mus):
        print(f"iw={iw}, imu={imu}")
        coeff += (iw * imu)

    #print(f"COEFF={coeff}")
    return coeff

def calcIntensity(I0, att_coeff, distance, density):                                                                                                                                                             
    
    return I0*np.exp(-(att_coeff)*distance*density)   

#------------------------------------------------------

fair = open('scintiData/AirMassAttCoeff.dat','r')
fargon = open('scintiData/ArgonMassAttCoeff.dat','r')
fcarbon = open('scintiData/CarbonMassAttCoeff.dat','r')
foxygen = open('scintiData/OxygenMassAttCoeff.dat','r')

Eair, Ear, Ec, Eo = [],[],[],[]
muAir, muAr, muC, muO = [],[],[],[]

for line in fair:
    if('#' in line):
        continue
    words = line.split(',')
    Eair.append(float(words[0]))
    muAir.append(float(words[1]))
    line = None
    words = None

fair.close()

for line in fargon:
    if('#' in line):
        continue
    words = line.split(',')
    Ear.append(float(words[0]))
    muAr.append(float(words[1]))
    line = None
    words = None

fargon.close()

for line in fcarbon:
    if('#' in line):
        continue
    words = line.split(',')
    Ec.append(float(words[0]))
    muC.append(float(words[1]))
    line = None
    words = None

fcarbon.close()

for line in foxygen:
    if('#' in line):
        continue
    words = line.split(',')
    Eo.append(float(words[0]))
    muO.append(float(words[1]))
    line = None
    words = None

foxygen.close()

##### tryna' get mu/rho for CO2 #####
#          Carbon, Oxygen
weights = [0.2729, 0.7270]
CO2_murho = []
for mu_oxy, mu_carb in zip(muO,muC):
    CO2_murho.append(getCompoundAttCoeff(weights,[mu_carb,mu_oxy]))

#print(f"CO2 coeffs at i=3 -> {CO2_murho[3]}, i=4->{CO2_murho[4]}")
# pruning data sets to energies above 1e-1 MeV
pruned_muAr = muAr[-20:]
pruned_muCO2 = CO2_murho[-20:]
pruned_energies = Ear[-20:]
print(pruned_muAr)
print(len(pruned_muAr))
print(pruned_muCO2)
print(len(pruned_muCO2))

#####################################
# now tryna' to get Ar:CO2 80:20 curve

#ArCO2weights = [0.8,0.2]
ArCO2weights = [0.4,0.6]
ArCO2_murho = []
for imu_Ar, imu_co2 in zip(pruned_muAr,pruned_muCO2):
    #ArCO2_murho.append(getCompoundAttCoeff([0.8,0.2], [imu_Ar, imu_co2]))
    ArCO2_murho.append(getCompoundAttCoeff(ArCO2weights, [imu_Ar, imu_co2]))

for i,j,k,l in zip(pruned_energies, pruned_muAr, pruned_muCO2, ArCO2_murho):
    print(f"{i} -> {j:.4f} / {k:.4f} => {j*0.8:.4f} + {k*0.2:.4f} = {j*0.8+k*0.2:.4f} [{l:.4f}]")


print("Calcualted ArCO2 (80:20) x-ray attenuation coefficients\nEnergy [MeV]\t(mu/rho)")
for e,m in zip(pruned_energies,ArCO2_murho):
    print(f"{e:.3f},\t{m:.8f}")

print(pruned_energies)
print(ArCO2_murho)

#print(ArCO2_murho)
#=====================================

plt.figure(figsize=(10,10))
plt.scatter(Eair,muAir, marker='+', color='black', label=r"$(\mu/\rho)_{Air}$")
plt.scatter(Ear, muAr, marker='o', color='red', label=r"$(\mu/\rho)_{Argon}$")
plt.scatter(Ec, muC, marker='v', color='peru', label=r"$(\mu/\rho)_{Carbon}$")
plt.scatter(Eo, muO, marker='^', color='dodgerblue', label=r"$(\mu/\rho)_{Carbon}$")
plt.plot(Eo,CO2_murho, label='CO2 attCoeff')
plt.plot(pruned_energies, ArCO2_murho, c='forestgreen', label='ArCO2')
plt.xlabel('X-Ray Energy, [MeV]')
plt.ylabel(r'$(\mu/\rho)$')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('ComparingAttenumationCoefs.png')
plt.xlim([1e-1,20])
plt.ylim([1e-2,1])
#plt.xscale('linear')
#plt.yscale('linear')
plt.savefig('ComparingAttenumationCoefs-zoom100keV-20MeV.png')
plt.close()

##### side calc ######################

E_fe55 = 5.9 # keV
muRho_air_Fe55 = 1.875e-1
rhoAir = 1.225 # g/cm3
Intensities = []
distances = np.arange(0.1,100,0.1)
for d in distances:
    Intensities.append(calcIntensity(1,muRho_air_Fe55,d, rhoAir))

plt.figure()
plt.plot(distances,Intensities)

plt.yscale('log')
plt.grid(True)
plt.savefig("Fe55Xray_intensity_vs_distance_air.png")
plt.close()

