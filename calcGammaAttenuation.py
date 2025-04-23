import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sortLists(listA, listB):
    
    # use as l1_sorted, l2_sorted = sortLists(listA,listB)
    
    return zip(*sorted(zip(listA,listB)))

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

def calcIntensity(I0, att_coeff, distance):

    return I0*np.exp(-(att_coeff*distance))

def calcAttenuation(intensityDrop, attFactor, material_density):

    return np.log(intensityDrop)/(attFactor*material_density)
    #  desired drop in intensity xN / mu/rho * material density

def exponent(x,a,b):
    return a*np.exp(b*x)

def exponent_woffset(x,a,b,c):
    return a + b*np.exp(c*x)

def linear(x, a,b):
    return a*x + b

def squareFunc(x,a,b,c):
    return a*x**2 + b*x + c

#################################################################
shield_thickness = np.linspace(0,15,1)

folder = 'scintiData/'

### reading Pb data for mass-attenuation coefficients ##################

pbfile = open(folder+"lead_gamma_attenuation_const.txt",'r')
wfile = open(folder+"tungsten_gamma_attenuation_coeff.csv")

W_rho = 19.3 #   g/cm3
Pb_rho = 11.35 # g/cm3
Energies, muRho = [], []

w_energies, w_muRho = [], []

for line in pbfile:
    words = line.split(',')
    #print(words)
    iEnergy = float(words[0])
    iconst = float(words[1])
    Energies.append(iEnergy)
    muRho.append(iconst)

pbfile.close()

for wline in wfile:
    if('#' in wline):
        continue
    words = wline.split(',')
    w_energies.append(float(words[0]))
    w_muRho.append(float(words[1]))

wfile.close()

plt.figure()
#plt.scatter(Energies, muRho)
plt.plot(Energies, muRho)
plt.xlabel(r"$E_{\gamma}$, MeV")
plt.ylabel(r"$\mu/\rho, cm^{2}/g$")
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.savefig("Pb_attenuation-coefficients-vs-energy.png")

# plotting tungsten and lead atttenuation in comparison plot

plt.figure()
plt.plot(Energies, muRho, label="Pb coeff.")
plt.plot(w_energies, w_muRho, label="W coeff.")
plt.xlabel(r"$E_{\gamma}$, MeV")
plt.ylabel(r"$\mu/\rho, cm^{2}/g$")
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig("attenuation_comparison_W_vs_Pb.png")

#-----------------------------------
thisPbX = 10 #cm
attCoeff = findClosest(0.1, Energies, muRho)
Intensity = 1 # a.u.

Intensities_0 = np.linspace(1,1e5, 100)
Intensities_i = []

I = calcIntensity(Intensity, attCoeff, thisPbX)

print(f"for 15 cm of lead attenuation of 0.1 MeV is {I}")

for i in Intensities_0:
    Intensities_i.append(calcIntensity(i, attCoeff, thisPbX))

plt.figure()
plt.plot(Intensities_0,Intensities_i)
plt.xlabel(r'Intensity $I_{0}$')
plt.ylabel(r'Attenuated intensity $I$')
plt.grid(True)
plt.savefig("GammaAttenuation_vs_Intensity_15cmPb.png")

L_range = np.linspace(0, 15, 16)
print(L_range)
Intesity_0 = 1
I_2 = []
EnergyList = np.linspace(0.1, 2, 10)
print(EnergyList)

plt.figure(figsize=(8,6))
for E in EnergyList:
    for l in L_range:
    
        attCoeff2 = findClosest(E, Energies, muRho)
        I_2.append(calcIntensity(Intesity_0, attCoeff2, l))
    plt.plot(L_range, I_2, label=f"E={round(E,2)} MeV")
    I_2 = []      
  

plt.xlabel('Pb thickness, [cm]')
plt.ylabel(r'Attenuated intensity $I_{l}$')
plt.grid(True)
plt.ylim([0,1.2])
plt.text(6,1.0, r'$I=I_{0}\cdot e^{\frac{\mu}{\rho}\cdot x}$', fontsize=12)
#plt.xscale('log')
plt.legend(loc='upper right')
plt.savefig("GammaAttenuation_vs_deltaLPb.png")


### reading Si data fro mass-attenuation coefficients ##################
#
#Sifile = open()
#
## reading scintillator energy loss data ################

prot_file = open(folder+"plastic_scinti_proton_stopping_power.txt", "r")
Units = None
pEnergy, pStopPow, pRange = [], [], []
scinti_rho = 1.032 # g/cm3

plcnt = 0
for pline in prot_file:
    if(len(pline)<4 or "PSTAR" in pline or "PLASTIC" in pline):
        plcnt+=1
        continue
    if("CSDA" in pline):
        print(f"found header - {pline}")
        words = pline.split(",")
        Units = words
        plcnt+=1
        continue
    words = pline.split(",")
    pEnergy.append(float(words[0]))
    pStopPow.append(float(words[1]))
    pRange.append(float(words[2])/scinti_rho)
    plcnt+=1

plt.figure()
plt.plot(pEnergy,pRange,color='green')
plt.xlabel(Units[0])
plt.ylabel("Range, cm")
plt.xscale('log')
plt.yscale('log')
plt.savefig("proton_range_in_plastic_scintillator.png")

########### chekin BC-408 scinti light yield for protons ########################################
# data from https://scintillator.lbl.gov/ej-20x-bc-40x-and-ne-110-quenching-data/i
pYieldFile = open(folder+'plastic_scinti_light_yield_protons.txt','r')
Eprotons, LightYield, LY_err = [], [], []

# data normalised vs 477 keV electron response of BC-408
# for 1 MeV e- this would be multiplied by a cator of 1 MeV/0.477MeV = 2096
scaleFactor = 2096.0

for yline in pYieldFile:
    if("#" in yline or "$" in yline):
        continue
    words = yline.split(',')    
    Eprotons.append(float(words[0]))
    #LightYield.append(float(words[1])*scaleFactor)
    LightYield.append(float(words[1]))
    LY_err.append(float(words[2]))

pYieldFile.close()

#def exponent(x,a,b,c):
#    return a + b*np.exp(c*x)
    
plt.figure()
plt.errorbar(Eprotons, LightYield, LY_err, fmt='o', linewidth=2, capsize=3, label='data, Laplace(2022)')
plt.xlabel("Proton Energy, [MeV]")
plt.ylabel("Light Yield , [MeVee]")
# lets fit some exponent
popt,pcov = curve_fit(exponent_woffset, Eprotons, LightYield)
print(popt)
fit = exponent_woffset(np.array(Eprotons), *popt)
plt.plot(Eprotons, fit, color='green', label=f'a+b*exp(x*c)')

spopt,spcov = curve_fit(squareFunc, Eprotons, LightYield)
print(spopt)
sfit = squareFunc(np.array(Eprotons), *spopt)
plt.plot(Eprotons, sfit, color='red', label=f'a*x^2+bx+c')

plt.hlines(0.447, 0,4, colors='black', linestyles='dashed')

plt.legend()
plt.savefig("scintiLightYield-protons.png")

##############################################################
#### scinti calculations end here ############################
##############################################################


efile = open(folder+'electrons_Pb_NIST.csv', 'r')

E_elec, e_CSDA, e_range = [],[],[]

for eline in efile:
    if("#" in eline):
        continue
    words = eline.split(',')    
    E_elec.append(float(words[0]))
    #e_CSDA.append(float(words[2])*Pb_rho)
    e_CSDA.append(float(words[2]))
    #print(f"reading {float(words[0])} -> {float(words[2])}")

efile.close()

plt.figure()
plt.scatter(E_elec, e_CSDA, marker='v', c='g')
plt.xlabel(r"$E_{electron}$, [Mev]")
plt.ylabel("Range, [cm]")
plt.xlim([0.1, 5])
plt.ylim([0, 5])
plt.title('Secondary electron range vs secondary electron energy')
plt.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--', linewidth=0.25)
plt.savefig("electron_range_in_lead.png")

##############################################################
##############################################################


print("THUS,")

Ebeam = 13 # MeV
LY_beam_protons_exp = exponent_woffset(Ebeam, *popt)
LY_beam_protons_sq = squareFunc(Ebeam, *spopt)

Ebeam_low = 7 # MeV
LY_beam_protons_exp_low = exponent_woffset(Ebeam_low, *popt)
LY_beam_protons_sq_low = squareFunc(Ebeam_low, *spopt)

print(f"the relative LY can be {LY_beam_protons_exp} or {LY_beam_protons_sq} MeVee")
plt.scatter([Ebeam, Ebeam_low], [LY_beam_protons_exp, LY_beam_protons_exp_low], marker="x", c='b', label=f'LY projected by exp.fit')
plt.scatter([Ebeam, Ebeam_low], [LY_beam_protons_sq, LY_beam_protons_sq_low], marker='x', c='g', label=f'LY projected by square fit')

plt.legend()
plt.savefig("scintiLightYield-protonsi-wprojected.png")


#############################################

########## xray ranges in air ###############

xrfile_air = open(folder+'gamma_stopping_power_air.csv','r')

Exray, muRho_air = [], []

for aline in xrfile_air:
    if('#' in aline):
        continue

    words = aline.split(',')
    Exray.append(float(words[0]))
    muRho_air.append(float(words[1]))

xrfile_air.close()

plt.figure()
plt.scatter(Exray, muRho_air, marker='^', c='m')
plt.xlabel(r"$E_{X-ray}$, [MeV]")
plt.ylabel(r'$\mu/\rho$'+r", $cm^{2}/g$")
plt.title('X-ray mass attenuation coeff. in dry air at sea level vs. X-ray energy')
plt.xscale('log')
plt.yscale('log')
plt.savefig('x-ray-stoping-power-dry-air.png')

Ifull = 1.0
E_test = 0.2
dx = 15
attCoeff_air = findClosest(E_test, Exray, muRho_air)
print(f'mass attenuation coefficient for 20keV X-rays in dry air is: {attCoeff}')
Iresult = calcIntensity(Ifull, attCoeff_air, dx)

desired_reduction = 2
air_density = 1.26
reduction_distance = calcAttenuation(desired_reduction, attCoeff_air, air_density)

print(f" in 15cm of air the intensity of 20keV xrays drops to {Iresult*100}")
print(f" To attenuate 20keV xrays by factor {desired_reduction} one needs {reduction_distance} cm of air")

fprange = open(folder+'p_range_in_BC-408_upd.csv','r')

Ep, prange = [], []

for pline in fprange:
    words = pline.split(',')
    Ep.append(float(words[0])) # in MeV
    prange.append(float(words[1])) # in cm

fprange.close()

plt.figure()
plt.scatter(Ep, prange, marker='x', c='r')
plt.xlabel(r'$E_{P}$, [MeV]')
plt.ylabel('Range, [mm]')
plt.title('Proton range in BC-408 vs proton energy')
#plt.xscale('log')
#plt.yscale('log')
#plt.savefig('proton_range_extracted.png')
plt.savefig('proton_range_extracted_fullErange.png')

Ep_sorted, prange_sorted = zip(*sorted(zip(Ep,prange)))

Ep_zoom, prange_zoom = [],[]

for i in range(len(Ep)):
    if(Ep_sorted[i] > 0.9 and Ep_sorted[i] < 20.1):
        Ep_zoom.append(Ep_sorted[i])
        prange_zoom.append(prange_sorted[i])

print(Ep_zoom)
print(prange_zoom)

plt.figure()
#plt.scatter()
plt.scatter(Ep_zoom, prange_zoom, marker='x', c='m')
plt.xlabel(r'$E_{P}$, [MeV]')
plt.ylabel('Range, [mm]')
plt.title('Proton range in BC-408 vs proton energy')
plt.xlim([0,15])
plt.ylim([0,2.6])

fopt, fcov = curve_fit(exponent_woffset, Ep_zoom, prange_zoom)
#fopt, fcov = curve_fit(linear, Ep, prange)
print(fopt)
prange_fit = exponent_woffset(np.array(Ep_zoom), *fopt)
parA = round(fopt[0],2)
parB = round(fopt[1],2)
parC = round(fopt[2],2)

#plt.plot(Ep_zoom, prange_fit, color='blue', label=f'fit={parA}*exp({parB}*x)')
plt.plot(Ep_zoom, prange_fit, color='blue', label=f'fit={parA}+{parB}*exp({parC}*x)')
#plt.plot(Ep, prange_fit, color='blue', label=f'fit=ln({parA})*x + ln({parB})')
cycloE_p = 13.16
cycloE_range = round(exponent_woffset(cycloE_p, *fopt),2)
plt.scatter([cycloE_p],[cycloE_range],marker='v', c='g',label=f'proton range @13.16 MeV= {cycloE_range} mm')
plt.hlines(cycloE_range,0,cycloE_p, colors='black', linestyles='dashed')
plt.vlines(cycloE_p,0,cycloE_range, colors='black', linestyles='dashed')

plt.legend()

plt.savefig('proton_range_extracted_zoom.png')

fLOprotons = open(folder+'rel_LO_protons_in_BC-408.csv','r')

LO_protons, proton_energy = [],[]

for loline in fLOprotons:
    words = loline.split(',')
    proton_energy.append(float(words[0]))
    LO_protons.append(float(words[1]))

fLOprotons.close()

sort_Eprotons, sort_LO_protons = sortLists(proton_energy, LO_protons) 

plt.figure()
plt.scatter(sort_Eprotons, sort_LO_protons, marker='x', c='g')
plt.xlabel('Proton Energy, [MeV]')
plt.ylabel('Relative Proton LO, [MeVee]')
plt.title("Relative LO of Protons vs Proton energy")
plt.savefig("protn_LO.png")

lowEp = sort_Eprotons[0:3]
lowEp_LO = sort_LO_protons[0:3]

plt.figure()
plt.scatter(lowEp, lowEp_LO, marker='x', c='g')
plt.xlabel('Proton Energy, [MeV]')
plt.ylabel('Relative Proton LO, [MeVee]')
plt.title("Relative LO of Protons vs Proton energy")

loopt, locov = curve_fit(linear, lowEp, lowEp_LO)
lowE_Fit = linear(np.array(lowEp), *loopt)
plt.plot(lowEp, lowE_Fit, color='red')

cycloP_LO = linear(13.16, *loopt)
print(f"Relative LO for cyclotron protons @ 13.16 MeV is {cycloP_LO}")

plt.savefig("protn_LO.png")

###############################################################################
#      plottin proton/alpha LY 
###############################################################################

fPLY = open(folder+'proton_LY_in_plastic.csv','r')
fALY = open(folder+'alpha_LY_in_plastic.csv','r')

protonLY, pEnergy_LY = [], []
alphaLY, alphaEnergy_LY = [], []

for lyLine in fPLY:
    words = lyLine.split(',')
    pEnergy_LY.append(float(words[0]))
    protonLY.append(float(words[1]))

fPLY.close()

for alyLine in fALY:
    words = alyLine.split(',')
    alphaEnergy_LY.append(float(words[0]))
    alphaLY.append(float(words[1]))

fALY.close()

plt.figure()
plt.scatter(pEnergy_LY, protonLY, marker='x',c='g', label="Protons")
plt.scatter(alphaEnergy_LY, alphaLY, marker='+',c='b', label='Alphas')
plt.xlabel('Particle Energy, MeV')
plt.ylabel(r'Light Yield (BC-400), $N_{photons}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('LY_protons_alphas.png')

pEnergy_LY, protonLY = sortLists(pEnergy_LY, protonLY)

pEnergy_LY_zoom, protonLY_zoom = [],[]

for i in range(len(pEnergy_LY)):
    if(pEnergy_LY[i] > 0.9 and pEnergy_LY[i] < 20.1):
        pEnergy_LY_zoom.append(pEnergy_LY[i])
        protonLY_zoom.append(protonLY[i])

plt.figure()
#plt.scatter(pEnergy_LY_zoom, protonLY_zoom, marker='o', c='black')
plt.scatter(pEnergy_LY_zoom, np.array(protonLY_zoom)*0.001, marker='o', c='black')
plt.xlabel(r'$E_{proton}$, MeV')
plt.ylabel(r'$LY_{protons}$, $N_{photons}\times 10^{3}$')

lyopt, lycov = curve_fit(exponent_woffset, pEnergy_LY_zoom, protonLY_zoom)
LY_fit = exponent_woffset(np.array(pEnergy_LY_zoom), *lyopt)

plt.plot(pEnergy_LY_zoom, LY_fit*0.001, color='red', linestyle='dashed', label="expo.fit")

LY_cyclorptons = round(exponent_woffset(13.16, *lyopt))
plt.scatter([13.16],[LY_cyclorptons*0.001], marker='*', c='green', label=f' proton LY @13.16 MeV = {LY_cyclorptons} photons')
plt.hlines(LY_cyclorptons*0.001,0,13.16, colors='blue', linestyles='dashed')
plt.vlines(13.16,-2.5,LY_cyclorptons*0.001, colors='blue', linestyles='dashed')
plt.xlim([0,20.5])
plt.ylim([-2.5,120])
plt.legend()
plt.grid()
plt.savefig('protonLY_zoom.png')

#######################################################
# x-ray attenuation in scinti material
# using data for H and C separately and combinig by weighted average
###########################################
#
#xrfile_H = open(folder+'hydrogen_mass_att_coeff_xrays.csv','r')
#xrfile_C = open(folder+'carbon_mass_att_coeff_xrays.csv','r')
#
#Exray_H, Exray_C, muRho_H, nuRho_C = [], [], [], []
#
#for hline in xrfile_H:
#    if('#' in hline):
#        continue
#
#    words = hline.split(',')
#    Exray_H.append(float(words[0]))
#    muRho_H.append(float(words[1]))
#
#xrfile_H.close()
#
#for cline in xrfile_C:
#    if('#' in cline):
#        continue
#
#    words = cline.split(',')
#    Exray_C.append(float(words[0]))
#    muRho_C.append(float(words[1]))
#
#xrfile_H.close()
#
## weighting 
#for Eh, Ec in zip(Exray_H,Exray_C):
#    if(Eh != Ec):
#        print(f"Energy_H={Eh} not equal to Erergy_C={Ec} !")
#        exit(0)
#
#comb_scintiCoeffs = []
#rel
#
#for 
#
# x-ray attenuation in scinti material


xrfile_scinti = open(folder+'scinti_massAttCoeff_NIST.csv','r')

Exray_scinti, muRho_scinti = [], []

for sciline in xrfile_scinti:
    if('#' in sciline):
        continue

    words = sciline.split(',')
    Exray_scinti.append(float(words[0]))
    muRho_scinti.append(float(words[1]))

xrfile_scinti.close()

plt.figure(figsize=(16,8))
plt.scatter(Exray_scinti, muRho_scinti, marker='^', c='firebrick')
plt.xlabel(r"$E_{X-ray}$, [MeV]")
plt.ylabel(r'$\mu/\rho$'+r", $cm^{2}/g$")
plt.title('X-ray mass attenuation in scintillator vs X-ray energy')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.savefig('x-ray-attenuation-scintillator.png')




