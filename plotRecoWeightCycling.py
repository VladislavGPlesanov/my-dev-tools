import numpy as np
import sys
import matplotlib.pyplot as plt

logfile = sys.argv[1]
picname = sys.argv[2]
ptype = sys.argv[3]

fRMIN, fRMAX, fWEIGHT = False, False, False

if("rmin" in ptype or "Rmin" in ptype or "Rin" in ptype or "rin" in ptype):
    fRMIN = True
if("rmax" in ptype or "Rmax" in ptype or "Rout" in ptype or "rout" in ptype):
    fRMAX = True
if("weight" in ptype or "Weight" in ptype or "wght" in ptype or "WGHT" in ptype):
    fWEIGHT = True

if(not fRMIN and not fRMAX and not fWEIGHT):
    print(f"Input argument [3] = {ptype} si invalid for all plot checks.")
    print("Use:\n rmin, Rmin, Rin, rin -> inner circle data,\nrmax, Rmax, Rout, rout -> outer circel data,\nweight,Weight, wght, WGHT -> weight data")
    exit(0)

if((fRMIN and fRMAX) or 
   (fRMIN and fWEIGHT) or 
   (fRMAX and fWEIGHT) or
   (fRMAX and fWEIGHT and fRMIN)):
    print('Can only use one option for argv[3] !')
    exit(0)

f=open(logfile,'r')

# ===========================================================
# these are copierd from "~/TPA/cycleParameters.sh"
weights = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0]
Rmin = [ 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0 ]
Rmax = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]
# ===========================================================

mu, muErr = [],[]
muOrig, muErrOrig = [], []

absX, absY, absDX, absDY = [], [],[],[]
chired, chiredOrig = [], []

phi, phierr = [], []
phiOrig, phiOrigErr = [], []

mu_cnt = 0
muErr_cnt = 0
nrun = 0

fModCut = False

for line in f:

    if("Absorption peak" in line):
        dataXY = line.split(":")[1].split(",")
        absDX.append(float(dataXY[0]))
        absDY.append(float(dataXY[1]))

    if("Mod_Cut" in line):
        data = line.split(":")[1]
        params = data.split(",")
        for par in params:
            if("mu=" in par):
                mu.append(float(par.split("=")[1])*100)
                mu_cnt+=1
            if("muErr=" in par):
                muErr.append(float(par.split("=")[1])*100)
            if("phi=" in par):
                phi.append(float(par.split("=")[1]))
            if("phierr=" in par):
                phierr.append(float(par.split("=")[1]))
            if("chired=" in par):
                chired.append(float(par.split("=")[1])) 

    if("Mod_Orig" in line):
        data = line.split(":")[1]
        params = data.split(",")
        for par in params:
            if("mu=" in par):
                muOrig.append(float(par.split("=")[1])*100)
            if("muErr=" in par):
                muErrOrig.append(float(par.split("=")[1])*100)
            if("phi=" in par):
                phiOrig.append(float(par.split("=")[1]))
            if("phierr=" in par):
                phiOrigErr.append(float(par.split("=")[1]))
            if("chired=" in par):
                chiredOrig.append(float(par.split("=")[1])) 


mean_mu = np.mean(mu)
mu_std = np.std(mu)
mean_mu_orig = np.mean(muOrig)
mu_orig_std = np.std(muOrig)

if(fWEIGHT):
    # plotting stufff vs weighting 
    fig = plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    
    ax.errorbar(weights[0:mu_cnt], muOrig, yerr=muErrOrig, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\overline{\mu_{Orig}}=$"+f"{mean_mu_orig:.2f} std=({mu_orig_std:.2f})")
    ax.errorbar(weights[0:mu_cnt], mu, yerr=muErr, color='orange', ecolor='black', fmt='*',capsize=4, label=r"$\overline{\mu_{Cut}}=$"+f"{mean_mu:.2f} std=({mu_std:.2f})")#r"$\mu$, cut")
    
    ax.set_title("Estimated Modualtion Factors Based on Weighting")
    ax.set_xlabel("Weight")
    ax.set_ylabel(r"$\mu$,[%]")
    ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
    ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
    ax.legend(loc='lower right')
    plt.savefig(f"ModFacs-vs-Weights-{picname}.png",dpi=800)
    plt.close()

if(not fWEIGHT):
    # plotting stufff vs R-min / R-max
    fig = plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    if(fRMIN):
        ax.errorbar(Rmin[0:mu_cnt], muOrig, yerr=muErrOrig, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\overline{\mu_{Orig}}=$"+f"{mean_mu_orig:.2f} std=({mu_orig_std:.2f})")
        ax.errorbar(Rmin[0:mu_cnt], mu, yerr=muErr, color='orange', ecolor='black', fmt='*',capsize=4, label=r"$\overline{\mu_{Cut}}=$"+f"{mean_mu:.2f} std=({mu_std:.2f})")#r"$\mu$, cut")
        ax.set_title(r"Estimated Modualtion Factors Based on $R_{Min}$")
        ax.set_xlabel(r"$R_{Min}$")
    else:
        ax.errorbar(Rmax[0:mu_cnt], muOrig, yerr=muErrOrig, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\overline{\mu_{Orig}}=$"+f"{mean_mu_orig:.2f} std=({mu_orig_std:.2f})")
        ax.errorbar(Rmax[0:mu_cnt], mu, yerr=muErr, color='orange', ecolor='black', fmt='*',capsize=4, label=r"$\overline{\mu_{Cut}}=$"+f"{mean_mu:.2f} std=({mu_std:.2f})")#r"$\mu$, cut")
        ax.set_title(r"Estimated Modualtion Factors Based on $R_{Max}$")   
        ax.set_xlabel(r"$R_{Max}$")

    ax.set_ylabel(r"$\mu$,[%]")
    plt.ylim([0.0,35])
    ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
    ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
    ax.legend(loc='lower right')
    if(fRMIN):
        plt.savefig(f"ModFacs-vs-Rmin-{picname}.png",dpi=800)
    else:
        plt.savefig(f"ModFacs-vs-Rmax-{picname}.png",dpi=800)
    plt.close()


# plotting some chi reduced 
plt.figure()
if(fWEIGHT):
    # weights
    plt.scatter(weights[0:mu_cnt],chired, marker="+", c='red', ls='--')
    plt.scatter(weights[0:mu_cnt],chired, marker="+", c='orangered', ls='--', label="With cuts")
    plt.scatter(weights[0:mu_cnt],chiredOrig, marker="+", c='firebrick', ls='--', label='No cuts')
    plt.xlim([np.min(weights)-0.2,np.max(weights)+0.2])
    plt.xlabel(r"Q weight")
elif(fRMIN):
    # Rmin
    plt.scatter(Rmin[0:mu_cnt],chired, marker="+", c='orangered', ls='--', label="With cuts")
    plt.scatter(Rmin[0:mu_cnt],chiredOrig, marker="+", c='firebrick', ls='--', label='No cuts')
    plt.xlim([np.min(Rmin)-0.2,np.max(Rmin)+0.2])
    plt.xlabel(r"$R_{Min}$")
else:
    # Rmax
    plt.scatter(Rmax[0:mu_cnt],chired, marker="+", c='orangered', ls='--', label="With cuts")
    plt.scatter(Rmax[0:mu_cnt],chiredOrig, marker="+", c='firebrick', ls='--', label='No cuts')
    plt.xlim([np.min(Rmax)-0.2,np.max(Rmax)+0.2])
    plt.xlabel(r"$R_{Max}$")

plt.grid(True)
plt.legend()
plt.ylabel(r"Fit $\chi_{reduced}^{2}$")
if(fWEIGHT):
    plt.savefig(f"ModFacs-Weight-CHIRED-{picname}.png")
elif(fRMIN):
    plt.savefig(f"ModFacs-Rmin-CHIRED-{picname}.png")
else:
    plt.savefig(f"ModFacs-Rmax-CHIRED-{picname}.png")

plt.close()


## plotting absorption point stds for [weights] 
#plt.figure()
#plt.grid(True)
#plt.xlim([0.0,4.2])
#plt.xlabel("Weight")
#plt.ylabel(r"Absorption point reconstruction dev.")
#plt.savefig(f"ModFacs-AbsorptionPoint-STD-{picname}.png")
#plt.close()

## plotting absorption point stds for [Rmin]
plt.figure(figsize=(8,6))
if(fWEIGHT):
    # weights
    plt.scatter(weights[0:mu_cnt],absDX[0:mu_cnt], marker="+", c='forestgreen', ls='--')
    plt.scatter(weights[0:mu_cnt],absDY[0:mu_cnt], marker="+", c='royalblue', ls='--')
    plt.xlim([np.min(weights)-0.2,np.max(weights)+0.2])
    plt.xlabel("Q weight [a.u.]")
elif(fRMIN):
    #Rmin
    plt.scatter(Rmin[0:mu_cnt],absDX[0:mu_cnt], marker="+", c='forestgreen', ls='--')
    plt.scatter(Rmin[0:mu_cnt],absDY[0:mu_cnt], marker="+", c='royalblue', ls='--')
    plt.xlim([np.min(Rmin)-0.2,np.max(Rmin)+0.2])
    plt.xlabel(r"$R_{Min}$")
else:
    #Rmax
    plt.scatter(Rmax[0:mu_cnt],absDX[0:mu_cnt], marker="+", c='forestgreen', ls='--')
    plt.scatter(Rmax[0:mu_cnt],absDY[0:mu_cnt], marker="+", c='royalblue', ls='--')
    plt.xlim([np.min(Rmax)-0.2,np.max(Rmax)+0.2])
    plt.xlabel(r"$R_{Max}$")


plt.ylabel(r"$\sigma$of the Beam Spot Reconstruction ")
plt.grid(True)
if(fWEIGHT):
    plt.savefig(f"AbsorptionPoint-STDEV-weights-{picname}.png")
elif(fRMIN):
    plt.savefig(f"AbsorptionPoint-STDEV-Rmin-{picname}.png")
else:
    plt.savefig(f"AbsorptionPoint-STDEV-Rmax-{picname}.png")
plt.close()

## plotting reconstructed angles for [weights]
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)
THE_ANGLE = 45*np.pi/180

hmax, hmin = None, None

if(fWEIGHT):
    # weights
    ax.errorbar(weights[0:mu_cnt], phi, yerr=phierr, color='darkorange', ecolor='black', fmt='*',capsize=4, label=r"$\phi$ (Cut)")
    ax.errorbar(weights[0:mu_cnt], phiOrig, yerr=phiOrigErr, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\phi$ (Original)")
    hmax = np.max(weights)+0.2
    hmin = np.min(weights)-0.2
    ax.set_xlabel("Q weight")
elif(fRMIN):
    # Rmin
    ax.errorbar(Rmin[0:mu_cnt], phi, yerr=phierr, color='darkorange', ecolor='black', fmt='*',capsize=4, label=r"$\phi$ reconstructed ")
    ax.errorbar(Rmin[0:mu_cnt], phiOrig, yerr=phiOrigErr, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\phi$ (uncut) reconstructed ")
    hmax = np.max(Rmin)+0.2
    hmin = np.min(Rmin)-0.2
    ax.set_xlabel(r"$R_{Min}$")

else:
    # Rmax
    ax.errorbar(Rmax[0:mu_cnt], phi, yerr=phierr, color='darkorange', ecolor='black', fmt='*',capsize=4, label=r"$\phi$ reconstructed ")
    ax.errorbar(Rmax[0:mu_cnt], phiOrig, yerr=phiOrigErr, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\phi$ (uncut) reconstructed ")
    hmax = np.max(Rmax)+0.2
    hmin = np.min(Rmax)-0.2
    ax.set_xlabel(r"$R_{Max}$")

avg_phi = np.average(np.array(phi))
avg_phi_deg = avg_phi*180.0/np.pi
ax.hlines(avg_phi, hmin, hmax, linestyles='--', colors='firebrick', label=r"$\phi$ AVG "+f"= {avg_phi:.4f}[rad]=>{avg_phi_deg:.4f}"+r"[$^{\circ}$]")

ax.grid(True)
ax.set_xlim([hmin,hmax])
ax.set_ylim([-0.5,0.5])
ax.set_ylabel(r"$\phi$, [radian]")
ax.legend(loc='lower right')
if(fWEIGHT):
    plt.savefig(f"recoAngles-weights-{picname}.png")
elif(fRMIN):
    plt.savefig(f"recoAngles-Rmin-{picname}.png")
else:
    plt.savefig(f"recoAngles-Rmax-{picname}.png")
plt.close()


