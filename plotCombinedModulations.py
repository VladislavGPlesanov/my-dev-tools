import numpy as np
import sys
import matplotlib.pyplot as plt

logfile = sys.argv[1]
picname = sys.argv[2]
xvalues = sys.argv[3]
#piclabels = sys.argv[4]

f=open(logfile,'r')

xvalues = xvalues.split(",")
xdata = [float(i) for i in xvalues]
print(f"xvalues:{xdata}, where ith member is [{type(xdata[0])}]")

#piclabels = piclabels.split(",")

mu, muErr = [],[]
muCut, muCutErr = [], []

absX, absY, absDX, absDY = [], [],[],[]
chired = []

phi, phierr = [], []
phiCut, phiCutErr = [], []

mu_cnt = 0
muErr_cnt = 0
nrun = 0

for line in f:

    if("Absorption peak" in line):
        dataXY = line.split(":")[1].split(",")
        absDX.append(float(dataXY[2]))
        absDY.append(float(dataXY[3]))

    if("Mod_Orig" in line):
        data = line.split(":")[1]
        params = data.split(",")
        for par in params:
            if("mu=" in par):
                mu.append(float(par.split("=")[1])*100.0)
            if("muErr=" in par):
                muErr.append(float(par.split("=")[1])*100.0)
            if("phi=" in par):
                phi.append(float(par.split("=")[1]))
            if("phierr=" in par):
                phierr.append(float(par.split("=")[1]))

    if("Mod_Cut" in line):
        data = line.split(":")[1]
        params = data.split(",")
        for par in params:
            if("mu=" in par):
                muCut.append(float(par.split("=")[1])*100.0)
            if("muErr=" in par):
                muCutErr.append(float(par.split("=")[1])*100.0)
            if("phi=" in par):
                phiCut.append(float(par.split("=")[1]))
            if("phierr=" in par):
                phiCutErr.append(float(par.split("=")[1]))

print(mu)
print(muErr)

fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
#if(len(piclabels)>3):
#    ax.errorbar(xdata[0:len(mu)], mu, yerr=muErr, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=piclabels[3])
#else:
#    ax.errorbar(xdata[0:len(mu)], mu, yerr=muErr, color='forestgreen', ecolor='black', fmt='*',capsize=4)
ax.errorbar(xdata[0:len(mu)], mu, yerr=muErr, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\mu$(Global)")
ax.errorbar(xdata[0:len(mu)], muCut, yerr=muCutErr, color='orange', ecolor='black', fmt='*',capsize=4, label=r"$\mu$(Cut)")
ax.set_title("Modulation factors VS Detector Rotation Angle")
ax.set_xlabel("Angle [deg]")
ax.set_ylabel(r"Reconstructed $\mu$")

#ax.set_title(piclabels[0])
#ax.set_xlabel(piclabels[1])
#ax.set_ylabel(piclabels[2])
ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax.set_ylim([0,50])
#if(len(piclabels)>3):
ax.legend(loc='lower right')
plt.savefig(f"ModFacs-{picname}.png",dpi=1000)
plt.close()

# plotting reconstructed angles
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
#THE_ANGLE = 45*np.pi/180

phi_deg = abs(np.array(phi)*180/np.pi)
phierr_deg = np.array(phierr)*180/np.pi
phiCut_deg = abs(np.array(phiCut)*180/np.pi)
phiCutErr_deg = np.array(phiCutErr)*180/np.pi

ax.errorbar(xdata[0:len(phi_deg)], phi_deg, yerr=phierr_deg, color='forestgreen', ecolor='black', fmt='*',capsize=4)
ax.errorbar(xdata[0:len(phiCut_deg)], phiCut_deg, yerr=phiCutErr_deg, color='orange', ecolor='black', fmt='*',capsize=4)

#ax.plot(xdata[0:len(phi)],xdata[0:len(phi)],color='red', linestyle='--')
#ax.scatter([0,30,60,90],[0,30,60,90],marker="+", c='firebrick',s=128)
ax.hlines(np.mean(phi_deg), np.min(xdata), np.max(xdata), label='mean, uncut, angle', linestyles=':')
ax.grid(True)
ax.set_ylim([70,110])
ax.set_xlabel(r"Real $\phi$")
ax.set_ylabel(r"Reconstructed $\phi$, [radian]")
plt.savefig(f"RecoAngles-{picname}.png")
plt.close()



