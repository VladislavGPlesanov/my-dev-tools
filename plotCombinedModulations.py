import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import re
import argparse as ap


parser = ap.ArgumentParser()
parser.add_argument("-p", "--picname", type=str, default="ALALA")
parser.add_argument("-l","--logfile",type=str,default="")
parser.add_argument("-x","--xval",type=str,default="")
parser.add_argument("-t","--type",type=str.lower,
                    choices=["rate","angles","gain","weights"],
                    help="Enter type of run (angles,rate,weights,gain)",
                    required=True)
parser.add_argument("--single",action='store_true')
parser.add_argument("--custom_title",type=str,default="")
parser.add_argument("--custom_xlab",type=str,default="")
parser.add_argument("--custom_ylab",type=str,default="")

args = parser.parse_args()

#logfile = sys.argv[1]
#picname = sys.argv[2]
#xvalues = sys.argv[3]

logfile = args.logfile
picname = args.picname
xvalues = args.xval
fSingle = args.single
runtype = args.type
cuTitle = args.custom_title
cuXlabel = args.custom_xlab
cuYlabel = args.custom_ylab

fCustomData = False
if(len(cuTitle)>0 and len(cuXlabel)>0 and len(cuYlabel)>0):
    print("Custom data flag set...")
    fCustomData = True
    if(len(xvalues)==0):
        print("Input from option -x / --xval is required!")
        
f=open(logfile,'r')

xdata = None
if(len(xvalues)>0):
    xdata = [float(i) for i in xvalues.split(",")]

datacheckstr = f"xvalues:{xdata}, plotting {runtype}"
if(fCustomData):
    datacheckstr+=f"[CUSTOM DATA]"
print(f"{datacheckstr}")
fRate, fAngles, fGain = False, False, False
fweight = False

voltages, rates = [], []
weights, angles  = [], []
#weights = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0]

if(runtype=="rate"):
    fRate = True
if(runtype=="angles"):
    fAngles = True
if(runtype=="gain"):
    fGain = True
if(runtype=="weights"):
    fweight = True

if(len(cuTitle)>0):
    print("USing custom labels and data\nsetting other run type flags to [FALSE]")
    fweight = False
    fAngles = False
    fGain = False
    fRate = False

#plotlabels = []
##if(not fRate and not fAngles and not fGain and not fweight):
#if(len(customlab)>0):
#    labels = customlab.split(":")
#    for lab in labels:
#        plotlabels.append(re.sub("-"," ",lab))

mu, muErr = [],[]
muCut, muCutErr = [], []

Bp = None

absX, absY, absDX, absDY = [], [],[],[]

chired, chiredmod = [], []

ndata, ndatacut = [], []

I, Imod, Ierr, Imoderr = [], [], [], []
U, Umod, Uerr, Umoderr = [], [], [], []
Q, Qmod, Qerr, Qmoderr = [], [], [], []

phi, phierr = [], []
phiCut, phiCutErr = [], []

isSimulation = []

mu_cnt = 0
muErr_cnt = 0
nrun = 0

fValid = True
fDrift = False

for line in f:

    if("-SUCCESS-" in line):
        break

    if("Absorption peak" in line):
        dataXY = line.split(":")[1].split(",")
        absDX.append(float(dataXY[2]))
        absDY.append(float(dataXY[3]))

    if(fweight and "Angle-Reco-const" in line):
        recodata = line.split(":")[1].split(",")
        weights.append(float(recodata[-1]))

    if("Current file:" in line):
        if("SIM" in line.split(":")[1]):
            isSimulation.append(1)
        else:
            isSimulation.append(0)

    if(fGain and "Current file:" in line):
        cutChar = -1
        fname = line.split(":")[1].split(".")[0]
        if("MP" in fname and "Vcm" in fname):
            cutChar = -3
            fDrift = True
        volt = fname.split("-")[-1][:cutChar] # removing "V" from thestring to enable int() conversion
        voltages.append(float(volt))

    if(fRate and "Current file" in line):
        fname = line.split(":")[1].split(".")[0]
        if("Hz" in fname):
            irate = fname.split("-")[-2][:-2] # getting xxxxHz and removing "Hz" part
            rates.append(int(irate))
        else:
            print(f"Cant' find rate in the file name {fname}!")
            exit(0)
        
    if(fAngles and "Current file" in line):
        fname = line.split(":")[1].split(".")[0]
        if("deg" in fname):
            iangle = fname.split("-")[-1][:-3] # getting xxxxHz and removing "Hz" part
            angles.append(int(iangle))
        else:
            print(f"Cant' find rate in the file name {fname}!")
            exit(0)
 

    if("N_clusters" in line):
        ndata.append(float(line.split("=")[1]))
    if("Data after cuts" in line):
        ndatacut.append(float(line.split(":")[1]))

    if("Mod_Orig" in line):
        data = line.split(":")[1]
        params = data.split(",")
        for par in params:
            if("mu=" in par):
                mu.append(float(par.split("=")[1])*100.0)
            if("Bp=" in par and Bp is None):
                print("Found Bp!")
                Bp = float(par.split("=")[1])
            if("muErr=" in par):
                muErr.append(float(par.split("=")[1])*100.0)
            if("phi=" in par):
                phi.append(float(par.split("=")[1]))
            if("phierr=" in par):
                phierr.append(float(par.split("=")[1]))
            if("chired=" in par):
                chired.append(float(par.split("=")[1]))           
            if("I=" in par):
                I.append(float(par.split("=")[1]))
            if("Ierr=" in par):
                Ierr.append(float(par.split("=")[1]))
            if("U=" in par):
                U.append(float(par.split("=")[1]))
            if("Uerr=" in par):
                Uerr.append(float(par.split("=")[1]))
            if("Q=" in par):
                Q.append(float(par.split("=")[1]))
            if("Qerr=" in par):
                Qerr.append(float(par.split("=")[1]))


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
            if("chired=" in par):
                chiredmod.append(float(par.split("=")[1]))           
            if("I=" in par):
                Imod.append(float(par.split("=")[1]))
            if("Ierr=" in par):
                Imoderr.append(float(par.split("=")[1]))
            if("U=" in par):
                Umod.append(float(par.split("=")[1]))
            if("Uerr=" in par):
                Umoderr.append(float(par.split("=")[1]))
            if("Q=" in par):
                Qmod.append(float(par.split("=")[1]))
            if("Qerr=" in par):
                Qmoderr.append(float(par.split("=")[1]))


print(mu)
print(len(mu))
print(muErr)
print(len(muErr))
print(U)
print(len(U))
print(Uerr)
print(len(Uerr))
print(Q)
print(len(Q))
print(Qerr)
print(len(Qerr))
print(f"Data numbers lists pre={len(ndata)}, post={len(ndatacut)}, len(mu)={len(mu)}")

print(f"Bp={Bp}")

if(fGain):
    xdata = voltages
if(fRate):
    xdata = rates
if(fweight):
    xdata = weights
if(fAngles):
    xdata = angles

print(f"ndata:\n{ndata}")
print(f"ndatacut:\n{ndatacut}")
print(f"xdata:\n{xdata}")

############################################################
# tryna plot things with second chisquare plot underneath
############################################################

fig = plt.figure(figsize=(10,8))
#gs = GridSpec(2,1, width_ratios=[2,1,0.2], height_ratios=[4,1], hspace=0.05, wspace=0.05)
gs = GridSpec(3,1, width_ratios=[1], height_ratios=[3,1,1], hspace=0.02, wspace=0.05)
ax_top = fig.add_subplot(gs[0,0])
ax_bot = fig.add_subplot(gs[1,0], sharex=ax_top)
ax_bot2 = fig.add_subplot(gs[2,0], sharex=ax_top)

# adjusting modulation main mod factor plots on top
ax_top.errorbar(xdata[0:len(mu)], mu, yerr=muErr, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"Before cuts")
ax_top.errorbar(xdata[0:len(mu)], muCut, yerr=muCutErr, color='orange', ecolor='black', fmt='*',capsize=4, label=r"After cuts")
ax_top.set_title("Modulation Factors VS Xray Beam Polarization Angle")
ax_top.set_ylabel(r"Reconstructed $\mu$")
if(fGain):
    ax_top.set_title("Modulation Factors vs Grid Voltage")
if(fRate):
    ax_top.set_title("Modulation Factors vs Incident X-ray Beam Rate")
if(fweight):
    ax_top.set_title("Modulation Factors vs Charge Weighting in Reconstruction")
if(not fRate and not fAngles and not fGain and not fweight):
    ax_top.set_title(f"Modulation Factors vs {cuTitle}")

ax_top.set_ylim([0,max(xdata[0:len(mu)])*1.1])
ax_top.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax_top.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax_top.tick_params(labelbottom=False)
ax_top.legend()

# adjusting chisquare plot below
ax_bot.scatter(xdata[0:len(mu)], chired, color='forestgreen', marker='+', s=44, label=r"$\chi^{2}_{reduced} (Original)$")
ax_bot.scatter(xdata[0:len(mu)], chiredmod, color='orange', marker='+', s=44, label=r"$\chi^{2}_{reduced} (After cuts)$")
ax_bot.set_ylabel(r'$\chi^{2}_{reduced}$')
ax_bot.set_ylim([1e-1,np.max([np.max(chired),np.max(chiredmod)])*1.5])
ax_bot.set_yscale('log')
ax_bot.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax_bot.tick_params(labelbottom=False)

# adding statistics plot even more bottom

if(fRate):
    barwidth_cold = (np.array(xdata[0:len(mu)])-2)/5
    barwidth_hot = (np.array(xdata[0:len(mu)])+2)/5
    ax_bot2.bar(np.array(xdata[0:len(mu)]), ndata, color='blue', width=barwidth_cold, edgecolor='gray', label="Before cuts")
    ax_bot2.bar(np.array(xdata[0:len(mu)]), ndatacut, color='red', width=barwidth_hot, edgecolor='gray', label="After cuts")
elif(fGain):
    ax_bot2.bar(np.array(xdata[0:len(mu)])-1, ndata, color='blue', width=2, edgecolor='gray', label='Before cuts')
    ax_bot2.bar(np.array(xdata[0:len(mu)])+1, ndatacut, color='red', width=2, edgecolor='gray', label='After cuts')
elif(fAngles):
    ax_bot2.bar(np.array(xdata[0:len(mu)])-2.5, ndata, color='blue', width=5, edgecolor='gray', label = 'Before cuts')
    ax_bot2.bar(np.array(xdata[0:len(mu)])+2.5, ndatacut, color='red', width=5, edgecolor='gray', label='After cuts')
elif(len(cuTitle)>0):
    print("ALLEEEGG")
    xdata_width = np.sum(np.diff(xdata))/(len(xdata)-1)/4
    ax_bot2.bar(np.array(xdata[0:len(mu)])+xdata_width/4, ndata, color='blue', width=xdata_width/2, edgecolor='gray', label='Before cuts')
    ax_bot2.bar(np.array(xdata[0:len(mu)])-xdata_width/4, ndatacut, color='red', width=xdata_width/2, edgecolor='gray', label='After cuts')
else:
    #xdata_width = 0.2
    xdata_width = np.sum(np.diff(xdata))/(len(xdata)-1)/4
    ax_bot2.bar(np.array(xdata[0:len(mu)])+xdata_width/4, ndata, color='blue', width=xdata_width/2, edgecolor='gray', label = 'Before cuts')
    ax_bot2.bar(np.array(xdata[0:len(mu)])-xdata_width/4, ndatacut, color='red', width=xdata_width/2, edgecolor='gray', label='After cuts')

ax_bot2.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax_bot2.set_ylabel(r'$N_{clusters}$')
ax_bot2.set_xlabel(r'Angles, [$^{\circ}$]')
if(fRate):
    ax_bot2.set_xlabel('Rate, [Hz]')
    ax_bot2.set_xscale('log')
    ax_bot2.set_yscale('log')
if(fGain):
    ax_bot2.set_xlabel(r'$V_{Grid}$, [V]')
    ax_bot2.set_yscale('log')
if(fweight):
    ax_bot2.set_xlabel(r'$w_{Q}$, [a.u]')
if(not fRate and not fAngles and not fGain and not fweight):
    ax_bot2.set_xlabel(cuXlabel)
ax_bot2.legend()

plt.tight_layout()
plt.savefig(f"ModFacs-IMPROVED-{picname}.png")
plt.close()

######################################################################################
#  plotting simlified data for MODULATION FACTORS without auxiliary plots
######################################################################################
if(fSingle):
    # separate, modulations only single  plot
   
    # pre-firing the error bar case where they exceed modulation by 100%
    # first for original mod factors (before cuts)

    mu_ylow = np.array(muErr)
    mu_yhigh = []
    for mc, mce in zip(mu, muErr):
        if(mc+mce>=100):
            mu_yhigh.append(mce-(mc+mce-100.0))
        else:
            mu_yhigh.append(mce)
    mc,mce = None, None
    
    asym_muErr = np.array([mu_ylow, mu_yhigh])

    print(mu_ylow)
    print(muErr)
    print(mu_yhigh)

    # same for the mod factors after cuts
    mucut_ylow = np.array(muCutErr)
    mucut_yhigh = []
    for mc, mce in zip(muCut, muCutErr):
        if(mc+mce>=100):
            mucut_yhigh.append(mce-(mc+mce-100.0))
        else:
            mucut_yhigh.append(mce)
    print(mucut_ylow)
    print(muCutErr)
    print(mucut_yhigh)
    asym_muCutErr = np.array([mucut_ylow, mucut_yhigh])

    #  
    fig = plt.figure(figsize=(8,7))
    
    ax = fig.add_subplot(111)
    
    #ax.text(120,29,"WORK IN PROGRESS", color='red', backgroundcolor='lightgray', fontweight='semibold', fontsize=12)
    #ax.errorbar(xdata[0:len(mu)], mu, yerr=muErr, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\mu$ Before cuts")
    #ax.errorbar(np.array(xdata[0:len(mu)]) +2.5, mu, yerr=asym_muErr, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\mu$ Before cuts")
    ax.errorbar(xdata[0:len(mu)], mu, yerr=asym_muErr, color='forestgreen',ms=8, ecolor='darkgreen', fmt='*',capsize=6, label=r"$\mu$ Before cuts")
    #ax.errorbar(xdata[0:len(mu)], muCut, yerr=muCutErr, color='orange', ecolor='black', fmt='*',capsize=4, label=r"$\mu$ After cuts")
    ax.errorbar(xdata[0:len(mu)], muCut, yerr=asym_muCutErr, color='orange',ms=8,  ecolor='darkgoldenrod', fmt='*',capsize=6, label=r"$\mu$ After cuts")
    ax.set_ylabel(r"Reconstructed $\mu$")

    if(fAngles):
        ax.set_title("Modulation Factors VS Xray Beam Polarization Angle")
        ax.set_xlabel(r"Angle of Detector Rotation, [$^{\circ}$]")
    if(fGain):
        ax.set_title("Modulation Factors vs Grid Voltage")
        ax.set_xlabel(r"$V_{\mathrm{grid}}$, [V]")
    if(fRate):
        ax.set_title("Modulation Factors vs Incident X-ray Beam Rate")
        ax.set_xscale('log')
        ax.set_xlabel("X-ray Beam Rate, [Hz]")
    if(not fRate and not fAngles and not fGain):
        ax.set_title(f"Modulation Factors vs {cuTitle}")
        ax.set_title(cuXlabel)

    ax.minorticks_on()
    ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
    ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
    ax.set_ylim([0,110])
    ax.legend()
    
    plt.savefig(f"ModFacs-SINGLE-{picname}.png", dpi=400)
    plt.close()

#########################################
# plotting Stokes params for each rate
#########################################
Unorm = np.array(U)/np.array(I)
Umodnorm = np.array(Umod)/np.array(Imod)
Unormerr = np.array(Uerr)/np.array(I)
Umodnormerr = np.array(Umoderr)/np.array(I)

Qnorm = np.array(Q)/np.array(I)
Qmodnorm = np.array(Qmod)/np.array(Imod)
Qnormerr = np.array(Qerr)/np.array(I)
Qmodnormerr = np.array(Qmoderr)/np.array(I)

#Stokes for DB from anayzer
#------------------------------

Uanal = 0.978512314947401 
Uanalerr = 0.010527069843376625

Qanal = 0.005452992210846639 
Qanalerr = 0.009862157795284005

#------------------------------
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)

ax.errorbar(xdata[0:len(mu)], Unorm, yerr=Unormerr, color='forestgreen', ecolor='black', fmt='s',capsize=4, label=r"U (Global)")
ax.errorbar(xdata[0:len(mu)], Umodnorm, yerr=Umodnormerr, color='orange', ecolor='black', fmt='s',capsize=4, label=r"U (Cut)")
ax.errorbar(xdata[0:len(mu)], Qnorm, yerr=Qnormerr, color='green', ecolor='black', fmt='v',capsize=4, label=r"Q (Global)")
ax.errorbar(xdata[0:len(mu)], Qmodnorm, yerr=Qmodnormerr, color='goldenrod', ecolor='black', fmt='v',capsize=4, label=r"Q (Cut)")

#ax.errorbar(xdata[0], Uanal, yerr=Uanalerr, color='red', ecolor='black', fmt='s',capsize=4, label=r"U (Analyzer)")
#ax.errorbar(xdata[0], Qanal, yerr=Qanalerr, color='firebrick', ecolor='black', fmt='v',capsize=4, label=r"Q (Analyzer)")

#ax.scatter([], [], color="white", label="Analyzer results")
ax.hlines(Uanal, np.min(xdata), np.max(xdata), label=f"U={Uanal:.4f}", linestyles=":", color="red")
ax.hlines(Qanal, np.min(xdata), np.max(xdata), label=f"Q={Qanal:.4f}", linestyles=":", color="firebrick")
#ax.hlines(np.mean(phi_deg), np.min(xdata), np.max(xdata), label='mean, uncut, angle', linestyles=':')

if(fAngles):
    ax.set_title("Normalized Stokes Parameters for Reconstructed Angles")
    ax.set_xlabel(r"Incident Angle, [$\circ$]")
    
    # plotting ref to Q/U development along progression 
    lineQ, lineU = [], []
    linedots = np.linspace(np.min(xdata),np.max(xdata),100)
    #Bnorm = Bp/iI[0]
    print(type(Bp))
    for dot in linedots:
        radian = dot*np.pi/180
        lineQ.append((1/(mu[0]/100))*(Bp/2)*np.cos(2*radian)/I[0])
        lineU.append((1/(mu[0]/100))*(Bp/2)*np.sin(2*radian)/I[0])
    ax.plot(linedots, lineQ, linestyle=':', alpha=0.4)
    ax.plot(linedots, lineU, linestyle=':', alpha=0.4)

if(fGain):
    ax.set_title("Normalized Stokes Parameters for Applied Grid Voltages")
    ax.set_xlabel(r"$V_{Grid}$, [V]")
if(fRate):
    ax.set_title("Normalized Stokes Parameters vs X-ray Beam Rate")
    ax.set_xlabel("Beam Rate, [Hz]")
    ax.set_xscale('log')
if(fweight):
    ax.set_title("Normalized Stokes Parameters for Chosen Weights in Reconstruction")
    ax.set_xlabel(r"$w_{Q}$, [a.u.]")
if(not fRate and not fGain and not fAngles and not fweight):
    ax.set_title(f"Stokes Parameters vs {cuTitle}")
    ax.set_xlabel(cuXlabel)

ax.set_ylabel(r"Stokes Parameters Normalized by Intensity")

ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax.legend(loc='lower right')
if(fGain):
    ax.legend(loc='center right')
plt.savefig(f"Stokes-vs-Rate-{picname}.png",dpi=1000)
plt.close()

############################################################
# tryna plot things with second chisquare plot underneath
############################################################
fig = plt.figure(figsize=(8,8))
gs = GridSpec(3,1, width_ratios=[1], height_ratios=[3,1,1], hspace=0.02, wspace=0.05)
ax_top = fig.add_subplot(gs[0,0])
ax_bot = fig.add_subplot(gs[1,0], sharex=ax_top)
ax_bot2 = fig.add_subplot(gs[2,0], sharex=ax_top)

plane_offset = 2.5 #for now assuming 2.5 deg rotational offset of the readout plane
plane_offset_rad = 2.5*np.pi/180 #same in radians 

phi_deg = abs((np.array(phi)-plane_offset_rad)*180/np.pi)
phierr_deg = np.array(phierr)*180/np.pi
phiCut_deg = abs((np.array(phiCut)-plane_offset_rad)*180/np.pi)
phiCutErr_deg = np.array(phiCutErr)*180/np.pi

if(fAngles):
    ax_top.plot(xdata[0:len(phi)],xdata[0:len(phi)],color='firebrick', linestyle='--', label=r"$\phi(\mathrm{reco})=\phi(\mathrm{expected})$")
if(fGain):
    if(fDrift):
        ax_top.plot(xdata[0:len(phi)],[90 for i in range(len(phi))],color='firebrick', linestyle='--', label="Zero Degree Beam")
    else:
        ax_top.plot(xdata[0:len(phi)],[0 for i in range(len(phi))],color='firebrick', linestyle='--', label="Zero Degree Beam")
ax_top.errorbar(xdata[0:len(phi_deg)], phi_deg, yerr=phierr_deg, color='forestgreen', ecolor='black', fmt='*',capsize=4, label='Before Cuts')
ax_top.errorbar(xdata[0:len(phiCut_deg)], phiCut_deg, yerr=phiCutErr_deg, color='orange', ecolor='black', fmt='*',capsize=4, label='After Cuts')

ax_top.set_title("Reconstructed Polarization Angle VS Detector X-axis Rotation Angle")
if(fGain):
    ax_top.set_title("Reconstructed Polarization Angle vs Grid Voltage")
if(fweight):
    ax_top.set_title("Reconstructed Polarization Angle vs Chosen Charge Weighting")

if(not fRate and not fAngles and not fGain and not fweight):
    ax_top.set_title("Reconstructed Polarization Angle VS {cuTitle}")
    ax_top.set_title(cuXlabel)

ax_top.set_ylim([0,max(phi_deg)*1.1])
ax_top.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax_top.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax_top.tick_params(labelbottom=False)
ax_top.minorticks_on()
ax_top.legend()
if(fRate):
    ax_top.legend(loc='upper left')
if(fAngles):
    ax_top.legend(loc='lower right')

# adjusting chisquare plot below
ax_bot.scatter(xdata[0:len(mu)], chired, color='forestgreen', marker='+', s=44, label=r"$\chi^{2}_{reduced} (Original)$")
ax_bot.scatter(xdata[0:len(mu)], chiredmod, color='orange', marker='+', s=44, label=r"$\chi^{2}_{reduced} (After cuts)$")
ax_bot.set_ylabel(r'$\chi^{2}_{reduced}$')
ax_bot.set_ylim([1e-1,np.max([np.max(chired),np.max(chiredmod)])*1.5])
ax_bot.set_yscale('log')
ax_bot.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax_bot.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax_bot.minorticks_on()
ax_bot.tick_params(labelbottom=False)

# adding statistics plot even more bottom

if(fRate):
    barwidth_cold = (np.array(xdata[0:len(mu)])-2)/5
    barwidth_hot = (np.array(xdata[0:len(mu)])+2)/5
    ax_bot2.bar(np.array(xdata[0:len(mu)]), ndata, color='blue', width=barwidth_cold, edgecolor='gray', label='Before cuts')
    ax_bot2.bar(np.array(xdata[0:len(mu)]), ndatacut, color='red', width=barwidth_hot, edgecolor='gray', label='After cuts')
elif(fGain):
    ax_bot2.bar(np.array(xdata[0:len(mu)])-1, ndata, color='blue', width=2, edgecolor='gray', label='Before cuts')
    ax_bot2.bar(np.array(xdata[0:len(mu)])+1, ndatacut, color='red', width=2, edgecolor='gray', label='After cuts')
elif(fAngles):
    ax_bot2.bar(np.array(xdata[0:len(mu)])-2.5, ndata, color='blue', width=5, edgecolor='gray', label='Before cuts')
    ax_bot2.bar(np.array(xdata[0:len(mu)])+2.5, ndatacut, color='red', width=5, edgecolor='gray', label='After cuts')
elif(len(cuTitle)>0):
    print("ALLEEEGG")
    xdata_width = np.sum(np.diff(xdata))/(len(xdata)-1)/4
    ax_bot2.bar(np.array(xdata[0:len(mu)])+xdata_width/4, ndata, color='blue', width=xdata_width/2, edgecolor='gray', label='Before cuts')
    ax_bot2.bar(np.array(xdata[0:len(mu)])-xdata_width/4, ndatacut, color='red', width=xdata_width/2, edgecolor='gray', label='After cuts')
else:
    xdata_width = np.sum(np.diff(xdata))/(len(xdata)-1)/4
    ax_bot2.bar(np.array(xdata[0:len(mu)])+xdata_width/4, ndata, color='blue', width=xdata_width/2, edgecolor='gray', label='Before cuts')
    ax_bot2.bar(np.array(xdata[0:len(mu)])-xdata_width/4, ndatacut, color='red', width=xdata_width/2, edgecolor='gray', label='After cuts')

ax_bot2.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax_bot2.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax_bot2.minorticks_on()
ax_bot2.set_ylabel(r'$N_{clusters}$')
if(fGain):
    ax_bot2.set_xlabel(r'$V_{Grid}$, [V]') 
if(fAngles):
    ax_bot2.set_xlabel(r'Angles, [$^{\circ}$]')
if(fRate):
    ax_bot2.set_xlabel('Rate, [Hz]')
    ax_bot2.set_xscale('log')
    ax_bot2.set_yscale('log')
if(fweight):
    ax_bot2.set_xlabel(r'$w_{Q}$, [a.u.]')

if(not fRate and not fAngles and not fGain and not fweight):

    ax_bot2.set_xlabel(cuXlabel)
ax_bot2.legend()

plt.tight_layout()
plt.savefig(f"RecoAngles-IMPROVED-{picname}.png")
plt.close()

######################################################################################
#  plotting simlified data for MODULATION FACTORS without auxiliary plots
######################################################################################
if(fSingle):
    # separate, modulations only single  plot
    
    fig = plt.figure(figsize=(8,7))
    
    ax = fig.add_subplot(111)
    
    #ax.text(120,29,"WORK IN PROGRESS", color='red', backgroundcolor='lightgray', fontweight='semibold', fontsize=12)
    ax.errorbar(xdata[0:len(mu)], phi_deg, yerr=phierr_deg, color='forestgreen', ecolor='black', ms=8, fmt='*',capsize=4, label=r"$\phi$ Before cuts")
    ax.errorbar(xdata[0:len(mu)], phiCut_deg, yerr=phiCutErr_deg, color='orange', ecolor='black', ms=8, fmt='*',capsize=4, label=r"$\phi$ After cuts")
    ax.set_ylabel(r"$\phi_{\mathrm{Reconstructed}}$,"+r"[$^{\circ}$]")
    if(fAngles):
        ax.plot(xdata[0:len(phi)],xdata[0:len(phi)],color='firebrick', linestyle='--', label=r"$\phi(\mathrm{reco})=\phi(\mathrm{expected})$")
        ax.set_title("Reconstructed Polarization Angles VS X-ray Beam Polarization Angle")
        ax.set_xlabel(r"$\phi_{\mathrm{Expected}}$, [$^{\circ}$]")
    if(fGain):
        ax.set_title("Recosntructed Polarization Angles vs Grid Voltage")
        ax.set_xlabel(r"$V_{\mathrm{Grid}}$, [V]")
    if(fRate):
        ax.set_title("Reconstructed Polarization Angles vs X-ray Beam Rate")
        ax.set_xscale('log')
        ax.set_xlabel("X-ray Beam Rate, [Hz]")
    if(fweight):
         ax.set_xlabel(r'$w_{Q}$, [a.u.]')
    if(not fRate and not fAngles and not fGain and not fweight):
        custom_xlabel = input("Proivide xlabel: ")
        ax.set_title(custom_xlabel)

    ax.minorticks_on()
    ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
    ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
    ax.set_ylim([0,100])
    ax.legend()
    
    plt.savefig(f"RecoSngles-SINGLE-{picname}.png", dpi=400)
    plt.close()




