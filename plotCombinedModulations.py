import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import re
import argparse as ap

from scipy.optimize import curve_fit

from MyPlotter import mySymbols
from MyPlotter import myColors

AXIS_LABEL_SIZE = 12
AXIS_FONT_SIZE = 12

MS = mySymbols()
MC = myColors()

def getMaxPos(array):

    numlist, strlist = None, None

    if(type(array)==numpy.ndarray):
        numlist = array.tolist() 
        strlist = str(numlist)
    else:       
        strlist = str(array)
    
    maxnum = np.max(array)
    maxpos = strlist.index(str(maxnum))
    
    return maxpos

def getGlobalMax(datalist, ctrl_list):

    glob_max = -np.inf
    glob_arr = None
    glob_idx = None

    for iarr, arr in enumerate(datalist):

        arr = np.asarray(arr)

        local_idx = np.argmax(arr)
        local_max = arr[local_idx]
        
        if(local_max > glob_max):
            glob_max = local_max
            glob_arr = arr
            glob_idx = local_idx


    print(f"Max value in the set = {glob_max}")
    print(f"{type(glob_arr)}")
    print(f"{glob_arr}")
    print(f"{type(glob_idx)}")


    #return glob_max, glob_arr, glob_idx
    return ctrl_list[glob_arr][glob_idx]

def getGlobalMin(datalist, ctrl_list):

    glob_min = np.inf
    glob_arr = None
    glob_idx = None

    for iarr, arr in enumerate(datalist):

        arr = np.asarray(arr)

        local_idx = np.argmin(arr)
        local_min = arr[local_idx]
        
        if(local_min < glob_min):
            glob_min = local_min
            glob_arr = arr
            glob_idx = local_idx

    print(f"Min value in the set = {glob_min}")
    print(f"{type(glob_arr)}")
    print(f"{glob_arr}")
    
    print(f"{type(glob_idx)}")

    return ctrl_list[glob_arr][glob_idx]


def linfunc(x,a,b):
    return x*a + b

def fitData(function, xdata, ydata, yerr):

    par, cov = curve_fit(function,xdata, ydata)
    
    expected_ydata = function(xdata,*par)
    ndof = len(ydata) - len(par)
    chisq = np.sum((ydata - expected_ydata)**2/yerr**2)
    chired = chisq/ndof
    uncert = np.sqrt(np.diag(cov))
    
    return par, uncert, chired

def fancyStateBox(descr, flags):
    
    fancystring = ""
    cnt = 0
    for d ,f in zip(descr, flags):
        if(f):
            fancystring+=f"{MC.GREEN}{d}{MC.RST}"
        else:
            fancystring+=f"{MC.RED}{d}{MC.RST}"
        cnt+=1        
        if(cnt!=len(descr)):
            fancystring+="\n"

    return fancystring

def getPhiFromStokes(Q, U):

    return 0.5*np.arctan(U/Q)

def calcPhiErr(Q, Qerr, U, Uerr):

    return np.sqrt(Qerr**2*((0.5*Q)/(2*(Q**2+U**2)))**2 + Uerr**2*((0.5*U)/(2*(Q**2+U**2)))**2)

def getModFromStokes(Q, U):

    return np.sqrt(Q**2 + U**2)

def calcModErr(mu, Q, Qerr, U, Uerr):

    return np.sqrt((Q**2/mu**2)*Qerr**2 + (U**2/mu**2)*Uerr**2)

def getPolDegFromStokes(I, Q, U):

    return np.sqrt(Q**2 + U**2)/I

def radToDeg(radian):

    return radian*180.0/np.pi 

##### main below ####

choices = ["rate","angles","gain","weights","custom","atten","drift"]
parser = ap.ArgumentParser()
parser.add_argument("-p", "--picname", type=str, default="ALALA")
parser.add_argument("-l","--logfile",type=str,default="")
parser.add_argument("-x","--xval",type=str,default="")
parser.add_argument("-t","--type",type=str.lower,
                    choices=choices,
                    help="Enter type of run (angles,rate,weights,gain)",
                    required=True)
parser.add_argument("--single",action='store_true')
parser.add_argument("--notitle",action='store_true')
parser.add_argument("--fitangles",action='store_true')

args = parser.parse_args()

logfile = args.logfile
picname = args.picname
xvalues = args.xval
fSingle = args.single
runtype = args.type
fNoTitle= args.notitle
fFitAngles = args.fitangles

if(runtype not in choices):
    print(f"{MC.RED}ALO!{MC.RST}, {runtype} is not in the list of default run types!")
    
cuTitle, cuXlabel = None, None

fSimulationData = True if "SIM" in logfile else False

fRate, fAngles, fGain = False, False, False
fweight, fAtten, fDrift = False, False, False

if(runtype=="rate"):
    fRate = True
if(runtype=="angles"):
    fAngles = True
if(runtype=="gain"):
    fGain = True
if(runtype=="weights"):
    fweight = True
if(runtype=="atten"):
    fAtten = True
if(runtype=="drift"):
    fDrift = True

fCustomData = False

if(runtype=="custom"):
    print("Custom data flag set...")
    fCustomData = True
    cuTitle = input("SetTitle: ") 
    cuXlabel = input("SetXlabel: ")
    #cuYlabel = input("SetTitle: ") 
    if(len(xvalues)==0):
        print("Input from option -x / --xval is required!")
        exit(0)        

f=open(logfile,'r')

xdata = None
if(len(xvalues)>0):
    xdata = [float(i) for i in xvalues.split(",")]

datacheckstr = f"xvalues:{xdata}, plotting {runtype}"
if(fCustomData):
    datacheckstr+=f"[CUSTOM DATA]"
print(f"{datacheckstr}")

voltages, rates = [], []
weights, angles  = [], []
atten, edrift = [], []
#weights = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0]

if(fCustomData):
    print("USing custom labels and data\nsetting other run type flags to [FALSE]")
    fweight = False
    fAngles = False
    fGain = False
    fRate = False
    fAtten = False

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

stokes_mu, stokes_muErr = [], []
stokes_phi, stokes_phiErr = [], []

Wstokes_mu, Wstokes_muErr = [], []
Wstokes_phi, Wstokes_phiErr = [], []

isSimulation = []

mu_cnt = 0
muErr_cnt = 0
nrun = 0

fValid = True
#fDrift = False
 
# reference values of crystal-based analyzer (CBA) extracted from Stokes parameters
# for DB, CP, MP pre-runs
# (units = degrees)

DB_CBA_Q = 0.9785
DB_CBA_Qe = 0.0105
DB_CBA_U = -0.0055
DB_CBA_Ue = 0.0099

CP_CBA_Q = 1.0353
CP_CBA_Qe = 0.0375
CP_CBA_U = -0.0733
CP_CBA_Ue = 0.0351

MP_CBA_Q = -1.0028
MP_CBA_Qe = 0.0499
MP_CBA_U = -0.1904
MP_CBA_Ue = 0.0386

ref_phi_db = radToDeg(getPhiFromStokes(DB_CBA_Q, DB_CBA_U))
ref_phi_dberr = radToDeg(calcPhiErr(DB_CBA_Q, DB_CBA_Qe, DB_CBA_U, DB_CBA_Ue))

ref_phi_cp = radToDeg(getPhiFromStokes(CP_CBA_Q, CP_CBA_U))
ref_phi_cperr = radToDeg(calcPhiErr(CP_CBA_Q, CP_CBA_Qe, CP_CBA_U, CP_CBA_Ue))

ref_phi_mp = radToDeg(getPhiFromStokes(MP_CBA_Q, MP_CBA_U))
ref_phi_mperr = radToDeg(calcPhiErr(MP_CBA_Q, MP_CBA_Qe, MP_CBA_U, MP_CBA_Ue))


ref_mu_db = getModFromStokes(DB_CBA_Q, DB_CBA_U)
ref_mu_dberr = calcModErr(ref_mu_db, DB_CBA_Q, DB_CBA_Qe, DB_CBA_U, DB_CBA_Ue )

ref_mu_cp = getModFromStokes(CP_CBA_Q, CP_CBA_U)
ref_mu_cperr = calcModErr(ref_mu_cp, CP_CBA_Q, CP_CBA_Qe, CP_CBA_U, CP_CBA_Ue)

ref_mu_mp = getModFromStokes(MP_CBA_Q, MP_CBA_U)
ref_mu_mperr = calcModErr(ref_mu_mp, MP_CBA_Q, MP_CBA_Qe, MP_CBA_U, MP_CBA_Ue)

ref_phi_db_0deg = -0.1610
ref_phierr_db_0deg = -0.1537
#ref_mu_db = 0.09785
#ref_phi_dberr = 5.5533 #0.0969 # [rad]
#ref_mu_dberr = 0.0103
#ref_mu_cp = 1.0379
#ref_mu_cperr = 0.0388
#ref_phi_cp = -2.0249
#ref_phi_cperr = 5.4607 #0.0953 # [rad]
#
#ref_mu_mp = 1.0207
#ref_mu_mperr = 0.0492
#ref_phi_mp = -5.3753
#ref_phi_mperr = 0.0489 
#

#### in case have sliding region data ###
fHaveStokesModulation = False

regions = []
reg_mus, reg_muerrs, reg_chis = [], [], []
fSlidingMod = False

fChargePeak, fMagnPeak, fDirectBeam = False, False, False
fGainVar = False

fZeroDegDbPlot, fSixtyDegDbPlot = False, False

for line in f:

    if("ANALYZING" in line):
        repList = ["[","]","'","\n"]
        line = line.split(":")[1]
        for char in repList:
            line = line.replace(char,"")
        files = line.split(",")
        print(f"LOGFILE HAS DATA FOR FILES:\n{files}\n")
        if(all("DB" in file for file in files)):
            fDirectBeam = True
            fZeroDegDbPlot = all("-0deg.h5" in fn for fn in files)
            fSixtyDegDbPlot = all("-60deg.h5" in fn for fn in files)
        elif(all("CP" in file for file in files)):
            fChargePeak = True
        elif(all("MP" in file for file in files)):
            fMagnPeak = True
        elif(all("GainVar" in file for file in files)):
            fGainVar = True
        else:
            print("data from different run types...")
        msg = "Detected Data Types:"
        types = ["Direct Beam", "Charge peak", "Magnetic Peak", "Gain Variation"]
        fanstr = fancyStateBox(types, [fDirectBeam,fChargePeak,fMagnPeak,fGainVar])
        print(f"{msg}\n{fanstr}")

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

    if(fAtten and "atten_fact" in line):
        iatt = line.split("=")[1] # atten_fact = xxxx,xxxx
        #print(f"Found atten < {iatt} >")
        #theDigit = iatt.split(".")[0]
        #print(f"number =  < {theDigit} >")

        #numIsFloat = all(char.isdigit() for char in iatt.split(".")[0])
        if("#" not in iatt):
            atten.append(float(iatt))
        else:
            atten.append(-9999)

    if(fGain and "Current file:" in line):
        cutChar = -1
        fname = line.split(":")[1].split(".")[0]
        #if("MP" in fname and "Vcm" in fname):
        #    cutChar = -3
        #    fDrift = True
        volt = fname.split("-")[-1][:cutChar] # removing "V" from thestring to enable int() conversion
        voltages.append(float(volt))

    if(fDrift and "Current file:" in line):
        cutChar = -3
        fname = line.split(":")[1].split(".")[0] 
        print(fname)
        volt = fname.split("-")[-1][:cutChar] # removing "Vcm" from thestring to enable int() conversion
        print(volt)
        edrift.append(float(volt))

    if(fRate and "Current file" in line):
        fname = line.split(":")[1].split(".")[0]
        if("Hz" in fname):
            irate = fname.split("-")[-2][:-2] # getting xxxxHz and removing "Hz" part
            rates.append(int(irate))
        else:
            print(f"Cant' find rate in the file name {fname}!")
            #exit(0)
     
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
                #print("Found Bp!")
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

    if("origstokes" in line):
        data = line.split(":")[1].split(",")
        for d in data:
            if("S_mu=" in d):
                stokes_mu.append(float(d.split("=")[1]))
            if("S_dmu=" in d):
                stokes_muErr.append(float(d.split("=")[1]))
            if("S_phi=" in d):
                stokes_phi.append(float(d.split("=")[1]))
            if("S_dphi=" in d):
                stokes_phiErr.append(float(d.split("=")[1]))

    if("weightstokes" in line):
        data = line.split(":")[1].split(",")
        for d in data:
            if("S_mu=" in d):
                Wstokes_mu.append(float(d.split("=")[1]))
            if("S_dmu=" in d):
                Wstokes_muErr.append(float(d.split("=")[1]))
            if("S_phi=" in d):
                Wstokes_phi.append(float(d.split("=")[1]))
            if("S_dphi=" in d):
                Wstokes_phiErr.append(float(d.split("=")[1]))



    if("sliding" in line and "modulation" in line):
        fSlidingMod = True

    if(fSlidingMod and "mu" in line and "muerr" in line and "chired" in line):
        regname = line.split(":")[0]
        regvalues = lien.split(":")[1].split(",")
        for entry in regvalues:
            if("mu" in entry):
                reg_mus.append(float(entry.split("=")[1]))
            if("muerr" in entry):
                reg_muerrs.append(float(entry.split("=")[1]))
            if("chired" in entry):
                reg_chis.append(float(entry.split("=")[1]))


print(f"mod factor list ({len(mu)} entries) :\n(ORIG) {mu}\n(CUT) {muCut}")
print(f"reco angle list ({len(mu)} entries) :\n(ORIG) {phi}\n(CUT) {phiCut}")
print(f"Data numbers lists pre={len(ndata)}, post={len(ndatacut)}, len(mu)={len(mu)}")

if(fGain):
    xdata = voltages
if(fRate):
    xdata = rates
if(fweight):
    xdata = weights
if(fAngles):
    xdata = angles
if(fAtten):
    xdata = atten
if(fDrift):
    xdata = edrift 

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
ax_top.errorbar(xdata[0:len(mu)], mu, yerr=muErr, color='forestgreen', ecolor='forestgreen', fmt='*',capsize=4, label=r"Before cuts")
ax_top.errorbar(xdata[0:len(mu)], muCut, yerr=muCutErr, color='orange', ecolor='orange', fmt='*',capsize=4, label=r"After cuts")
ax_top.set_ylabel(r"Reconstructed $\mu$")

if(not fNoTitle):
    ax_top.set_title("Modulation Factors VS Xray Beam Polarization Angle")
    if(fGain):
        ax_top.set_title("Modulation Factors vs Grid Voltage")
    if(fRate):
        ax_top.set_title("Modulation Factors vs Incident X-ray Beam Rate")
    if(fweight):
        ax_top.set_title("Modulation Factors vs Charge Weighting in Reconstruction")
    if(fAtten):
        ax_top.set_title("Modulation Factors vs First Harmonic Attenuation Factor")
    if(fDrift):
        ax_top.set_title("Modulation Factors vs Drift Field Strength")
    if(fCustomData):
        ax_top.set_title(f"Modulation Factors vs {cuTitle}")

ax_top.set_ylim([0,max([np.max(mu),np.max(muCut)])*1.5])
ax_top.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax_top.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax_top.tick_params(labelbottom=False)
ax_top.minorticks_on()
ax_top.legend()

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
    ax_bot2.bar(np.array(xdata[0:len(mu)]), ndata, color='blue', width=barwidth_cold, edgecolor='gray', label="Before cuts")
    ax_bot2.bar(np.array(xdata[0:len(mu)]), ndatacut, color='red', width=barwidth_hot, edgecolor='gray', label="After cuts")
elif(fGain):
    ax_bot2.bar(np.array(xdata[0:len(mu)])-1, ndata, color='blue', width=2, edgecolor='gray', label='Before cuts')
    ax_bot2.bar(np.array(xdata[0:len(mu)])+1, ndatacut, color='red', width=2, edgecolor='gray', label='After cuts')
elif(fAngles):
    ax_bot2.bar(np.array(xdata[0:len(mu)])-2.5, ndata, color='blue', width=5, edgecolor='gray', label = 'Before cuts')
    ax_bot2.bar(np.array(xdata[0:len(mu)])+2.5, ndatacut, color='red', width=5, edgecolor='gray', label='After cuts')
elif(fCustomData):
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
ax_bot2.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax_bot2.minorticks_on()
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
if(fAtten):
    ax_bot2.set_xlabel('Attenuation Factor, [a.u]')
if(fAtten):
    ax_bot2.set_xlabel('Drift field strength, [V/cm]')
if(fCustomData):
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

    print(f"(ORIGINAL)  mu_low  = {mu_ylow}")
    print(f"(ORIGINAL)  muErr   = {muErr}")
    print(f"(ORIGINAL)  mu_high = {mu_yhigh}")

    # same for the mod factors after cuts
    mucut_ylow = np.array(muCutErr)
    mucut_yhigh = []
    for mc, mce in zip(muCut, muCutErr):
        if(mc+mce>=100):
            mucut_yhigh.append(mce-(mc+mce-100.0))
        else:
            mucut_yhigh.append(mce)
    print(f"(CUT) mu_low  = {mucut_ylow}")
    print(f"(CUT) muErr   = {muCutErr}")
    print(f"(CUT) mu_high = {mucut_yhigh}")
    asym_muCutErr = np.array([mucut_ylow, mucut_yhigh])

    highest_muErr = np.max([np.ravel(np.column_stack(asym_muCutErr)),
                           np.ravel(np.column_stack(asym_muErr))])

    xpoints = xdata[0:len(mu)]

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111)
  
#    ax = None
#
#    if(fAtten): 
#        ax = brokenaxes(xlims=((3e4,2e5),(4.1e6,5e6)))
#    else:
#        ax = fig.add_subplot(111)
    

    ax.errorbar(xpoints, mu, yerr=asym_muErr, color='forestgreen',ms=8, ecolor='darkgreen', fmt='*',capsize=6, label=r"$\mu$ Before cuts")
    ax.errorbar(xpoints, muCut, yerr=asym_muCutErr, color='orange',ms=8,  ecolor='darkgoldenrod', fmt='*',capsize=6, label=r"$\mu$ After cuts")
    ax.set_ylabel(r"Reconstructed $\mu$", fontsize=AXIS_FONT_SIZE)

    ylim_maxoffset = 5
    ylim_minoffset = 0.9

    if(not fNoTitle):
        if(fAngles):
            ax.set_title("Modulation Factors VS Xray Beam Polarization Angle")
        if(fGain):
            ax.set_title("Modulation Factors vs Grid Voltage")
        if(fRate):
            ax.set_title("Modulation Factors vs Incident X-ray Beam Rate")
        if(fAtten):
            ax.set_title("Modulation Factors vs First Harmonic Attenuation Factors")
        if(fAtten):
            ax.set_title("Modulation Factors vs Drift Field Strength")
        if(fCustomData):
            ax.set_title(f"Modulation Factors vs {cuTitle}")

    if(fAngles):
        ax.set_xlabel(r"Angle of Detector Rotation, [$^{\circ}$]", fontsize=AXIS_FONT_SIZE)
    if(fGain):
        ax.set_xlabel(r"$V_{\mathrm{grid}}$, [V]", fontsize=AXIS_FONT_SIZE)
    if(fRate):
        ax.set_xscale('log')
        ax.set_xlabel("X-ray Beam Rate, [Hz]", fontsize=AXIS_FONT_SIZE)

    if(fAtten):
        ax.set_xlabel("Attenuation Factor, [a.u.]", fontsize=AXIS_FONT_SIZE)
        ax.set_xscale('log')  
    if(fDrift):
        ax.set_xlabel(r"$\mathrm{V}_{\mathrm{drfit}}$, [V/cm]", fontsize=AXIS_FONT_SIZE)
    if(fCustomData):
        ax.set_xlabel(cuXlabel, fontsize=AXIS_FONT_SIZE)
        ylim_maxoffset = 2
        ylim_minoffset = 0.5
        if("Energy" in cuTitle or "energy" in cuTitle):
            #dE = np.diff(xpoints)
            #ax.set_xlim([np.min(xpoints)-dE/2, np.max(xpoints)+dE/2])  
            ax.set_title("")
            if("SIM" in picname):
                ax.set_xlim([9.5,12.5]) 
            else:
                avgxdist = np.sum(np.diff(xpoints))/(len(xpoints)-1)
                print(f"PADLA: {avgxdist}")
                print(f"PADLA: {np.min(xpoints)-avgxdist/2}")
                print(f"PADLA: {np.max(xpoints)+avgxdist/2}")
                ax.set_xlim([np.min(xpoints)-avgxdist/2, np.max(xpoints)+avgxdist/2])
    
    ax.minorticks_on()
    ax.tick_params(axis='x',labelsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis='y',labelsize=AXIS_LABEL_SIZE)

    ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
    ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
    if("-60deg-" in picname):
        minmu, maxmu = 15,55
    #elif(("Energy" in cuTitle or "energy" in cuTitle) and "SIM" not in picname):
    elif("custom" in runtype and "SIM" not in picname):
        minmu, maxmu = 25,40 
    else:
        maxmu = np.ceil(np.max([np.max(mu), np.max(muCut)])+highest_muErr*ylim_maxoffset)
        minmu = np.floor(np.min([np.min(mu),np.min(muCut)])-highest_muErr*ylim_minoffset)
    
    ax.set_ylim([minmu,maxmu])
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

ax.errorbar(xdata[0:len(mu)], Unorm, yerr=Unormerr, color='forestgreen', ecolor='black', fmt='s',capsize=4, label=r"U (Raw reconstruction)")
ax.errorbar(xdata[0:len(mu)], Qnorm, yerr=Qnormerr, color='green', ecolor='black', fmt='v',capsize=4, label=r"Q (Raw reconstruction)")
ax.errorbar(xdata[0:len(mu)], Qmodnorm, yerr=Qmodnormerr, color='goldenrod', ecolor='black', fmt='v',capsize=4, label=r"Q (Cuts applied)")
ax.errorbar(xdata[0:len(mu)], Umodnorm, yerr=Umodnormerr, color='orange', ecolor='black', fmt='s',capsize=4, label=r"U (Cuts applied)")

pmstr=r"$\pm$"
ax.hlines(Uanal, np.min(xdata), np.max(xdata), label=r"$\mathrm{U}_{\mathrm{CBA}}(0^{\circ})$"+f"={Uanal:.4f}{pmstr}{Uanalerr:.4f}", linestyles=":", color="red")
ax.hlines(Qanal, np.min(xdata), np.max(xdata), label=r"$\mathrm{Q}_{\mathrm{CBA}}(0^{\circ})$"+f"={Qanal:.4f}{pmstr}{Qanalerr:.4f}", linestyles=":", color="firebrick")
#ax.hlines(np.mean(phi_deg), np.min(xdata), np.max(xdata), label='mean, uncut, angle', linestyles=':')

legPos = "lower right"
if(not fNoTitle):
    if(fAngles):
        ax.set_title("Normalized Stokes Parameters for Reconstructed Angles")
    if(fGain):
        ax.set_title("Normalized Stokes Parameters for Applied Grid Voltages")
    if(fAtten):
        ax.set_title("Normalized Stokes Parameters for Attenuation Factors of the First Harmonics")
    if(fRate):
        ax.set_title("Normalized Stokes Parameters vs X-ray Beam Rate")
    if(fweight):
        ax.set_title("Normalized Stokes Parameters for Chosen Weights in Reconstruction")
    if(fweight):
        ax.set_title("Normalized Stokes Parameters for Drfit Field Variation")
    if(fCustomData):
        ax.set_title(f"Stokes Parameters vs {cuTitle}")

if(fAngles):
    ax.set_xlabel(r"Incident Angle, [$\circ$]", fontsize=AXIS_FONT_SIZE)
    legPos = "lower left"  
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
    ax.set_xlabel(r"$V_{Grid}$, [V]", fontsize=AXIS_FONT_SIZE)
    legPos = "center right"
if(fAtten):
    ax.set_xlabel("Attenuation Factor [a.u.]", fontsize=AXIS_FONT_SIZE)
if(fRate):
    ax.set_xlabel("Beam Rate, [Hz]", fontsize=AXIS_FONT_SIZE)
    ax.set_xscale('log')
    legPos = "lower center"
if(fweight):
    ax.set_xlabel(r"$w_{Q}$, [a.u.]", fontsize=AXIS_FONT_SIZE)
if(fCustomData):
    ax.set_xlabel(cuXlabel, fontsize=AXIS_FONT_SIZE)
if(fDrift):
    ax.set_xlabel(r"$\mathrm{V}_{\mathrm{drfit}}$, [V/cm]", fontsize=AXIS_FONT_SIZE)

ax.set_ylabel(r"Stokes Parameters Normalized by Intensity", fontsize=AXIS_FONT_SIZE)
ax.tick_params(axis='x',labelsize=AXIS_LABEL_SIZE)
ax.tick_params(axis='y',labelsize=AXIS_LABEL_SIZE)

ax.set_ylim([-1.1,1.1])
ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax.minorticks_on()
ax.legend(loc=legPos)
#if(fGain):
#    ax.legend(loc='center right')
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

phi_deg, phiCut_deg = None, None
if(fAngles):
    phi_deg = abs(np.array(phi)*180/np.pi)
else:
    phi_deg = np.array(phi)*180/np.pi
phierr_deg = np.array(phierr)*180/np.pi

if(fAngles):
    phiCut_deg = abs(np.array(phiCut)*180/np.pi)
else:
    phiCut_deg = np.array(phiCut)*180/np.pi
phiCutErr_deg = np.array(phiCutErr)*180/np.pi

if(fAngles):
    ax_top.plot(xdata[0:len(phi)],xdata[0:len(phi)],color='firebrick', linestyle='--', label=r"$\phi(\mathrm{reco})=\phi(\mathrm{expected})$")
if(fGain):
    ax_top.plot(xdata[0:len(phi)],[0 for i in range(len(phi))],color='firebrick', linestyle='--', label="Zero Degree Beam")
if(fDrift):
    ax_top.plot(xdata[0:len(phi)],[90 for i in range(len(phi))],color='firebrick', linestyle='--', label="Zero Degree Beam")

ax_top.errorbar(xdata[0:len(phi_deg)], phi_deg, yerr=phierr_deg, color='forestgreen', ecolor='black', fmt='*',capsize=4, label='Before Cuts')
ax_top.errorbar(xdata[0:len(phiCut_deg)], phiCut_deg, yerr=phiCutErr_deg, color='orange', ecolor='black', fmt='*',capsize=4, label='After Cuts')

if(not fNoTitle):
    ax_top.set_title("Reconstructed Polarization Angle VS Detector X-axis Rotation Angle")
    if(fGain):
        ax_top.set_title("Reconstructed Polarization Angle vs Grid Voltage")
    if(fweight):
        ax_top.set_title("Reconstructed Polarization Angle vs Chosen Charge Weighting")
    if(fAtten):
        ax_top.set_title("Reconstructed Polarization Angle vs Attenuation Factors of First Harmonics")
    
    if(fCustomData):
        ax_top.set_title("Reconstructed Polarization Angle VS {cuTitle}")

ax_top.set_ylim([0,max(phi_deg)*1.5])
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
elif(fCustomData):
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
if(fAtten):
    ax_bot2.set_xlabel('Attenuation Factor, [a.u.]')
    #ax_bot2.set_xscale('log')
if(fDrift):
    ax_bot2.set_xlabel(r"$\mathrm{V}_{\mathrm{drfit}}$, [V/cm]")

if(fCustomData):
    ax_bot2.set_xlabel(cuXlabel)
ax_bot2.legend()

#plt.tight_layout()
plt.savefig(f"RecoAngles-IMPROVED-{picname}.png")
plt.close()

######################################################################################
#  plotting simlified data for RECONTRUCTED ANGLES without auxiliary plots
######################################################################################
if(fSingle):
    # separate, modulations only single  plot

    maxphi = np.max([np.max(phi_deg), np.max(phiCut_deg)])
    minphi = np.min([np.min(phi_deg), np.min(phiCut_deg)])
    maxphierr = np.max([np.max(phierr_deg),np.max(phiCutErr_deg)])*1.05
    
    #philist = [phi_deg.tolist(), phiCut_deg.tolist()]
    #phierrlist = [phierr_deg.tolist(), phiCutErr_deg.tolist()]

    #globmaxphi = getGlobalMax(philist, phierrlist)
    #globminphi = getGlobalMin(philist, phierrlist)
    
    print("\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    print(maxphi)
    print(maxphierr)
   
    if(fAngles):
        # fitting lines to the resulting data points
       
        arrx_ang = np.array(xdata[0:len(mu)])
        arry_phi = np.array(phi_deg)
        arry_phiCut = np.array(phiCut_deg)
        arry_phierr = np.array(phierr_deg)
        arry_phicuterr = np.array(phiCutErr_deg)
    
        pars_phi, unc_phi, chired_phi = fitData(linfunc,arrx_ang,arry_phi,arry_phierr)
        pars_phiCut, unc_phiCut, chired_phiCut = fitData(linfunc,arrx_ang,arry_phiCut,arry_phicuterr)
    
        matpm = r"$\pm$"
    
        a_phi = pars_phi[0]
        b_phi = pars_phi[1]
        a_phiCut = pars_phiCut[0]
        b_phiCut = pars_phiCut[1]
        
        Erra_phi = unc_phi[0]
        Errb_phi = unc_phi[1]
        Erra_phiCut = unc_phiCut[0]
        Errb_phiCut = unc_phiCut[1]
    
        label_phi = f"a={a_phi:.2f}"+matpm+f"{Erra_phi:.2f}, b={b_phi:.2f}"+matpm+f"{Errb_phi:.2f}; {MS.chi}(red)={chired_phi:.2f}"
        label_phiCut = f"a={a_phiCut:.2f}"+matpm+f"{Erra_phiCut:.2f}, b={b_phiCut:.2f}"+matpm+f"{Errb_phiCut:.2f}; {MS.chi}(red)={chired_phiCut:.2f}"

    # ------------------------------------------------------------------------------------- 

    fig = plt.figure(figsize=(8,7))
    
    ax = fig.add_subplot(111)
 
    xpoints = xdata[0:len(mu)]  

    for x,p in zip(xpoints,phi_deg):
        print(f"{x} -> {p:.4f}")

    if(fRate and fZeroDegDbPlot):
        phi_deg = abs(phi_deg)
        phiCut_deg = abs(phiCut_deg)

    #ax.text(120,29,"WORK IN PROGRESS", color='red', backgroundcolor='lightgray', fontweight='semibold', fontsize=12)
    ax.errorbar(xpoints, phi_deg, yerr=phierr_deg, color='forestgreen', ecolor='forestgreen', ms=8, fmt='*',capsize=4, label=r"$\phi_{\mathrm{GP3}}$ Before cuts")
    ax.errorbar(xpoints, phiCut_deg, yerr=phiCutErr_deg, color='orange', ecolor='orange', ms=8, fmt='*',capsize=4, label=r"$\phi_{\mathrm{GP3}}$ After cuts")
    ax.set_ylabel(r"$\phi_{\mathrm{Reconstructed}}$,"+r"[$^{\circ}$]", fontsize=AXIS_FONT_SIZE)
    if(fAngles):
        ax.plot(xdata[0:len(phi)],xdata[0:len(phi)],color='firebrick', linestyle='--', label=r"$\phi(\mathrm{reco})=\phi(\mathrm{expected})$")
        if(not fNoTitle):
            ax.set_title("Reconstructed Polarization Angles VS X-ray Beam Polarization Angle")
        ax.set_xlabel(r"$\phi_{\mathrm{Expected}}$, [$^{\circ}$]", fontsize=AXIS_FONT_SIZE)
    if(fGain):
        if(not fNoTitle):
            ax.set_title("Recosntructed Polarization Angles vs Grid Voltage")
        ax.set_xlabel(r"$V_{\mathrm{Grid}}$, [V]", fontsize=AXIS_FONT_SIZE)
        phi_cba, phierr_cba = ref_phi_cp, ref_phi_cperr
        degstr = r"$^{\circ}$"
        ax.hlines(phi_cba,
                  np.min(xdata),
                  np.max(xdata), 
                  label=r"$\phi_{CBA}$="+f"{phi_cba:.4f}{degstr}"+r"$\pm$"+f"{phierr_cba:.4f}{degstr}", 
                  linestyles="--", 
                  color='firebrick')
        ax.axhspan(phi_cba-phierr_cba, phi_cba+phierr_cba, color='red', alpha=0.05)

    if(fRate):
        phi_cba, phierr_cba = None, None
        degstr = r"$^{\circ}$"
        if(fDirectBeam):
            if(fZeroDegDbPlot):
                print("---SUKA---")
                phi_cba, phierr_cba = ref_phi_db_0deg, ref_phierr_db_0deg
                ax.hlines(phi_cba,
                          np.min(xdata),
                          np.max(xdata), 
                          label=r"$\phi_{CBA}$="+f"{phi_cba:.4f}{degstr}"+r"$\pm$"+f"{phierr_cba:.4f}{degstr}", 
                          linestyles="--", 
                          color='firebrick')
                ax.axhspan(phi_cba-phierr_cba, phi_cba+phierr_cba, color='red', alpha=0.05)
            elif(fSixtyDegDbPlot):
                print("---PADLA---")
                ax.hlines(60.0,
                          np.min(xdata),
                          np.max(xdata), 
                          label=r"$\phi_{\mathrm{expected}}$="+f"60{degstr}", 
                          linestyles="--", 
                          color='firebrick')
            else:
                print("---PIDR---")
                ax.hlines(np.mean(phi_deg),
                          np.min(xdata),
                          np.max(xdata), 
                          label=r"$\phi_{\mathrm{average}}$="+f"{np.mean(phi_deg)}{degstr}", 
                          linestyles="--", 
                          color='firebrick')

            #print(f"{MC.BLUE}AAAAAAAAAAAAAAAA{MC.RST}")
        elif(fChargePeak):
            phi_cba, phierr_cba = ref_phi_cp, ref_phi_cperr 
        elif(fMagnPeak):
            phi_cba, phierr_cba = ref_phi_mp, ref_phi_mperr
        else: 
            phi_cba, phierr_cba = 0.0, 0.0

        if(not fNoTitle):
            ax.set_title("Reconstructed Polarization Angles vs X-ray Beam Rate")
        ax.set_xscale('log')
        ax.set_xlabel("X-ray Beam Rate, [Hz]", fontsize=AXIS_FONT_SIZE)
    if(fweight):
         ax.set_xlabel(r'$w_{Q}$, [a.u.]', fontsize=AXIS_FONT_SIZE)
    if(fAtten):
        if(not fNoTitle):
            ax.set_title("Reconstructed Polarization Angles vs Attenuiation Factors of First Harmonics")
        ax.set_xlabel('Attenuation Factor, [a.u.]', fontsize=AXIS_FONT_SIZE)
    if(fCustomData):
        if(not fNoTitle):
            ax.set_title(f"Recosntructed Polarization Angles vs {cuTitle}", fontsize=AXIS_FONT_SIZE)
        ax.set_xlabel(f"{cuXlabel}")
    if(fDrift):
        ax.set_xlabel(r"$\mathrm{V}_{\mathrm{drfit}}$, [V/cm]", fontsize=AXIS_FONT_SIZE)
        if(not fNoTitle):
            ax.set_title(f"Recosntructed Polarization Angles vs Drift Field Variation")

    if(fAngles and fFitAngles):
        ax.plot(xpoints, linfunc(arrx_ang,*pars_phi), color='limegreen', linestyle='--',label=label_phi)
        ax.plot(xpoints, linfunc(arrx_ang,*pars_phiCut), color='goldenrod', linestyle='--',label=label_phiCut)

    ax.minorticks_on()
    ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
    ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
    deflim = [0,100]
    if(fGain):
        #ax.set_ylim([(phi_cba-phierr_cba)*1.1,maxphi+maxphierr])        
        ax.set_ylim([-8,2])
    elif(fDrift):
        ax.set_ylim([75,100])
    elif(fDirectBeam and fRate):
        if(fSixtyDegDbPlot):
            ax.set_ylim([40,80])
        else:
            ax.set_ylim([-2,100])
    elif(fCustomData):
        ax.set_ylim([minphi-maxphierr*1.2,
                     maxphi+maxphierr*1.2])
        #if(("Energy" in cuTitle or "energy" in cuTitle) and not "SIM" in picname):
        if("Energy" in cuTitle or "energy" in cuTitle):
          
            dE = None
            if(xpoints[0]>xpoints[1]):
                dE = np.diff(sorted(xpoints))
            else: 
                dE = np.diff(xpoints)
            minxspan = np.min(xpoints)-dE/2
            maxxspan = np.max(xpoints)+dE/2

            ax.set_title("")
            
            phi_cba, phierr_cba = None, None
            if(fSimulationData):
                phi_cba, phierr_cba = 0.0,0.1 # for simulation
            else:
                phi_cba, phierr_cba = ref_phi_cp, ref_phi_cperr # for data 

            degstr = r"$^{\circ}$"
            ax.hlines(phi_cba,
                      minxspan,
                      maxxspan, 
                      label=r"$\phi_{CBA}$="+f"{phi_cba:.4f}{degstr}"+r"$\pm$"+f"{phierr_cba:.4f}{degstr}", 
                      linestyles="--", 
                      color='firebrick')
            ax.axhspan(phi_cba-phierr_cba, phi_cba+phierr_cba, color='red', alpha=0.05)
            
            if(fSimulationData):
                ax.set_ylim([-6,6]) # for simulation
            else:
                ax.set_ylim([-5,0]) # for data 
            ax.set_xlim([minxspan, maxxspan])  

    else:
        print("OLEG")
        ax.set_ylim(deflim)

    ax.tick_params(axis='x',labelsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis='y',labelsize=AXIS_LABEL_SIZE)
    ax.legend()
    
    plt.savefig(f"RecoAngles-SINGLE-{picname}.png", dpi=400)
    plt.close()

print(f"{len(stokes_mu)}, {len(stokes_muErr)}, {len(stokes_phi)}, {len(stokes_phiErr)}")

fHaveStokesModulation = True if (len(stokes_mu)>0 and len(stokes_phi)>0) else False
print(f"Do i have a Stokes-based modulation/angle data = {fHaveStokesModulation}")

if(fHaveStokesModulation):
    # ========================================================
    # plotting modulations from stokes parameters...
    # ========================================================

    fig = plt.figure(figsize=(8,7))
    
    ax = fig.add_subplot(111)
    
    ax.errorbar(xdata[0:len(stokes_mu)], np.array(stokes_mu)*100.0, yerr=np.array(stokes_muErr)*100.0, color='blue',ms=8, ecolor='royalblue', fmt='*',capsize=6, label=r"$\mu$ from clean Stokes")
    ax.errorbar(xdata[0:len(Wstokes_mu)], np.array(Wstokes_mu)*100.0, yerr=np.array(Wstokes_muErr)*100.0, color='purple',ms=8,  ecolor='indigo', fmt='*',capsize=6, label=r"$\mu$ from weighted Stokes")
    ax.set_ylabel(r"Reconstructed $\mu$")

    ylim_maxoffset = 5
    ylim_minoffset = 0.9

    if(not fNoTitle):
        if(fAngles):
            ax.set_title("Modulation Factors VS Xray Beam Polarization Angle")
        if(fGain):
            ax.set_title("Modulation Factors vs Grid Voltage")
        if(fRate):
            ax.set_title("Modulation Factors vs Incident X-ray Beam Rate")
        if(fAtten):
            ax.set_title("Modulation Factors vs First Harmonic Attenuation Factors")
        if(fCustomData):
            ax.set_title(f"Modulation Factors vs {cuTitle}")

    if(fAngles):
        ax.set_xlabel(r"Angle of Detector Rotation, [$^{\circ}$]", fontsize=AXIS_FONT_SIZE)
    if(fGain):
        ax.set_xlabel(r"$V_{\mathrm{grid}}$, [V]", fontsize=AXIS_FONT_SIZE)
    if(fRate):
        ax.set_xscale('log')
        ax.set_xlabel("X-ray Beam Rate, [Hz]", fontsize=AXIS_FONT_SIZE)
    if(fAtten):
        ax.set_xscale('log')
        #ax.set_xlim([0.0, ])
        ax.set_xlabel("Attenuation Factor, [a.u.]", fontsize=AXIS_FONT_SIZE)
    if(fDrift):
        ax.set_xlabel(r"$\mathrm{V}_{\mathrm{drfit}}$, [V/cm]", fontsize=AXIS_FONT_SIZE)

    if(fCustomData):
        ax.set_xlabel(cuXlabel)

    ax.minorticks_on()
    ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
    ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
  
    ax.set_ylim([0,50])
    ax.legend()

    plt.savefig(f"StokesMOD-SINGLE-{picname}.png", dpi=400)
    plt.close()

    # ========================================================
    # plotting angles from stokes parameters...
    # ========================================================

    stks_phi_deg, Wstks_phi_deg = None, None
    if(fAngles):
        stks_phi_deg = abs(np.array(stokes_phi)*180/np.pi)
    else:
        stks_phi_deg = np.array(stokes_phi)*180/np.pi
    stks_phiErr_deg = np.array(stokes_phiErr)*180/np.pi
    if(fAngles):
        Wstks_phi_deg = abs(np.array(Wstokes_phi)*180/np.pi)
    else:
        Wstks_phi_deg = np.array(Wstokes_phi)*180/np.pi
    Wstks_phiErr_deg = np.array(Wstokes_phiErr)*180/np.pi

    fig = plt.figure(figsize=(8,7))
    
    ax = fig.add_subplot(111)
    
    #ax.text(120,29,"WORK IN PROGRESS", color='red', backgroundcolor='lightgray', fontweight='semibold', fontsize=12)
    ax.errorbar(xdata[0:len(stokes_phi)], stks_phi_deg, yerr=stks_phiErr_deg, color='blue', ecolor='royalblue', ms=8, fmt='*',capsize=6, label=r"$\phi_{\mathrm{GP3}}$ from clean Stokes")
    ax.errorbar(xdata[0:len(Wstokes_phi)], Wstks_phi_deg, yerr=Wstks_phiErr_deg, color='purple', ecolor='indigo', ms=8, fmt='*',capsize=6, label=r"$\phi_{\mathrm{GP3}}$ from weighted Stokes")
    ax.set_ylabel(r"$\phi_{\mathrm{Reconstructed}}$,"+r"[$^{\circ}$]")
    if(fAngles):
        ax.plot(xdata[0:len(phi)],xdata[0:len(phi)],color='firebrick', linestyle='--', label=r"$\phi(\mathrm{reco})=\phi(\mathrm{expected})$")
        if(not fNoTitle):
            ax.set_title("Reconstructed Polarization Angles VS X-ray Beam Polarization Angle")
        ax.set_xlabel(r"$\phi_{\mathrm{Expected}}$, [$^{\circ}$]", fontsize=AXIS_FONT_SIZE)
    if(fGain):
        if(not fNoTitle):
            ax.set_title("Recosntructed Polarization Angles vs Grid Voltage")
        ax.set_xlabel(r"$V_{\mathrm{Grid}}$, [V]", fontsize=AXIS_FONT_SIZE)
    if(fRate):
        if(not fNoTitle):
            ax.set_title("Reconstructed Polarization Angles vs X-ray Beam Rate")
        ax.set_xscale('log')
        ax.set_xlabel("X-ray Beam Rate, [Hz]", fontsize=AXIS_FONT_SIZE)
    if(fweight):
         ax.set_xlabel(r'$w_{Q}$, [a.u.]', fontsize=AXIS_FONT_SIZE)
    if(fAtten):
        if(not fNoTitle):
            ax.set_title("Reconstructed Polarization Angles vs Attenuiation Factors of First Harmonics")
        ax.set_xlabel('Attenuation Factor, [a.u.]', fontsize=AXIS_FONT_SIZE)
    if(fDrift):
        ax.set_xlabel(r"$\mathrm{V}_{\mathrm{drfit}}$, [V/cm]", fontsize=AXIS_FONT_SIZE)
        if(not fNoTitle):
            ax.set_title(f"Recosntructed Polarization Angles vs Drift Field Variation")

    if(fCustomData):
        if(not fNoTitle):
            ax.set_title(f"Recosntructed Polarization Angles vs {cuTitle}")
        ax.set_xlabel(f"{cuXlabel}", fontsize=AXIS_FONT_SIZE)

    ax.legend()    
    plt.savefig(f"StokesAngles-SINGLE-{picname}.png", dpi=400)
    plt.close()


