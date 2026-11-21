import sys
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#names = ["Analyzer","Polarimeter","Corrected"]
names = ["Analyzer","Polarimeter"]

##################################################
# Charge Peak data
AnalyzerQ, AnalyzerU = [1.035326625390043],[-0.07328935414960994]
AnalyzerQ_err, AnalyzerU_err = [0.03752109044786], [0.03509137717815376]

# older, unknown run
#PolarQ, PolarU = [0.9984],[-0.0558]
#PolarQ_err, PolarU_err = [0.0009],[0.0146]

# more recent, CPmain2H data, ~1e6 events
PolarQ, PolarU = [0.9903],[-0.1391]
PolarQ_err, PolarU_err = [0.02],[0.0129]

# rerun of CPmain2H data with correlated uncertainty case, ~1e6 events
PolarQ, PolarU = [0.9915],[-0.1304]
PolarQ_err, PolarU_err = [0.0023],[0.0003]

# Direct beam data 
AnalDB_0_Q, AnalDB_0_Qerr = [0.978512314947401], [0.010527069843376625]
AnalDB_0_U, AnalDB_0_Uerr = [0.005452992210846639], [0.009862157795284005]

## unknown run
#PolarDB_0_Q, PolarDB_0_Qerr = [0.9976], [0.0016]
#PolarDB_0_U, PolarDB_0_Uerr = [-0.0690], [0.0170]

# direct beam 15kHz, 0 degrees
PolarDB_0_Q, PolarDB_0_Qerr = [0.9986], [0.0035]
PolarDB_0_U, PolarDB_0_Uerr = [-0.0533], [0.0002]

# Magnetic Peak data 
# !!! fro now, WEEEERY preliminary data

AnalMP_Q, AnalMP_Qerr = [-1.0028147295713346], [0.04986157301568596]
AnalMP_U, AnalMP_Uerr = [-0.19042628692040245], [0.03860217416190682]

# older data
#PolarMP_Q, PolarMP_Qerr = [-0.9880],[0.0076]
#PolarMP_U, PolarMP_Uerr = [-0.1545], [0.0294]

# from fit of MP-500Vcm +MP2 + MP3 + MP4
#PolarMP_Q, PolarMP_Qerr = [-0.9911],[0.0055]
#PolarMP_U, PolarMP_Uerr = [-0.1329], [0.0208]

# from fit of MP-500Vcm +MP2 + MP3 + MP4 - rerun with correlated uncertainties
PolarMP_Q, PolarMP_Qerr = [-0.9904],[0.0066]
PolarMP_U, PolarMP_Uerr = [-0.1380], [0.0009]

## as a test adding stokes calc for corrected Q,U via q_source = (q(0) - q(90))/2 & u_source = (u(0) - u(90))/2

DBcorrQ, DBcorrU = 0.8963 ,  0.2772
CPcorrQ, CPcorrU = 0.9969 , -0.0273
MPcorrQ, MPcorrU = 0.0036 , -0.0328

##################################################

figsize=(12,9)
#figsize=(16,12)
#figsize=(9,7)
#ncols, nrows = 3,1
ncols, nrows = 2,1
#wratios = [1,1,1]
wratios = [1,1]

##################################################

defxlim = [0,3]#[0,4]
defylim = [-1.2,1.2]

##################################################

DEF_FONT_SIZE = 14
DEF_TICK_SIZE = 14
DEF_TICK_LEN = 5
DEF_TICK_WIDTH = 1.25

##################################################

fig = plt.figure(figsize=figsize)
gs = GridSpec(nrows,ncols, width_ratios=wratios, height_ratios=[1], wspace=0.1)

ax_db = fig.add_subplot(gs[0,0])
ax_cp = fig.add_subplot(gs[0,1], sharey=ax_db)
#ax_mp = fig.add_subplot(gs[0,2], sharey=ax_db)

#x = [1,2,3]
x = [1,2]

#################### CP DATA ##################################################
ax_cp.errorbar([],[],xerr=[],yerr=[],color="white",label="P09 Analyzer")
ax_cp.errorbar(x[0],AnalyzerQ,yerr=AnalyzerQ_err,color='red', ecolor="black", fmt="s",capsize=6, label=f"Q = {AnalyzerQ[0]:.4f}"+r"$\pm$"+f"{AnalyzerQ_err[0]:.4f}")
ax_cp.errorbar(x[0],AnalyzerU,yerr=AnalyzerU_err,color='firebrick', ecolor="black", fmt="o",capsize=6, label=f"U = {AnalyzerU[0]:.4f}"+r"$\pm$"+f"{AnalyzerU_err[0]:.4f}")
ax_cp.errorbar([],[],yerr=[],color="white",label="GridPix3")
ax_cp.errorbar(x[1],PolarQ,yerr=PolarQ_err,color='blue', ecolor="black", fmt="s",capsize=6, label=f"Q = {PolarQ[0]:.4f}"+r"$\pm$"+f"{PolarQ_err[0]:.4f}")
ax_cp.errorbar(x[1],PolarU,yerr=PolarU_err,color='royalblue', ecolor="black", fmt="o",capsize=6, label=f"U = {PolarU[0]:.4f}"+r"$\pm$"+f"{PolarU_err[0]:.4f}")

# corrections for spurious modulation
#ax_cp.scatter(x[2], CPcorrQ, color='forestgreen', marker="s", label=r"$Q_{sm,corr}$"+f"={CPcorrQ}")
#ax_cp.scatter(x[2], CPcorrU, color='limegreen', marker="o", label=r"$U_{sm,corr}$"+f"={CPcorrU}")

ax_cp.hlines(AnalyzerQ[0],0,3,colors='red', linestyles='dashed')#,label=f"Q={AnalyzerQ[0]:.4f}")
ax_cp.hlines(PolarQ[0],0,3,colors='blue', linestyles='dashed')#,label=f"Q={PolarQ[0]:.4f}")
ax_cp.axhspan(AnalyzerQ[0]-AnalyzerQ_err[0], AnalyzerQ[0]+AnalyzerQ_err[0], color='red', alpha=0.05)
ax_cp.axhspan(AnalyzerQ[0]-AnalyzerQ_err[0]*2, AnalyzerQ[0]+AnalyzerQ_err[0]*2, color='red', alpha=0.05)
ax_cp.axhspan(AnalyzerQ[0]-AnalyzerQ_err[0]*3, AnalyzerQ[0]+AnalyzerQ_err[0]*3, color='red', alpha=0.05)
ax_cp.hlines(AnalyzerU[0],0,3,colors='red', linestyles='dashed')
ax_cp.hlines(PolarU[0],0,3,colors='blue', linestyles='dashed')
ax_cp.axhspan(AnalyzerU[0]-AnalyzerU_err[0], AnalyzerU[0]+AnalyzerU_err[0], color='red', alpha=0.05)
ax_cp.axhspan(AnalyzerU[0]-AnalyzerU_err[0]*2, AnalyzerU[0]+AnalyzerU_err[0]*2, color='red', alpha=0.05)
ax_cp.axhspan(AnalyzerU[0]-AnalyzerU_err[0]*3, AnalyzerU[0]+AnalyzerU_err[0]*3, color='red', alpha=0.05)

ax_cp.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax_cp.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax_cp.minorticks_on()
#ax_cp.legend(loc='lower right')
ax_cp.legend()
ax_cp.tick_params(labelleft=False)
ax_cp.set_xticks(x)
ax_cp.set_xticklabels(names, fontsize=DEF_FONT_SIZE)
#ax_cp.tick_params(labelsize=DEF_TICK_SIZE)
ax_cp.tick_params(labelsize=DEF_TICK_SIZE, length=DEF_TICK_LEN, width=DEF_TICK_WIDTH)
#ax_cp.set_ylabel("Normalized Parameter values")
#ax_cp.set_title("Sokes Parameters for P09 Analyzer and GridPix3 Polarimeter")
ax_cp.set_title("Charge Scattering", fontsize=DEF_FONT_SIZE)
ax_cp.set_xlim(defxlim)
ax_cp.set_ylim(defylim)

#################### DB DATA ##################################################
ax_db.errorbar([],[],xerr=[],yerr=[],color="white",label="P09 Analyzer")
ax_db.errorbar(x[0],AnalDB_0_Q,yerr=AnalDB_0_Qerr, color='red', ecolor="black", fmt="s", capsize=6, label=f"Q = {AnalDB_0_Q[0]:.4f}"+r"$\pm$"+f"{AnalDB_0_Qerr[0]:.4f}")
ax_db.errorbar(x[0],AnalDB_0_U,yerr=AnalDB_0_Uerr, color='firebrick', ecolor="black", fmt="o",capsize=6, label=f"U = {AnalDB_0_U[0]:.4f}"+r"$\pm$"+f"{AnalDB_0_Uerr[0]:.4f}")
ax_db.errorbar([],[],xerr=[],yerr=[],color="white",label="GridPix3")
ax_db.errorbar(x[1],PolarDB_0_Q,yerr=PolarDB_0_Qerr, color='blue', ecolor="black", fmt="s", capsize=6, label=f"Q = {PolarDB_0_Q[0]:.4f}"+r"$\pm$"+f"{PolarDB_0_Qerr[0]:.4f}")
ax_db.errorbar(x[1],PolarDB_0_U,yerr=PolarDB_0_Uerr, color='royalblue', ecolor="black", fmt="o",capsize=6, label=f"U = {PolarDB_0_U[0]:.4f}"+r"$\pm$"+f"{PolarDB_0_Uerr[0]:.4f}")

# spurious mod correction
#ax_db.scatter(x[2], DBcorrQ, color='forestgreen', marker="s", label=r"$Q_{sm,corr}$"+f"={DBcorrQ}")
#ax_db.scatter(x[2], DBcorrU, color='limegreen', marker="o", label=r"$U_{sm,corr}$"+f"={DBcorrU}")

ax_db.hlines(AnalDB_0_Q[0],0,3,colors='red', linestyles='dashed')#,label=f"Q={AnalDB_0_Q[0]:.4f}")
ax_db.hlines(PolarDB_0_Q[0],0,3,colors='blue', linestyles='dashed')#,label=f"Q={PolarDB_0_Q[0]:.4f}")
ax_db.axhspan(AnalDB_0_Q[0]-AnalDB_0_Qerr[0], AnalDB_0_Q[0]+AnalDB_0_Qerr[0], color='red', alpha=0.05)
ax_db.axhspan(AnalDB_0_Q[0]-AnalDB_0_Qerr[0]*2, AnalDB_0_Q[0]+AnalDB_0_Qerr[0]*2, color='red', alpha=0.05)
ax_db.axhspan(AnalDB_0_Q[0]-AnalDB_0_Qerr[0]*3, AnalDB_0_Q[0]+AnalDB_0_Qerr[0]*3, color='red', alpha=0.05)
ax_db.hlines(AnalDB_0_U[0],0,3,colors='red', linestyles='dashed')
ax_db.hlines(PolarDB_0_U[0],0,3,colors='blue', linestyles='dashed')
ax_db.axhspan(AnalDB_0_U[0]-AnalDB_0_Uerr[0], AnalDB_0_U[0]+AnalDB_0_Uerr[0], color='red', alpha=0.05)
ax_db.axhspan(AnalDB_0_U[0]-AnalDB_0_Uerr[0]*2, AnalDB_0_U[0]+AnalDB_0_Uerr[0]*2, color='red', alpha=0.05)
ax_db.axhspan(AnalDB_0_U[0]-AnalDB_0_Uerr[0]*3, AnalDB_0_U[0]+AnalDB_0_Uerr[0]*3, color='red', alpha=0.05)

ax_db.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax_db.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax_db.minorticks_on()
#ax_db.legend(loc='lower right')
ax_db.legend()
ax_db.set_xticks(x)
ax_db.set_xticklabels(names, fontsize=DEF_FONT_SIZE)
#ax_db.tick_params(labelsize=DEF_TICK_SIZE)
ax_db.tick_params(labelsize=DEF_TICK_SIZE, length=DEF_TICK_LEN, width=DEF_TICK_WIDTH)
ax_db.set_ylabel("Normalized Parameter values", fontsize=DEF_FONT_SIZE)
#ax_db.set_title("Sokes Parameters for P09 Analyzer and GridPix3 Polarimeter")
ax_db.set_title("Direct Beam", fontsize=DEF_FONT_SIZE)
ax_db.set_xlim(defxlim)
ax_db.set_ylim(defylim)

##################### MP DATA ##################################################
#ax_mp.errorbar([],[],xerr=[],yerr=[],color="white",label="P09 Analyzer")
#ax_mp.errorbar(x[0],AnalMP_Q,AnalMP_Qerr,[0.0],color='red', ecolor="black", fmt="s",capsize=4, label=f"Q = {AnalMP_Q[0]:.4f}"+r"$\pm$"+f"{AnalMP_Qerr[0]:.4f}")
#ax_mp.errorbar(x[0],AnalMP_U,AnalMP_Uerr,[0.0],color='firebrick', ecolor="black", fmt="o",capsize=4, label=f"U = {AnalMP_U[0]:.4f}"+r"$\pm$"+f"{AnalMP_Uerr[0]:.4f}")
#ax_mp.errorbar([],[],xerr=[],yerr=[],color="white",label="GridPix3")
#ax_mp.errorbar(x[1],PolarMP_Q,PolarMP_Qerr,[0.0],color='blue', ecolor="black", fmt="s",capsize=4, label=f"Q = {PolarMP_Q[0]:.4f}"+r"$\pm$"+f"{PolarMP_Qerr[0]:.4f}")
#ax_mp.errorbar(x[1],PolarMP_U,PolarMP_Uerr,[0.0],color='royalblue', ecolor="black", fmt="o",capsize=4, label=f"U = {PolarMP_U[0]:.4f}"+r"$\pm$"+f"{PolarMP_Uerr[0]:.4f}")
#
## correction for spurious modulation
#ax_mp.scatter(x[2], MPcorrQ, color='forestgreen', marker="s", label=r"$Q_{sm,corr}$"+f"={MPcorrQ}")
#ax_mp.scatter(x[2], MPcorrU, color='limegreen', marker="o", label=r"$U_{sm,corr}$"+f"={MPcorrU}")
#
#ax_mp.hlines(AnalMP_Q[0],0,3,colors='red', linestyles='dashed')#,label=f"Q={AnalMP_Q[0]:.4f}")
#ax_mp.hlines(PolarMP_Q[0],0,3,colors='blue', linestyles='dashed')#,label=f"Q={PolarMP_Q[0]:.4f}")
#ax_mp.axhspan(AnalMP_Q[0]-AnalMP_Qerr[0], AnalMP_Q[0]+AnalMP_Qerr[0], color='red', alpha=0.05)
#ax_mp.axhspan(AnalMP_Q[0]-AnalMP_Qerr[0]*2, AnalMP_Q[0]+AnalMP_Qerr[0]*2, color='red', alpha=0.05)
#ax_mp.axhspan(AnalMP_Q[0]-AnalMP_Qerr[0]*3, AnalMP_Q[0]+AnalMP_Qerr[0]*3, color='red', alpha=0.05)
#ax_mp.hlines(AnalMP_U[0],0,3,colors='red', linestyles='dashed')
#ax_mp.hlines(PolarMP_U[0],0,3,colors='blue', linestyles='dashed')
#ax_mp.axhspan(AnalMP_U[0]-AnalMP_Uerr[0], AnalMP_U[0]+AnalMP_Uerr[0], color='red', alpha=0.05)
#ax_mp.axhspan(AnalMP_U[0]-AnalMP_Uerr[0]*2, AnalMP_U[0]+AnalMP_Uerr[0]*2, color='red', alpha=0.05)
#ax_mp.axhspan(AnalMP_U[0]-AnalMP_Uerr[0]*3, AnalMP_U[0]+AnalMP_Uerr[0]*3, color='red', alpha=0.05)
#
#ax_mp.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
#ax_mp.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
#ax_mp.minorticks_on()
#ax_mp.legend()
##ax_mp.legend(loc='lower right')
#ax_mp.tick_params(labelleft=False)
#ax_mp.set_xticks(x)
#ax_mp.set_xticklabels(names)
##ax_mp.set_ylabel("Normalized Parameter values")
##ax_mp.set_title("Sokes Parameters for P09 Analyzer and GridPix3 Polarimeter")
#ax_mp.set_title("Magnetic Peak Data")
#ax_mp.set_xlim(defxlim)
##ax_mp.set_ylim(defylim)

#plt.savefig("StokesCompared.png",dpi=400)
#plt.savefig("StokesCompared-wSMcorrection.png",dpi=400)

plt.savefig("StokesCompared-DB-CP.png",dpi=512)
plt.close()

