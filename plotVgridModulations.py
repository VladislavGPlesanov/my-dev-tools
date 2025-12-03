import sys
import numpy as np
import matplotlib.pyplot as plt

Vgrid = [450,460,470,480,490,500]

Modfac = [7.22,8.58, 9.77,10.42,11.46,11.71] #All events
Mf_err = [0.001, 0.001, 0.001,0.001,0.001,0.001]

Modfac_wcuts = [9.22,9.32, 10.17,11.30,11.73,12.57] #Position cut, AbsPoints near beamspot
Mf_wcuts_err = [0.01, 0.001,0.001,0.001,0.001,0.001]

Modfac_wcuts_hT = [6.60, 8.10, 9.64, 11.02, 11.53, 12.27] #Position + hit + sumTOT cuts
Mf_wcuts_err_hT = [0.01, 0.001, 0.001, 0.001, 0.001, 0.001]

Modfac_wcuts_largeSigma = [5.89, 7.74, 9.54, 10.54, 11.51, 12.17] #(x_abs,y_abs) <= 2.5 sigmaX, + hits + sumTOT
Mf_wcuts_err_largeSigma = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

MFAC_hitsOnly = [7.23, 8.63, 9.99, 10.68, 11.72, 11.87]
MFAC_hitsOnly_err = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

def_err = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

MFAC_diffRad = [8.11, 9.71, 10.89, 10.90, 12.47, 13.14]

MFAC_newRad = [0.08644722753742028, 0.10809366753026833, 0.12636240866828868, 0.12840247490827855, 0.14597011795011872,0.1549121771714411 ]
MFAC_newRadErr = [0.004796311511243695, 0.004487462923205948, 0.00409709498539382, 0.004472242766047134, 0.0047952900634751225, 0.004219247685837028]

recoAngle = [ -0.032425629560060336, -0.061696126251769616, -0.019377138903903107, -0.022087722591740737, -0.026305424239759963, -0.02941906755382352  ]
recoAngle_err = [ 0.02758707811865441, 0.02057774533974994, 0.0160209591799983, 0.017203510988188515, 0.016169228733732513, 0.013379512402437035 ]

#########################################################################

fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)

ax.errorbar(Vgrid, Modfac, yerr=Mf_err, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"No cuts, Original $\mu$")
ax.errorbar(Vgrid, Modfac_wcuts, yerr=Mf_wcuts_err, color='firebrick', ecolor='black', fmt='*',capsize=4, label=r"Cut: (x,y)_ap<=1.5")
ax.errorbar(Vgrid, Modfac_wcuts_hT, yerr=Mf_wcuts_err_hT, color='royalblue', ecolor='black', fmt='*',capsize=4, label=r"Cut: (x,y)_ap<=1.5 + hits +"+r"$\Sigma$(TOT)")
ax.errorbar(Vgrid, Modfac_wcuts_largeSigma, yerr=Mf_wcuts_err_largeSigma, color='goldenrod', ecolor='black', fmt='*',capsize=4, label=r"Cut: (x,y)_ap<=2.5 + hits +"+r"$\Sigma$(TOT)")
ax.errorbar(Vgrid, MFAC_hitsOnly, yerr=MFAC_hitsOnly_err, color='magenta', ecolor='black', fmt='*',capsize=4, label=r"Cut: hits")
ax.errorbar(Vgrid, MFAC_diffRad, yerr=def_err, color='yellow', ecolor='black', fmt='*',capsize=4, label=r"Cut: hits, Rin=1.6,Rout=3.5")
ax.errorbar(Vgrid, np.asarray(MFAC_newRad)*100, yerr=(np.asarray(MFAC_newRadErr))*100, color='cyan', ecolor='black', fmt='*',capsize=4, label=r"Cut: hits, Rin=1.5, Rout=3.5, Weight=2.5")
ax.set_title("Reconsturcted Modulation Factors vs. Grid Voltage")
ax.set_xlabel(r"$V_{Grid}$, [V]")
ax.set_ylabel(r"$\mu$,[%]")
ax.set_xlim([440,510])
ax.set_ylim([0,20])
ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax.legend(loc='lower right')

plt.savefig("Reco-ModFacs-vs-VGrid.png",dpi=800)
plt.close()

#------ reco angles -----------

fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)

ax.errorbar(Vgrid, recoAngle, yerr=recoAngle_err, color='forestgreen', ecolor='black', fmt='*',capsize=4)
ax.set_title("Reconsturcted Angles vs. Grid Voltage")
ax.set_xlabel(r"$V_{Grid}$, [V]")
ax.set_ylabel(r"$\phi$,[radian]")
ax.set_xlim([440,510])
ax.set_ylim([-np.pi/4,np.pi/4])
ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
#ax.legend(loc='lower right')
plt.savefig("Reco-Angles-vs-VGrid.png",dpi=1000)
plt.close()

