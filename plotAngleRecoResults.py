import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.gridspec import GridSpec 

def toDegree(angle):
    return angle*180/np.pi

logfile_data = sys.argv[1]
logfile_sim = sys.argv[2]
picname = sys.argv[3]

DEF_FONT_SIZE = 12
DEF_TICK_SIZE = 12

sim_phi = []
sim_phi_err = []

rawphi, rawphierr = [], []
phi, phierr = [], []
phicorr = []

rawmu, rawmu_err = [], []
mu, muerr = [], []
sim_mu, sim_muerr = [], []

def_angles = [0,30,60,90]

dfile = open(logfile_data, "r")

for line in dfile:

    if("-SUCCESS-" in line):
        break

    if("Mod_Orig" in line):
        data = line.split(":")[1]
        params = data.split(",")
        for par in params:
            if("mu=" in par):
                rawmu.append(float(par.split("=")[1])*100.0)
            if("muErr=" in par):
                rawmu_err.append(float(par.split("=")[1])*100.0)
            if("phi=" in par):
                rawphi.append(toDegree(abs(float(par.split("=")[1]))))
            if("phierr=" in par):
                rawphierr.append(toDegree(abs(float(par.split("=")[1]))))

    if("Mod_Cut" in line):
        data = line.split(":")[1]
        params = data.split(",")
        for par in params:
            if("mu=" in par):
                mu.append(float(par.split("=")[1])*100.0)
            if("muErr=" in par):
                muerr.append(float(par.split("=")[1])*100.0)
            if("phi=" in par):
                phi.append(toDegree(abs(float(par.split("=")[1]))))
#                phicorr.append(toDegree(abs(float(par.split("=")[1]))-0.0341))
            if("phierr=" in par):
                phierr.append(toDegree(abs(float(par.split("=")[1]))))

dfile.close()

print(f"data(PHI):\nphi={phi} [{len(phi)}]\nphierr={phierr} [{len(phierr)}]")
print(f"data(MODFACS):\nmu={mu} [{len(mu)}]\nmuerr={muerr} [{len(muerr)}]")

sfile = open(logfile_sim, "r")

for line in sfile:

    if("-SUCCESS-" in line):
        break

    if("Mod_Cut" in line):
        data = line.split(":")[1]
        params = data.split(",")
        for par in params:
            if("mu=" in par):
                sim_mu.append(float(par.split("=")[1])*100.0)
            if("muErr=" in par):
                sim_muerr.append(float(par.split("=")[1])*100.0)
            if("phi=" in par):
                sim_phi.append(toDegree(abs(float(par.split("=")[1]))))
            if("phierr=" in par):
                sim_phi_err.append(toDegree(abs(float(par.split("=")[1]))))

sfile.close()
print(f"SIM(PHI):\nphi={sim_phi} [{len(sim_phi)}]\nphierr={sim_phi_err} [{len(sim_phi_err)}]")
print(f"SIM(MODFACS):\nphi={sim_mu} [{len(sim_mu)}]\nmuerr={sim_muerr} [{len(sim_muerr)}]")

###############################################################################################
# plotting reconstructed angle comparison
###############################################################################################


fig = plt.figure(figsize=(9,7))
gs = GridSpec(2,1, width_ratios=[1], height_ratios=[4,1], hspace=0.02, wspace=0.05)
ax_main = fig.add_subplot(gs[0,0])
ax_bot = fig.add_subplot(gs[1,0], sharex=ax_main)

ax_main.tick_params(labelbottom=False)
ax_main.errorbar(def_angles, phi, yerr=phierr, color='orange', ecolor='black', fmt='o', capsize=6, label="After Cuts")
ax_main.errorbar(def_angles, rawphi, yerr=rawphierr, color='green', ecolor='black', fmt='o', capsize=6, label="Raw Data")
#ax_main.errorbar(def_angles, phicorr, yerr=phierr, color='darkblue', ecolor='black', fmt='o', capsize=6, label="Data(+offset)")
ax_main.errorbar(def_angles, sim_phi, yerr=sim_phi_err, color='black', ecolor='black', fmt='^', capsize=6,  label="Simulation")
ax_main.plot(def_angles, def_angles, color='blue', linestyle=":", alpha=0.5, label=r"$\phi(\mathrm{expected})=\phi(\mathrm{reco})$")
#ax_main.set_title("Reconstructed and Simulated Polarization Angles")
#ax_main.set_xlabel(r"Expected $\phi$, [$^{\circ}$]")
ax_main.set_ylabel(r"Reconstructed $\phi$, [$^{\circ}$]", fontsize=DEF_FONT_SIZE)
ax_main.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
ax_main.grid(which='minor', color='gray', linestyle='--', linewidth=0.25)
ax_main.minorticks_on()
ax_main.tick_params(axis='y', labelsize=DEF_TICK_SIZE)
ax_main.set_xlim([-10, 100.0])
ax_main.set_ylim([-10, 100.0])
ax_main.legend(loc='upper left')

# below plotting distance from data to simulation in units of (n x delta_phi(data))
tension_data = np.abs((np.array(phi) - np.array(def_angles)))/np.array(phierr)
tension_rawdata = np.abs((np.array(rawphi) - np.array(def_angles)))/np.array(phierr)
#tension_datacorr = np.abs((np.array(phicorr) - np.array(def_angles)))/np.array(phierr)
tension_sim = np.abs((np.array(sim_phi) - np.array(def_angles)))/np.array(sim_phi_err)

ax_bot.scatter(def_angles, tension_data, color='orange', label='Data')
ax_bot.scatter(def_angles, tension_rawdata, color='green', label='Data')
#ax_bot.scatter(def_angles, tension_datacorr, color='darkblue', label='Data(+offset)')
ax_bot.scatter(def_angles, tension_sim, color='black', label="Simulation")

maxdev = np.max([np.max(tension_data),np.max(tension_sim),np.max(tension_rawdata)])

ax_bot.set_ylabel(r"$|\frac{\phi(\mathrm{data}) - \phi(\mathrm{sim})}{\sigma_{\phi}(\mathrm{data})}|$", fontsize=DEF_FONT_SIZE)
ax_bot.set_xlabel(r"Expected $\phi$, [$^{\circ}$]", fontsize=DEF_FONT_SIZE)
ax_bot.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
ax_bot.grid(which='minor', color='gray', linestyle='--', linewidth=0.25)
ax_bot.set_ylim([0.0, maxdev*1.2])
ax_bot.minorticks_on()
ax_bot.tick_params(labelbottom=True)
ax_bot.tick_params(axis='x', labelsize=DEF_TICK_SIZE)
ax_bot.tick_params(axis='y', labelsize=DEF_TICK_SIZE)
#ax_bot.set_yscale('log')
#ax_bot.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f"RecoAngles-SIMvsDATA-{picname}.png", dpi=200)
plt.close()

###############################################################################################
# plotting modulation factor comparison
###############################################################################################

fig = plt.figure(figsize=(9,7))
gs = GridSpec(2,1, width_ratios=[1], height_ratios=[4,1], hspace=0.02, wspace=0.05)
ax_main = fig.add_subplot(gs[0,0])
ax_bot = fig.add_subplot(gs[1,0], sharex=ax_main)

mean_sim_mu = np.mean(sim_mu)
def_mu = [mean_sim_mu for i in range(len(sim_mu))]

ax_main.tick_params(labelbottom=False)
ax_main.errorbar(def_angles, mu, yerr=muerr, color='orange', ecolor='black', fmt='o', capsize=6, label="After Cuts")
ax_main.errorbar(def_angles, rawmu, yerr=rawmu_err, color='green', ecolor='black', fmt='o', capsize=6, label="Raw Data")
#ax_main.errorbar(def_angles, mucorr, yerr=muerr, color='darkblue', ecolor='black', fmt='o', capsize=6, label="Data(+offset)")
ax_main.errorbar(def_angles, sim_mu, yerr=sim_muerr, color='black', ecolor='black', fmt='^', capsize=6,  label="Simulation")
ax_main.plot(def_angles, def_mu, color='blue', linestyle=":", alpha=0.5, label=r"$\mu(\mathrm{expected})=\overline{\mu(\mathrm{sim})}$")
#ax_main.set_title("Reconstructed and Simulated Modulation Factors")
#ax_main.set_xlabel(r"Expected $\mu$, [$^{\circ}$]")
ax_main.set_ylabel(r"Modulation, $\mu$, [%]", fontsize=DEF_FONT_SIZE)
ax_main.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
ax_main.grid(which='minor', color='gray', linestyle='--', linewidth=0.25)
ax_main.minorticks_on()
ax_main.tick_params(axis='y', labelsize=DEF_TICK_SIZE)
ax_main.set_xlim([-5, 95])
ax_main.set_ylim([0, 100.0])
ax_main.legend(loc='upper left')

# below plotting distance from data to simulation in units of (n x delta_mu(data))
tension_data = np.abs((np.array(mu) - mean_sim_mu))/np.array(muerr)
tension_rawdata = np.abs((np.array(rawmu) - mean_sim_mu))/np.array(rawmu_err)
#tension_datacorr = np.abs((np.array(mucorr) - np.array(def_angles)))/np.array(muerr)
tension_sim = np.abs((np.array(sim_mu) - mean_sim_mu))/np.array(sim_muerr)

mu_maxdev = np.max([np.max(tension_data),np.max(tension_sim),np.max(tension_rawdata)])

ax_bot.scatter(def_angles, tension_data, color='orange', label='After Cuts')
ax_bot.scatter(def_angles, tension_rawdata, color='green', label='Raw Data')
#ax_bot.scatter(def_angles, tension_datacorr, color='darkblue', label='Data(+offset)')
ax_bot.scatter(def_angles, tension_sim, color='black', label="Simulation")
ax_bot.set_ylabel(r"$|\frac{\mu(\mathrm{data}) - \mu(\mathrm{sim})}{\sigma_{\mu}(\mathrm{data})}|$", fontsize=DEF_FONT_SIZE)
ax_bot.set_xlabel(r"Detector $\phi$, [$^{\circ}$]", fontsize=DEF_FONT_SIZE)
ax_bot.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
ax_bot.grid(which='minor', color='gray', linestyle='--', linewidth=0.25)
ax_bot.set_ylim([0.0, mu_maxdev*1.2])
ax_bot.minorticks_on()
ax_bot.tick_params(labelbottom=True)
ax_bot.tick_params(axis='x', labelsize=DEF_TICK_SIZE)
ax_bot.tick_params(axis='y', labelsize=DEF_TICK_SIZE)

#ax_bot.set_yscale('log')
#ax_bot.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f"ModFacs-SIMvsDATA-{picname}.png", dpi=200)
plt.close()






