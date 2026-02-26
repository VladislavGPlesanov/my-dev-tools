import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.gridspec import GridSpec 

def toDegree(angle):
    return angle*180/np.pi

logfile_data = sys.argv[1]
logfile_sim = sys.argv[2]
picname = sys.argv[3]

sim_phi = []
sim_phi_err = []

phi, phierr = [], []
phicorr = []
mu, muerr = [], []

def_angles = [0,30,60,90]

dfile = open(logfile_data, "r")

for line in dfile:

    if("-SUCCESS-" in line):
        break

    if("Mod_Cut" in line):
        data = line.split(":")[1]
        params = data.split(",")
        for par in params:
            #if("mu=" in par):
            #    muCut.append(float(par.split("=")[1])*100.0)
            #if("muErr=" in par):
            #    muCutErr.append(float(par.split("=")[1])*100.0)
            if("phi=" in par):
                phi.append(toDegree(abs(float(par.split("=")[1]))))
                phicorr.append(toDegree(abs(float(par.split("=")[1]))-0.0341))
            if("phierr=" in par):
                phierr.append(toDegree(abs(float(par.split("=")[1]))))

dfile.close()

print(f"data:\nphi={phi} [{len(phi)}]\nphierr={phierr} [{len(phierr)}]")

sfile = open(logfile_sim, "r")

for line in sfile:

    if("-SUCCESS-" in line):
        break

    if("Mod_Cut" in line):
        data = line.split(":")[1]
        params = data.split(",")
        for par in params:
            #if("mu=" in par):
            #    muCut.append(float(par.split("=")[1])*100.0)
            #if("muErr=" in par):
            #    muCutErr.append(float(par.split("=")[1])*100.0)
            if("phi=" in par):
                sim_phi.append(toDegree(abs(float(par.split("=")[1]))))
            if("phierr=" in par):
                sim_phi_err.append(toDegree(abs(float(par.split("=")[1]))))

sfile.close()
print(f"SIM:\nphi={sim_phi} [{len(sim_phi)}]\nphierr={sim_phi_err} [{len(sim_phi_err)}]")

fig = plt.figure(figsize=(9,7))
gs = GridSpec(2,1, width_ratios=[1], height_ratios=[4,1], hspace=0.02, wspace=0.05)
ax_main = fig.add_subplot(gs[0,0])
ax_bot = fig.add_subplot(gs[1,0], sharex=ax_main)

ax_main.tick_params(labelbottom=False)
ax_main.errorbar(def_angles, phi, yerr=phierr, color='orange', ecolor='black', fmt='o', capsize=6, label="Data")
ax_main.errorbar(def_angles, phicorr, yerr=phierr, color='darkblue', ecolor='black', fmt='o', capsize=6, label="Data(+offset)")
ax_main.errorbar(def_angles, sim_phi, yerr=sim_phi_err, color='black', ecolor='black', fmt='^', capsize=6,  label="Simulation")
ax_main.plot(def_angles, def_angles, color='blue', linestyle=":", alpha=0.5, label=r"$\phi(expected)=\phi(reco)$")
ax_main.set_title("Reconstructed and Simulated Polarization Angles")
#ax_main.set_xlabel(r"Expected $\phi$, [$^{\circ}$]")
ax_main.set_ylabel(r"Reconstructed $\phi$, [$^{\circ}$]")
ax_main.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
ax_main.grid(which='minor', color='gray', linestyle='--', linewidth=0.25)
ax_main.minorticks_on()
ax_main.set_xlim([-10, 100.0])
ax_main.set_ylim([-10, 100.0])
ax_main.legend(loc='upper left')

# below plotting distance from data to simulation in units of (n x delta_phi(data))
tension_data = np.abs((np.array(phi) - np.array(def_angles)))/np.array(phierr)
tension_datacorr = np.abs((np.array(phicorr) - np.array(def_angles)))/np.array(phierr)
tension_sim = np.abs((np.array(sim_phi) - np.array(def_angles)))/np.array(sim_phi_err)

ax_bot.scatter(def_angles, tension_data, color='orange', label='Data')
ax_bot.scatter(def_angles, tension_datacorr, color='darkblue', label='Data(+offset)')
ax_bot.scatter(def_angles, tension_sim, color='black', label="Simulation")
ax_bot.set_ylabel(r"$\frac{\phi(data) - \phi(sim)}{\sigma_{\phi}(data)}}$")
ax_bot.set_xlabel(r"Expected $\phi$, [$^{\circ}$]")
ax_bot.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
ax_bot.grid(which='minor', color='gray', linestyle='--', linewidth=0.25)
ax_bot.set_ylim([0.0, np.max([np.max(tension_data), np.max(tension_sim)])*1.2])
ax_bot.minorticks_on()
ax_bot.tick_params(labelbottom=True)
#ax_bot.set_yscale('log')
ax_bot.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f"RecoAngles-SIMvsDATA-{picname}.png", dpi=200)
plt.close()







