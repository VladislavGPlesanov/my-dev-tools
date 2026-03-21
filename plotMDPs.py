import numpy as np
import matplotlib.pyplot as plt 
import sys

minutes = float(sys.argv[1])
picname = sys.argv[2]

#sig_rates = np.linspace(10.0,1e6,100)
sig_rates = [i*10 for i in range(100)]
sig_rates = np.array(sig_rates)

fix_sigrates = [5, 10, 100, 1000, 10000]

bgr_rates = np.linspace(1.0,1e5,100)
mus = np.linspace(0.0,1.0,100)
times = np.linspace(60.0,18000.0,100)

fix_bgr = [10,100,1000,10000,100000]

const_bgr = 5000
const_mu = 0.5
const_t = minutes*60.0

factor = 4.29

mdp_fix_bgr = [] 
labels = []

# calc MDP with const-> mu, BGR, time
for rbg in fix_bgr:
    tmp_mdp = []
    for r in sig_rates:
    
        imdp = factor/(const_mu*r)*np.sqrt((r+rbg)/const_t)
        tmp_mdp.append(imdp)
        imdp = None

    labels.append(f"BGR={rbg}Hz")
    mdp_fix_bgr.append(tmp_mdp)
    tmp_mdp = None

for l in mdp_fix_bgr:
    print(len(l))

colors = ["firebrick","darkgreen","limegreen", "royalblue", "goldenrod", ]
plt.figure(figsize=(10,8))

for mdp, lab, clr in zip(mdp_fix_bgr, labels, colors):
    print(len(sig_rates))
    print(len(mdp))
    plt.scatter(sig_rates, np.array(mdp)*100.0, color=clr, marker="o", label=lab)

plt.hlines(100,0,np.max(sig_rates),colors="red",linestyles=":")
char_t = r"$t_{\mathrm{meas}}$"
char_mu = r"$\mu$"
plt.title(f"MDPs vs Signal Rate ({char_t}={const_t}[s], {char_mu}={const_mu})")
plt.xlabel("Signal Rate, [Hz]")
plt.ylabel("MDP, [%]")
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='upper right')
plt.minorticks_on()
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)

defpicname = "HUYA"
savename = "MDPs-vs-Rate-"
if(picname is not None):
    savename+=picname
else:
    savename+=defpicname

plt.savefig(f"{savename}.png")
plt.close()
##########################################################
##########################################################
##########################################################
mdp_fix_bgr_var_time = []
labels_vartime = []

for fsig in fix_sigrates:
    tmp_mdp = []
    for t in times:
    
        imdp = factor/(const_mu*fsig)*np.sqrt((fsig+const_bgr)/t)
        tmp_mdp.append(imdp)
        imdp = None

    labels_vartime.append(f"SIG={fsig}Hz")
    mdp_fix_bgr_var_time.append(tmp_mdp)
    tmp_mdp = None

for l in mdp_fix_bgr_var_time:
    print(len(l))


plt.figure(figsize=(10,8))

markerlist = ["o","s","*","v","^"]

for mdp, lab, mark in zip(mdp_fix_bgr_var_time, labels_vartime, markerlist):
    #print(len(times))
    #print(len(mdp))
    plt.scatter(times, np.array(mdp)*100.0, marker=mark, label=lab)

plt.vlines(300,0,100,color="violet", label='5 min')
plt.vlines(1800,0,100,color="indigo", label='30 min')
plt.vlines(3600,0,100,color="crimson", label='60 min')
plt.title(f"MDPs vs Measurement Time (BGR={const_bgr},{char_mu}={const_mu})")
plt.xlabel("Time, [s]")
plt.ylabel("MDP, [%]")
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='upper right')
plt.minorticks_on()
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)

defpicname = "HUYA"
savename = "MDPs-vs-Time-"
if(picname is not None):
    savename+=picname
else:
    savename+=defpicname

plt.savefig(f"{savename}.png")
plt.close()


