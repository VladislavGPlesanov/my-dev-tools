import numpy as np
import sys
import matplotlib.pyplot as plt

logfile = sys.argv[1]
picname = sys.argv[2]

f=open(logfile,'r')

weights = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8, 4.0]

legacyWeightMu= [0.006784951172223514, 0.023233193967273257, 0.04695941469078984, 0.07338712621207003, 0.098360011622454, 0.12314777405793377, 0.13968129365868032, 0.15773956382996515, 0.17431349000895993, 0.18774042722350842, 0.197698987294928, 0.20533714765961372, 0.20761480782986239, 0.20804087473248956, 0.206004348807681, 0.204946838196388, 0.20319121701384757, 0.20156688139237414, 0.19737404083277252, 0.19229207420383612]
muErr_leg=[ 0.060328217519696956, 0.01115394065892499, 0.006615237763553174, 0.005968077558587299, 0.00566517058076359, 0.006394318267010783, 0.007086439749119372, 0.006985120696469713, 0.006881611483056141, 0.006966653790174502, 0.006573199188674454, 0.006931472646413423, 0.006820980515620939, 0.006319025211948317, 0.006013492347582723, 0.006640901923554398, 0.0063733475716472015, 0.0066510337553274245, 0.006743712740850833, 0.006580529657392775 ]

muOptimized=[0.016359739002260176, 0.022653795124854208, 0.044488844613151966, 0.0772143001444764, 0.10343261401984187, 0.12843630656942318, 0.1494536775628951, 0.16495656285874902, 0.17480050559549828, 0.18524436084583448, 0.19163104592846336, 0.19278495260603748, 0.19300368830363052, 0.19180863674816062, 0.1918195440830514, 0.18985470003732352, 0.18566808738631418, 0.18192652451885052, 0.17566989718826598, 0.17252528300083708]
muOpt_err=[0.05801632254894696, 0.010617952302588583, 0.0067898523411826055, 0.006608428450676273, 0.007009770513065712, 0.006735994014370134, 0.006407098346676904, 0.006633452314217917, 0.0069538112278850804, 0.006539143874714734, 0.006353498233782299, 0.006469387518735112, 0.006248423833405628, 0.00644585240747988, 0.006283991090867773, 0.006778973453771231, 0.00690075167929272, 0.006537145979135311, 0.0062732941730196555, 0.006575212455598443 ]

mu, muErr = [],[]

absX, absY, absDX, absDY = [], [],[],[]
chired = []

phi, phierr = [], []

mu_cnt = 0
muErr_cnt = 0
nrun = 0

fModCut = False



for line in f:

    if("Absorption peak" in line):
        dataXY = line.split(":")[1].split(",")
        #print(dataXY)
        absDX.append(float(dataXY[2]))
        absDY.append(float(dataXY[3]))

    if("mu=" not in line and "muErr" not in line and "chired" not in line and "phi" not in line and "phierr" not in line):
        if("ANALYZING:" in line):
            nrun+=1
        elif("Mod_Cut" in line):
            fModCut = True
        else:
            continue
    else:
        print("OLEG")
        if(fModCut):
            print(line)
            if("mu=" in line):
                mu.append(float(line.split("=")[1]))
                mu_cnt+=1
            elif("chired=" in line):
                print(line)
                chired.append(float(line.split("=")[1]))
            elif("phi=" in line):
                phi.append(float(line.split("=")[1]))
            elif("phierr=" in line):
                phierr.append(float(line.split("=")[1]))
                fModCut=False #reset flag after last entry in branch "Mod_Cut"
            elif("muErr=" in line):
                muErr.append(float(line.split("=")[1]))
                muErr_cnt+=1
#            elif("Uerr=" in line):
#                fModCut=False #reset flag after last entry in branch "Mod_Cut"
            else:
                continue
        else:
            continue

print(mu)
print(mu_cnt)
print(muErr)

print(chired)
#print(len(mu))
#print(weights[0:mu_cnt])

print(len(legacyWeightMu))
print(len(mu))
print(len(weights))

print(len(chired))

print(len(phi))
print(len(phierr))

fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
#ax.errorbar(weights[0:mu_cnt], mu, yerr=muErr, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\mu,  R_{in}=1.4, R_{out}=2.5$")
ax.errorbar(weights[0:mu_cnt], mu, yerr=muErr, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\mu,  R_{in}=1.4, R_{out}=3.5$")
ax.errorbar(weights, legacyWeightMu, yerr=muErr_leg, color='firebrick', ecolor='black', fmt='*',capsize=4, label=r"$\mu,  R_{in}=1.6, R_{out}=3.5$")
ax.errorbar(weights, muOptimized, yerr=muOpt_err, color='orange', ecolor='black', fmt='*',capsize=4, label=r"$\mu,  R_{in}=1.4, R_{out}=2.5$")
ax.set_title("Estimated Modualtion Factors Based on Weighting")
ax.set_xlabel("Weight")
ax.set_ylabel(r"$\mu$,[%]")
ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
ax.legend(loc='lower right')
plt.savefig(f"ModFacs-vs-Weights-{picname}.png",dpi=800)
plt.close()

# plotting some chi reduced 
plt.figure()
plt.scatter(weights[0:mu_cnt],chired, marker="+", c='red', ls='--')
plt.grid(True)
plt.xlim([0.0,4.2])
plt.yscale('log')
plt.xlabel("Weight")
plt.ylabel(r"Fit $\chi_{reduced}^{2}$")
plt.savefig(f"ModFacs-CHIRED-{picname}.png")
plt.close()

# plotting absorption point stds 
plt.figure()
plt.scatter(weights[0:mu_cnt],absDX, marker="+", c='forestgreen', ls='--')
plt.scatter(weights[0:mu_cnt],absDY, marker="+", c='royalblue', ls='--')
plt.grid(True)
plt.xlim([0.0,4.2])
plt.xlabel("Weight")
plt.ylabel(r"Absorption point reconstruction dev.")
plt.savefig(f"ModFacs-AbsorptionPoint-STD-{picname}.png")
plt.close()

# plotting reconstructed angles
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)
THE_ANGLE = 45*np.pi/180

ax.errorbar(weights[0:mu_cnt], phi, yerr=phierr, color='forestgreen', ecolor='black', fmt='*',capsize=4, label=r"$\phi$ reconstructed ")
#ax.hlines(THE_ANGLE, 0,4, linestyles='--', colors='firebrick')
ax.hlines(0.78, 0,4, linestyles='--', colors='firebrick', label=r"$\phi$ simulated")
ax.grid(True)
ax.set_xlim([0.0,4.2])
ax.set_ylim([0,1.52])
ax.set_xlabel("Weight")
ax.set_ylabel(r"$\phi$, [radian]")
ax.legend(loc='lower right')
plt.savefig(f"ModFacs-recoAngle-{picname}.png")
plt.close()



