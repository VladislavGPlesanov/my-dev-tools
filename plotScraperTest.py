import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit

def getRateErr(rate, cts, t, t_e):
    # set sigma_current to be 1 pA
    return rate*np.sqrt((np.sqrt(cts)/cts)**2 + (t_e/t)**2)


infile = sys.argv[1]
picname = sys.argv[2]

header, current, position, rate = [],[],[],[]
rate_err, pos_err =  [], []

f = open(infile)

cnt = 0
for line in f:
    words = line.split(',')
        
    if(cnt==0):
        header = words
    else:
        if("#" in line):
            continue

        I = float(words[0])
        time = float(words[1])
        xpos = float(words[2])
        CNT_list = words[3].split('/')
        sfactor = float(words[4])
        mean_cts = None
        if(len(CNT_list)>1):
            tmp = []
            for i in CNT_list:
                if(len(i)>0):
                    tmp.append(float(i)*sfactor)
            mean_cts = np.mean(tmp)
        else:
            mean_cts = float(CNT_list[0]) 

        # -------------------------
        irate = mean_cts/time
        if(cnt<14):
            print(f"{I} -> {time} -> {xpos} -> {CNT_list}")
            print(f"Rate={irate}")

        const_time_error = 0.01
        rate.append(mean_cts/time)

        rate_err.append(getRateErr(irate, mean_cts, time, const_time_error))
        position.append(xpos)
        current.append(I)
        pos_err.append(0.001) 
        I, time, xpos, sfactor, CNT_list = None, None, None, None, None
    cnt+=1


plt.figure(figsize=(8,8))
#plt.scatter(position, rate, marker='o', color='r')
plt.errorbar(position, rate, xerr = pos_err, yerr = rate_err, fmt='x', color='red', ecolor='black', capsize=2)
plt.plot(position,np.array(current)*10, c='g')
#plt.yscale("log")
plt.title("Beam Position vs Total Recorded Event Rate")
plt.xlabel("Beam x position, [mm]")
plt.ylabel("Scinti rate, [Hz]")
#plt.xlim([22,30])
#plt.ylim([1e2,1e4])
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
plt.savefig(f"beam-profile-{picname}.png")

plt.xlim([16,28])
plt.yscale('log')
plt.title("Beam Position vs Total Recorded Event Rate (Zoom on peak)")
plt.savefig(f"beam-profile-logscale-{picname}.png")


####### Plotting Scraper Data ###################################################################
sfile = "/home/vlad/IonTestBeams/scraperRun/scraper-movement.csv"

spos, srate = [], []
epos, erate = [], []

sf = open(sfile)

for sline in sf:

    if("#" in sline):
        continue

    words = sline.split(",")

    x = float(words[0])
    clist = words[1].split("/")
    fact = float(words[2])

    avg_cts = None
    tmp = []
    for c in clist:
        tmp.append(float(c)*fact)
    avg_cts = np.mean(tmp)
    time = 10.0
    irate = avg_cts/time
    spos.append(x)
    epos.append(0.001)
    srate.append(irate)
    erate.append(getRateErr(irate, avg_cts, time, 0.001))

    irate, x, clist, fact, avg_cts = None, None, None, None, None


plt.figure(figsize=(8,8))
#plt.scatter(spos,srate,marker='x', color='b')
plt.errorbar(spos, srate, xerr = epos, yerr = erate, fmt='x', color='blue', ecolor='black', capsize=3)
plt.yscale("log")
plt.title("Scraper Position vs Total Recorded Event Rate")
plt.xlabel("Scraper position, [mm]")
plt.ylabel("Recordedd total rate, [Hz]")
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
plt.savefig(f"scraper-rate-{picname}.png")

# ===================================================================
print("Plotting previous results for 200um hole")

lastRun = open('/home/vlad/IonTestBeams/scintiRun/200umHole.csv')
               # ~/IonTestBeams/scintiRun/200umHole.csv
header = None
beamCurrent = None
xzero = None

lastposition, lastrate = [], []
e_rate = []
const_e = []

cnt = 0

for line in lastRun:

    if('xpos' in line):
        continue

    words = line.split(',')

    if(beamCurrent is None):
        beamCurrent = float(words[0])
    
    xpos = float(words[1])
    itime = words[2].split('/')
    cts = float(words[3])
    factor = float(words[4])
    
    if(xzero is None):
        xzero = xpos
    
    rel_x = xpos - xzero
    
    mean_time = None
    sigma_time = 0.01
    if(len(itime)>1):
       tmp = []
       for i in itime:
            if(len(i)>0):
                tmp.append(float(i))
       mean_time = np.mean(tmp)
       sigma_time = np.sqrt(3*(0.01**2))
    else:
       mean_time = float(itime[0])
    
    irate = (cts*factor)/mean_time
    
    lastposition.append(xpos)            
    #position.append(rel_x)            
    #rate.append((cts*factor)/mean_time)
    const_e.append(sigma_time)
    e_rate.append(getRateErr(irate, cts, mean_time, sigma_time)) 
    lastrate.append(irate)

    cnt+=1

plt.figure(figsize=(8,8))
plt.errorbar(position, rate, xerr = pos_err, yerr = rate_err, fmt='x', color='red', ecolor='black', capsize=2, label='New run')
plt.errorbar(lastposition, lastrate, xerr = const_e, yerr = e_rate, fmt='x', color='blue', ecolor='black', capsize=2, label="Old run")
plt.xlabel('x position, [mm]')
plt.ylabel('Rate, [Hz]')
#plt.title('')
plt.yscale('log')
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
#plt.legend(loc='center left')
plt.legend(loc='upper left')
  
plt.savefig("RunComparison-200um-"+picname+".png")

# ============================================================================================

IbeamCalibfile = '/home/vlad/IonTestBeams/scraperRunXY/IbeamVsSX1position.csv'

NpbaseCurrent = (40*1e-9)/1.602e-19
print(f'base current of 40nA translates to {np.floor(NpbaseCurrent)} protons')

Ibeam, scx1_pos = [], []
Nprotons = []
ibfile = open(IbeamCalibfile,'r')

for line in ibfile:

    if('#' in line):
        continue

    words = line.split(',')

    Ibeam.append(float(words[1]))
    scx1_pos.append(float(words[0]))
    Np = (float(words[1])*1e-12)/1.602e-19 
#    print()
    Nprotons.append(Np)
    #Nprotons.append((float(words[1])*1e-12)/1.602e-19)


#nplot = 5
plt.figure(figsize=(10,8))
plt.scatter(scx1_pos, Ibeam, marker='*', color='r', s=80)
#plt.scatter(scx1_pos[1:nplot], Ibeam[1:nplot], marker='*', color='r')
plt.xlabel('SCX1 position, [mm]')
plt.ylabel(r'$I_{beam}$ (by SEM), [pA]')
plt.yscale('log')
plt.ylim([1e0,3e4])
plt.xlim([8,18])
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.savefig('BeamCurrentCalibCurve.png')

plt.figure(figsize=(8,6))
plt.scatter(scx1_pos[1:], Nprotons[1:], marker='d', color='green',s=40)
plt.xlabel('SCX1 position, [mm]')
plt.ylabel(r'$N_{protons}$, [cnt]')
plt.yscale('log')
plt.xlim([0,18])
plt.ylim([1e3,1e11])
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.savefig('EstimatedNprotons.png')

# ----------------------------------------------------------------------------------
file_Np_direct = '/home/vlad/IonTestBeams/scraperRunXY/Nprotons-direct.csv'
file_Np_wY = '/home/vlad/IonTestBeams/scraperRunXY/Nprotons-direct-withYsraper.csv'
file_Np_wY_wDia = '/home/vlad/IonTestBeams/scraperRunXY/Nprotons-direct-withYsraper-wDiaph200um.csv'
file_Np_wY_wDia_wPb = '/home/vlad/IonTestBeams/scraperRunXY/Nprotons-direct-withYsraper-wDiaph200um-afterPb.csv'


xpos_direct, Np_direct = [], []
err_xpos, err_Np = [], []

xpos_wY, Np_wY = [], []
err_xpos_wY, err_Np_wY = [], []

xpos_wY_wDia, Np_wY_wDia = [], []
err_xpos_wY_wDia, err_Np_wY_wDia = [], []

xpos_wY_wDia_wPb, Np_wY_wDia_wPb = [], []
err_xpos_wY_wDia_wPb, err_Np_wY_wDia_wPb = [], []

const_err_xpos = 0.001 #mm based on linear stage controll
const_t_err =0.001 #seconds based on func-gen waveforms

# --------
fnpd = open(file_Np_direct,'r')

t_meas = 10.0

for line in fnpd:

    if('#' in line):
        continue

    words = line.split(',')

    x = float(words[0])
    cnt_list = words[1].split('/')
    fact = float(words[2])

    tmp = []
    for i in cnt_list:
        if(len(i)>0):
            tmp.append(float(i)*fact)
    rate = np.mean(tmp)/t_meas
    xpos_direct.append(x)
    Np_direct.append(rate)
    #adding errors
    err_xpos.append(const_err_xpos)
    sigma_rate = getRateErr(rate, np.mean(tmp), t_meas, const_t_err)
    err_Np.append(sigma_rate)

    x, fact = None, None
    tmp = None

# ------------------------------

fnpd_wY = open(file_Np_wY,'r')
for line in fnpd_wY:

    if('#' in line):
        continue

    words = line.split(',')

    x = float(words[0])
    cnt_list = words[1].split('/')
    fact = float(words[2])

    tmp = []
    for i in cnt_list:
        if(len(i)>0):
            tmp.append(float(i)*fact)
    rate = np.mean(tmp)/t_meas
    xpos_wY.append(x)
    #Np_wY.append(np.mean(tmp)/t_meas)
    Np_wY.append(rate)
    #adding errors
    err_xpos_wY.append(const_err_xpos)
    sigma_rate = getRateErr(rate, np.mean(tmp), t_meas, const_t_err)
    err_Np_wY.append(sigma_rate)

    x, fact = None, None
    tmp = None

# ------------------------------

fnpd_wY_wDia = open(file_Np_wY_wDia,'r')

npr=0

ratioSurv, xSurv = [], []

print("Looking for survivors")
for line in fnpd_wY_wDia:

    if('#' in line):
        continue

    words = line.split(',')

    x = float(words[0])
    cnt_list = words[1].split('/')
    fact = float(words[2])

    tmp = []
    for i in cnt_list:
        if(len(i)>0):
            tmp.append(float(i)*fact)
    xpos_wY_wDia.append(x)
    rate = np.mean(tmp)/t_meas
    #Np_wY_wDia.append(np.mean(tmp)/t_meas)
    Np_wY_wDia.append(rate)

    #adding errors
    err_xpos_wY_wDia.append(const_err_xpos)
    sigma_rate = getRateErr(rate, np.mean(tmp), t_meas, const_t_err)
    err_Np_wY_wDia.append(sigma_rate)

    if(x in scx1_pos):
        idx = scx1_pos.index(x)
        Nsurv = (np.mean(tmp)/t_meas)/Nprotons[idx]
        print(f'at x={x} there are {Nsurv} protons')
        ratioSurv.append(Nsurv)
        xSurv.append(scx1_pos[idx])

    x, fact = None, None
    tmp = None

    npr+=1

# ------------------------------

fnpd_wY_wDia_wPb = open(file_Np_wY_wDia_wPb,'r')
for line in fnpd_wY_wDia_wPb:

    if('#' in line):
        continue

    words = line.split(',')

    x = float(words[0])
    cnt_list = words[1].split('/')
    fact = float(words[2])

    tmp = []
    for i in cnt_list:
        if(len(i)>0):
            tmp.append(float(i)*fact)
    xpos_wY_wDia_wPb.append(x)
    rate = np.mean(tmp)/t_meas
    #Np_wY_wDia_wPb.append(np.mean(tmp)/t_meas)
    Np_wY_wDia_wPb.append(rate)

    #adding errors
    err_xpos_wY_wDia_wPb.append(const_err_xpos)
    sigma_rate = getRateErr(rate, np.mean(tmp), t_meas, const_t_err)
    err_Np_wY_wDia_wPb.append(sigma_rate)

    x, fact = None, None
    tmp = None


#-------------- plotting ratio of survivors after diaphragm ------------------

plt.figure(figsize=(10,8))
plt.scatter(xSurv, ratioSurv, marker='+',color='black')
plt.xlabel('SCX1 position [mm]')
plt.ylabel('Ratio of protons after diaphragm')
plt.ylim([0,5e-5])
plt.savefig('Survivors.png')

# ---------------------------------
fig = plt.figure(figsize=(10,8))
#fig = plt.figure()
ax = plt.subplot(111)

ax.errorbar([],[],xerr=[],yerr=[], color='white', label=r'$t_{counting}$=10[s], $I_{beam}\approx$ 40pA')
ax.errorbar(xpos_direct, Np_direct, xerr = err_xpos, yerr=err_Np, color='red', ecolor ='black', fmt='*', capsize=2, label='Direct beam on scinti')
ax.errorbar(xpos_wY, Np_wY, xerr= err_xpos_wY, yerr=err_Np_wY, color='blue', ecolor='black', fmt='+', capsize=2, label='with SCY1=0.5[mm]')
ax.errorbar(xpos_wY_wDia[0:4], Np_wY_wDia[0:4], xerr=err_xpos_wY_wDia[0:4], yerr=err_Np_wY_wDia[0:4], color='green',ecolor='black', fmt='v', capsize=2, label='SCY1=0.5[mm]')
ax.errorbar([],[],xerr=[],yerr=[],color='white', label='+ 200um diaphragm')
ax.errorbar(xpos_wY_wDia_wPb[0:4], Np_wY_wDia_wPb[0:4], xerr = err_xpos_wY_wDia_wPb[0:4], yerr=err_Np_wY_wDia_wPb[0:4], color='orange', ecolor ='black', fmt='^', capsize=2, label='SCY1=0.5[mm]')
ax.errorbar([],[],xerr=[],yerr=[],color='white', label='+ 200um diaphragm + (5.5 x 5) mm hole + 15 cm air')

ax.set_title('Recorded total counting rates vs SCX1 scraper [SHORT RANGE] positions')
ax.set_xlabel('SCX1 position, [mm]')
ax.set_ylabel('Total rate, [Hz]')
ax.set_yscale('log')
ax.set_ylim([1e-2,3e6])
ax.legend(loc='center left')
ax.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
ax.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
plt.savefig(f'ScintiIradiation-{picname}.png')

# full scale plot for last 2 measurements
#
fig = plt.figure(figsize=(10,8))
ax1 = plt.subplot(111)
#plt.figure(figsize=(10,8))
plt.errorbar(xpos_wY_wDia, Np_wY_wDia, xerr=err_xpos_wY_wDia, yerr=err_Np_wY_wDia, color='green',ecolor='black', fmt='v', capsize=2, label='SCY1=0.5[mm]')
plt.errorbar([],[],xerr=[],yerr=[],color='white', label='+ 200um diaphragm')
plt.errorbar(xpos_wY_wDia_wPb, Np_wY_wDia_wPb, xerr = err_xpos_wY_wDia_wPb, yerr=err_Np_wY_wDia_wPb, color='orange', ecolor ='black', fmt='^', capsize=2, label='SCY1=0.5[mm]')
plt.errorbar([],[],xerr=[],yerr=[],color='white', label='+ 200um diaphragm + (5.5 x 5) mm hole + 15 cm air')
plt.scatter(scx1_pos[1:], Nprotons[1:], color='red', s=60, label=r'$N_{protons}$ based on SEM')
plt.scatter([],[], color='w', label='with SCY1 open')

plt.text(10,1e11, r"Estimated as $N_{protons} = \frac{I_{beam}}{q\cdot e}$", fontsize=13,color='red')

plt.title('Recorded total counting rates vs. SCX1 scraper [LONG RANGE] positions')
plt.xlabel('SCX1 position, [mm]')
plt.ylabel('Total rate, [Hz]')
plt.yscale('log')
plt.ylim([1e-2,1e12])
#plt.legend(loc='upper left')
plt.legend(loc='center left')
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.savefig(f'ScintiIradiation-wDiaphragm-{picname}.png')

