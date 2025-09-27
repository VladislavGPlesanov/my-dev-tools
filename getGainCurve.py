import sys
import glob
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'


def plotAsciiHist(nuarray, name):

    print(f" ======== Plotting ASCII hist for {name} ======== ")
    for i in nuarray:
        picstr = ""
        for j in range(int(i/100.0)):
            picstr+="|"
        print(f"{picstr}")
    print(" =================================================")



def estimateHalfPeak(counts, nbins, peakbin):

    halfPeakBin = None

    for i in range(nbins-peakbin):
        diff = counts[peakbin] - counts[peakbin+i]
        thr = counts[peakbin]/2.0
        if(diff <= thr):
            halfPeakBin = peakbin+i-1

    return halfPeakBin

def getBinDistance(nbins, maxbin, minbin):

    return (maxbin-minbin)/nbins

def gauss(x, A, mu, sigma):

    return A*np.exp(-((x-mu)**2)/(2*sigma**2))

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)

def getMaxBin(numbers):

    maxcounts = 0
    maxbin = 0
    cnt = 0
    for n in numbers:
        if(n>maxcounts):
            maxcounts = n
            maxbin = cnt
            cnt+=1
        else:
            cnt+=1

    return maxbin

def simpleHist(nuarray, nbins, minbin, maxbin, labels, picname):

    plt.figure()

    fit_pars = None

    counts, bin_edges = np.histogram(nuarray, bins=nbins, range=(minbin,maxbin))

    plotAsciiHist(counts, "HUYA")

    print(f"Retrieved array of counts:\n{counts}")

    #plt.hist(nuarray, nbins, range=(minbin,maxbin), histtype='stepfilled', facecolor='b')
    plt.hist(bin_edges[:-1], weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', facecolor='b')
    plt.figsize=(8,8)
    maxbin_cnt = np.max(counts)
    minbin_cnt = np.min(counts)
    #minval, maxval = max, counts[len(counts)-1]
    val_ibin = (maxbin-minbin)/nbins
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    print(f"Range: {minbin} -> {maxbin}, with {nbins} bins gives {val_ibin} per bin")

    ########################################################
    model = None

    peakbin = getMaxBin(counts)

    print(f"Found maximum bin at {peakbin} = {counts[peakbin]}")
    model = Model(gauss)
    
    pars = model.make_params(A=maxbin_cnt, mu=peakbin*val_ibin, sigma=np.std(counts))
    #pars['A'].min = maxbin_cnt*0.8
    #pars['A'].min = maxbin_cnt*1.2

    #pars['mu'].min = peakbin*0.6*val_ibin
    #pars['mu'].min = peakbin*1.4*val_ibin
 
    #pars['sigma'].min = np.std(counts)*0.6
    #pars['sigma'].min = np.std(counts)*1.4

    print(f"Gauss Fit: Setting: A={maxbin_cnt}, mu={peakbin*val_ibin}, sigma={np.std(counts)}")

    #print(f"\n\ncounts={counts} \n\n")
    print(maxbin_cnt)
    print(minbin_cnt)
    print(f"max_amplitude={maxbin_cnt - minbin_cnt}")

    #====================================================

    f_failed = False

    lowcut, uppercut = None, None
    peak_pos_half = estimateHalfPeak(counts, nbins, peakbin)
    binwidth = getBinDistance(nbins,maxbin,minbin)
    est_sigma = (peak_pos_half - peakbin)*binwidth

    if("TOT" in picname): 
        lowcut = peakbin*binwidth-peak_pos_half*20
        uppercut = peakbin*binwidth+peak_pos_half*20
    if("HITS" in picname):
        lowcut = peakbin*binwidth-peak_pos_half
        uppercut = peakbin*binwidth+peak_pos_half

    print("########## FITTING GASUSS FUNCTION #############")
    result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
    counts_trunc, bin_centers_trunc = None, None
    print(result.fit_report()) 
    if(result.params['mu']<=0 or result.params['sigma']<=0):
        f_failed = True
        print("\tFIT FAILED: restricting fit parameters and re-fitting")

        pars['A'].min = maxbin_cnt*0.8
        pars['A'].max = maxbin_cnt*1.2

        pars['mu'].min = peakbin*0.8*val_ibin
        pars['mu'].max = peakbin*1.2*val_ibin
 
        #pars['sigma'].min = np.std(counts)*0.8
        #pars['sigma'].max = np.std(counts)*1.2

        #peak_pos_half = estimateHalfPeak(counts, nbins, peakbin)
        #binwidth = getBinDistance(nbins,maxbin,minbin)
        #est_sigma = (peak_pos_half - peakbin)*binwidth

        print(f"peak at {peakbin}, half-peak position at bin {peak_pos_half}")
        print(f"Bin width = {binwidth}")
        print(f"peak width should be {2*est_sigma} ")

        #pars['sigma'].min = est_sigma*0.8
        #pars['sigma'].max = est_sigma*1.2

        sigma_to_peak = np.abs(result.params['sigma']) / result.params['A']

        if(result.params['sigma']<0 and sigma_to_peak < 1):
 
            pars['sigma'].min = np.abs(result.params['sigma'])*0.8
            pars['sigma'].max = np.abs(result.params['sigma'])*1.2

        else:

            pars['sigma'].min = est_sigma*0.8
            pars['sigma'].max = est_sigma*1.2


        #lowcut = peakbin-peak_pos_half*2
        #uppercut = peakbin+peak_pos_half*2

        #counts_trunc = counts[lowcut:uppercut]
        #bin_centers_trunc = bin_centers[lowcut:uppercut]

        
        result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
        #result = model.fit(counts_trunc, pars, x=bin_centers_trunc)
        #result = model.fit(counts[peakbin-10:peakbin+10], pars, x=bin_centers[peakbin-10:peakbin+10])

        print(result.fit_report()) 
    ########################################################            

    fitlab = ""
    miny,maxy = 0, 0

    #print("########## FITTING GAUS FUNCTION #############")
    A = result.params["A"].value
    err_A = result.params["A"].stderr
    mu = result.params["mu"].value
    err_mu = result.params["mu"].stderr
    sigma = result.params["sigma"].value
    err_sigma = result.params["sigma"].stderr

    fit_pars = [A, err_A,  mu, err_mu, sigma, err_sigma]

    ax = plt.gca()
    miny,maxy = ax.get_ylim()
    #plt.yscale('log')
    #if(counts_trunc is None):
    plt.plot(bin_centers[:-1], result.best_fit, '--r')
    #else:
    #    plt.plot(bin_centers_trunc, result.best_fit, '--r')

    plt.text(minbin*1.1, maxy*0.9, f"A={round(A,2)}, {G_mu}={round(mu,2)}, {G_sigma}={round(sigma,2)}")

    #plt.legend(loc='upper right')
    plt.vlines(lowcut, 0, maxbin_cnt+10, colors='magenta', linestyles='dashed')
    plt.vlines(uppercut, 0, maxbin_cnt+10, colors='magenta', linestyles='dashed')
    if(f_failed):
        plt.text(minbin*1.1, maxy*0.7, "FIT FAILED INITIALLY")

    f_failed = False
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    #plt.yscale('log')
    plt.grid()
    plt.savefig(f"GainRuns-{picname}.png")

    return fit_pars

########################################################################

location = sys.argv[1]
picname = sys.argv[2]

dir_files = glob.glob(location+"*.h5")

for i in dir_files:
    print(i)
inputlist = sorted(dir_files)

peak_amp = []
peak_amp_err = []
peak_mu = []
peak_mu_err = []
peak_sigma = []
peak_sigma_err = []

hit_peak_amp = []
hit_peak_amp_err = []
hit_peak_mu = []
hit_peak_mu_err = []
hit_peak_sigma = []
hit_peak_sigma_err = []

Vgrid_list = []

totHistList = []

hitHistList = []

fcnt = 0

for file in inputlist:

    TOT, HITS = [], []

    with tb.open_file(file) as f:

        print(f"Reading {f}")
        groups = f.walk_groups('/')
        grouplist = []
        for gr in groups:
            print(f'found {gr}')
            grouplist.append(gr)
        main_group = str(grouplist[len(grouplist)-1])
        print(f"last entry in walk_groups = \n{main_group}")
        grouplist = None 

        basewords = main_group.split('(')
        print(basewords)

        base_group_name = basewords[0][:-1]+'/'
        #                              ^ removes space at the end of 'run_xxx/chip0 '
        print(f'base group name is : <{base_group_name}>')
        bgn_split = base_group_name.split('/')
        print(bgn_split)
        run_name = bgn_split[2]
        print(f"<{run_name}>")
        run_num = int(run_name[4:])
        print(f'run number is {run_num}')
        basewords = None
        #Gettin voltage
        fname = str(inputlist[fcnt])
        words = fname.split("-")
        #print(words)
        Vgrid = int(words[len(words)-1][:-4])# removin V.h5 from voltage number
        print(f"Found Vgrid={Vgrid} V")

        ###############################################
        sumTOT = f.get_node(base_group_name+"sumTot")
        print(f"found sumTOT {type(sumTOT)}")
        print(f"Contains {len(sumTOT)} events/hits")
        for totsum in sumTOT:
            TOT.append(totsum)
        sumTOT = None

        hits = f.get_node(base_group_name+"hits")
        print(f"found nhits {type(hits)}")
        print(f"Contains {len(hits)} events/hits")
        for h in hits:    
            HITS.append(h)
        hits = None


    #print(f"--------------------DONE----------------------")
   
    print("=============== FITTING TOT ================================")
    peak_info = simpleHist(TOT, 100,0, 6000, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)

    print("=============== FITTING HITS ================================")
    hit_info = simpleHist(HITS, 50,0, 150, [f"Hits per event (Vgrid={Vgrid}[V])", r"$N_{HITS}$", "N"], f"HITS-{Vgrid}-"+picname)

    totHistList.append([TOT,Vgrid])
    hitHistList.append([HITS,Vgrid])

    Vgrid_list.append(Vgrid)

    peak_amp.append(peak_info[0])
    peak_amp_err.append(peak_info[1])
    peak_mu.append(peak_info[2])
    peak_mu_err.append(peak_info[3])
    peak_sigma.append(peak_info[4])
    peak_sigma_err.append(peak_info[5])
    
    hit_peak_amp.append(hit_info[0])
    hit_peak_amp_err.append(hit_info[1])
    hit_peak_mu.append(hit_info[2])
    hit_peak_mu_err.append(hit_info[3])
    hit_peak_sigma.append(hit_info[4])
    hit_peak_sigma_err.append(hit_info[5])
 
    hit_info = None
    peak_info = None
    TOT = None
    HITS = None

    fcnt+=1


cminbin = 0
cmaxbin = 6000
cnbins = 100

plt.figure(figsize=(10,8))
plt.hist([],weights=[], bins=cnbins, range=(cminbin, cmaxbin), color='white', label=r"$\mathrm{V}_{\mathrm{Grid}}:$")
for data in totHistList:
    counts, bin_edges = np.histogram(data[0], bins=cnbins, range=(cminbin,cmaxbin))
    weights = counts/sum(counts)
    #plt.hist(bin_edges[:-1], weights=counts, bins=101, range=(0,10000), align='left', histtype='stepfilled', alpha=0.2, label=f"{data[1]} [V]")
    plt.hist(bin_edges[:-1], weights=weights, bins=cnbins, range=(cminbin, cmaxbin), align='left', histtype='stepfilled', alpha=0.2, label=f"{data[1]} [V]")
    
plt.title("Fe55 Energy Spectrum")
plt.xlabel("TOT cycles (charge)")
plt.ylabel(r"$N_{events}$")
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(f"TOT-Fe55-gainScan-Combined-{picname}.png")
plt.close()
# ============= plotting hit histograms ====================

pnbins = 50
pmaxbin = 250
pminbin = 0

plt.figure(figsize=(10,8))
plt.hist([],weights=[], bins=cnbins, range=(pminbin, pmaxbin), color='white', label=r"$\mathrm{V}_{\mathrm{Grid}}:$")
for data in hitHistList:
    counts, bin_edges = np.histogram(data[0], bins=pnbins, range=(pminbin,pmaxbin))
    weights = counts/sum(counts)
    plt.hist(bin_edges[:-1], weights=weights, bins=pnbins, range=(pminbin, pmaxbin), align='left', histtype='stepfilled', alpha=0.2, label=f"{data[1]} [V]")
    
plt.title("Fe55 - Hits per event")
plt.xlabel(r"$N_{Hits}$")
plt.ylabel(r"$N_{events}$")
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(f"HITS-Fe55-gainScan-Combined-{picname}.png")

#============================================================

print(len(Vgrid_list))
print(len(peak_mu))
print(len(peak_mu_err))

xmin = int(Vgrid_list[0])-10
xmax = int(Vgrid_list[len(Vgrid_list)-1])+10

plt.figure(figsize=(8,8))

fig, ax = plt.subplots()

ax.errorbar(Vgrid_list, peak_mu, peak_mu_err, fmt='o', mfc='green', linewidth=2, capsize=6)

ax.set_title(r"GridPix3 - $^{55}$Fe - TOT peak position vs. $\mathrm{V}_{\mathrm{Grid}}$")
ax.set_xlabel("Grid voltage, [V]")
ax.set_ylabel(r"Fit $\mu$, [$\Sigma$(TOT) per Event]")
ax.set_xlim([xmin, xmax])
#ax.set_ylim([13000, 28000])
ax.grid(True)
plt.savefig("GainCurve.png",dpi=400)
plt.close()
###################################################
# combined hit histogram means of the fits
plt.figure(figsize=(8,8))

fig, ax = plt.subplots()

ax.errorbar(Vgrid_list, hit_peak_mu, hit_peak_mu_err, fmt='o', mfc='red', linewidth=2, capsize=6)
ax.set_title(r"GridPix3 - $^55$Fe - Hit peak position vs. $\mathrm{V}_{\mathrm{Grid}}$")
ax.set_xlabel("Grid voltage, [V]")
ax.set_ylabel(r"Fit ($\mu$), [$\mathrm{N}_{\mathrm{Hits}}$]")
ax.set_xlim([xmin, xmax])
#ax.set_ylim([13000, 28000])
ax.grid(True)
plt.savefig("HitsCurve.png",dpi=400)
plt.close()

###################################################

plt.figure(figsize=(8,8))

fig, ax = plt.subplots()
ax.errorbar(peak_mu, hit_peak_mu, hit_peak_mu_err,peak_mu_err, fmt='x', linewidth=2, capsize=6)
ax.set_title("Hits vs TOT")
ax.set_ylabel(r"Fit ($\mu$), [$\mathrm{N}_{\mathrm{Hits}}$]")
ax.set_xlabel(r"Fit $\mu$, [$\Sigma$(TOT) per Event]")
ax.set_xlim([250, 3000])
ax.set_ylim([15,100])
ax.grid(True)
plt.savefig("MuVsHits-gainMeas.png")
plt.close()

print("____________OLEG________________")
for m,h,em,eh in zip(peak_mu, hit_peak_mu, peak_mu_err, hit_peak_mu_err):
    print(f" {m} +- {em} -> {h} +- {eh}")


###################################################
E_res = []
E_res_err = []
cnt = 0
for mu in peak_mu:
    res = peak_sigma[cnt]/mu
    res_err = np.sqrt((peak_mu_err[cnt]/mu)**2 + (peak_sigma_err[cnt]/peak_sigma[cnt])**2)
    #print(f"calculating resolution => item={cnt}, {mu:.4f} +- {peak_mu_err[cnt]:.4f}, {peak_sigma[cnt]:.4f} +- {peak_sigma_err[cnt]:.4f}")
    print(f"{cnt}: [PEAK-MEAN]={mu:.4f} +- {peak_mu_err[cnt]:.4f}, [PEAK-SIGMA]={peak_sigma[cnt]:.4f} +- {peak_sigma_err[cnt]:.4f}")
    E_res.append(res*100)
    E_res_err.append(res_err*100)
    cnt+=1

#print(len(E_res))
#print(E_res)
#print(len(E_res_err))
#print(E_res_err)
#print(len(Vgrid_list))
#print(Vgrid_list)

print("=============================================")
for v, res, eres in zip(Vgrid_list,E_res,E_res_err):
    print(f"Vgrid={v} | {res:.2f}% +- {eres:.2f}%")
print("=============================================")

plt.figure(figsize=(8,8))

fig, ax = plt.subplots()

ax.errorbar(Vgrid_list, E_res, E_res_err, fmt='o', linewidth=2, capsize=6)
ax.set_xlabel("Grid voltage, [V]")
ax.set_ylabel(f"{G_delta}E/E, [%]")
ax.set_title("Energy resolution vs Grid voltage")

#ax.set_ylim([0,20])
ax.set_xlim([xmin, xmax])
plt.grid()
plt.savefig("Energy_resolutions.png")

