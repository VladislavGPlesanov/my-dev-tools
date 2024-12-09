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

    #plt.hist(nuarray, nbins, range=(minbin,maxbin), histtype='stepfilled', facecolor='b')
    plt.hist(bin_edges[:-1], weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', facecolor='b')
    plt.figsize=(8,8)
    maxbin_cnt = np.max(counts)
    minbin_cnt = np.min(counts)
    #minval, maxval = max, counts[len(counts)-1]
    val_ibin = (maxbin-minbin)/nbins
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    print(f"Range= {minbin} -> {maxbin}, with {nbins} bins gives {val_ibin} per bin")

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

    print(f"\n\ncounts={counts} \n\n")
    print(maxbin_cnt)
    print(minbin_cnt)
    print(f"max_amplitude={maxbin_cnt - minbin_cnt}")

    print("########## FITTING GASUSS FUNCTION #############")
    result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
    print(result.fit_report()) 
    if(result.params['mu']<=0):

        #pars['A'].min = maxbin_cnt*0.8
        #pars['A'].max = maxbin_cnt*1.2

        pars['mu'].min = peakbin*0.8*val_ibin
        pars['mu'].max = peakbin*1.2*val_ibin
 
        #pars['sigma'].min = np.std(counts)*0.8
        #pars['sigma'].max = np.std(counts)*1.2

        print("FIT FAILED: restricting fit parameters and re-fitting")
        result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
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
    plt.plot(bin_centers[:-1], result.best_fit, '--r')
    plt.text(minbin*1.1, maxy*0.9, f"A={round(A,2)}, {G_mu}={round(mu,2)}, {G_sigma}={round(sigma,2)}")

    #plt.legend(loc='upper right')

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

Vgrid_list = []

totHistList = []

fcnt = 0

for file in inputlist:

    TOT = []

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

    print(f"--------------------DONE----------------------")
   
    print("===============================================")
    peak_info = simpleHist(TOT, 101,0, 40000, [f"TOT per event (Vgrid[fcnt])", "TOT cycles", "N"], f"{Vgrid}-"+picname)

    totHistList.append([TOT,Vgrid])
    Vgrid_list.append(Vgrid)

    peak_amp.append(peak_info[0])
    peak_amp_err.append(peak_info[1])
    peak_mu.append(peak_info[2])
    peak_mu_err.append(peak_info[3])
    peak_sigma.append(peak_info[4])
    peak_sigma_err.append(peak_info[5])
    
    peak_info = None
    TOT = None

    fcnt+=1

plt.figure(figsize=(8,8))
for data in totHistList:
    counts, bin_edges = np.histogram(data[0], bins=101, range=(0,35000))
    plt.hist(bin_edges[:-1], weights=counts, bins=101, range=(0,35000), align='left', histtype='stepfilled', alpha=0.2, label=f"Vgrid = {data[1]} [V]")
    
plt.title("Energy spectrum, all runs")
plt.xlabel("TOT cycles")
plt.ylabel("N")
plt.legend(loc='upper left')
plt.savefig(f"TOT-Fe55-gainScan-Combined-{picname}.png")

print(len(Vgrid_list))
print(len(peak_mu))
print(len(peak_mu_err))

plt.figure(figsize=(8,8))

fig, ax = plt.subplots()

ax.errorbar(Vgrid_list, peak_mu, peak_mu_err, fmt='o', linewidth=2, capsize=6)
ax.set_xlabel("Grid voltage [V]")
ax.set_ylabel("Peak mu [TOT sum per event]")

plt.savefig("GainCurve.png",dpi=400)
###################################################
E_res = []
E_res_err = []
cnt = 0
for mu in peak_mu:
    res = peak_sigma[cnt]/mu
    res_err = np.sqrt((peak_mu_err[cnt]/mu)**2 + (peak_sigma_err[cnt]/peak_sigma[cnt])**2)
    E_res.append(res*100)
    E_res_err.append(res_err*100)
    cnt+=1

print(len(E_res))
print(E_res)
print(len(E_res_err))
print(E_res_err)
print(len(Vgrid_list))
print(Vgrid_list)

plt.figure(figsize=(8,8))

fig, ax = plt.subplots()

ax.errorbar(Vgrid_list, E_res, E_res_err, fmt='o', linewidth=2, capsize=6)
ax.set_xlabel("Grid voltage [V]")
ax.set_ylabel(f"{G_delta}E/E [%]")
ax.set_title("Energy resolution vs Grid voltage")

ax.set_ylim([4,16])
ax.set_xlim([405,435])
plt.grid()
plt.savefig("Energy_resolutions.png")

