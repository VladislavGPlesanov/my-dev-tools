import sys
import glob
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.special import gamma
from scipy.optimize import curve_fit

import ROOT as r

###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

def polya_MG(x, K, G, T):
    Tp1 = T + 1
    return (K / G) * (np.power(Tp1, Tp1))/(gamma(Tp1)) * np.power((x/G), T) * np.exp(-(Tp1*x)/(G))

def chargeperpixel(charge, picname, electrons):
    # Definition of the plot size
    fig_width, fig_height = 7, 6  # in inches

    # Relative font size
    font_size = fig_height * 2  # Beispiel: 2 mal die Breite der Figur

    # Set font size
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
    })

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.cla()
    maximum = charge.max()
    unique_data = np.sort(np.unique(charge))
    bin_edges = np.concatenate((unique_data - 0.5, [unique_data[-1] + 0.5]))

    hist, bins = np.histogram(charge, bins=bin_edges, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fit_range_start = np.nanmin(charge)
    charge_cap = np.ceil(np.median(charge))
    #fit_range = (bin_centers >= fit_range_start)
    fit_range = (bin_centers <= charge_cap*4)
    fit_range_end = len(bin_centers[fit_range])-1

    initAmp, initMean, initStd = 1000, 2000, 1

    #params, params_covariance = curve_fit(polya_MG, bin_centers[fit_range], hist[fit_range], p0=[10000, 2000, 1])
    params, params_covariance = curve_fit(polya_MG, bin_centers[fit_range], hist[fit_range], p0=[initAmp, initMean, initStd])
    params_errors = np.sqrt(np.diag(params_covariance))
    #x_values = np.linspace(fit_range_start, max(bin_edges), 1000)
    x_values = np.linspace(fit_range_start, fit_range_end, 1000)
    polya_fit = polya_MG(x_values, *params)

    residuals = hist[fit_range] - polya_MG(bin_centers[fit_range], *params)
    reduced_chi_squared = np.sum((residuals ** 2) / polya_MG(bin_centers[fit_range], *params)) / (len(hist[fit_range]) - len(params))

    # -----------------------------------------------------

    r.gROOT.SetBatch(True) # so it does not try to show anything....

    rbins = len(bin_edges) - 1 
    roothist = r.TH1D("hcharge", "Charge per Pixel", rbins, bin_edges)

    for q in charge:
        roothist.Fill(q)

    polya_expr = ("([0]/[1])"
    "* ( pow( ( ([1]*[1]) / ([2]*[2]) ), ( ([1]*[1]) / ([2]*[2]) ) ) / TMath::Gamma( ( [1]*[1] ) / ( [2]*[2] ) ) )"
    "* pow( x/[1], ( ( [1]*[1] ) / ( [2]*[2] ) - 1 ) )"
    "* exp( - ( ( [1]*[1] ) / ( [2]*[2] ) ) * ( x / [1] ) )" 
    )

    polya_expr = (
    "([0] / [1]) * "
    "TMath::Power([1]/[2], [1]/[2]) / TMath::Gamma([1]/[2]) * "
    "TMath::Power(x/[1], [1]/[2] - 1) * "
    "TMath::Exp(-( [1]/[2] ) * (x/[1]))"
    )

    #rootpolya = r.TF1("polya",
    #          "([0] / [1]) *(((([1]*[1])/([2]*[2]))^(([1]*[1])/([2]*[2]))) /(TMath::Gamma((([1]*[1])/([2]*[2]))))) * ((x /[1])^((([1]*[1])/([2]*[2]))-1)) * exp(-(([1]*[1])/([2]*[2])) *(x / [1])))",
    #          bin_edges[0], bin_edges[-1])


    xmin = np.nanmin(charge)
    xmax = np.nanmax(charge)
    #rootpolya = r.TF1("polya", polya_expr, bin_edges[0], bin_edges[-1])
    rootpolya = r.TF1("polya", polya_expr, xmin, xmax)

    #amp_scaling = initAmp
    #amp_gain = initMean
    #amp_width = initStd
    amp_scaling = charge.size * 0.1
    amp_gain = np.median(charge)
    amp_width = 0.5

    rootpolya.SetParameter(0,amp_scaling)
    rootpolya.SetParameter(1,amp_gain)
    rootpolya.SetParameter(2,amp_width)

    r.gStyle.SetOptFit(1)
    fit_result = roothist.Fit(rootpolya, "S")

    scaling_fit = rootpolya.GetParameter(0)
    gain_fit = rootpolya.GetParameter(1)
    width_fit = rootpolya.GetParameter(2)

    scaling_err = rootpolya.GetParError(0)
    gain_err = rootpolya.GetParError(1)
    width_err = rootpolya.GetParError(2)

    rootpolya.SetParLimits(0, 0.0, 1e9)
    rootpolya.SetParLimits(1, 1e-6, 1e9)
    rootpolya.SetParLimits(2, 1e-6, 1e9)

    print("===========================================================================")
    print("ROOT fit:")
    print(f"Amplification scale: {scaling_fit:.4f}"+r" $\pm$ "+f"{scaling_err:.4f}")
    print(f"Amplification gain: {gain_fit:.4f}"+r" $\pm$ "+f"{gain_err:.4f}")
    print(f"Amplification width: {width_fit:.4f}"+r" $\pm$ "+f"{width_err:.4f}")
    print("===========================================================================")

    # -----------------------------------------------------

    info_K = f"K = {params[0]:.3f}"+" $\pm$ "+f"{params_errors[0]:.3f}\n"
    info_G = f"G = {params[1]:.3f}"+" $\pm$ "+f"{params_errors[1]:.3f}\n"
    info_theta = r"$\theta$ = "+f"{params[2]:.3f} "+r"$\pm$"+f" {params_errors[2]:.3f}\n"
    info_chired = ""
    if(reduced_chi_squared>1e5):
        info_chired += r"$\chi^2_{{\mathrm{{red}}}}$ > 1e5"
    elif(reduced_chi_squared<0.0):
        info_chired += r"$\chi^2_{{\mathrm{{red}}}}$ < 0"
    else:
        info_chired += r"$\chi^2_{{\mathrm{{red}}}}$"+f" = {reduced_chi_squared:.3f}"

    fit_info = info_K+info_G+info_theta+info_chired
    
    plt.gca().text(0.68, 0.95, fit_info, transform=plt.gca().transAxes, 
               fontsize=10, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))

    ax.hist(charge, bins=bin_edges, density=True)
    plt.plot(x_values, polya_fit, 'r-', lw=2)
    if electrons:
        ax.set_xlabel("Charge per pixel [electrons]")
    else:
        ax.set_xlabel("Charge per pixel [clock cycles]")
    ax.set_ylabel("Normalised number of events")
    ax.set_ylim([1e-8,1])
    ax.set_yscale('log')
    ax.grid(which='both')
    plt.grid(True)
    plt.savefig(f"ChargePerPixel-{picname}.png", bbox_inches='tight', pad_inches=0.08)


def getPolyaChiSquared(bin_centers, counts, count_err, fitparams):

    expected_cts = polya(bin_centers, *fitparams)
    chisq = np.sum(((counts - expected_cts)**2)/count_err**2)
    dof = len(counts) - len (fitparams)
    chired = chisq / dof

    return chired

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

def polya(n, mean, var):
#def polya(mean, var):
     #from sigma^2 = mean*(theta+1) express (theta+1), which is var^2/mean^2 !
     # in mathematical nomenclature it is 
     # negative binomial distribution     
     #
     return (1/mean)*(np.power((var**2/mean**2),(var**2/mean**2)))/(gamma(var**2/mean**2))*np.power((n/mean),(var**2/(mean**2 - 1)))*np.exp(-(var**2/mean**2)*(n/mean))

    ## compute Polya parameter (θ)
    #theta = (mean**2 / var) - 1
    #if theta <= 0:
    #    return np.zeros_like(n)  # avoid invalid values

    ## proper Polya/Gamma PDF form
    #coeff = (1 / mean) * ((theta + 1) ** (theta + 1)) / gamma(theta + 1)
    #pdf = coeff * (n / mean) ** theta * np.exp(-(theta + 1) * n / mean)
    #return pdf

def model_counts(bin_centers, mean, theta, Ntot, bin_width):

    return polya_pdf(bin_centers, mean, theta) * Ntot * bin_width


def simpleGainPlot(nuarray, nbins, labels, picname):

    print(f"PLOTTING: histogram => {picname}")

    nuarray = np.array(nuarray)
    unique_data = np.sort(np.unique(nuarray))
    bin_edges_estim = np.concatenate((unique_data - 0.5, [unique_data[-1] + 0.5]))
    counts, bin_edges = np.histogram(nuarray, bins=bin_edges_estim, density=True)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
   
    Ntot = counts.sum() 
    mean_estim = nuarray.mean() if len(nuarray) > 0 else 1.0
    var_estim = nuarray.var(ddof=1) if len(nuarray) > 0 else max(mean_estim*0.1, 1.0)

    p0 = [mean_estim, var_estim]

    mask = counts > 0 
    xfit = bin_centers[mask]
    yfit = counts[mask]

    #count_errors = np.sqrt(yfit)
    count_errors = np.sqrt(counts)

    minbin = 0
    maxbin = np.ceil(np.median(nuarray) * 4)

    fitrange_start = np.nanmin(nuarray)
    fitrange = (bin_centers <= maxbin)

    #params, covariance = curve_fit(polya, bin_centers, counts, bounds=([1e-6,1e-6],[np.inf, np.inf]), maxfev=1000000)
    #params, covariance = curve_fit(polya, bin_centers[fitrange], counts[fitrange], p0=p0, bounds=([1e-6,1e-6],[np.inf, np.inf]), maxfev=1000000)
    params, covariance = curve_fit(polya, bin_centers[fitrange], counts[fitrange], p0=p0, maxfev=1000000)
    #params, covariance = curve_fit(polya, bin_centers, counts, maxfev=1000000)
    
    mean, var = params
    mean_err, var_err = np.sqrt(np.diag(covariance))
 
    CHISQRED = getPolyaChiSquared(bin_centers, counts, count_errors, params)

    # plotting below

    plt.figure(figsize=(8,6))
    plt.hist(bin_edges[:-1], 
             weights=counts, 
             bins=bin_edges_estim,    
             align='left', 
             histtype='stepfilled', 
             facecolor='tomato')

    #plt.plot(bin_centers[:-1],polya(bin_centers[:-1], mean, var), '--y')
    plt.plot(bin_centers[fitrange],polya(bin_centers[fitrange], mean, var), '--y')

    ax = plt.gca()
    _, maxy = ax.get_ylim()
    plt.text(0.70, 0.96, f"mean={mean:.2f}\nvar={var:.2f}")
    plt.vlines(mean, 0, maxy*0.9, linestyles='--', colors="blue")

    print(f"BLACK line at {bin_centers[fitrange[len(fitrange)-1]]}")
    print(fitrange)
    print(len(fitrange))
    print(bin_centers[fitrange])
    print(len(bin_centers[fitrange]))
    fitrange_end = len(bin_centers[fitrange])-1
    
    plt.vlines(bin_centers[fitrange_end], 0, maxy*0.5, linestyles='-', colors="black")

    plt.title("Fit: Negative Binomial Distribution")
    plt.xlabel("TOT, [cts]")
    plt.ylabel(r"$N_{Entries}$")
    plt.ylim([1e-8,1])
    plt.yscale('log')
    plt.savefig(f"POLYAFIT-{picname}.png")
    plt.close()

    #return [gmean, gmean_err, gtheta, gtheta_err]
    return [mean, mean_err, var, var_err]

#def simpleHist(nuarray, nbins, minbin, maxbin, labels, picname):
def simpleHist(nuarray, nbins, labels, picname):

    print(f"PLOTTING: histogram => {picname}")

    plt.figure()

    fit_pars = None

    arr_median = np.median(nuarray)

    maxbin = np.ceil(arr_median*4)
    minbin = 0

    counts, bin_edges = np.histogram(nuarray, bins=nbins, range=(minbin,maxbin))

    #plotAsciiHist(counts, "HUYA")

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

    peakbin = getMaxBin(counts[1:])

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
        lowcut = peakbin*binwidth-peak_pos_half*2
        if(lowcut < 0):
            lowcut = 0
        uppercut = peakbin*binwidth+peak_pos_half*2
    if("HITS" in picname):
        lowcut = peakbin*binwidth-peak_pos_half
        if(lowcut < 0):
            lowcut = 0
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
    #plt.vlines(lowcut, 0, maxbin_cnt+10, colors='magenta', linestyles='dashed')
    #plt.vlines(uppercut, 0, maxbin_cnt+10, colors='magenta', linestyles='dashed')
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

negbin_mean, negbin_var = [], []
negbin_mean_err, negbin_var_err = [], []

Vgrid_list = []
suffixes = []
suflabels = []

totHistList = []

hitHistList = []

ref_hits_lab = {"450":[29,6], "460":[37,7], "470":[50,8], "480":[65,8], "490":[78,9], "500":[83,12]}
ref_hits_p09 = {"450":[29,7], "460":[45,11], "470":[64,11], "480":[85,11], "490":[108,11], "500":[128,13]}
#ref_hits_sigma = {"450":6, "460":7, "470":8, "480":8, "490":9, "500":12}

fcnt = 0

for file in inputlist:

    rawTOT = None
    TOT, HITS = [], []
    pruned_tot = []

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
        fname_pruned = fname.split("/")[-1]
        words = fname_pruned.split("-")
        anonolist = ['reco', 'weighted']
        #print(words)
        isuff = None
        if("GHOSTS" in file):
            words[len(words)-1] = words[len(words)-1][:-3]
            for ano in anonolist:
                words.remove(ano)
            isuff = " ".join(words)
        else:
            Vgrid = int(words[len(words)-1][:-4])# removin V.h5 from voltage number
            print(f"Found Vgrid={Vgrid} V")

        ###############################################
        rawTOT = f.get_node(base_group_name+"ToT") 
        #rawTOT = np.concatenate(rawTOT)
        sumTOT = f.get_node(base_group_name+"sumTot")    
        hits = f.get_node(base_group_name+"hits")
        x = f.get_node(base_group_name+"x")
        y = f.get_node(base_group_name+"y")
        print(f"found sumTOT {type(sumTOT)}")
        print(f"Contains {len(sumTOT)} events/hits")
        hits_mean, hits_sigma = None, None
        if("LabRuns" in location):
            hits_mean = ref_hits_lab[str(Vgrid)][0]
            hits_sigma = ref_hits_lab[str(Vgrid)][1]
        elif("P09" in location):
            hits_mean = ref_hits_p09[str(Vgrid)][0]
            hits_sigma = ref_hits_p09[str(Vgrid)][1]
        else:    
            hits_mean = 120
            hits_sigma = 120

        hits_min = hits_mean - hits_sigma
        hits_max = hits_mean + hits_sigma
        
        icyc = 0
        for ts, h in zip(sumTOT, hits):
            if(h <= hits_max and h >= hits_min):
                TOT.append(ts)
            #TOT.append(ts)
            HITS.append(h)
            icyc+=1
            if("1p46MHz" in file and icyc == 75000):
                break

        nevt, cnt = len(x), 0
        
        for ix, iy, itot in zip(x,y,rawTOT):
            for jx,jy,jtot in zip(ix,iy,itot):
               if(jx>=50 and jx<=250 and jy>=50 and jy<=250):
                    pruned_tot.append(jtot)
            cnt+=1
            print(f"\r[{cnt/nevt*100:.2f}%]", flush=True, end="")

        sumTOT = None
        hits = None

        charge = np.concatenate(rawTOT)
        chpp_name = None
        if("GHOSTS" in file):
            reass_name = "-".join(words)
            suflabels.append(reass_name)
            chpp_name = f"{reass_name}-"+picname
        else:
            chpp_name = f"{str(Vgrid)}V-"+picname
        chargeperpixel(charge, chpp_name, False) 

    #print(f"--------------------DONE----------------------")

    #exit(0)

    print(f"Accumulated: {len(HITS)} hits, {len(TOT)} sumTOT entries")   
    print("=============== FITTING TOT ================================")
    #peak_info = simpleHist(TOT, 100,0, 6000, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)
    if("GHOSTS" in file): 
        peak_info = simpleHist(TOT, 100, [f"TOT per event ({isuff})", "TOT cycles", "N"], f"TOT-{isuff}-"+picname)
    else:
        peak_info = simpleHist(TOT, 100, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)

    print("=============== FITTING TOT (NegBinDist) ===================")
    #peak_info = simpleHist(TOT, 100,0, 6000, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)
    #gain_info = simpleGainPlot(TOT, 100, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)
    if("GHOSTS" in file):
        gain_info = simpleGainPlot(np.array(pruned_tot), 100, [f"TOT per event ({isuff})", "TOT cycles", "N"], f"TOT-{isuff}-"+picname)
    else:
        gain_info = simpleGainPlot(np.array(pruned_tot), 100, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)
    #gain_info = simpleGainPlot(rawTOT, 100, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)

    #print("CHINAZES!")
    #exit(0)
    
    print("=============== FITTING HITS ===============================")
    #hit_info = simpleHist(HITS, 50,0, 150, [f"Hits per event (Vgrid={Vgrid}[V])", r"$N_{HITS}$", "N"], f"HITS-{Vgrid}-"+picname)
    if("GHOSTS" in file):
        hit_info = simpleHist(HITS, 100, [f"Hits per event ({isuff})", r"$N_{HITS}$", "N"], f"HITS-{isuff}-"+picname)
    else:
        hit_info = simpleHist(HITS, 100, [f"Hits per event (Vgrid={Vgrid}[V])", r"$N_{HITS}$", "N"], f"HITS-{Vgrid}-"+picname)

    if("GHOSTS" in file):
        totHistList.append([TOT,isuff])
        hitHistList.append([HITS,isuff])
        suffixes.append(isuff)
    else:
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
 
    negbin_mean.append(gain_info[0])
    negbin_var.append(gain_info[2])
    negbin_mean_err.append(gain_info[1])
    negbin_var_err.append(gain_info[3])

    gain_info = None
    hit_info = None
    peak_info = None
    TOT = None
    HITS = None
    rawTOT = None

    fcnt+=1


fSuffix = False
if(len(suffixes)>0):
    fSuffix = True
print("======================================================")
print("Hit peak positions per voltage")
if(fSuffix):
    print("Hit peak positions per data set")
    for suf, peak, sig in zip(suffixes, hit_peak_mu, hit_peak_sigma):
        print(f"{suf} -> {peak} +- {sig}")
else:
    print("Hit peak positions per voltage")
    for vg, peak, sig in zip(Vgrid_list, hit_peak_mu, hit_peak_sigma):
        print(f"{vg} -> {peak} +- {sig}")
print("======================================================")
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
    
plt.title("Energy Spectrum")
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
xmin, xmax = None, None

if(fSuffix):
    print(len(Vgrid_list))
else:
    print(len(Vgrid_list))
    xmin = int(Vgrid_list[0])-10
    xmax = int(Vgrid_list[len(Vgrid_list)-1])+10

print(len(peak_mu))
print(len(peak_mu_err))

plt.figure(figsize=(8,8))

fig, ax = plt.subplots()

if(fSuffix):
    ax.errorbar([int(i) for i in range(len(suffixes))], peak_mu, peak_mu_err, fmt='o', mfc='green', linewidth=2, capsize=6, label="Gauss means")
    ax.scatter([int(i) for i in range(len(suffixes))], negbin_mean, marker='x', c='m', label="Neg.Bin.Dist. means")
else:
    ax.errorbar(Vgrid_list, peak_mu, peak_mu_err, fmt='o', mfc='green', linewidth=2, capsize=6, label="Gauss means")
    ax.scatter(Vgrid_list, negbin_mean, marker='x', c='m', label="Neg.Bin.Dist. means")

#ax.set_title(r"GridPix3 - $^{55}$Fe - TOT peak position vs. $\mathrm{V}_{\mathrm{Grid}}$")
if(fSuffix):
    ax.set_title(r"GridPix3 - TOT peak position for Different Data Sets$")
    ax.set_xlabel("Data Set")
else:
    ax.set_title(r"GridPix3 - $^{55}$Fe - TOT peak position vs. $\mathrm{V}_{\mathrm{Grid}}$")
    ax.set_xlabel("Grid voltage, [V]")
ax.set_ylabel(r"Fit $\mu$, [$\Sigma$(TOT) per Event]")
if(not fSuffix):
    ax.set_xlim([xmin, xmax])
ax.grid(True)
plt.legend(loc='upper left')
plt.savefig(f"GainCurve-{picname}.png",dpi=400)
plt.close()

###################################################
# combined hit histogram means of the fits
plt.figure(figsize=(8,8))

fig, ax = plt.subplots(figsize=(10,8))
if(fSuffix):
    ax.errorbar([int(i) for i in range(len(suffixes))], hit_peak_mu, hit_peak_mu_err, fmt='o', mfc='red', linewidth=2, capsize=6)
    ax.set_title(r"GridPix3 - Hit peak position$")
    indexes = [int(i) for i in range(len(suffixes))]
    for su, idx in zip(suffixes, indexes):
        ax.scatter([],[],color='white',label=f"{idx}: {su}")
    ax.set_xlabel("Data Set")
    ax.legend()
else:
    ax.errorbar(Vgrid_list, hit_peak_mu, hit_peak_mu_err, fmt='o', mfc='red', linewidth=2, capsize=6)
    ax.set_title(r"GridPix3 - $^55$Fe - Hit peak position vs. $\mathrm{V}_{\mathrm{Grid}}$")
    ax.set_xlabel("Grid voltage, [V]")
ax.set_ylabel(r"Fit ($\mu$), [$\mathrm{N}_{\mathrm{Hits}}$]")
if(not fSuffix):
    ax.set_xlim([xmin, xmax])
#ax.set_ylim([13000, 28000])
ax.grid(True)
plt.savefig(f"HitsCurve-{picname}.png",dpi=400)
plt.close()

###################################################

plt.figure(figsize=(8,8))

fig, ax = plt.subplots()
ax.errorbar(peak_mu, hit_peak_mu, hit_peak_mu_err, peak_mu_err, fmt='x', linewidth=2, capsize=6)
ax.set_title("Hits vs TOT")
ax.set_ylabel(r"Fit ($\mu$), [$\mathrm{N}_{\mathrm{Hits}}$]")
ax.set_xlabel(r"Fit $\mu$, [$\Sigma$(TOT) per Event]")
#ax.set_xlim([250, 3000])
ax.set_ylim([15,200])
ax.grid(True)
plt.savefig(f"MuVsHits-gainMeas-{picname}.png")
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

if(fSuffix):
    print("=============================================")
    for suf, res, eres in zip(suffixes,E_res,E_res_err):
        print(f"{suf} | {res:.2f}% +- {eres:.2f}%")
    print("=============================================")
else:
    print("=============================================")
    for v, res, eres in zip(Vgrid_list,E_res,E_res_err):
        print(f"Vgrid={v} | {res:.2f}% +- {eres:.2f}%")
    print("=============================================")

plt.figure(figsize=(8,8))

fig, ax = plt.subplots()
if(fSuffix):
    ax.errorbar([int(i) for i in range(len(suffixes))], E_res, E_res_err, fmt='o', linewidth=2, capsize=6)
    ax.set_xlabel("Data Set")
    ax.set_title("Energy Resolution")
else:
    ax.errorbar(Vgrid_list, E_res, E_res_err, fmt='o', linewidth=2, capsize=6)
    ax.set_xlabel("Grid voltage, [V]")
    ax.set_title("Energy resolution vs Grid voltage")
ax.set_ylabel(f"{G_delta}E/E, [%]")
if(not fSuffix):
    ax.set_xlim([xmin, xmax])
plt.grid()
plt.savefig(f"Energy_resolutions-{picname}.png")

