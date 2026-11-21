import sys
import glob
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.special import gamma, erf
from scipy.optimize import curve_fit
from MyPlotter import myUtils
from MyPlotter import myColors
from MyPlotter import mySymbols
###### using qt plotting
from MyPlotter import myMonitor
from pyqtgraph.Qt import QtWidgets
#from PyQt5 import QtWidgets, QtCore

######

import ROOT as r

###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

def check_node(f,node):

    return node in f

def polya_MG(x, K, G, T):
    Tp1 = T + 1
    return (K / G) * (np.power(Tp1, Tp1))/(gamma(Tp1)) * np.power((x/G), T) * np.exp(-(Tp1*x)/(G))

def estimate_nbins_sturge(nobs):
        # using Sturges's rule
        return int(np.round(1 + np.emath.logn(2, nobs)))

def estimate_nbins_rice(nobs):
        # using Rice rule
        return int(np.round(2 * np.power(nobs, 1/3)))

def getFWHM(sigma):

    return 2 * np.sqrt(2*np.log(2))*sigma

#def rootplot_chargeperpixel():
#    ## -----------------------------------------------------
#
#    #r.gROOT.SetBatch(True) # so it does not try to show anything....
#
#    #rbins = len(bin_edges) - 1 
#    #roothist = r.TH1D("hcharge", "Charge per Pixel", rbins, bin_edges)
#
#    #for q in charge:
#    #    roothist.Fill(q)
#
#    #polya_expr = ("([0]/[1])"
#    #"* ( pow( ( ([1]*[1]) / ([2]*[2]) ), ( ([1]*[1]) / ([2]*[2]) ) ) / TMath::Gamma( ( [1]*[1] ) / ( [2]*[2] ) ) )"
#    #"* pow( x/[1], ( ( [1]*[1] ) / ( [2]*[2] ) - 1 ) )"
#    #"* exp( - ( ( [1]*[1] ) / ( [2]*[2] ) ) * ( x / [1] ) )" 
#    #)
#
#    #polya_expr = (
#    #"([0] / [1]) * "
#    #"TMath::Power([1]/[2], [1]/[2]) / TMath::Gamma([1]/[2]) * "
#    #"TMath::Power(x/[1], [1]/[2] - 1) * "
#    #"TMath::Exp(-( [1]/[2] ) * (x/[1]))"
#    #)
#
#    ##rootpolya = r.TF1("polya",
#    ##          "([0] / [1]) *(((([1]*[1])/([2]*[2]))^(([1]*[1])/([2]*[2]))) /(TMath::Gamma((([1]*[1])/([2]*[2]))))) * ((x /[1])^((([1]*[1])/([2]*[2]))-1)) * exp(-(([1]*[1])/([2]*[2])) *(x / [1])))",
#    ##          bin_edges[0], bin_edges[-1])
#
#
#    #xmin = np.nanmin(charge)
#    #xmax = np.nanmax(charge)
#    ##rootpolya = r.TF1("polya", polya_expr, bin_edges[0], bin_edges[-1])
#    #rootpolya = r.TF1("polya", polya_expr, xmin, xmax)
#
#    ##amp_scaling = initAmp
#    ##amp_gain = initMean
#    ##amp_width = initStd
#    #amp_scaling = charge.size * 0.1
#    #amp_gain = np.median(charge)
#    #amp_width = 0.5
#
#    #rootpolya.SetParameter(0,amp_scaling)
#    #rootpolya.SetParameter(1,amp_gain)
#    #rootpolya.SetParameter(2,amp_width)
#
#    #r.gStyle.SetOptFit(1)
#    #fit_result = roothist.Fit(rootpolya, "S")
#
#    #scaling_fit = rootpolya.GetParameter(0)
#    #gain_fit = rootpolya.GetParameter(1)
#    #width_fit = rootpolya.GetParameter(2)
#
#    #scaling_err = rootpolya.GetParError(0)
#    #gain_err = rootpolya.GetParError(1)
#    #width_err = rootpolya.GetParError(2)
#
#    #rootpolya.SetParLimits(0, 0.0, 1e9)
#    #rootpolya.SetParLimits(1, 1e-6, 1e9)
#    #rootpolya.SetParLimits(2, 1e-6, 1e9)
#
#    #print("===========================================================================")
#    #print("ROOT fit:")
#    #print(f"Amplification scale: {scaling_fit:.4f} {MS.plusminus} {scaling_err:.4f}")
#    #print(f"Amplification gain: {gain_fit:.4f} {MS.plusminus} {gain_err:.4f}")
#    #print(f"Amplification width: {width_fit:.4f} {MS.plusminus} {width_err:.4f}")
#    #print("===========================================================================")
#
#    ## -----------------------------------------------------


def chargeperpixel(charge, picname, fElectrons=False):
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
    bin_edges = None
    if():
        bin_edges = 1000
    else:
        bin_edges = np.concatenate((unique_data - 0.5, [unique_data[-1] + 0.5]))

    hist, bins = np.histogram(charge, bins=bin_edges, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    #hist_max = np.nanmax(hist)
    range_factor = 4;
    hist_info =""
    if(fElectrons):
        hist_info+=f"max={np.nanmax(charge):.4f}\n"
        hist_info+=f"mean={np.mean(charge):.4f}\n"
        hist_info+=f"med={np.median(charge):.4f}\n"  
        hist_info+=f"std={np.std(charge):.4f}\n" 

    fit_range_start = np.nanmin(charge)
    charge_cap = None
    if(fElectrons):
        charge_cap = 15000
    else:
        charge_cap = np.ceil(np.median(charge))*range_factor

    fit_range = (bin_centers <= charge_cap)
    prun_centers = bin_centers[fit_range]
    fit_range_end = prun_centers[-1]
    prun_centers = None

#    if(fElectrons):
#        print(f"{MC.RED}bin_centers={bin_centers}{MC.RST}")
#        print(f"{MC.BLUE}length={len(bin_centers)}{MC.RST}")
#        print(f"{MC.YELLOW}fit_range={fit_range}{MC.RST}")

    initAmp, initMean, initStd = None, None, None
    if(fElectrons):
        maxbin = np.argmax(hist)
        initAmp = hist[maxbin]
        initMean = 0.5*(bin_centers[maxbin]+bin_centers[maxbin])
        initStd = (np.mean(charge)/np.std(charge))**2
        hist_info+=f"maxbin={maxbin}\n"
        hist_info+=f"initA={initAmp:.4f}\n"
        hist_info+=f"initM={initMean:.4f}\n"
        hist_info+=f"initS={initStd:.4f}\n"
        hist_info+=f"rangeStart={fit_range[0]:.1f}\n"
        hist_info+=f"rangeStop={fit_range_end:.1f}\n"
        

    else:
        initAmp = 1000
        initMean = 2000
        initStd = 1
       

    #params, params_covariance = curve_fit(polya_MG, bin_centers[fit_range], hist[fit_range], p0=[10000, 2000, 1])
    params, params_covariance = curve_fit(polya_MG, bin_centers[fit_range], hist[fit_range], p0=[initAmp, initMean, initStd])
    #if(fElectrons):
    #    params, params_covariance = curve_fit(polya_MG, bin_centers, hist, p0=[initAmp, initMean, initStd])
    #else:
    #    params, params_covariance = curve_fit(polya_MG, bin_centers[fit_range], hist[fit_range], p0=[initAmp, initMean, initStd])

    params_errors = np.sqrt(np.diag(params_covariance))
    #x_values = np.linspace(fit_range_start, max(bin_edges), 1000)
    x_values = np.linspace(fit_range_start, fit_range_end, 1000)
    polya_fit = polya_MG(x_values, *params)

    residuals = hist[fit_range] - polya_MG(bin_centers[fit_range], *params)
    reduced_chi_squared = np.sum((residuals ** 2) / polya_MG(bin_centers[fit_range], *params)) / (len(hist[fit_range]) - len(params))

    K, G, theta = params[0], params[1], params[2]
    errK, errG, errtheta = params_errors[0], params_errors[1], params_errors[2]

    info_K = f"K = {K:.3f}"+" $\pm$ "+f"{errK:.3f}\n"
    info_G = f"G = {G:.3f}"+" $\pm$ "+f"{errG:.3f}\n"
    info_theta = r"$\theta$ = "+f"{theta:.3f} "+r"$\pm$"+f" {errtheta:.3f}\n"
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
    if(fElectrons):
        ax.text(22000,1e-6,hist_info,fontsize=11)
    ax.set_ylabel("Normalised number of events")
    #ax.set_ylim([1e-8,1])
    #ax.set_yscale('log')
    ax.grid(which='both')
    plt.grid(True)
    plt.savefig(f"ChargePerPixel-{picname}.png", bbox_inches='tight', pad_inches=0.08)
    plt.close()

    #return K, G, theta, errK, errG, errtheta
    return G, errG

def getPolyaChiSquared(bin_centers, counts, count_err, fitparams):

    expected_cts = polya(bin_centers, *fitparams)
    chisq = np.sum(((counts - expected_cts)**2)/count_err**2)
    dof = len(counts) - len (fitparams)
    chired = chisq / dof

    return chired

def getSkewGaussChisquared(bin_centers, counts, count_err, fitparams):

    # TODO: 
    # need to restrict fit range otherwise the empty bins 
    # throw ZeroDivisionError and result in X^2_red to be -nan

    #funcname = 'getSkewGaussChisquared'
    #print(f"[{funcname}]: fitparams={fitparams}")
    expected_cts = skewed_gauss(bin_centers, *fitparams)
    #print(f"expected={expected_cts}")
    chisq = np.sum(((counts - expected_cts)**2)/count_err**2)
    #print(f"chisq={chisq}")
    dof = len(counts) - len (fitparams)
    #print(f"degs of freedom={dof}")
    chired = chisq / dof
    #print(f"chired={chired}")    

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

def skewed_gauss(x, A, mu, sigma, alpha):

    defgauss = np.exp(-(x-mu)**2/(2*sigma**2))
    skew = 1 + erf(alpha*(x-mu)/(np.sqrt(2)*sigma))
    return A*defgauss*skew

def calcEResolution(mean, sigma):

    return 2.355*(sigma/mean)*100.0

def calcDetRes(mu, sigma):

    return (sigma/mu)*100.0

def calcEResolutionCorUnc(mu,sigma,err_mu,err_sigma,cov_sigma_mu):

    sigmaE = np.sqrt(
        ((2.355/mu)**2 * err_sigma**2)
        +
        ((2.355*sigma/err_mu**2)**2 * err_mu**2)
        -
        ((2*(2.355**2)*sigma*cov_sigma_mu)/mu**3)
        ) * 100.0

    return sigmaE

def calcDetResUnc(mu, sig, emu, esig, sigmucov):

    dR = np.sqrt(
            (1/mu)**2 * esig**2 
            + 
            (sig/mu**2)**2 * emu**2
            -2 * (sig * sigmucov)/mu**3 
            )

    return dR

def calcEResolutionUnc(mu,sigma,emu,esigma):

    return 2.355*sigma/mu*np.sqrt((esigma/sigma)**2+(emu/mu)**2)

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

def fitAndPlot(nuarray, # numpy.array/numpy.ndarray
                nbins, # int
                binrange, # tuple (int,int)
                labels, # [str,str,str]
                picname, # str
                plotxlim=None, # [int,int] for ax.set_xlim(...)
                plotsize=None, # tuple (int,int)
                fit="gauss", # str -> "gauss"- default, later dajust for others, like "skewgauss", etc
                annotation=None, # str -> input for ax.text(..)
                bincolor=None, # str -> face color of bins
                fDensity=False, # bool -> eanble density histograming
                fCorUnc=False, # bool -> return correlation factors
                fDebug=False): # bool -> enable debug printouts

    # in this alternative plotting trying to user "curve_fit" 
    # instead of lmfit.Model ....

    if(fDebug):
        print(f"[fitAndPlot]: processing <{picname}>...")
     
    clear_data = nuarray[~np.isnan(nuarray)]
    counts, edges = None, None
    if(fDensity):
        counts, edges = np.histogram(clear_data, bins=nbins, range=binrange, density=True)
    else:
        counts, edges = np.histogram(clear_data, bins=nbins, range=binrange)

    centers = (edges[:-1]+edges[1:])/2
    count_err = np.sqrt(counts)
    # establishing initial guesses...
    hist_mean = np.mean(clear_data)
    hist_med = np.median(clear_data)
    hist_std = np.std(clear_data)
    hist_max = np.max(counts)   

    # fitting here

    params, covariance = None, None
    fitfunc = None
    equationstr = ""

    fitmask = None

    if(fit=="gauss"):
        fitfunc = gauss
        #    Amp,        mu     sigma
        p0=[hist_max,hist_mean,hist_std]
        bounds = ([0,0,0],[np.inf,np.inf,np.inf])
        equationstr = r"$\frac{1}{\sqrt{2\pi}}\cdot\exp{-\frac{x^{2}}{2}}$"
        if(fDebug):
            print(f"[fitandplot]: fitting \"gauss\" function....")
    elif(fit=="skewgauss"):
        fitfunc=skewed_gauss
        #     amp,      mu,       sigma   alpha
        p0=[hist_max, hist_mean, hist_std, 0]
        bounds = ([0,0,0,-20],[np.inf,np.inf,np.inf,20])
        #equationstr = r"2$(\frac{1}{\sqrt{2\pi}}\cdot\exp{-\frac{x^{2}}{2}})\cdot(\frac{1}{2}[1+\mathrm{erf(\frac{x}{\sqrt{2}})})$"
        equationstr = r"$2\phi(x)\Phi(\alpha\,x)$"
        if(fDebug):
            print(f"[fitandplot]: fitting \"skew_gauss\" function....")

        # try automatic range later ...
        #range_low = np.percentile(nuarray,5)
        #range_high = np.percentile(nuarray,95)
        #fitmask = (centers>range_low) & (centers<range_high)

        if("450" in picname):
            fitmask = (centers > 1.25e4) & (centers < 1e5)
        elif("460" in picname):
            fitmask = (centers > 4.5e4) & (centers < 1.6e5)
        elif("470" in picname):
            fitmask = (centers > 8e4) & (centers < 2.5e5)
        elif("480" in picname):
            fitmask = (centers > 1.5e5) & (centers < 3.5e5)
        elif("490" in picname):
            fitmask = (centers > 2.3e5) & (centers < 4.5e5)
        elif("500" in picname):
            fitmask = (centers > 3.2e5) & (centers < 5.6e5)
        else:
            fitmask = (counts > 0)

    else:
        raise ValueError(f"There ain't no fits for {fit}")
        exit(0)

    params,covariance = curve_fit(
                        fitfunc,
                        centers[fitmask],
                        counts[fitmask],
                        p0=p0,
                        bounds=bounds,
                        maxfev=1000000)

    # doing chisquares here

    expected_cts = fitfunc(centers[fitmask], *params)
    chisquare = np.sum(((counts[fitmask]-expected_cts)**2)/count_err[fitmask]**2)
    dof = len(counts) - len(params)
    chired = chisquare/dof

    # fit param uncertainties

    param_unc = np.sqrt(np.diag(covariance))

    # some defaults for histogram
    thisplotsize = (9,8)
    thisbincol = 'royalblue'
    if(plotsize is not None):
        thisplotsize = plotsize
    if(bincolor is not None):
        thisbincol = bincolor
    
    #A, mu, sig, alpha = None, None, None, None
    #eA, emu, esig, ealpha = None, None, None, None
    alpha, ealpha = None ,None    

    A, mu, sig = params[0], params[1], params[2]
    eA, emu, esig = param_unc[0], param_unc[1], param_unc[2]
    if(fit=="skewgauss"):
        alpha, ealpha = params[3] , param_unc[3]

    textparams=""
    pmstr = r"$\pm$"
    #textparams = f"{fittype}, nbins={nbins}\n"
    textparams+=f"Amp={A:.2f}{pmstr}{eA:.2f}\n"
    textparams+=f"{MS.mu}={mu:.2f}{pmstr}{emu:.2f}\n"
    textparams+=f"{MS.sigma}={sig:.2f}{pmstr}{esig:.2f}\n"
    if(fit=="skewgauss"):
        textparams+=f"{MS.alpha}={alpha:.2f}{pmstr}{ealpha:.2f}\n"
        textparams+=r"$\chi^{2}$"+f"={chisquare:.2f}\n"
        textparams+=f"NDOF={dof:.2f}\n"
        textparams+=r"$\chi^{2}_{\mathrm{red.}}$"+f"={chired:.2f}\n"
    if("ELEC" in picname):
        cov_mu_sig = covariance[2,1]
        deltaEoverE = calcEResolution(mu,sig)
        err_deltaEoverE = calcEResolutionCorUnc(mu,sig,emu,esig,cov_mu_sig)
        textparams+=r"$\frac{FWHM}{\mu}$="+f"{deltaEoverE:.2f}{pmstr}{err_deltaEoverE:.2f}\n"
        deltaEoverE, err_deltaEoverE = None, None


    # plotting here
    fig,ax = plt.subplots(figsize=thisplotsize)
    
    ax.hist(centers,
            weights=counts,
            bins=nbins,
            range=binrange,
            align='left',
            histtype='stepfilled',
            facecolor=thisbincol)
    ax.plot(centers[fitmask], fitfunc(centers[fitmask],*params), '--r',label=f"Fit:{equationstr}")
    miny,maxy = ax.get_ylim()
    _,xmax = ax.get_xlim()
    ax.text(xmax*0.65, maxy*0.5,textparams)
    if(annotation is not None):
        ax.text(0,0,f"{annotation}")
    ax.set_title(labels[0])
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[2])
    if(plotxlim is not None):
        ax.set_xlim(plotxlim)
    ax.grid()

    plt.savefig(f"fitAndPlot-{picname}",dpi=256)
    plt.close()

    return params, param_unc, covariance

def simpleHist(nuarray, nbins, labels, picname, fittype="gauss",debug=False):

    if(debug):
        print(f"PLOTTING: histogram => {picname}")

    plt.figure(figsize=(8,6))

    fit_pars = None

    arr_median = np.median(nuarray)

    minbin = 0
    maxbin = None
    if("Electrons" in labels[0]):
        maxbin = 1e6
    else:
        if("HITS" in picname):
            maxbin = 250
            #if("450-P09" in picname):
            #    maxbin = 80
            #elif("460-P09" in picname):
            #    maxbin = 100
            #elif("470-P09" in picname):
            #    maxbin = 100
            #elif("480-P09" in picname):
            #    maxbin = 100
            #elif("490-P09" in picname):
            #    maxbin = 100
            #elif("500-P09" in picname):
            #    maxbin = 100
            #else:
            #    maxbin = np.ceil(arr_median*4)
        elif("TOT" in picname):
            maxbin = 7500
            #if("450-P09" in picname):
            #    maxbin = 2200
            #elif("460-P09" in picname):
            #    maxbin = 3000
            #elif("470-P09" in picname):
            #    maxbin = 4000
            #elif("480-P09" in picname):
            #    maxbin = 5000
            #elif("490-P09" in picname):
            #    maxbin = 6000
            #elif("500-P09" in picname):
            #    maxbin = 7500
            #else:
            #    maxbin = np.ceil(arr_median*4)
        else:
            maxbin = np.ceil(arr_median*4)

    counts, bin_edges = None, None
    fAutoRange = False
    if(type(nbins)==str):
        fAutoRange = True
        counts, bin_edges = np.histogram(nuarray, bins=nbins)
        nbins = len(bin_edges)-1
    else:
        counts, bin_edges = np.histogram(nuarray, bins=nbins, range=(minbin,maxbin))

    fitmask = counts > 10
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    #count_err = np.sqrt(counts)
    count_err = np.sqrt(np.maximum(counts,1))

    #plotAsciiHist(counts, "HUYA")

    if(debug):
        print(f"Retrieved array of counts:\n{counts}")

    if(fAutoRange):
        #plt.hist(bin_edges[:-1], weights=counts, bins=nbins, align='left', histtype='stepfilled', facecolor='b')
        plt.hist(bin_centers, weights=counts, bins=nbins, align='left', histtype='stepfilled', facecolor='b')
    else:
        plt.hist(bin_centers, weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', facecolor='b')
    plt.figsize=(8,8)
    maxbin_cnt = np.max(counts)
    minbin_cnt = np.min(counts)
    val_ibin = (maxbin-minbin)/nbins

    if(debug):
        print(f"Range: {minbin} -> {maxbin}, with {nbins} bins gives {val_ibin} per bin")

    ########################################################
    model = None

    peakbin = getMaxBin(counts[1:])
    
    if(debug):
        print(f"Found maximum bin at {peakbin} = {counts[peakbin]}")

    model, pars = None, None

    if(fittype=="gauss"):
        model = Model(gauss)
        pars = model.make_params(A=maxbin_cnt, 
                                mu=peakbin*val_ibin, 
                                sigma=np.std(nuarray))
 
    elif(fittype=="skewgauss"):
        model = Model(skewed_gauss)
        pars = model.make_params(A=maxbin_cnt, 
                                mu=peakbin*val_ibin, 
                                sigma=np.std(nuarray),
                                alpha=1.0)

        pars['sigma'].min = 1e-6
        pars['alpha'].min = -20
        pars['alpha'].max = 20

    else:
        raise ValueError(f"Unkown fit type: {fittype}...")

    
    #pars['A'].min = maxbin_cnt*0.8
    #pars['A'].min = maxbin_cnt*1.2
    #pars['mu'].min = peakbin*0.6*val_ibin
    #pars['mu'].min = peakbin*1.4*val_ibin
    #pars['sigma'].min = np.std(counts)*0.6
    #pars['sigma'].min = np.std(counts)*1.4

    if(debug):
        print(f"Gauss Fit: Setting: A={maxbin_cnt}, mu={peakbin*val_ibin:.4f}, sigma={np.std(counts):.4f}")
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

#    dataunder_sigma = np.std(nuarray)
#    dataunder_med = np.median(nuarray)
#    dataunder_mean = np.mean(nuarray)

    THR = None
    fitmask = None
    if("ELEC" in picname):
        #THR = 490
        if("450" in picname):
            fitmask = (bin_centers >= 1.25e4) & (bin_centers <= 1e5)
        elif("460" in picname):
            fitmask = (bin_centers >= 4.5e4) & (bin_centers <= 1.6e5)
        elif("470" in picname):
            fitmask = (bin_centers >= 8e4) & (bin_centers <= 2.5e5)
        elif("480" in picname):
            fitmask = (bin_centers >= 1.5e5) & (bin_centers <= 3.5e5)
        elif("490" in picname):
            fitmask = (bin_centers >= 2.3e5) & (bin_centers <= 4.5e5)
        elif("500" in picname):
            fitmask = (bin_centers >= 3.2e5) & (bin_centers <= 5.6e5)
        else:
            fitmask = (bin_centers > 0)

    elif("HITS" in picname):
        if("450" in picname):
            fitmask = (bin_centers >= 11) & (bin_centers <= 50)
        elif("460" in picname):
            fitmask = (bin_centers >= 25) & (bin_centers <= 73)
        elif("470" in picname):
            fitmask = (bin_centers >= 38) & (bin_centers <= 90)
        elif("480" in picname):
            fitmask = (bin_centers >= 60) & (bin_centers <= 110)
        elif("490" in picname):
            fitmask = (bin_centers >= 85) & (bin_centers <= 145)
        elif("500" in picname):
            fitmask = (bin_centers >= 100) & (bin_centers <= 165)
        else:
            fitmask = (bin_centers > 0)

    elif("TOT" in picname):
        #THR = 200
        if("450" in picname):
            fitmask = (bin_centers >= 280) & (bin_centers <= 1000)
        elif("460" in picname):
            fitmask = (bin_centers >= 550) & (bin_centers <= 1550)
        elif("470" in picname):
            fitmask = (bin_centers >= 950) & (bin_centers <= 2200)
        elif("480" in picname):
            fitmask = (bin_centers >= 1500) & (bin_centers <= 3000)
        elif("490" in picname):
            fitmask = (bin_centers >= 2300) & (bin_centers <= 5000)
        elif("500" in picname):
            fitmask = (bin_centers >= 3000) & (bin_centers <= 6100)
        else:
            fitmask = (bin_centers > 0)
    else:
        #THR = np.std(counts)
        fitmask = (bin_centers > 0)

    #fitmask = counts > THR

    #result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
    result = model.fit(counts[fitmask], pars, x=bin_centers[fitmask], weights=1/count_err[fitmask])
    counts_trunc, bin_centers_trunc = None, None
    if(debug):
        print(result.fit_report()) 
    if(result.params['mu']<=0 or result.params['sigma']<=0):
        f_failed = True
        print("\tFIT FAILED: restricting fit parameters and re-fitting")

        pars['A'].min = maxbin_cnt*0.8
        pars['A'].max = maxbin_cnt*1.2

        pars['mu'].min = peakbin*0.8*val_ibin
        pars['mu'].max = peakbin*1.2*val_ibin
 
        if(debug):
            print(f"peak at {peakbin}, half-peak position at bin {peak_pos_half}")
            print(f"Bin width = {binwidth}")
            print(f"peak width should be {2*est_sigma} ")

        sigma_to_peak = np.abs(result.params['sigma']) / result.params['A']

        if(result.params['sigma']<0 and sigma_to_peak < 1):
 
            pars['sigma'].min = np.abs(result.params['sigma'])*0.8
            pars['sigma'].max = np.abs(result.params['sigma'])*1.2

        else: 

            pars['sigma'].min = est_sigma*0.8
            pars['sigma'].max = est_sigma*1.2

       
        #result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
        result = model.fit(counts[fitmask], pars, x=bin_centers[fitmask],weights=1/count_err[fitmask])
        #result = model.fit(counts_trunc, pars, x=bin_centers_trunc)
        #result = model.fit(counts[peakbin-10:peakbin+10], pars, x=bin_centers[peakbin-10:peakbin+10])

        if(debug):
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

    # checking fit quality from the lmfit.Model itself...?    
    chisq = result.chisqr
    ndof = result.nfree
    chired_auto = result.redchi

    fit_pars = None
    chired = None
    cov_sigma_mu = None
    if(fittype=="gauss"):
        fit_pars = [A, err_A,  mu, err_mu, sigma, err_sigma]

    elif(fittype=="skewgauss"):
        if("Electrons" in labels[0] or "TOT" in picname):
            pnames = result.var_names    
            cov_idx_mu = pnames.index("mu")
            cov_idx_sigma = pnames.index("sigma")
            #cov_sigma_mu = result.covar[cov_idx_mu,cov_idx_sigma]        
            cov_sigma_mu = result.covar[cov_idx_sigma,cov_idx_mu]        
            if(debug):
                print(f"GOT covariance for sigma/mu from index [{cov_idx_sigma},{cov_idx_mu}] as [{cov_sigma_mu}]")

        alpha = result.params['alpha'].value    
        err_alpha = result.params['alpha'].stderr    
        if("Electrons" in labels[0]):
            fit_pars = [A, err_A,  mu, err_mu, sigma, err_sigma, alpha, err_alpha, cov_sigma_mu]
        else:
            fit_pars = [A, err_A,  mu, err_mu, sigma, err_sigma, alpha, err_alpha]
        useparams = A, mu ,sigma, alpha
        chired = getSkewGaussChisquared(bin_centers[fitmask], counts[fitmask], count_err[fitmask], useparams) 
    else:
        raise ValueError(f"No return of fit_params for fit type [{fittype}]")

    pmstr = r"$\pm$"
    plottext = f"{fittype}, nbins={nbins}\n"
    plottext+=f"A={A:.2f}{pmstr}{err_A:.2f}\n"
    plottext+=f"{MS.mu}={mu:.2f}{pmstr}{err_mu:.2f}\n"
    plottext+=f"{MS.sigma}={sigma:.2f}{pmstr}{err_sigma:.2f}\n"
    if(fittype=="skewgauss"):
        plottext+=f"{MS.alpha}={alpha:.2f}{pmstr}{err_alpha:.2f}\n"
        plottext+=r"$\chi^{2}$"+f"={chisq:.2f}\n"
        plottext+=r"$\mathrm{N}_{\mathrm{d.o.f.}}$"+f"={ndof:.2f}\n"
        plottext+=r"$\chi^{2}_{\mathrm{Red}}$(auto)"+f"={chired_auto:.2f}\n"
        #plottext+=r"$\chi^{2}_{\mathrm{Red}}$(calc)"+f"={chired:.2f}\n"
        # ok, auto and calc approiaches give EXACTLY the same number (nice)
    if("Electrons" in labels[0] or "TOT" in picname):
        #pnames = result.var_names    
        #cov_idx_mu = pnames.index("mu")
        #cov_idx_sigma = pnames.index("sigma")
        ##cov_sigma_mu = result.covar[cov_idx_mu,cov_idx_sigma]        
        #cov_sigma_mu = result.covar[cov_idx_sigma,cov_idx_mu]        

        #deltaEoverE = (sigma*2.355/mu)*100.0
        #err_deltaEoverE = (np.sqrt((err_mu/mu)**2+2.355*(err_sigma/sigma)**2))*100.0
        deltaEoverE = calcEResolution(mu,sigma)
        #err_deltaEoverE = calcEResolutionCorUnc(mu,sigma,err_mu,err_sigma,cov_sigma_mu)
        err_deltaEoverE = calcEResolutionUnc(mu,sigma,err_mu,err_sigma)

        plottext+=r"$\frac{\mathrm{FWHM}}{\mu}$="+f"{deltaEoverE:.2f}{pmstr}{err_deltaEoverE:.2f}\n"
        deltaEoverE, err_deltaEoverE = None, None

    ax = plt.gca()
    miny,maxy = ax.get_ylim()
    _,xmax = ax.get_xlim()
    #plt.plot(bin_centers[:-1], result.best_fit, '--r')
    plt.plot(bin_centers[fitmask], result.best_fit, '--r')
    plt.text(xmax*0.65, 
            maxy*0.4, 
            plottext)

    if(f_failed):
        plt.text(minbin*1.1, maxy*0.7, "FIT FAILED INITIALLY")

    f_failed = False
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    #plt.yscale('log')
    plt.grid()
    plt.savefig(f"GainRuns-{picname}.png")
    plt.close()

    return fit_pars

########################################################################
fDebug = False
#######################
location = sys.argv[1]
picname = sys.argv[2]
if(len(sys.argv)>3):
    okval = ['debug', '1', 'inspect']
    chars = sys.argv[3]
    if(chars in okval):
        fDebug = True

MU = myUtils()
MC = myColors()
MS = mySymbols()

# will do a propper upgrade with live plotting later...
#app = QtWidgets.QApplication([])
#mymon = myMonitor()
##app.exec()
#mymon.win.show()
##mymon.app.processEvents()
#QtWidgets.QApplication.processEvents()

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

ele_amp, ele_mu, ele_sig = [], [], []
ele_amp_err, ele_mu_err, ele_sig_err = [], [], []
ele_covariances = []

#egain_K, egain_G, egain_theta = []
#egain_errK, egain_errG, egain_errtheta = []

egain_G, egain_errG = [], []

negbin_mean, negbin_var = [], []
negbin_mean_err, negbin_var_err = [], []

Vgrid_list = []
suffixes = []
suflabels = []

totHistList = []
elecHistList = []

hitHistList = []

#
detres, e_detres = [] ,[] 

ref_hits_lab = {"450":[29,6], "460":[37,7], "470":[50,8], "480":[65,8], "490":[78,9], "500":[83,12]}
ref_hits_p09 = {"450":[29,7], "460":[45,11], "470":[64,11], "480":[85,11], "490":[108,11], "500":[128,13]}
#ref_hits_sigma = {"450":6, "460":7, "470":8, "480":8, "490":9, "500":12}

fcnt = 0

for file in inputlist:

    #QtWidgets.QApplication.processEvents()

    rawTOT, electrons = None, None

    TOT, HITS, ELEC = [], [], []
    pruned_tot, pruned_elec = [], []

    with tb.open_file(file) as f:

        print(f"Reading {f}")
        groups = f.walk_groups('/')
        grouplist = []
        for gr in groups:
            #print(f'found {gr}')
            grouplist.append(gr)
        main_group = str(grouplist[len(grouplist)-1])
        #print(f"last entry in walk_groups = \n{main_group}")
        grouplist = None 

        basewords = main_group.split('(')
        #print(basewords)

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
        if(check_node(f,base_group_name+"charge")):
            electrons = f.get_node(base_group_name+"charge")
            print(f"found calibrated charge in {file} ...")
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

        if(electrons is not None):
            for ievent in electrons:
                ELEC.append(np.sum(ievent))
            elecHistList.append([ELEC,Vgrid])

        nevt, cnt = len(x), 0
        
        elecs_iter = None
        if(electrons is None):
            elecs_iter = [None] * len(rawTOT)
        else:
            elecs_iter = electrons

        for ix, iy, itot, ielec in zip(x, y, rawTOT, electrons):

            jelec = None
            if(ielec is None):
                jelec = [None]*len(itot)

            for jx,jy,jtot,jelec in zip(ix,iy,itot, ielec):
               if(jx>=50 and jx<=250 and jy>=50 and jy<=250):
                    pruned_tot.append(jtot)
                    pruned_elec.append(jelec)
            cnt+=1
            print(f"\r[{cnt/nevt*100:.2f}%]", flush=True, end="")
            MU.progress_bar(cnt,nevt)

        sumTOT = None
        hits = None

        tot_charge = np.concatenate(rawTOT)
        chpp_name = None
        if("GHOSTS" in file):
            reass_name = "-".join(words)
            suflabels.append(reass_name)
            chpp_name = f"{reass_name}-"+picname
        else:
            if(electrons is not None):
                elec_charge = np.concatenate(electrons)
                chpp_name = f"{str(Vgrid)}V-ELECTRO-"+picname
            else:
                chpp_name = f"{str(Vgrid)}V-TOT-"+picname

        _,_ =  chargeperpixel(tot_charge, chpp_name, fElectrons=False) 
        if(electrons is not None):
            g, eg = chargeperpixel(elec_charge, chpp_name, fElectrons=True) 
            egain_G.append(g)
            egain_errG.append(eg)
            ## checking if myMonitor works
            #mymon.updateScatter(Vgrid,g)
            #QtWidgets.QApplication.processEvents()


    #print(f"--------------------DONE----------------------")

    #exit(0)

    print(f"Accumulated: {len(HITS)} hits, {len(TOT)} sumTOT entries")   
    print("=============== FITTING TOT ================================")
    #peak_info = simpleHist(TOT, 100,0, 6000, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)
    if("GHOSTS" in file): 
        peak_info = simpleHist(TOT,
                               375,#estimate_nbins_sturge(len(TOT)),#"fd", #200, # apaprently numpy can auto-adjust binning based on data (pass "fd" to bins parameter)
                               [f"TOT per event ({isuff})", "CTS, [TOT cycles]", r"$\mathrm{N}_{\mathrm{tracks}}$"], 
                               f"TOT-{isuff}-"+picname,
                               fittype="skewgauss",
                               debug=fDebug)
    else:
        peak_info = simpleHist(TOT, 
                               375,#estimate_nbins_sturge(len(TOT)),#"fd",#200, 
                               [f"TOT per event (Vgrid={Vgrid}[V])", "CTS, [TOT cycles]", r"$\mathrm{N}_{\mathrm{tracks}}$"], 
                               f"TOT-{Vgrid}-"+picname,
                               fittype="skewgauss",
                               debug=fDebug)

    if(len(ELEC)>0):    
        elecfitinfo = simpleHist(ELEC, 
                                500,
                                [f"Electrons per event (Vgrid={Vgrid}[V])", r"$N_{e^{-}}$", "N_tracks"], 
                                f"ELEC-{Vgrid}-"+picname,
                                fittype="skewgauss",
                                debug=fDebug)

        ele_amp.append(elecfitinfo[0])       
        ele_mu.append(elecfitinfo[2])       
        ele_sig.append(elecfitinfo[4])       
        ele_amp_err.append(elecfitinfo[1])       
        ele_mu_err.append(elecfitinfo[3])       
        ele_sig_err.append(elecfitinfo[5])       
        ele_covariances.append(elecfitinfo[8])
        
        fitpar, fitpar_unc, fit_covar = fitAndPlot(np.array(ELEC),
                                                   250,
                                                   #(np.nanmin(ELEC),np.percentile(ELEC,99)),
                                                   (np.nanmin(ELEC),650000),
                                                   [f"Electrons per event (Vgrid={Vgrid}[V])", r"$N_{e^{-}}$", "N_tracks"], 
                                                   f"ELEC-{Vgrid}-"+picname,
                                                   #plotxlim=[0,1e6],
                                                   fit="skewgauss",
                                                   bincolor='indigo')

        print("Testing \"fitAndPlot\" results")
        parnames = ['A','mu','sig','alpha']
        for name, par, err in zip(parnames, fitpar, fitpar_unc):
            print(f"{name} = {par:.2f} +- {err:.2f}")

        detres.append(calcDetRes(fitpar[1],fitpar[2]))
        covMuSig = fit_covar[2,1]
        e_detres.append(calcDetResUnc(fitpar[1],fitpar[2],fitpar_unc[1],fitpar_unc[2],covMuSig))
               

#    print("=============== FITTING TOT (NegBinDist) ===================")
#    #peak_info = simpleHist(TOT, 100,0, 6000, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)
#    #gain_info = simpleGainPlot(TOT, 100, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)
#    if("GHOSTS" in file):
#        gain_info = simpleGainPlot(np.array(pruned_tot), 100, [f"TOT per event ({isuff})", "TOT cycles", "N"], f"TOT-{isuff}-"+picname)
#    else:
#        gain_info = simpleGainPlot(np.array(pruned_tot), 100, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)
#        elec_gain_info = simpleGainPlot(np.array(pruned_elec), 100, [f"Electrons per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-ELEC-"+picname)
#    #gain_info = simpleGainPlot(rawTOT, 100, [f"TOT per event (Vgrid={Vgrid}[V])", "TOT cycles", "N"], f"TOT-{Vgrid}-"+picname)
#
#    #print("CHINAZES!")
#    #exit(0)
    
    print("=============== FITTING HITS ===============================")
    #hit_info = simpleHist(HITS, 50,0, 150, [f"Hits per event (Vgrid={Vgrid}[V])", r"$N_{HITS}$", "N"], f"HITS-{Vgrid}-"+picname)
    if("GHOSTS" in file):
        hit_info = simpleHist(HITS, 
                                250,#estimate_nbins_sturge(len(HITS)),#"fd",#200,
                                [f"Hits per event ({isuff})", r"$N_{HITS}$", "N"], 
                                f"HITS-{isuff}-"+picname,
                                fittype='skewgauss',
                                debug=fDebug)
    else:
        hit_info = simpleHist(HITS, 
                            250,#estimate_nbins_sturge(len(HITS)),#"fd",#200, 
                            [f"Hits per event (Vgrid={Vgrid}[V])", r"$N_{HITS}$", "N"], 
                            f"HITS-{Vgrid}-"+picname,
                            fittype='skewgauss',
                            debug=fDebug)

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
 
    #negbin_mean.append(gain_info[0])
    #negbin_var.append(gain_info[2])
    #negbin_mean_err.append(gain_info[1])
    #negbin_var_err.append(gain_info[3])

    gain_info = None
    elec_gain_info = None
    hit_info = None
    peak_info = None
    TOT = None
    HITS = None
    rawTOT = None
    electrons = None

    fcnt+=1


# ============ do absolute gain plot if have calibrated charge  data =======================================

if(len(egain_G)>0):
    print("Plotting propper gain vs grid voltage")
    fig,ax = plt.subplots(figsize=(9,8))
    
    ax.errorbar(Vgrid_list,
                egain_G,
                yerr=egain_errG,
                fmt='o',
                color='darkgreen',
                ecolor='forestgreen',
                capsize=4)
    ax.set_title("Amplification Gain vs Grid Voltage")
    ax.set_xlabel(r"$\mathrm{V}_{\mathrm{grid}}$, [V]")
    ax.set_ylabel(r"Gain, [$\mathrm{N}_{e^{-}}$]")
    ax.grid(True)
    ax.set_xlim([440,510])
    plt.savefig(f"GainVsVGrid-{picname}.png", dpi=128)
    plt.close()

# ===================================================
fSuffix = False
if(len(suffixes)>0):
    fSuffix = True
print("======================================================")
print("Hit peak positions per voltage")
if(fSuffix):
    print("Hit peak positions per data set")
    for suf, peak, sig in zip(suffixes, hit_peak_mu, hit_peak_sigma):
        print(f"{suf:.4f} -> {peak:.4f} +- {sig:.4f}")
else:
    print("Hit peak positions per voltage")
    for vg, peak, sig in zip(Vgrid_list, hit_peak_mu, hit_peak_sigma):
        print(f"{vg:.4f} -> {peak:.4f} +- {sig:.4f}")
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
plt.savefig(f"TOT-gainScan-Combined-{picname}.png")
plt.close()

# plotting electron sum per event of exists...
if(len(elecHistList)>0):

    enbins = 1000
    emaxbin = 8e5
    eminbin = 0
    
    plt.figure(figsize=(10,8))
    plt.hist([],weights=[], bins=enbins, range=(eminbin, emaxbin), color='white', label=r"$\mathrm{V}_{\mathrm{Grid}}:$")
    for data in elecHistList:
        counts, bin_edges = np.histogram(data[0], bins=enbins, range=(eminbin,emaxbin))
        weights = counts/sum(counts)
        plt.hist(bin_edges[:-1], weights=weights, bins=enbins, range=(eminbin, emaxbin), align='left', histtype='stepfilled', alpha=0.2, label=f"{data[1]} [V]")
        
    plt.title(r"$\sum(e^{-})$ per Track")
    plt.xlabel(r"$N_{e^{-}}$")
    plt.ylabel(r"$N_{tracks}$")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f"SUM-ELEC-gainScan-{picname}.png")
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
plt.savefig(f"HITS-gainScan-Combined-{picname}.png")
plt.close()

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
    #ax.scatter([int(i) for i in range(len(suffixes))], negbin_mean, marker='x', c='m', label="Neg.Bin.Dist. means")
else:
    ax.errorbar(Vgrid_list, peak_mu, peak_mu_err, fmt='o', mfc='green', linewidth=2, capsize=6, label="Gauss means")
    #ax.scatter(Vgrid_list, negbin_mean, marker='x', c='m', label="Neg.Bin.Dist. means")

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
# plotting gain curve with electron data (calibrated charge)
#####################################################

fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar(Vgrid_list,ele_mu,yerr=ele_mu_err,fmt='o',color='darkblue',ecolor='royalblue',capsize=4)
ax.set_title("GridPix3 - Gain Curve")
ax.set_xlabel(r"$\mathrm{V}_{\mathrm{grid}}$, [V]")
ax.set_ylabel(r"Fit $\mu$ ($\sum$(e) per track)")
ax.set_xlim([xmin,xmax])
ax.grid(True)
plt.savefig(f"GainCurve-Electrons-{picname}.png",dpi=400)
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
    print(f" {m:.4f} +- {em:.4f} -> {h:.4f} +- {eh:.4f}")

##### plotin' hits vs charge ######

fig, ax = plt.subplots(figsize=(9,8))
ax.errorbar(hit_peak_mu, 
            ele_mu, 
            yerr=ele_mu_err, 
            xerr=hit_peak_mu_err,
            fmt='s',
            color='darkorange',
            ecolor='goldenrod',
            capsize=4)
ax.set_title(r"Electron peak $\mu$ vs Hit peak $\mu$")
ax.set_xlabel(r"$\mu_{\mathrm{Hits}}, [\mathrm{n}]$")
ax.set_ylabel(r"$\mu_{\mathrm{Electrons}}, [\mathrm{n}_\mathrm{e}^{-}]$")
ax.grid()
plt.savefig(f"MuElec-vs-MuHits-{picname}.png")

###################################################
E_res = []
E_res_err = []
cnt = 0
for mu in peak_mu:
    res = peak_sigma[cnt]/mu
    res_err = np.sqrt((peak_mu_err[cnt]/mu)**2 + (peak_sigma_err[cnt]/peak_sigma[cnt])**2)
    #print(f"calculating resolution => item={cnt}, {mu:.4f} +- {peak_mu_err[cnt]:.4f}, {peak_sigma[cnt]:.4f} +- {peak_sigma_err[cnt]:.4f}")
    #print(f"{cnt}: [PEAK-MEAN]={mu:.4f} +- {peak_mu_err[cnt]:.4f}, [PEAK-SIGMA]={peak_sigma[cnt]:.4f} +- {peak_sigma_err[cnt]:.4f}")
    print(f"{cnt}: peak_{MS.mu}={mu:.4f} +- {peak_mu_err[cnt]:.4f}, peak_{MS.sigma}={peak_sigma[cnt]:.4f} +- {peak_sigma_err[cnt]:.4f}")
    E_res.append(res*100)
    E_res_err.append(res_err*100)
    cnt+=1

#print(len(E_res))
#print(E_res)
#print(len(E_res_err))
#print(E_res_err)
#print(len(Vgrid_list))
#print(Vgrid_list)

# PLOTTING ENERGY RES FRO ELECTRON FITS AND SKEWED FITS
elec_Eres, elec_EresErr = [], []
for mu, emu, sig, esig, covar in zip(ele_mu, ele_mu_err, ele_sig, ele_sig_err,ele_covariances):
    #fwhm = getFWHM(sig)
    #ires = fwhm/mu
    ires = calcEResolution(mu,sig)
    #ireserr = ires*np.sqrt((emu/mu)**2+(esig/sig)**2)*100.0
    #ireserr = calcEResolutionCorUnc(mu,sig,emu,esig,covar)
    ireserr = calcEResolutionUnc(mu,sig,emu,esig)
    #ireserr = calcER
    elec_Eres.append(ires)    
    elec_EresErr.append(ireserr)    

plt.figure(figsize=(8,6))
plt.errorbar(Vgrid_list,
             elec_Eres,
             yerr=elec_EresErr,
             fmt='o',
             color='crimson',
             ecolor='red',
             linewidth=2,
             capsize=4)
plt.title(f"Energy Resolution "+r"$\frac{\mathrm{FWHM}}{\mu}$"+"(electron-based)")
plt.xlabel(r"$V_{\mathrm{grid}}$,[V]")
plt.ylabel("FWHM(E)/E,[%]")
plt.grid()
plt.savefig(f"Energy_resolutions-FWHM_electrons-{picname}.png")
plt.close()

###### 
# E resolution based on pure sigma/mu (! NOT FWHM/E) #########
######
plt.figure(figsize=(8,6))
plt.errorbar(Vgrid_list,
             detres,
             yerr=e_detres,
             fmt='o',
             color='indigo',
             ecolor='purple',
             linewidth=2,
             capsize=4)
plt.title(f"Energy Resolution "+r"$\frac{\sigma}{\mu}$"+" (electron-based)")
plt.xlabel(r"$V_{\mathrm{grid}}$,[V]")
plt.ylabel(r"$\sigma$E/E,[%]")
plt.grid()
plt.savefig(f"Energy_resolutions-SIGMA_electrons-{picname}.png")
plt.close()


###################################################################################

if(fSuffix):
    print("=============================================")
    for suf, res, eres in zip(suffixes,E_res,E_res_err):
        print(f"{suf} | {res:.2f}% +- {eres:.2f}%")
    print("=============================================")
else:
    print("============== TOT-based ===============================")
    for v, res, eres in zip(Vgrid_list,E_res,E_res_err):
        print(f"Vgrid={v} | {res:.2f}% +- {eres:.2f}%")
    print("=============================================")
    v, res, eres = None, None, None
    print("============== Electron-based ===============================")
    for v, res, eres in zip(Vgrid_list,elec_Eres,elec_EresErr):
        print(f"Vgrid={v} | {res:.2f}% +- {eres:.2f}%")
    print("=============================================")


fig, ax = plt.subplots(figsize=(8,6))
if(fSuffix):
    ax.errorbar([int(i) for i in range(len(suffixes))], E_res, E_res_err, fmt='o', linewidth=2, capsize=6)
    ax.set_xlabel("Data Set")
    ax.set_title("Energy Resolution")
else:
    ax.errorbar(Vgrid_list, 
                E_res, 
                E_res_err, 
                fmt='o', 
                color='darkblue',
                ecolor='royalblue',
                linewidth=2, 
                capsize=6)
    ax.set_xlabel("Grid voltage, [V]")
    ax.set_title("Energy resolution vs Grid voltage "+r"($\frac{\sigma}{\mu}$, TOT-based)")
ax.set_ylabel(r"$\sigma_{\mathrm{E}}$/E, [%]")
if(not fSuffix):
    ax.set_xlim([xmin, xmax])
plt.grid()
plt.savefig(f"Energy_resolutions-TOT-based-{picname}.png")
plt.close()

