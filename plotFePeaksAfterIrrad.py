import numpy as np
import sys
import glob
import tables as tb
import matplotlib.pyplot as plt
from MyPlotter import myUtils  
from lmfit import Model

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'


def gauss(x, A, mu, sigma):

    return A*np.exp(-((x-mu)**2)/(2*sigma**2))


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

def simpleHistSpectrum(nuarray, nbins, minbin, maxbin, labels, picname, scale=None):

    plt.figure()
  
    plt.figsize=(8,8)

    counts, bin_edges = np.histogram(nuarray, bins=nbins, range=(minbin,maxbin))
    maxbin_cnt = np.max(counts)
    minbin_cnt = np.min(counts)

    val_ibin = (maxbin-minbin)/nbins

    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    print(f"Range= {minbin} -> {maxbin}, with {nbins} bins gives {val_ibin} per bin")

    model = None

    peakbin = getMaxBin(counts[1:])
    peakbin+=1

    print(f"Found maximum bin at {peakbin} = {counts[peakbin]}")
    model = Model(gauss) 
    pars = model.make_params(A=maxbin_cnt, mu=peakbin*val_ibin, sigma=np.std(counts))
    print(f"Gauss Fit: Setting: A={maxbin_cnt}, mu={peakbin*val_ibin}, sigma={np.std(counts)}")   
 
    print(len(bin_centers[:-1]))
    print(len(counts))
    plt.hist(bin_centers, weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', facecolor='b')

    print(f"\n\ncounts={counts} \n\n")
    print(maxbin_cnt)
    print(minbin_cnt)
    print(f"max_amplitude={maxbin_cnt - minbin_cnt}")
    print("########## FITTING GASUSS FUNCTION #############")
    result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
    print(result.fit_report()) 
    if(result.params['mu']<=0):

        pars['mu'].min = peakbin*0.8*val_ibin
        pars['mu'].max = peakbin*1.2*val_ibin
 
        print("FIT FAILED: restricting fit parameters and re-fitting")
        result = model.fit(counts[:-1], pars, x=bin_centers[:-1] )

        print(result.fit_report()) 

    fitlab = ""
    miny,maxy = 0, 0
    A = result.params["A"].value
    mu = result.params["mu"].value
    sigma = result.params["sigma"].value 
    ax = plt.gca()
    miny,maxy = ax.get_ylim()
    plt.plot(bin_centers[:-1], result.best_fit, '--r')
    plt.text((maxbin/2.0)*1.5, maxy*0.9, f"A={round(A,2)}")
    plt.text((maxbin/2.0)*1.5, maxy*0.85, f"{G_mu}={round(mu,2)}")
    plt.text((maxbin/2.0)*1.5, maxy*0.8, f"{G_sigma}={round(sigma,2)}")

    #plt.legend(loc='upper right')

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    if(scale is not None):
        plt.yscale(scale)
    plt.grid()
    plt.savefig(f"GausFit-{picname}.png")
    plt.savefig(f"GausFit-{picname}.pdf")

    plt.close() 

def getData(file):

    dlist = []
    hitlist = []
    hits_reduced = []

    with tb.open_file(file, 'r') as f: 
    
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
    
        sumTOT = f.get_node(base_group_name+"sumTot")
        print(f"found sumTOT {type(sumTOT)}")
        hits = f.get_node(base_group_name+"hits")
        print(f"found hits {type(hits)}")

        #x = f.get_node(base_group_name+"x")
        #print(f"found x {type(x)}")
        #y = f.get_node(base_group_name+"y")
        #print(f"found y {type(y)}")

        ievt = 0    
        for event in sumTOT:
            dlist.append(event)        
            hitlist.append(hits[ievt])
            hits_reduced.append(event/hits[ievt])
            ievt+=1

    f.close()

    return dlist, hitlist, hits_reduced


################################################
picname = sys.argv[1]

main_path = "/home/vlad/readoutSW/tpx3-anal/TimepixAnalysis/"
fbefore = "reco-ParProtBeam-006663-Fe55-BeforeIrrad-23Jul2025-Morning.h5"
fafter1 = "reco-ParProtBeam-006723-AfterFirstIrrad-Fe55-24Jul2025-Morning.h5"
fafter2 = "reco-ParProtBeam-AfterSecondIrrad-Fe55-24Jul2025-Evening.h5"

TOT_before, hits_before, redhits_before = getData(main_path+fbefore)
TOT_after1, hits_after1, redhits_after1 = getData(main_path+fafter1)
TOT_after2, hits_after2, redhits_after2 = getData(main_path+fafter2)

simpleHistSpectrum(np.array(TOT_before), 
                    101, 
                    0, 
                    60000, 
                    [r'$^{55}$Fe X-Ray Charge Spectrum (Before Irradiation)', 'TOT cycles','N'], 
                    "Fe55BeforeIrradiation", 
                    scale="linear")
simpleHistSpectrum(np.array(TOT_after1), 
                    101, 
                    0, 
                    60000, 
                    [r'$^{55}$Fe X-Ray Charge Spectrum (After Run1)', 'TOT cycles','N'], 
                    "Fe55AfterRun1", 
                    scale="linear")
simpleHistSpectrum(np.array(TOT_after2), 
                    101, 
                    0, 
                    60000, 
                    [r'$^{55}$Fe X-Ray Charge Spectrum (After Run2)', 'TOT cycles','N'], 
                    "Fe55AfterRun2", 
                    scale="linear")
# ------------- plotting hits here --------------------
simpleHistSpectrum(np.array(hits_before), 
                    100, 
                    0, 
                    400, 
                    [r'$^{55}$Fe X-Ray Hits per Frame (Before Irradiation)', r'$N_{hits}$',r'$N_{frames}$'], 
                    "HITS-Fe55BeforeIrradiation", 
                    scale="linear")
simpleHistSpectrum(np.array(hits_after1), 
                    100, 
                    0, 
                    400, 
                    [r'$^{55}$Fe X-Ray Hits per Frame (Wednesday evening)', r'$N_{hits}$',r'$N_{frames}$'], 
                    "HITS-Fe55AfterRun1", 
                    scale="linear")
simpleHistSpectrum(np.array(hits_after2), 
                    100, 
                    0, 
                    400, 
                    [r'$^{55}$Fe X-Ray Hits per Frame (Thursday Evening)', r'$N_{hits}$',r'$N_{frames}$'], 
                    "HITS-Fe55AfterRun2", 
                    scale="linear")

# ------------- plotting reduced hits here --------------------
simpleHistSpectrum(np.array(redhits_before), 
                    100, 
                    0, 
                    800, 
                    [r'$^{55}$Fe X-Ray Reduced Hits per Frame (Before Irradiation)', r'$N_{hits}$',r'$N_{frames}$'], 
                    "REDHITS-Fe55BeforeIrradiation", 
                    scale="linear")
simpleHistSpectrum(np.array(redhits_after1), 
                    100, 
                    0, 
                    800, 
                    [r'$^{55}$Fe X-Ray Reduced Hits per Frame (Wednesday evening)', r'$N_{hits}$',r'$N_{frames}$'], 
                    "REDHITS-Fe55AfterRun1", 
                    scale="linear")
simpleHistSpectrum(np.array(redhits_after2), 
                    100, 
                    0, 
                    800, 
                    [r'$^{55}$Fe X-Ray Reduced Hits per Frame (Thursday Evening)', r'$N_{hits}$',r'$N_{frames}$'], 
                    "REDHITS-Fe55AfterRun2", 
                    scale="linear")

# =======================================================================================

minbin, maxbin = 0, 60000

plt.figure(figsize=(8,6))
plt.hist(np.array(TOT_before), 100, range=(minbin,maxbin), alpha=0.2, label="Before Irradiation")
plt.hist(np.array(TOT_after1), 100, range=(minbin,maxbin), alpha=0.2, label="After run 1")
plt.hist(np.array(TOT_after2), 100, range=(minbin,maxbin), alpha=0.2, label="After run2")
plt.title(r"Uncalibrated Charge Spectrum of $^{55}$Fe X-Rays (Before and After Proton Beam)")
plt.xlabel("TOT cycles")
plt.ylabel("CTS")
plt.legend()
plt.savefig(f"CombinedFe55Spectrum-{picname}.png")
plt.savefig(f"CombinedFe55Spectrum-{picname}.pdf")

#======= combined reduced hit spectra ===============
plt.figure(figsize=(8,6))
plt.hist(np.array(redhits_before), 100, range=(0,1000), alpha=0.2, label="Before Irradiation")
plt.hist(np.array(redhits_after1), 100, range=(0,1000), alpha=0.2, label="After run 1")
plt.hist(np.array(redhits_after2), 100, range=(0,1000), alpha=0.2, label="After run2")
plt.title(r"Reduced Hit Spectrum of $^{55}$Fe X-Rays (Before and After Proton Beam)")
plt.xlabel(r"$\Sigma$(TOT)/$N_{hits}$")
plt.ylabel("CTS")
plt.legend()
plt.savefig(f"CombinedFe55-RedHits-Spectrum-{picname}.png")
plt.savefig(f"CombinedFe55-RedHits-Spectrum-{picname}.pdf")




    
