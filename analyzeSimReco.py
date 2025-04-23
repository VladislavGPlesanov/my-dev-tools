import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tables as tb
from sklearn.cluster import DBSCAN 
from scipy.optimize import curve_fit
from lmfit import Model


###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

######
def cosfunc(x, Ap, Bp, phi0):

    return Ap + Bp*np.cos(x - phi0)**2

def gauss(x, A, mu, sigma):

    return A*np.exp(-((x-mu)**2)/(2*sigma**2))

def modFac(Ap,Bp):

    return Bp/(2*Ap+Bp)

def polDeg(Ap, Bp, mu):

    return (1/mu)*(Bp/((2*Ap)+Bp))

def check_node(gr_path, file):

    return gr_path in file

def toDegrees(angle):

    return angle*180.0/3.14159 

def toRadian(angle):

    return angle*3.14159/180

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

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)

def getClusterCenter(matrix):

    x,y = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), indexing='ij')
    
    norm_factor = matrix.sum()

    mean_x = (matrix * x).sum() / norm_factor
    mean_y = (matrix * y).sum() / norm_factor

    std_x = np.sqrt( (matrix * (x - mean_x)**2).sum() / norm_factor)
    std_y = np.sqrt( (matrix * (y - mean_y)**2).sum() / norm_factor)

    return mean_x, mean_y, std_x, std_y


def multiHist(nuarray_list, nbins, binbounds, labels, legends, logy ,picname, odir):

    plt.figure()

    cnt = 0 
    for arr in nuarray_list:
        plt.hist(arr, nbins, range=(binbounds[0],binbounds[1]), alpha=0.125, label=f"{legends[cnt]}")
        cnt+=1

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    if(logy):
        plt.yscale('log')
    plt.legend()
    plt.savefig(f"{odir}/multiHist-{picname}.png")   

def simpleHist(nuarray, nbins, minbin, maxbin, labels, picname, odir,fit=None):

    plt.figure()

    counts, bin_edges = np.histogram(nuarray, bins=nbins, range=(minbin,maxbin))

    #plt.hist(nuarray, nbins, range=(minbin,maxbin), histtype='stepfilled', facecolor='b')
    plt.hist(bin_edges[:-1], weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', facecolor='b')
    if(fit):
        plt.figsize=(8,8)
        maxbin_cnt = np.max(counts)
        minbin_cnt = np.min(counts)
        #minval, maxval = max, counts[len(counts)-1]
        val_ibin = (maxbin-minbin)/nbins
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

        print(f"Range= {minbin} -> {maxbin}, with {nbins} bins gives {val_ibin} per bin")

        ########################################################
        model = None
        if(fit=="gaus"): 
            peakbin = getMaxBin(counts[1:])
            peakbin+=1

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

        elif(fit=="cosfunc"):            
            model = Model(cosfunc)
            pars = model.make_params(Ap=minbin_cnt*0.9,Bp=maxbin_cnt*1.1,phi0=1.5)
            pars['Ap'].min = minbin_cnt*0.8
            #pars['Ap'].max = minbin*1.5
            #pars['Bp'].min = minbin_cnt*1.1
            #pars['Bp'].max = maxbin*1.5
            pars['phi0'].min = 0.0
            pars['phi0'].max = 3.14

        print(f"\n\ncounts={counts} \n\n")
        print(maxbin_cnt)
        print(minbin_cnt)
        print(f"max_amplitude={maxbin_cnt - minbin_cnt}")

        if(fit=="cosfunc"):
            print("########## FITTING COS^2 FUNCTION #############")
        if(fit=="gaus"):
            print("########## FITTING GASUSS FUNCTION #############")
        result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
        print(result.fit_report()) 
        if(fit=="gaus" and result.params['mu']<=0):

            #pars['A'].min = maxbin_cnt*0.8
            #pars['A'].max = maxbin_cnt*1.2

            pars['mu'].min = peakbin*0.8*val_ibin
            pars['mu'].max = peakbin*1.2*val_ibin
 
            #pars['sigma'].min = np.std(counts)*0.8
            #pars['sigma'].max = np.std(counts)*1.2

            print("FIT FAILED: restricting fit parameters and re-fitting")
            result = model.fit(counts[:-1], pars, x=bin_centers[:-1] )
            #result = model.fit(counts[peakbin-10:peakbin+10], pars, x=bin_centers[peakbin-10:peakbin+10])

            print(result.fit_report()) 
        ########################################################            

        fitlab = ""
        miny,maxy = 0, 0
        if(fit=="cosfunc"):
            #print("########## FITTING COS^2 FUNCTION #############")
            Ap = result.params['Ap'].value
            Bp = Ap + result.params['Bp'].value
            phi = result.params['phi0'].value
            fitlab+=f"Ap={round(Ap,2)}, Bp={round(Bp,2)}, {G_phi}={round(phi,2)}"
            plt.plot(bin_centers[:-1], result.best_fit, '--r')
            #plt.plot(bin_edges[:-1], result.best_fit, '--r')
            #plt.hlines(Ap, -0.1, 3.15, colors='g', label=f"Ap={round(Ap,2)}")
            #plt.hlines(Bp, -0.1, 3.15, colors='y', label=f"Bp={round(Bp,2)}")
            #plt.vlines(phi, 0, np.max(counts)*1.2, colors='m',label=f"{G_phi}={round(phi,2)}")
            #plt.vlines(phi, 0, np.max(counts)*1.2, colors='m',label=f"{G_phi}={round(phi,2)}({round(toDegrees(phi),2)} deg)")
            ax = plt.gca()
            miny,maxy = ax.get_ylim()
            mu = modFac(Ap,Bp)
            P = polDeg(Ap,Bp,mu)
            plt.text(0.05, maxy*0.9, f"{G_mu}={round(mu*100,2)}%, P={round(P,2)}")
            plt.text(0.05, maxy*0.95, r"$N(\phi) = A_{P} + B_{P}\cdot cos^2(\phi-\phi_{0})$")
        elif(fit=="gaus"):
            #print("########## FITTING GAUS FUNCTION #############")
            A = result.params["A"].value
            mu = result.params["mu"].value
            sigma = result.params["sigma"].value 
            ax = plt.gca()
            miny,maxy = ax.get_ylim()
            #plt.yscale('log')
            plt.plot(bin_centers[:-1], result.best_fit, '--r')
            plt.text(minbin*1.1, maxy*0.9, f"A={round(A,2)}, {G_mu}={round(mu,2)}, {G_sigma}={round(sigma,2)}")

        plt.legend(loc='upper right')


    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    #plt.yscale('log')
    plt.grid()
    plt.savefig(f"{odir}/1DHist-{picname}.png")


def plot2Dhist(x,y,labels,picname,odir):
    
    plt.figure()

    #hist,xedges,yedges = np.histogram2d(x,y,bins=100)

    #plt.imshow(hist.T, 
    #            origin="lower", 
    #            cmap="plasma", 
    #            extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    #plt.colorbar(label="cnt")
    plt.hist2d(x,y,bins=(51,51),cmap="plasma")

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.xlim(0,2)
    plt.ylim(0,1)
    plt.plot()
    plt.savefig(f"{odir}/2Dhist-{picname}.png")


def plotDbscan(index, dblabels, ievent, nfound, odir):

    #plt.figure(figsize=(8,8))
    plt.figure()
    plt.scatter(index[:, 0], index[:, 1], c=labels, cmap='plasma', s=10)
    plt.gca().invert_yaxis() 
    plt.gca().invert_xaxis() 
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0,64)
    plt.ylim(0,64)
    plt.title(f"DBSCAN on smeared matrix event {ievent}")
    plt.text(2,55 , f"n hits = {len(index)},\n ncluster={nfound}", fontsize=10, color="black")
    plt.colorbar(label='Cluster Label')
    plt.savefig(f"{odir}/DBSCAN-{ievent}.png")


def plot2dEvent(nuarray, info, picname, odir, plotMarker=None):
 
    # ---- matrix 2d hist ----
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.86,0.1,0.05,0.8])
    #ms = ax.matshow(nuarray, cmap='plasma')
    ms = None
    if("total" in picname):
        ms = ax.matshow(nuarray.T, cmap='hot')
    else:
        ms = ax.matshow(nuarray, cmap='hot')
    fig.colorbar(ms,cax=cax,orientation='vertical')
    start = 120
    #ax.text(-90, start, info, fontsize=10,color='black' )    
    #if(plotMarker is not None):
    #    ax.scatter(plotMarker[0],plotMarker[1],c='blue', marker='*')
    #    ax.text(-90,200, f"redX={plotMarker[0]}\nredY={plotMarker[1]}", fontsize=10, color='black')

    if("total" in picname):
        ax.invert_yaxis()

    plt.plot()
    fig.savefig(f"{odir}/reco-event-{picname}.png")
    plt.close()


def other2Dplot(nuarray, picname, odir):

    #non_zero_index = np.nonzero(nuarray)

    #xmin, xmax = non_zero_index[0].min(), non_zero_index[0].max()
    #ymin, ymax = non_zero_index[1].min(), non_zero_index[1].max()

    #ROI = nuarray[xmin-10:xmax+10, ymin-10:ymax+10]
    ####################################################################

    mean_x, mean_y, std_x, std_y = getClusterCenter(nuarray)
    #print("{},{}: {},{}".format(mean_x,mean_y,std_x,std_y))

    maxdev = 0
    if(std_x > std_y):
        maxdev = np.floor(std_x)
    else:
        maxdev = np.floor(std_y)

    xmin = int(np.floor(mean_x) - 4*maxdev) 
    xmax = int(np.floor(mean_x) + 4*maxdev)

    ymin = int(np.floor(mean_y) - 4*maxdev)
    ymax = int(np.floor(mean_y) + 4*maxdev)

    ROI = nuarray[xmin-2:xmax+2, ymin-2:ymax+2]
 
    #print("\n{},{}: {},{}\n".format(xmin,xmax,ymin,ymax))
    #ROI = nuarray[xmin:xmax, ymin:ymax]

    plt.figure(figsize=(8,8))

    #plt.imshow(ROI, cmap='viridis', origin='lower', extent=(ymin,ymax,xmin,xmax))
    plt.imshow(ROI, cmap='turbo', origin='lower', extent=(ymin,ymax,xmin,xmax))
    plt.gca().invert_yaxis() 
    plt.colorbar(orientation='vertical', label='Counts')

    #plt.title('Zoom-in')    
    plt.xlabel('x')
    plt.ylabel('y')
 
    plt.plot()
    #if("-285-" in picname):
    plt.savefig(f"{odir}/reco-zoom-event-{picname}.png")
    #   plt.savefig(f"reco-zoom-event-{picname}.pdf")
    #else:
    #   plt.savefig(f"{odir}/reco-zoom-event-{picname}.png", dpi=200)

def simpleSimRecoHist(data, nbins, minbin, maxbin, labels, picname, ylog, odir):

    plt.figure() 

    plt.hist(data, nbins, range=(minbin,maxbin), histtype='step', facecolor='b')
 
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
 
    if(ylog):                                                                                       
        plt.yscale('log')
                
    plt.savefig(f"{odir}/simpleSimRecoHist-{picname}.png")                                   


#####################################################################
recofile = sys.argv[1]
plotname = sys.argv[2]

outdir = f"tmp-{plotname}/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

# valid for NeCO2-45deg run only
#empty_ev = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
#prop_ev =[41, 42, 43, 44, 45, 46, 47, 183, 184, 185, 186, 193, 205, 206, 207, 322, 352, 366, 383, 384, 385, 386, 387, 583, 584, 585, 586, 587, 747, 748] 

#good_events = []

tot_list = np.zeros(51, dtype=np.int64)
tot_edges = np.linspace(0,1000, 51+1)

alt_rotang = []    
tot_reduced = []   
TOTarray=[]

with tb.open_file(recofile, 'r') as f:
   
    groups = f.walk_groups('/')
    grouplist = []
    for gr in groups:
        print(f'found {gr}')
        grouplist.append(gr)
    main_group = str(grouplist[len(grouplist)-1])
    print(f"last entry in walk_groups = \n{main_group}")
    #print(grouplist)
    #groupname_parts = grouplist[0].split('/')
    #print(groupname_parts)

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

    #exit(0)
    basewords = None
    # =========== gettin' shit ==================
    #base_group_name = '/reconstruction/run_9999/chip_0/'
    #base_group_name = '/reconstruction/run_1868/chip_0/'
    
    #comToA = f.get_node(base_group_name+"ToACombined")
    #print(f"found VLarray TOACombined of size {type(comToA)}")
    #ToA = f.get_node(base_group_name+"ToA")
    #print(f"found VLarray TOT of size {type(ToA)}")
    ToT = f.get_node(base_group_name+"ToT")
    print(f"found VLarray TOT of size {type(ToT)}")
    centerX = f.get_node(base_group_name+"centerX")
    print(f"found centerX {type(centerX)}")
    centerY = f.get_node(base_group_name+"centerY")
    print(f"found centerY {type(centerY)}")
    hits = f.get_node(base_group_name+"hits")
    print(f"found hits {type(hits)}")
    excent = f.get_node(base_group_name+"eccentricity")
    print(f"found excentricity {type(excent)}")
    FIRT = f.get_node(base_group_name+"fractionInTransverseRms")
    print(f"found FIRT {type(FIRT)}")
    kurtosisL = f.get_node(base_group_name+"kurtosisLongitudinal")
    print(f"found kurtsisL {type(kurtosisL)}")
    kurtosisT = f.get_node(base_group_name+"kurtosisTransverse")
    print(f"found kurtosisT {type(kurtosisT)}")
    length = f.get_node(base_group_name+"length")
    print(f"found length {type(length)}")
    LDRT = f.get_node(base_group_name+"lengthDivRmsTrans")
    print(f"found LDRT {type(LDRT)}")
    RMS_L = f.get_node(base_group_name+"rmsLongitudinal")
    print(f"found RMS_L {type(RMS_L)}")
    RMS_T = f.get_node(base_group_name+"rmsTransverse")
    print(f"found RMS_T {type(RMS_T)}")
    rotAng = f.get_node(base_group_name+"rotationAngle")
    print(f"found rotAng {type(rotAng)}")
    skewL = f.get_node(base_group_name+"skewnessLongitudinal")
    print(f"found skewL {type(skewL)}")
    skewT = f.get_node(base_group_name+"skewnessTransverse")
    print(f"found skewT {type(skewT)}")
    sumTOT = f.get_node(base_group_name+"sumTot")
    print(f"found sumTOT {type(sumTOT)}")
    width = f.get_node(base_group_name+"width")
    print(f"found width {type(width)}")
    x = f.get_node(base_group_name+"x")
    print(f"found x {type(x)}")
    y = f.get_node(base_group_name+"y")
    print(f"found y {type(y)}")

    #print(f"centerX is {type(centerX)}, centerY is {type(centerY)}")    
    #print(f"centerX shape={centerX.shape}")
    #print(f"first ten centerX={centerX[0:10]}")
    #print(f"centerY shape={centerY.shape}")
    #print(f"first ten centerY={centerY[0:10]}")

    #print(f"ToT is {type(ToT)}")
    #print(f"x is {type(x)}")
    #print(f"y is {type(y)}")

    n_x = len(x)
    print(f"rate = {n_x/60} per second")

    #print("cehcking some variable length arrays:")
    #for i in range(5):
    #    print(f"x={x[i]}, y={y[i]}, TOT={ToT[i]}")

    # =========== plottin' shit =================

    print("Printing global variable data sets")
    #simpleSimRecoHist(ToT.reshape(-1), 51, np.nanmin(ToT.reshape(-1)), np.nanmax(ToT.reshape(-1)), ['TOT_Cycles','TOT','N'], plotname+"-TOT", False, outdir)
    simpleSimRecoHist(centerX, 50, 6, 8, ['centerX','x','cnt'], plotname+"centerX", False, outdir)
    simpleSimRecoHist(centerY, 50, 6, 8, ['centerY','Y','cnt'], plotname+"centerY", False, outdir)
    # - -------------------------------------------------------------------------
    #simpleSimRecoHist(excent, 50, 0.9, 10 , ['Excentricity','','cnt'], plotname+"excent", True, outdir)
    #simpleSimRecoHist(FIRT, 50, np.min(FIRT), np.max(FIRT)+0.5, ['frac_In_rms_T','','cnt'], plotname+"fracInRmsTrans", True, outdir)
    #simpleSimRecoHist(kurtosisL, 50, np.nanmin(kurtosisL), np.nanmax(kurtosisL), ['kurtosisL','','cnt'], plotname+"kurtL", True, outdir)
    #simpleSimRecoHist(kurtosisT, 50, np.nanmin(kurtosisT), np.nanmax(kurtosisT), ['kurtosisT','','cnt'], plotname+"kurtT", True, outdir)
    #simpleSimRecoHist(length, 50, np.min(length), np.max(length), ['length','','cnt'], plotname+"length", False, outdir)
    #simpleSimRecoHist(length, 50, np.min(length), 6, ['length','','cnt'], plotname+"length", False, outdir)

    simpleSimRecoHist(hits, 100, np.min(hits), np.max(hits), ['hits','','cnt'], plotname+"hits", False, outdir)
    #simpleSimRecoHist(hits, 50, np.min(hits), 500, ['hits','','cnt'], plotname+"hits", False, outdir)
    #simpleSimRecoHist(hits, 100, np.min(hits), 200, ['hits','','cnt'], plotname+"hits", False, outdir)

    #simpleSimRecoHist(LDRT, 50,0,50, ['lenDivRmsTrans','','cnt'], plotname+"LDRT", True, outdir)

    #simpleSimRecoHist(RMS_L, 50,np.min(RMS_L), 2.0, ['RMS_L','',''], plotname+"RMS_L", False, outdir)
    #simpleSimRecoHist(RMS_T, 50,np.min(RMS_T), 1.5, ['RMS_T','',''], plotname+"RMS_T", False, outdir)

    #simpleSimRecoHist(rotAng*180/3.14159, 50,0,360, ['rotAng','',''], plotname+"rotAng", False, outdir)

    simpleSimRecoHist(rotAng, 50,np.min(rotAng),np.max(rotAng), ['rotAng','',''], plotname+"rotAng", False, outdir)
    #simpleSimRecoHist(skewL, 50,np.nanmin(skewL),np.nanmax(skewL), ['skewL','',''], plotname+"skewL", True, outdir)
    #simpleSimRecoHist(skewT, 50,np.nanmin(skewT),np.nanmax(skewT), ['skewT','',''], plotname+"skewT", True, outdir)
    #simpleSimRecoHist(sumTOT, 100 , 0, 6000, ['Total charge per event','TOT, [n cycles]','N'], plotname+"sumTOT", False, outdir)
    simpleSimRecoHist(sumTOT, 100,np.min(sumTOT),np.max(sumTOT), ['sumTOT','',''], plotname+"sumTOT", False, outdir)
    #simpleSimRecoHist(sumTOT, 100,np.min(sumTOT),40000, ['Total charge per event','TOT cycles','N'], plotname+"-sumTOT", False, outdir)
    #simpleSimRecoHist(sumTOT, 50, 0 ,7000, ['sumTOT','',''], plotname+"sumTOT", False, outdir)
    #simpleSimRecoHist(width, 50,np.min(width), 2, ['width','',''], plotname+"width", False, outdir)
    # -----------------------------------------------------------------------------
    print("---------- CHINAZES! ------------")

    print(f"\nFile has {len(hits)} clusters\n")

    #exit(0)

    mean_glob_x = np.mean(centerX)   
    mean_glob_y = np.mean(centerY)   
    std_centerX = np.std(centerX)
    std_centerY = np.std(centerY)

    print(f"Mean position of the \"beam\" x={mean_glob_x},y={mean_glob_y}")
    print(f"sigmmas of the \"beamspot\" {round(std_centerX,2)}, {round(std_centerY,2)}")
    #plot2Dhist(excent.reshape(-1),FIRT.reshape(-1), ["\u03B5 vs FIRT","excentricity","FIRT"], "epsilon-vs-FIRT", outdir)    

    #exit(0)
    ################################################################
    ################################################################
    ################################################################

    if(check_node(base_group_name+"angle_firststage", f)):
        print("Found reconstruction data!")
        
        firstangles = f.get_node(base_group_name+"angle_firststage")[:].T
        secondangles = f.get_node(base_group_name+"angle_secondstage")[:].T

        multiHist([firstangles,secondangles], 
                    62, 
                    [0.0,3.15],
                    ['xray angles','Angles [radian]','#'],
                    ['stage1','stage2'],
                    False,
                    f"RecoAngles-{plotname}",
                    outdir) 

        simpleHist(firstangles, 
                    31, 
                    0.0,
                    3.15,
                    ["Reco angles first stage", "Angle,[radian]", "N"],
                    "XPolFirstStage",
                    outdir,
                    fit="cosfunc")

        simpleHist(secondangles, 
                    31, 
                    0.0,
                    3.15,
                    ["Reco angles second stage", "Angle,[radian]", "N"],
                    "XPolSecondStage",
                    outdir,
                    fit="cosfunc")


        #exit(0)

    #matrix = np.zeros((256,256),dtype=np.uint16)
    matrixTotal = np.zeros((256,256),dtype=np.uint16)

    #matrixTOA = np.zeros((256,256),dtype=np.uint16)

    #eventString = ""

    #multiClusterEvents = []

    ntotal = ToT.shape[0]

    print(f"\nTOTAL nr of clusters: {ntotal}\n")
    ievent, npics, mcevents = 0, 0, 0
    n_good = 0
    n_tracks = 0
    for event in ToT:

        nhits = len(event)

        i_bincnt, _ = np.histogram(event, bins=tot_edges)
        tot_list += i_bincnt       

        #redX = int(Sum_Qx/float(sumTOT[ievent]))
        #redY = int(Sum_Qy/float(sumTOT[ievent]))

        #tot_reduced.append(sum(event)/hits[ievent])
        tot_reduced.append(sumTOT[ievent]/hits[ievent])
        TOTarray.append(sumTOT[ievent])

        # tryna' dbscan (fuck it slows down things...)
        #
        #smearedMat = matrix.reshape(64,4,64,4).mean(axis=(1,3))
        #nz_index = np.transpose(np.nonzero(smearedMat))
        #nholes = 1
        #min_hits = 2
        #db = DBSCAN(eps=nholes, min_samples=min_hits).fit(nz_index)
 
        #labels = db.labels_
        #nfound = len(set(labels)) - (1 if -1 in labels else 0)

        #if(nfound>1):
        #    multiClusterEvents.append(ievent)
        #    mcevents+=1
       
        rotAngDeg = round(rotAng[ievent]*180/3.1415,2)  
        characs = f"cX={round(centerX[ievent],2)}\ncY={round(centerY[ievent],2)}\n\u03B5={round(excent[ievent],2)}\nrotAng={rotAngDeg}\nnhits={nhits}\nlen={round(length[ievent],2)}\nwidth={round(width[ievent],2)}\nRMS:\n(T={round(RMS_T[ievent],2)},L={round(RMS_L[ievent],2)})\nFIRT={round(FIRT[ievent],2)}\nkurt:\nT={round(kurtosisT[ievent],2)}\n,L={round(kurtosisL[ievent],2)}\nLDRT={round(LDRT[ievent],2)}"        

        #---- working below -------
        #if(LDRT[ievent] > 10 and nhits>10 and FIRT[ievent]>0.01):
        #--------------------------
        #if(ievent in prop_ev or ievent in empty_ev):
        #suffix = None
        #if(ievent in prop_ev):
        #    suffix = "PROP"
        #else:
        #    suffix = "EMPTY"
        #good_events.append(ievent)

        #alt_rotang.append(rotAngDeg)
        alt_rotang.append(rotAng[ievent])

        #n_good+=1
        #if(length[ievent] < 10):
        #for i in range(nhits):
        #   #    matrix[x[ievent][i],y[ievent][i]] = event[i]
        #   matrixTotal[x[ievent][i],y[ievent][i]] += 1

        for xpos, ypos in zip(x[ievent],y[ievent]):
            np.add.at(matrixTotal, (xpos,ypos), 1)

        #eventString+=f"{ievent},"
        #if(npics < 50 and FIRT[ievent] > 0.01):
        
        goodX = (centerX[ievent] < 6.85 and centerX[ievent] > 7.4)        
        goodY = (centerY[ievent] < 6.75 and centerY[ievent] > 7.25)        

        #if(npics < 100 and excent[ievent] > 2 and nhits > 25 and goodX and goodY):      
        if(npics < 100 and nhits > 25 and excent[ievent] > 3):      
        #if(npics < 50 and ievent%100==0):      
        #if(npics < 50 and excent[ievent] > 3.0 and FIRT[ievent] < 0.1 and nhits > 20):      
            #matrixTOA = np.zeros((256,256),dtype=np.uint16)
            #matrixTOAcom = np.zeros((256,256),dtype=np.uint16)
            matrix = np.zeros((256,256),dtype=np.uint16)
            n_good+=1
            for i in range(nhits):
                matrix[x[ievent][i],y[ievent][i]] = event[i]
                #matrixTOA[x[ievent][i],y[ievent][i]] = ToA[ievent][i]
                #matrixTOAcom[x[ievent][i],y[ievent][i]] = comToA[ievent][i]
            

            #for t in range(nhits):
            #    matrixTOA[x[ievent][t],y[ievent][t]] = comToA[ievent][t]

            Sum_Qx = np.sum(event*x[ievent])
            Sum_Qy = np.sum(event*y[ievent])
 
            redX = Sum_Qx/np.sum(event)
            redY = Sum_Qy/np.sum(event)
       
            plot2dEvent(matrix, characs, f"cluster-{ievent}-{plotname}", outdir, [redX,redY]) 
            other2Dplot(matrix, f"cluster-{ievent}-{plotname}",outdir)

            #other2Dplot(matrixTOA,f"TOA-cluster-{ievent}-{plotname}",outdir)
           # other2Dplot(matrixTOAcom,f"TOAcom-cluster-{ievent}-{plotname}",outdir)
            

            #plotDbscan(nz_index, labels, ievent, nfound, outdir)
            npics+=1
            matrix = np.zeros((256,256),dtype=np.uint16)
            #matrixTOA = np.zeros((256,256),dtype=np.uint16)
            #matrixTOAcom = np.zeros((256,256),dtype=np.uint16)
        progress(ntotal, ievent)
        ievent+=1
        
        if(ievent==40000):
        #if("RomeRun" in recofile and ievent==30000):
            break

        #exit(0)

print(f"FOUND <{n_good}> events")

print(f"resudec TOT has <{len(tot_reduced)}> entries")

plot2dEvent(matrixTotal, "", "total-run",outdir)

#simpleHist(alt_rotang, 51, 0.0, 180.0, ["Selected Cluster inclination","angle[deg]","N"], "ALT_ROTANG", outdir, fit="cosfunc") 
#simpleHist(alt_rotang, 51, 0.0, 3.15, ["Selected Cluster inclination","angle[radian]","N"], "ALT_ROTANG", outdir, fit="cosfunc") 
simpleHist(alt_rotang, 51, 0.0, 3.15, ["","Reconstructed angle,[rad]","Events,[N]"], "ALT_ROTANG", outdir, fit="cosfunc") 
#simpleHist(tot_reduced, 101, 0, 400, ["TOT/nhits","N hits","#"], "TOT_REDUCED", outdir, fit="gaus")
simpleHist(tot_reduced, 51, np.nanmin(tot_reduced), np.nanmax(tot_reduced), ["TOT/nhits","N hits","#"], "TOT_REDUCED", outdir, fit="gaus")
simpleHist(TOTarray, 101, 0, np.nanmax(TOTarray), ['Total charge per event', 'TOT cycles','N'], "sumTOT", outdir, fit='gaus')

simpleHist(tot_list, 51, 0, 1000, ["raw TOT","TOT", "N"], "RAW_TOT", outdir)
#plt.figure()
#plt.hist(position_bin_edges[:-1], weights=x_bin_cnt, range=(0,255),bins=256, align='left',histtype='step',edgecolor='red')
#plt.savefig(f"{outdir}/EBALA.png")


#print(f"TOTAL matrix projections: \nstdev_x={stdxproj},\nstdev_y={stdyproj},\nexccentriccity={epsilon}")

#print(f"found {mcevents} multi cluster events")

#print(f"Interesting events have indices:{eventString[:-1]}")

#listFile = open(f"{plotname}-indices.txt", "w")
#listFile.write(eventString[:-1])
#listFile.write(eventString[:-1])
#listFile.close()



