import numpy as np
import numpy.ma as ma

import argparse as ap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors  import LogNorm
from matplotlib import cm
from matplotlib.gridspec import GridSpec

from scipy.signal import find_peaks

import glob
import os

from MyPlotter import myPlotter 
from MyPlotter import myUtils 

def initMatrix(size, dtype):
    return np.zeros((size,size),dtype=dtype)

def getSumTOT(matrix):
    return np.sum(matrix)

def getHits(matrix):
    return np.count_nonzero(matrix)

def getFrameTime(st,sr):

    return 256**sr * 46 * st / 4e7

def countIons(nx, ny):

    if(nx==ny):
        return nx
    if(nx>ny or ny>nx):
        nmax = nx if nx>ny else ny
        #nmin = ny if ny<nx else nx
        return nmax


def getClusterCenter(matrix):

    x,y = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), indexing='ij')
    
    norm_factor = matrix.sum()

    mean_x = (matrix * x).sum() / norm_factor
    mean_y = (matrix * y).sum() / norm_factor

    std_x = np.sqrt( (matrix * (x - mean_x)**2).sum() / norm_factor)
    std_y = np.sqrt( (matrix * (y - mean_y)**2).sum() / norm_factor)

    return mean_x, mean_y, std_x, std_y

def getClusterCenterMM(matrix_masked):
    
    #
    # use this one for masked arrays
    #

    valid = ~matrix_masked.mask
    if (not np.any(valid)):
        return -1,-1,-1,-1

    idx, idy = np.nonzero(valid)

    weights = matrix_masked[valid]
    norm = weights.sum()

    meanx = np.sum(idx*weights)/norm
    meany = np.sum(idy*weights)/norm
   
    stdx = np.sqrt(np.sum(weights*(idx-meanx)**2)/norm) 
    stdy = np.sqrt(np.sum(weights*(idy-meany)**2)/norm) 

    return meanx,meany,stdx,stdy


def checkClusterPosition(centerX, centerY, clusterX, clusterY, radius):

    r = np.sqrt((clusterX - centerX)**2 + (clusterY - centerY)**2)
    if(r>radius):
        return False
    else:
        return True

def getClusterCircleArea(radius):
    return np.pi*radius**2

def plotDetailedEvent(matrix,
                    picname,
                    outdir=None,
                    comment=None,
                    geometry=None,
                    markers=None):

    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(2,3, width_ratios=[1,3,0.15], height_ratios=[3,1], hspace=0.05, wspace=0.05)
    
    ax_mat = fig.add_subplot(gs[0,1])
    ax_xproj = fig.add_subplot(gs[1,1], sharex=ax_mat)
    ax_yproj = fig.add_subplot(gs[0,0], sharey=ax_mat)
    ax_cbar = fig.add_subplot(gs[0,2])

    if(comment is not None):
        ax_text = fig.add_subplot(gs[1,0])
        ax_text.axis('off')
        #lines = comment.split(":")
        #for line in lines:
        stats_text = (comment.replace(":",""))
        ax_text.text(-0.45,0.85, stats_text, transform=ax_text.transAxes,fontsize=11,verticalalignment='top',family='monospace')
    
    ms = ax_mat.matshow(matrix, cmap='jet')

    if(geometry is not None):
        for geo in geometry:
            ax_mat.add_patch(geo)
    if(markers is not None):
        for mar in markers:
            ax_mat.scatter(mar[0],mar[1],marker=mar[2],color=mar[3],label=mar[4])

    ax_mat.invert_yaxis()
    ax_mat.tick_params(labelbottom=False)
    ax_mat.tick_params(labelleft=False)
    ax_mat.set_xlim([0,256])
    ax_mat.set_ylim([0,256])

    cbar = plt.colorbar(ms,cax=ax_cbar,orientation='vertical')
    cbar.set_label("TOT")

    xhist = matrix.sum(axis=0)
    yhist = matrix.sum(axis=1)

    #noise_thr = 250 # threshold to trigger
    #min_height = 250 # threshold to trigger
    rel_height = 0.6 # threshold to trigger
    min_distance = 8 #pixels between peaks
    peak_width = 3 # width of the peaks
    #plat_offset = 10 # noise offset

    xpeaks, xprops = find_peaks(xhist, 
                        rel_height=rel_height,
                        distance=min_distance,
                        width=peak_width)

    ypeaks, yprops = find_peaks(yhist, 
                        rel_height=rel_height,
                        distance=min_distance,
                        width=peak_width)

    print(f"Npeaks_x={len(xpeaks)}, Npeaks_y={len(ypeaks)} ")
    xpromin = xprops["prominences"]
    xwidths = xprops['widths']
    ypromin = yprops["prominences"]
    ywidths = yprops['widths']
    print(f"xpeak: prominences = {xpromin}, widths={xwidths}")#, plateaus={xplateu}")
    print(f"ypeak: prominences = {ypromin}, widths={ywidths}")#, plateaus={yplateu}")

    ax_xproj.bar(np.arange(len(xhist)), xhist, width=1.0, color='darkblue', align='center')
    ax_xproj.scatter(xpeaks, xhist[xpeaks],marker="x",color='red')
    ax_xproj.set_xlabel("x,[pix]")
    ax_xproj.tick_params(axis='x', labelsize=8)

    ax_yproj.barh(np.arange(len(yhist)), yhist, height=1.0, color='darkblue', align='center')
    ax_yproj.scatter(yhist[ypeaks], ypeaks, marker="x",color='red')
    ax_yproj.set_ylabel("y,[pix]")
    ax_yproj.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    if(outdir is not None):
        plt.savefig(f"{outdir}{picname}.png")
    else: 
        plt.savefig(f"{picname}.png")

    plt.close()

def hasSparks(matrix,
              THR,
              ):

    nx,ny = matrix.shape
    fact_downscale = 4.0
    newX = nx//fact_downscale
    newY = ny//fact_downscale
    down_matrix = matrix.reshape(newX, fact_downscale, newY, fact_downscale).sum(axis=(1,3))

    projectionY = down_matrix.sum(axis=1)
    differences = np.diff(projectionY)

  
    

    return false

def getFramePeaks(matrix,        # matrix (original or pruned)
                  #oneSigmaPix,  # matrix with pixels inside one sigma from weight mean
                  #twoSigmaPix,  # matrix with pixels inside two sigma from weight mean
                  rel_height=0.7, # default relative height of the peaks 
                  min_dist=20,     # minimal distance between peaks (in bins)
                  peak_width=7,   # peak width at half prominence (whatever thefuck that means....)
                  fDebug=False):  # enable prints

    # projecting data in 1D onto x and y axis
    xhist = matrix.sum(axis=0)
    yhist = matrix.sum(axis=1)

    xpeaks, xprops = find_peaks(xhist, 
                        rel_height=rel_height,
                        distance=min_dist,
                        width=peak_width)

    ypeaks, yprops = find_peaks(yhist, 
                        rel_height=rel_height,
                        distance=min_dist,
                        width=peak_width)


    if(fDebug):
        print(f"Npeaks_x={len(xpeaks)}, Npeaks_y={len(ypeaks)} ")
        xpromin = xprops["prominences"]
        xwidths = xprops['widths']
        ypromin = yprops["prominences"]
        ywidths = yprops['widths']
        print(f"xpeak: prominences = {xpromin}, widths={xwidths}")#, plateaus={xplateu}")
        print(f"ypeak: prominences = {ypromin}, widths={ywidths}")#, plateaus={yplateu}")

    #npeaks_x = len(xpeaks)
    #npeaks_y = len(ypeaks)

    return xpeaks, ypeaks, xprops, yprops

############################################################
# MAIN
############################################################

parser = ap.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default='', help="loaction ofr the data runs")
parser.add_argument('-n', "--name", type=str, default='EBALA', help="suffix of the plots")
parser.add_argument("-SR", "--srbit", type=int, default=1, help="SR bits")
parser.add_argument("-ST", "--stbit", type=int, default=15, help="ST bits")
parser.add_argument("--nframes", type=int, default=10, help="Number of frames to plot")
parser.add_argument("--cutnoise", action='store_true' , help="cut noisy frames out")
parser.add_argument("--plotglobal", action='store_true' , help="Plot global parametere histograms")
parser.add_argument("--plotframes", action='store_true' , help="Plot single frames")
parser.add_argument("--makedir", action='store_true' , help="Dump plots in a custom directory")
parser.add_argument("--hitrate", action='store_true' , help="Plot hits vs time for a run")
parser.add_argument("--countTOT", action='store_true' , help="Count ions based on the average TOT in a single cluster events")
parser.add_argument("--countBS", action='store_true' , help="Also count bullshit frames. One frame = one ion")

args = parser.parse_args()

directory = args.dir
picname = args.name
SR = args.srbit
ST = args.stbit
nframes = args.nframes

fPlotGlobalHists = args.plotglobal
fCutNoise = args.cutnoise
fPlotFrames = args.plotframes
fMakeDir = args.makedir
fPlotHitRate = args.hitrate
fCountBS = args.countBS
fCIT = args.countTOT
fSingle = False

nSingleIons = 0
print("Flag check:")
print(f"GlobalPlots: {fPlotGlobalHists}")
print(f"CutNoise: {fCutNoise}")
print(f"MakeDir: {fMakeDir}")

if(fPlotFrames):
    print(f"PlotFrames: {fPlotFrames} : NEED {nframes}")
else:
    print(f"PlotFrames: {fPlotFrames}")

print("\n")

outdir = None
if (fMakeDir):
    outdir = f"NITRO-CHARGE-{picname}/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

filelist = glob.glob(directory+"*.txt")
nfiles = len(filelist)

matrix = initMatrix(256,int)
totmatrix = initMatrix(256,int)

t_dead = 0.048 # deadtime in seconds
t_frame = getFrameTime(ST,SR)

# lists for first step in gathering global characteristics of frames

runtime = []
hitrate = []
fucked_hitrate = []

frame_hits, frame_sumTOT = [], []
frame_avg_noise = []
frame_stdx, frame_stdy = [], []
frame_FI1STD, frame_FI2STD = [], [] # fraction of charge inside one and two sigmas from center
frame_TSA = [] # Area of 2 sigma of the cluster
frame_NPixA = [] # normalised active area in pixels normalized over 256x256
frame_qdensity = [] # charge density (sumTOT (inside 2 sigma) /nhits)
frame_QBsigmas = [] # fraction of charge between sigma1 and 2
# Accumulated frame data
frame_charge_weights = initMatrix(256,float) # centers fo the charge weight
#frame_charge_means = initMatrix(256,float) # centers of mean cluster positions


single_hits, single_sumTOT = [], []
single_QBsigma, single_Qdens = [], []
single_AreaRelPix = []

single_promx, single_promy = [], []
single_widthx, single_widthy = [], []

pruned_hits, pruned_sumTOT = [], []

##### peak finding histos ####
xprominences, xwidths = [], []
yprominences, ywidths = [], []
##############################

nSparks = 0
nIonPeaks = 0

occ_matrix = initMatrix(256,int)
#####################################
# instantiating MyPlotter
MU = myUtils()
MP = myPlotter()
##########################
NPIXELS =256*256
#### attributes for "find_peaks" shit.. ..
RHEIGHT = 0.6
MDIST = 8
PWIDTH = 3
#############

diff_traces = []
sum_traces = np.zeros((63,), dtype=int)
ifile = 0 
nskipped = 0
npics = 0
for file in filelist:

    ifile+=1
    tmp_matrix = np.loadtxt(file,dtype=int)

    # i-th frame hits and sum TOT
    nhits = np.count_nonzero(tmp_matrix)
    sumTOT = np.sum(tmp_matrix)

    time_iframe = None
    if(fPlotHitRate):
        time_iframe = ifile*0.048
    # for now lets see full picture
    # if have more than 30k htis : probably spark and bullshit frame - discard for first loop
    if(fCutNoise and (nhits > 30000 or nhits < 500)):
        if(nhits>30000 and fPlotHitRate):
            fucked_hitrate.append([time_iframe, nhits])
        nskipped+=1
        continue
    if(fPlotHitRate):
        fucked_hitrate.append([time_iframe,-999]) 
        hitrate.append([time_iframe, nhits]) 
        
    # fiiling global hit,sumTOT hists
    frame_hits.append(nhits)
    frame_sumTOT.append(sumTOT)
    
    # getting mean position of charge in frame [x,y]
    # and standard deciations of charge in frame [x,y]

    meanx, meany, stdx, stdy = getClusterCenter(tmp_matrix)

    # getting fraction of charge within certain region

    onesigmaTOT, twosigmaTOT = 0,0
    onesigmaHits, twosigmaHits = 0,0
    outTOT, outHits = 0, 0
    nhits_3sigma = 0
    nonzeropix = np.nonzero(tmp_matrix)
    # choose lowest deviation
    minRadius = stdx if stdx<=stdy else stdy
    # loop only through nonzero entris of the matrix
    #oneSigPixels, twoSigPixels = initMatrix(256,int),initMatrix(256,int)
    for x,y in zip(*nonzeropix):
        insideOneSigma = checkClusterPosition(x,y,meanx,meany,minRadius)
        insideTwoSigma = checkClusterPosition(x,y,meanx,meany,minRadius*2)
        if(insideOneSigma):
            onesigmaTOT+=tmp_matrix[x,y]
            onesigmaHits+=1
            #np.add.at(oneSigPixels,(x,y),1)
        if(insideTwoSigma):
            twosigmaTOT+=tmp_matrix[x,y]
            twosigmaHits+=1
            #np.add.at(twoSigPixels,(x,y),1)
        if(not insideTwoSigma):
            outTOT+=tmp_matrix[x,y]
            outHits+=1      

        np.add.at(occ_matrix, (x,y), 1)

    # recording fraction of charge in the cluster regions of 1 and 2 sigmas
    rfTOT_onesig = onesigmaTOT/sumTOT
    rfTOT_twosig = twosigmaTOT/sumTOT
    frame_FI1STD.append(rfTOT_onesig)
    frame_FI2STD.append(rfTOT_twosig)
    # calculating relative annoumt of charge inside area between sigma1 and sigma2 lines
    frac_QBsigmas = (twosigmaTOT - onesigmaTOT)/sumTOT
    frame_QBsigmas.append(frac_QBsigmas)

    # cheking average noise outside the 2sigma circle
    frame_avg_noise.append(outTOT/outHits)
   
    # getting charge density of hits within 2*sigma_min
    qdensity = twosigmaTOT/twosigmaHits
    frame_qdensity.append(qdensity)
    
    # approximate, circular clsuter area of 2 sigmas
    cluster_area = getClusterCircleArea(minRadius*2)
    # area as a fraction of pixels covered within 2 sigmas / total number of pixels
    cluster_area_pixfrac = twosigmaHits/NPIXELS

    # filling rest of lists/matrices
    frame_stdx.append(stdx)    
    frame_stdy.append(stdy)    

    frame_TSA.append(cluster_area)    
    frame_NPixA.append(cluster_area_pixfrac)
    np.add.at(frame_charge_weights,(int(np.round(meanx)),int(np.round(meany))),1)

    huya_matrix = tmp_matrix.reshape(64,4,64,4).sum(axis=(1,3))
    huya_projY = huya_matrix.sum(axis=1)
    huya_diffs = np.diff(huya_projY)
    #print(huya_diffs.shape)

    diff_traces.append(abs(huya_diffs))
    sum_traces+=abs(huya_diffs)

    MU.progress(nfiles, ifile)

    if(fPlotFrames and npics <= nframes):
        if(nhits>100 and nhits<10000):
            #print(f"IS SINGLE CLUSTER: [{fSingleCluster}]")
            comment = ""
            comment+=r"$N_{hits}$="+f"{nhits}\n:"
            comment+=r"$\Sigma$(TOT)="+f"{sumTOT}\n:"
            #comment+=r"$\bar{y}$="+f"{meanx:.2f}\n:"# coords inverted
            #comment+=r"$\bar{x}$="+f"{meany:.2f}\n:"# coords inverted
            #comment+=r"$\sigma_{x}$="+f"{stdx:.2f}\n:"
            #comment+=r"$\sigma_{y}$="+f"{stdy:.2f}\n:"
            #comment+=r"$Q_{1x\sigma}/Q_{total}$="+f"{rfTOT_onesig:.2f}\n:"
            #comment+=r"$Q_{2x\sigma}/Q_{total}$="+f"{rfTOT_twosig:.2f}\n:"
            #comment+=r"$Q_{A(\sigma_{2}-\sigma_{1})}$="+f"{frac_QBsigmas:.2f}\n:"
            #comment+=r"$\rho_{Q}$="+f"{qdensity:.2f}\n:"
            #comment+=r"$A(N_{cpix}/256^{2})$="+f"{cluster_area_pixfrac:.2f}\n:"

            # patches for MP.plotMatrix()
            circ_sig1 = plt.Circle((meany,meanx), #swapping x,y --> y,x for inverted axis of our matrix
                                    minRadius,
                                    edgecolor='blue',
                                    facecolor='none')#,
                                    #clip_on=True)

            circ_sig2 = plt.Circle((meany,meanx),
                                    minRadius*2,
                                    edgecolor='darkblue',
                                    facecolor='none')#, 
                                    #clip_on=True)

            #huya_matrix = tmp_matrix.reshape(64,4,64,4).sum(axis=(1,3))
            #huya_projY = huya_matrix.sum(axis=1)
            #huya_diffs = np.diff(huya_projY)
            #plt.figure()
            #plt.plot([i for i in range(len(huya_diffs))], abs(huya_diffs))
            #plt.title(f'FRAME[{ifile}], TOT diffs of downscaled sum of matrix')
            #plt.xlabel('step')
            #plt.ylabel(r'$\Delta$(TOT)')
            ##plt.yscale('log')
            #plt.savefig(f"DIFFs-{ifile}.png")
            #plt.close()

            MP.plotMatrix(tmp_matrix,
                          f"FRAME-{ifile}-{picname}",
                          labels=[f"EVENT-{ifile}", "x,[pixel]","y,[pixel]"],
                          info=comment,
                          infopos=[-90,25],
                          outdir=outdir,
                          figsize=(8,6),
                          #geomshapes=[circ_sig1,circ_sig2],
                          #plotMarker=[meany,meanx],
                          fLognorm=True,
                          fDebug=True)

        #end of raw frame plotting
    
    THR = 750 #250
    badIDX = np.where(tmp_matrix<THR)
    skippix = 5
    masked_matrix = np.ma.masked_array(tmp_matrix, np.zeros_like(tmp_matrix, dtype=int))
    masked_matrix[badIDX] = np.ma.masked # masking pix with TOT<THR
    masked_matrix[:skippix, :] = np.ma.masked # masking top Nskip
    masked_matrix[-skippix:,:] = np.ma.masked # bottom 
    masked_matrix[:, :skippix] = np.ma.masked # left
    masked_matrix[:, -skippix:] = np.ma.masked # right

    cutMeanX, cutMeanY, cutStdx, cutStdy = getClusterCenterMM(masked_matrix)

    cutMinRad = cutStdx if cutStdx<=cutStdy else cutStdy

    valid = ~masked_matrix.mask
    idx,idy = np.nonzero(valid)
    
    nhits_left = len(idx)
    cutSumTOT = masked_matrix[valid].sum()       
    
    cut_1STOT,  cut_2STOT=  0,0
    cut_1SHits, cut_2SHits = 0,0

    for ix,iy in zip(idx,idy):
        fOneSig = checkClusterPosition(ix,iy,cutMeanX,cutMeanY,cutMinRad)
        fTwoSig = checkClusterPosition(ix,iy,cutMeanX,cutMeanY,cutMinRad*2)
        if(fOneSig):
            cut_1STOT+=masked_matrix[ix,iy]            
            cut_1SHits+=1            
        if(fTwoSig):
            cut_2STOT+=masked_matrix[ix,iy]
            cut_2SHits+=1            

    # checking parameters after masked matrix checks

    try:
        cut_rfTOT_1s = cut_1STOT/cutSumTOT
    except ZeroDivisionError:
        cut_rfTOT_1s = -1

    try:
        cut_rfTOT_2s = cut_2STOT/cutSumTOT
    except ZeroDivisionError:
        cut_rfTOT_2s = -1    

    try:        
        cut_frac_qb = (cut_2STOT - cut_1STOT)/cutSumTOT 
    except ZeroDivisionError:
        cut_frac_qb = -1

    try:
        cut_qdens = cut_2STOT/cut_2SHits
    except ZeroDivisionError:
        cut_qdens = -1

    try:
        cut_relpixarea = cut_2SHits/NPIXELS
    except ZeroDivisionError:
        cut_relpixarea = -1


    peaksX, peaksY, propsX, propsY = getFramePeaks(masked_matrix, 
                                                    rel_height=RHEIGHT, 
                                                    min_dist=MDIST,
                                                    peak_width=PWIDTH)

    prelim_nxpeaks, prelim_nypeaks = len(peaksX), len(peaksY)

    nIonPeaks += countIons(len(peaksX),len(peaksY))

    for xp,xw in zip(propsX["prominences"],propsX['widths']):
        xprominences.append(int(xp))
        xwidths.append(float(xw))

    for yp,yw in zip(propsY["prominences"],propsY['widths']):
        yprominences.append(int(yp))
        ywidths.append(float(yw))

    #idx_s1,idy_s1 = np.nonzero(oneSigmaPix)
    #idx_s2,idy_s2 = np.nonzero(twoSigmaPix)

    ##oneSigmaHits = len(oneSigmaPix)
    #twoSigmaHits = len(idx_s2)
    #isumTOT = getSumTOT(matrix)
    #itot_onesigma = np.sum(matrix[idx_s1,idy_s1])
    #itot_twosigma = np.sum(matrix[idx_s2,idy_s2])
    #iHits = getHits(matrix)   
    #itot_fi1std = itot_onesigma/isumTOT # cluster TOT within one sigma
    #itot_fi2std = itot_twosigma/isumTOT # cluster TTO within two sigmas 
    #iqbsigma = (itot_fi2std - itot_fi1std)/isumTOT # cluster TOT within one ant two sigma slice
    #iqdensity = itot_twosigma/twoSigmaHits # Q per number of hits within two sigma 
    #iarea_pixfrac = twoSigmaHits/(256*256)# realtive number of pixels within 2 sigma (n/65k) 
 
    if(prelim_nxpeaks==1 and prelim_nypeaks==1):
        single_hits.append(nhits)
        single_sumTOT.append(sumTOT)
        single_QBsigma.append(frac_QBsigmas)
        single_Qdens.append(qdensity)
        single_AreaRelPix.append(cluster_area_pixfrac)
        single_promx.append(propsX['prominences']) 
        single_promy.append(propsY['prominences']) 
        single_widthx.append(propsX['widths']) 
        single_widthy.append(propsY['widths']) 
        nSingleIons+=1

    if(fPlotFrames and npics<=nframes):
        if(nhits>100 and nhits<10000):
            #################################################
            cut_comment = ""
            cut_comment+=r"$N_{hits}$="+f"{nhits_left}\n:"
            cut_comment+=r"$N_{hits}(2x\sigma)$="+f"{cut_2SHits:.2f}\n"
            cut_comment+=r"$\Sigma$(TOT)="+f"{cutSumTOT}\n:"
            #cut_comment+=r"$\bar{y}$="+f"{cutMeanX:.2f}\n:"# coords inverted
            #cut_comment+=r"$\bar{x}$="+f"{cutMeanY:.2f}\n:"# coords inverted
            cut_comment+=r"$\sigma_{x}$="+f"{cutStdx:.2f}\n:"
            cut_comment+=r"$\sigma_{y}$="+f"{cutStdy:.2f}\n"
            #cut_comment+=r"$N_{hits}(1x\sigma)$="+f"{cut_1SHits:.2f}\n"
            #cut_comment+=r"$\Sigma$(TOT,1x$\sigma$)="+f"{cut_1STOT:.2f}\n"
            cut_comment+=r"$\Sigma$(TOT,2x$\sigma$)="+f"{cut_2STOT:.2f}\n"
            #cut_comment+=r"Rel($\Sigma$TOT)1x$\sigma$"+f"={cut_rfTOT_1s:.2f}\n"
            cut_comment+=r"Rel($\Sigma$TOT)2x$\sigma$"+f"={cut_rfTOT_2s:.2f}\n"
            cut_comment+=r"$Q(2x\,\sigma/Q_{total})$"+f"={cut_frac_qb:.2f}\n"
            cut_comment+=r"$\rho_{Q}$(TOT/Npix)"+f"={cut_qdens:.2f}\n"
            cut_comment+=r"$A_{pix}$(Npix/$256^{2}$)"+f"={cut_relpixarea:.2f}\n"

            cut_comment+="X:"
            nxp=0
            for xp,xw in zip(propsX["prominences"],propsX['widths']):
                nxp+=1
                cut_comment+=f"[{nxp}]:A={int(xp)} w={xw:.2f}|"
            cut_comment+="\nY:"
            nyp=0
            for yp,yw in zip(propsY["prominences"],propsY['widths']):
                nyp+=1
                cut_comment+=f"[{nyp}]:A={int(yp)} w={yw:.2f}|"
            cut_comment+="\n"

            c_sigma1 = plt.Circle((cutMeanY,cutMeanX), #swapping x,y --> y,x for inverted axis of our matrix
                                    cutMinRad,
                                    edgecolor='blue',
                                    facecolor='none')#,
                                    #clip_on=True)

            c_sigma2 = plt.Circle((cutMeanY,cutMeanX),
                                    cutMinRad*2,
                                    edgecolor='darkblue',
                                    facecolor='none')#, 
                                    #clip_on=True)

            plotDetailedEvent(masked_matrix,
                              f"FRAME-CUT-WSTATS-{ifile}-{picname}",
                              outdir=outdir,
                              comment=cut_comment, 
                              geometry=[c_sigma1,c_sigma2],
                              markers=[[cutMeanY,cutMeanX,"+","black","Qweigth"]])
            npics+=1
    
    #if(npics==20):
    #    exit(0) 
    # for loop ends here
print("============================")
print(f"FOUND {nIonPeaks} IONS")
print("============================")

avg_tot_sions = sum(single_sumTOT)/nSingleIons
print(f"Found {nSingleIons} single ions")
print(f"average TOT per Ion = {avg_tot_sions:.2f}")

MP.plotMatrix(frame_charge_weights, 
              f"CHARGE_CENTERS-{picname}",
              labels=[f"Charge Weight Centers", "x, [pix]", "y, [pix]"],
              cmap='viridis',
              cbarname=r"$N_{frames}$",
              outdir=outdir,
              fDebug=True)

MP.plotMatrix(occ_matrix,
                f"OCCUPANCY-{picname}",
                labels=["Occupancy Matrix", "x,[pix]", "y.[pix]"],
                cbarname=r'$N_{hits}$',
                outdir=outdir,
                fLognorm=True,
                fDebug=True)

quiet_pixels = np.ones((256,256))
pixels_active = np.where(occ_matrix>0)
no_response_pix = np.ma.masked_array(quiet_pixels)
no_response_pix[pixels_active] = np.ma.masked
n_quiet = np.sum(no_response_pix)
dcomment = r"$N_{quiet}$"+f"={n_quiet}\n:"
if(n_quiet>0):
    print(f"FOUND {n_quiet} QUIET channels")
else:
    print("All pixels found to be active")

plt.figure(figsize=(9,6))
#sum_diffs = np.zeros((63,63), dtype=int)
fSetLabel = True
for idiff in diff_traces:
    if(fSetLabel):
        plt.plot([i for i in range(len(idiff))], idiff, alpha=0.05, color='red', label='ith frame, TOT diff.')
        fSetLabel = False
    else:
        plt.plot([i for i in range(len(idiff))], idiff, alpha=0.05, color='red')
       
avg_diffs = sum_traces/len(diff_traces)
plt.scatter([i for i in range(len(avg_diffs))], avg_diffs, color='green',marker='o',label='Averaged deviation')
plt.title(f'Combined TOT diffs of downscaled sum of matrix along Y axis')
plt.xlabel('step')
plt.ylabel(r'$\Delta$(TOT)')
#plt.yscale('log')
plt.grid(which='major', color='gray', linestyle='-',linewidth=0.5)
plt.grid(which='minor', color='gray', linestyle='--',linewidth=0.25)
plt.minorticks_on()
plt.legend(loc='upper right')
plt.savefig(f"Combined-Yaxis-DIFFs-{picname}.png")
plt.close()

MP.plotMatrix(no_response_pix,
                f"QUIET-PIXELS-{picname}",
                labels=["Pixels with 0 Activity over the Run Time", "x,[pix]", "y.[pix]"],
                figsize=(8,7),
                info=dcomment,
                infopos=[0,-20],
                outdir=outdir,
                cmap="gray",
                cbarname="ON/OFF",
                #fLognorm=True,
                fDebug=True)

if(fPlotGlobalHists):
    MP.simpleHist(
            np.array(frame_hits), 
            100, 
            0, 
            np.max(frame_hits), 
            ["Global Hits per Frame", r"$N_{hits}$", r"$N_{frames}$"],
            f"HITS-{picname}",
            outdir=outdir,
            ylog=True
            )
    
    MP.simpleHist(
            np.array(frame_sumTOT), 
            100, 
            0, 
            np.max(frame_sumTOT), 
            ["Global sumTOT per Frame", r"$\Sigma$(TOT)", r"$N_{frames}$"],
            f"SUMTOT-{picname}",
            outdir=outdir,
            ylog=True
            )
    
    MP.simpleHist(
        np.array(frame_stdx),
        100,
        0,
        np.max(frame_stdx),
        ["Global Deviation in X in a Frame", r"$\sigma_{x}$", "$N_{frames}$"],
        f"STDX-{picname}",
        outdir=outdir
        )
    
    MP.simpleHist(
        np.array(frame_stdy),
        100,
        0,
        np.max(frame_stdy),
        ["Global Deviation in Y in a Frame", r"$\sigma_{y}$", "$N_{frames}$"],
        f"STDY-{picname}",
        outdir=outdir
        )
    
    MP.simpleHist(
        np.array(frame_FI1STD),
        100,
        0,
        np.max(frame_FI1STD),
        [r"Global: Fraction of TOT inside $1\cdot\sigma_{min}$", r"Q fraction [$Q_{total}^{-1}$]", "$N_{frames}$"],
        f"FRAC-Q-ONESIGMA-{picname}",
        outdir=outdir
        )
    
    MP.simpleHist(
        np.array(frame_FI2STD),
        100,
        0,
        np.max(frame_FI2STD),
        [r"Global: Fraction of TOT inside $2\cdot\sigma_{min}$", r"Q fraction [$Q_{total}^{-1}$]", "$N_{frames}$"],
        f"FRAC-Q-twoSIGMA-{picname}",
        outdir=outdir
        )
    
    MP.simpleHist(
        np.array(frame_TSA),
        100,
        0,
        np.max(frame_TSA),
        [r"Global: Area of $2\cdot\sigma_{x/y}$ of a Cluster", "Area, [pix]", "$N_{frames}$"],
        f"AREA-CIRCLE-{picname}",
        outdir=outdir
        )
    
    MP.simpleHist(
        np.array(frame_NPixA),
        100,
        0,
        np.max(frame_NPixA),
        [r"Global: Area of Cluster in pixels Normalised to N(total pix.)", "Area,[frac]", "$N_{frames}$"],
        f"AREA-NORMPIX-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(frame_qdensity),
        100,
        0,
        np.max(frame_qdensity),
        [r"Global: Charge density within 2 sigma of a cluster", r"$\rho_{q}$[TOT/nhits]", "$N_{frames}$"],
        f"Q-DENSITY-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(frame_avg_noise),
        100,
        0,
        np.max(frame_avg_noise),
        [r"Global: Average TOT outside 2 sigma of a cluster", r"$\overline{TOT}$ [nCLK]", "$N_{frames}$"],
        f"SUMTOT-OUT-{picname}",
        outdir=outdir,
        getStats=True
        #ylog=True
        )

    MP.simpleHist(
        np.array(frame_QBsigmas),
        100,
        0,
        np.max(frame_QBsigmas),
        [r"Global: Relative Fraction of Q between $\sigma_{x1}$ and $\sigma_{x2}$", "Relative Q [1/sumTOT]", "$N_{frames}$"],
        f"Q-BETWEEN-SIGMAS-{picname}",
        outdir=outdir
        #getStats=True
        #ylog=True
        )

    print("Plotting peak finder's data")
    MP.simpleHist(
        np.array(xprominences),
        100,
        0,
        np.nanmax(xprominences),
        ["Peak Finder: X prominences", "Amplitude", "$N_{peaks}$"],
        f"XPROMINENCE-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(yprominences),
        100,
        0,
        np.nanmax(yprominences),
        ["Peak Finder: Y prominences", "Amplitude", "$N_{peaks}$"],
        f"YPROMINENCE-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(xwidths),
        100,
        0,
        np.nanmax(xwidths),
        ["Peak Finder: X widths", "widths", "$N_{peaks}$"],
        f"XWIDTH-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(ywidths),
        100,
        0,
        np.nanmax(ywidths),
        ["Peak Finder: Y widths", "widths", "$N_{peaks}$"],
        f"YWIDTH-{picname}",
        outdir=outdir
        )

    MP.plot2Dhist(np.array(xprominences),
                  np.array(xwidths),
                  100,
                  ["Peak finder: width vs prominence (X)","Prominence","Width"],
                  f"width-vs-promin-Xaxis-{picname}",
                  odir=outdir,
                  figsize=(9,8),
                  fLogNorm=True
                  #fDebug=True 
                )

    MP.plot2Dhist(np.array(yprominences),
                  np.array(ywidths),
                  100,
                  ["Peak finder: width vs prominence (Y)","Prominence","Width"],
                  f"width-vs-promin-Yaxis-{picname}",
                  odir=outdir,
                  figsize=(9,8),
                  fLogNorm=True
                  #fDebug=True 
                )

    print("TRYNA' PLOT single Cluster stats (prelim. selection by\"find_peaks\")")

    MP.simpleHist(
        np.array(single_hits),
        100,
        0,
        np.nanmax(single_hits),
        ["Single Clusters: Hits", r"$N_{hits}$", "$N_{frames}$"],
        f"SINGLE-NHITS-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(single_sumTOT),
        100,
        0,
        np.nanmax(single_sumTOT),
        [r"Single Clusters: $\Sigma$(TOT)", r"$\Sigma$(TOT)", "$N_{frames}$"],
        f"SINGLE-SUMTOT-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(single_QBsigma),
        100,
        0,
        np.nanmax(single_QBsigma),
        [r"Single Clusters: Q in Area $\pi\cdot(\sigma_{x2}-\sigma_{x1})^{2}$", r"Relative Q", "$N_{frames}$"],
        f"SINGLE-RelQ-betweenCirc-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(single_Qdens),
        100,
        0,
        np.nanmax(single_Qdens),
        [r"Single Clusters: Charge Density", r"$Q_{2\cdot\sigma_{min}}/Q_{total}$", "$N_{frames}$"],
        f"SINGLE-QDENSITY-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(single_AreaRelPix),
        100,
        0,
        np.nanmax(single_AreaRelPix),
        [r"Single Cluster: Cluster Area Relative to Full Matrix", r"$N_{pix,cluster}/N_{total}$", "$N_{frames}$"],
        f"SINGLE-RelQ-betweenCirc-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(single_promx),
        100,
        0,
        np.nanmax(single_promx),
        [r"Single Cluster: Peak Prominence [X]", "Ampl. Max. Bin", "$N_{frames}$"],
        f"SINGLE-PROMX-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(single_promy),
        100,
        0,
        np.nanmax(single_promy),
        [r"Single Cluster: Peak Prominence [Y]", "Ampl. Max. Bin", "$N_{frames}$"],
        f"SINGLE-PROMY-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(single_widthx),
        100,
        0,
        np.nanmax(single_widthx),
        [r"Single Cluster: Peak Width [X]", "N(x) Bins", "$N_{frames}$"],
        f"SINGLE-WIDTHX-{picname}",
        outdir=outdir
        )

    MP.simpleHist(
        np.array(single_widthy),
        100,
        0,
        np.nanmax(single_widthy),
        [r"Single Cluster: Peak Width [Y]", "N(y) Bins", "$N_{frames}$"],
        f"SINGLE-WIDTHY-{picname}",
        outdir=outdir
        )


    print("TRYNA' PLOT 2D:")

    MP.plot2Dhist(np.array(frame_hits),
                  np.array(frame_sumTOT),
                  100,
                  [r"$\Sigma$(TOT) vs Nhits in a frame",r"$N_{Hits}$",r"$\Sigma$(TOT)"],
                  f"sumTOT-vs-hits-{picname}",
                  odir=outdir,
                  figsize=(9,8),
                  fLogNorm=True
                  #fDebug=True 
                )

    MP.plot2Dhist(np.array(frame_hits),
                  np.array(frame_FI1STD),
                  100,
                  ["Fraction of Q inside 1sigma vs Nhits in a frame",r"$N_{Hits}$",r"$Q_{\sigma x1}/Q_{total}$"],
                  f"FI1STD-vs-HITS-{picname}",
                  odir=outdir,
                  figsize=(9,8),
                  fLogNorm=True
                  #fDebug=True 
                )
    MP.plot2Dhist(np.array(frame_hits),
                  np.array(frame_FI2STD),
                  100,
                  ["Fraction of Q inside 2sigma vs Nhits in a frame",r"$N_{Hits}$",r"$Q_{\sigma x2}/Q_{total}$"],
                  f"FI2STD-vs-HITS-{picname}",
                  odir=outdir,
                  figsize=(9,8),
                  fLogNorm=True
                  #fDebug=True 
                )

    MP.plot2Dhist(np.array(frame_hits),
                  np.array(frame_NPixA),
                  100,
                  [r"Npixels within 2x$\sigma_{min}$ vs Nhits",r"$N_{Hits}$",r"$N_{pix}/A_{Cluster}$"],
                  f"NPixA-vs-HITS-{picname}",
                  odir=outdir,
                  figsize=(9,8),
                  fLogNorm=True
                  #fDebug=True 
                )

if(fPlotHitRate):
    print("Plotting Hitrate...")
    plt.figure(figsize=(10,6))
    plt.plot([hitrate[i][0] for i in range(len(hitrate))],
             [hitrate[i][1] for i in range(len(hitrate))],
             color='darkblue', marker='s')
    
    plt.plot([fucked_hitrate[i][0] for i in range(len(fucked_hitrate))],
             [fucked_hitrate[i][1] for i in range(len(fucked_hitrate))],
             color='firebrick', marker='o')
    plt.title("Hits vs Time")
    plt.xlabel('Time,[s]')
    plt.ylabel(r'$N_{hits}$')
    plt.grid(True)
    plt.savefig(f"{outdir}HITRATE-{picname}.png")
    plt.close()

print(f"Read {nfiles}, skipped {nskipped}")

if(fCIT):

    N_Nitrogen = 0
   
    total_sumTOT = 0
 
    ifile = 0
    for f in filelist:
    
        ifile+=1
        tmp_matrix = np.loadtxt(file,dtype=int)
        nhits = np.count_nonzero(tmp_matrix)# using total hits 

        THR = 750 #250
        badIDX = np.where(tmp_matrix<THR)
        skippix = 5
        masked_matrix = np.ma.masked_array(tmp_matrix, np.zeros_like(tmp_matrix, dtype=int))
        masked_matrix[badIDX] = np.ma.masked # masking pix with TOT<THR
        masked_matrix[:skippix, :] = np.ma.masked # masking top Nskip
        masked_matrix[-skippix:,:] = np.ma.masked # bottom 
        masked_matrix[:, :skippix] = np.ma.masked # left
        masked_matrix[:, -skippix:] = np.ma.masked # right

        # using pixels with TOT above threshold
        cut_sumTOT = np.sum(masked_matrix)
        total_sumTOT+=cut_sumTOT
        if(nhits>3e4):
             if(fCountBS):
                N_Nitrogen+=1
             continue
    
        N_Nitrogen += np.floor(float(cut_sumTOT)/float(avg_tot_sions))

        nhits, cut_sumTOT = None, None
        tmp_matrix = None

    print(f"Separate Counting results in {N_Nitrogen} Ions in {ifile} frames")
    
    # Extrapolating how many ions collected in total

    total_runtime = (t_frame+t_dead)*ifile 
    total_nitrogen = (t_dead+t_frame)/t_frame*N_Nitrogen
    ionrate = total_nitrogen/total_runtime

    print("------ COUNTING (TOT-based) -------")
    print(f"Averge TOT-per-single cluster = {avg_tot_sions:.2f}")
    print(f"Total collected TOT = {total_sumTOT:.2f}")
    print(f"Resulting in {total_sumTOT/avg_tot_sions:.2f} ions\n")
    print(f"Ions observed = {N_Nitrogen:.2f}")
    print(f"Run time = {total_runtime:.2f} minutes")
    print(f"Dead-time scaling factor = {t_dead/t_frame:.2f}")
    print(f"rate={ionrate:.2f} [Hz]")




# Later make first for loop to evaluate some parameters by which automatically one could
# select single clusters and count their charge, then apply charge counting
# over all matrix for all frames (good ones)

#print("Commence step 2 to select specific events")
#
#for file in filelist:
#    
#    tmp_matrix = np.loadtxt(file, dtype=int)    
#
#    nhits = np.count_nonzero(tmp_matrix)
#    if(fCutNoise and (nhits > 30000 or nhits < 10)):
#        nskipped+=1
#        continue
# 
#    sumTOT = np.sum(tmp_matrix)
#

