import numpy as np
import numpy.ma as ma
import argparse as ap

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors  import LogNorm
from matplotlib import cm
from matplotlib.gridspec import GridSpec

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

import glob
import os

from MyPlotter import myPlotter 
from MyPlotter import myUtils 
from MyPlotter import myColors

# shite for detecting ion track blobs
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage import filters, measure, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.color import label2rgb

from scipy.spatial import cKDTree

# linear fit between 4096 (0.025s dead time) and 65536 (0.4s max)
td_slope = 6.103515620101e-6
td_offset = 1.268210634667e-10

ndiv = 4
npix_redreg = 256**2/ndiv**2

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

def estimateDeadTime(tframe, nframes, trun):

    # lets use seconds here...
    tdead = trun/nframes - tframe
    return tdead
    
def estimateFrameDeadTime(nhits):

    tdead_frame = td_slope*nhits + td_offset 

    return tdead_frame

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

def computeClusterDensity(coords, sigma=3.0):

    # this one computes density of neighbors for each pixel
    y,x = coords
    points = np.column_stack((x,y))
    N = len(points)
    density = np.zeros(N)
    for i,p in enumerate(points):
        dist2 = np.sum((points-p)**2, axis=1)
        density[i] = np.sum(np.exp(-dist2/(2*sigma**2)))

    return density

def computeChargeDensity(coords, charges, sigma=3.0, cutoff=3):

    # this one calculates charge density
    
    y,x = coords
    points = np.column_stack((x,y))
    charges = np.asarray(charges/np.max(charges))
    #charges = np.sqrt(charges)
    #charges = np.asarray(charges)
     
    tree = cKDTree(points)
    density = np.zeros(len(points))
    
    radius = cutoff * sigma

    for i,p in enumerate(points):

        idx = tree.query_ball_point(p, r=radius)
        neigh = points[idx]
        q = charges[idx]

        dist2 = np.sum((neigh - p)**2, axis=1)
        weights = np.exp(-dist2/(2*sigma**2))
        
        density[i] = np.sum(q * weights)

    return density

def plotDetailedEvent(matrix,
                    picname,
                    debug=False,
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

    if(debug):
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

    # TODO: finish this one later

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


# simple version of blob detection
#def detectBlobs(matrix):
#
#    # setting threshold
#    thr = filters.threshold_otsu(matrix)
#    binary = matrix > thr
#
#    # removing noise
#    binary = morphology.remove_small_objects(binary, min_size=30)
#
#    # labeling connected regions
#    labels = measure.label(binary)
#    regions = measure.regionprops(labels, intensity_image=matrix)
#    
#    good_blobs = []
#
#    for reg in regions:
#
#        width = reg.bbox[3] - reg.bbox[1]
#        height = reg.bbox[2] - reg.bbox[0]
#
#        aspect_ratio = height/width if width > 0 else 0 
#
#        # filter fro compact blobs
#        if(
#            10<=width<=40 and 
#            reg.area>100 and 
#            aspect_ratio < 2.5 and
#            reg.eccentricity < 0.85
#        ):
#            good_blobs.append(reg)
#
#    return good_blobs

# same, but with "watersheding" and plotting

def checkRegionCounts(matrix, nsplit=4):

    h, w = matrix.shape
    sh = h//nsplit
    sw = w//nsplit
    
    blocks = matrix.reshape(nsplit, sh, nsplit, sw)
    blocks = blocks.swapaxes(1,2)

    reg_cts = blocks.sum(axis=(2,3))

    return reg_cts


def detectBlobs(matrix,
                outdir,
                picname,
                min_blob_width = 5,
                max_blob_width = 40,
                min_area = 25,
                aspect_ratio_thr = 3,
                excent_thr = 3.5,
                show_separation = True
                ):

    # getting threshold
    thr = filters.threshold_otsu(matrix)
    #thr = 1000
    binmatrix = matrix > thr

    # removing "noise"
    binmatrix = morphology.remove_small_objects(binmatrix, min_size=20)

    # distance transform
    distances = distance_transform_edt(binmatrix)

    # smoothing
    distances = gaussian_filter(distances,sigma=1)

    # finding local maxima (peaks of the blobs)
    # min_distance -> controls separation sensitivity
    coords = peak_local_max(distances,min_distance=5,labels=binmatrix)
    markers = np.zeros_like(matrix, dtype=int)
    for i, (r,c) in enumerate(coords):
        markers[r,c] = i+1

    # watersheding 
    labels = watershed(-distances, markers, mask=binmatrix)

    # filtering regions
    regions = measure.regionprops(labels, intensity_image=matrix)

    good_regs = []
    spark = False
    spark_regions = []

    for reg in regions:
        
        w = reg.bbox[3] - reg.bbox[1]
        h = reg.bbox[2] - reg.bbox[0]
        aspect_ratio = h / w if w > 0 else 0 
        
        # tryna' detect BLOBS       
        if(
            min_blob_width <= w <= max_blob_width and
            reg.area >=min_area and
            aspect_ratio < aspect_ratio_thr and 
            reg.eccentricity < excent_thr
          ):

            good_regs.append(reg)

        # tryna' detect SPARKS
        if(w<=5 and h>50 and reg.eccentricity > 0.98):
            spark = True
            spark_regions.append(reg)

    # optionally plot separation shite
    if(show_separation):

        #fig, axes = plt.subplots(1,3, figsize = (15,6))
        fig, axes = plt.subplots(1,4, figsize = (20,6))

        # original frame
        ax0 = axes[0]
        ax0.imshow(matrix, cmap='jet')
        ax0.set_title("Original Frame")
        ax0.set_axis_off()
        
        # bounding boxes
        ax1 = axes[1]
        ax1.imshow(matrix, cmap='jet')
        ax1.set_title("Detected Blobs")
        ax1.set_axis_off()

        for r in good_regs:

            minr, minc, maxr, maxc = r.bbox
            rect = plt.Rectangle(
                    (minc,minr),
                    maxc - minc,
                    maxr - minr,
                    fill=False,
                    edgecolor='red',
                    linewidth=1.5 
            )
            ax1.add_patch(rect)

        if(spark):
            for sp in spark_regions:

                minr, minc, maxr, maxc = sp.bbox
                rect = plt.Rectangle(
                        (minc,minr),
                        maxc - minc,
                        maxr - minr,
                        fill=False,
                        edgecolor='firebrick',
                        linewidth=1 
                )
                ax1.add_patch(rect)
                ax1.text(maxc+1,maxr+1,"Spark",textcolor='white')

        # watershed shite
        ax2 = axes[2]
        colored_labels = label2rgb(labels,bg_label=0)
        ax2.imshow(colored_labels)
        ax2.set_title("Watershed Segmentation")
        ax2.set_axis_off()

        ax3 = axes[3]
        ax3.hist(matrix.ravel(),bins=64)
        ax3.axvline(thr,color='r')
        ax3.set_xlim([np.nanmin(matrix),np.nanmax(matrix)])
        ax3.set_yscale('log')
        
        #for ax in axes:
        #    ax.set_axis_off()

        plt.tight_layout()
        plt.savefig(f"{outdir}ION-SEPARATION-{picname}.png")
        plt.close()


    return good_regs, labels, spark

def simpleCountBlobs(matrix,
                    min_bwidth = 5,
                    max_bwidth = 40,
                    min_area = 50,
                    max_aspect_ratio = 2.5,
                    max_excent = 4
                    ):

    # getting threshold
    #thr = filters.threshold_otsu(matrix)

    thr = None
    binmatrix = None

    if(np.ma.isMaskedArray(matrix)):
        matrix =  matrix.filled(0)
        binmatrix = matrix > 0 
    else:
        thr = filters.threshold_otsu(matrix)
        binmatrix = matrix > thr 

    # removing "noise"
    binmatrix = morphology.remove_small_objects(binmatrix, min_size=20)

    # distance transform
    distances = distance_transform_edt(binmatrix)

    # smoothing
    distances = gaussian_filter(distances,sigma=1)

    # finding local maxima (peaks of the blobs)
    # min_distance -> controls separation sensitivity
    coords = peak_local_max(distances,min_distance=5,labels=binmatrix)
    markers = np.zeros_like(matrix, dtype=int)
    for i, (r,c) in enumerate(coords):
        markers[r,c] = i+1

    # watersheding 
    labels = watershed(-distances, markers, mask=binmatrix)
    nlabels = len(labels)

    # filtering regions
    regions = measure.regionprops(labels, intensity_image=matrix)

    nblobs = 0
 
    for reg in regions:
        
        w = reg.bbox[3] - reg.bbox[1]
        h = reg.bbox[2] - reg.bbox[0]
        aspect_ratio = h / w if w > 0 else 0 
        
        # tryna' detect BLOBS       
        if(
            min_bwidth <= w <= max_bwidth and
            reg.area >=min_area and
            aspect_ratio < max_aspect_ratio and 
            reg.eccentricity < max_excent
          ):

            nblobs+=1

    return nblobs, nlabels


def checkSparks(matrix, 
                min_col = 3,
                min_norm_height = 0.5,
                thr=None
                ):

    # Handle masked arrays
    if np.ma.isMaskedArray(matrix):
        matrix = matrix.filled(0)

    # Binary image
    if thr is None:
        binary = matrix > 0
    else:
        binary = matrix > threshold

    ny, nx = binary.shape

    # Active pixels per column
    colsum = np.sum(binary, axis=0)

    # Column must contain at least this many active pixels
    min_hits = int(min_norm_height * ny)

    suspicious = np.where(colsum >= min_hits)[0]

    if len(suspicious) == 0:
        return False, [], []

    # Find consecutive runs
    runs = []
    start = suspicious[0]
    prev = suspicious[0]

    for c in suspicious[1:]:
        if c == prev + 1:
            prev = c
        else:
            runs.append((start, prev))
            start = prev = c

    runs.append((start, prev))

    # Keep only sufficiently wide runs
    good_runs = []
    columns = []

    for s, e in runs:
        width = e - s + 1
        if width >= min_col:
            good_runs.append((s, e))
            columns.extend(range(s, e + 1))

    return len(good_runs) > 0, columns, good_runs

############################################################
# MAIN
############################################################

#####################################
# instantiating MyPlotter
MU = myUtils()
MP = myPlotter()
MC = myColors()
#######################

parser = ap.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default='', help="loaction ofr the data runs")
parser.add_argument('-n', "--name", type=str, default='EBALA', help="suffix of the plots")
parser.add_argument("-SR", "--srbit", type=int, default=1, help="SR bits")
parser.add_argument("-ST", "--stbit", type=int, default=15, help="ST bits")
parser.add_argument("--runtime", type=int, default=-1, help="recorded run time of the measurement (enter n minutes,\ndefault:-1, in this case using t_dead=44ms)")
parser.add_argument("--nframes", type=int, default=-1, help="Number of frames to plot (Default=-1), Enter positive integer")
parser.add_argument("--bsframes", type=int, default=-1, help="Number of bullshit frames to plot (Default=-1), Enter positive integer")
parser.add_argument("--cutnoise", action='store_true' , help="cut noisy frames out")
parser.add_argument("--plotglobal", action='store_true' , help="Plot global parametere histograms")
#parser.add_argument("--plotframes", action='store_true' , help="Plot single frames")
parser.add_argument("--makedir", action='store_true' , help="Dump plots in a custom-named directory")
parser.add_argument("--hitrate", action='store_true' , help="Plot hits vs time for a run")
parser.add_argument("--countTOT", action='store_true' , help="Count ions based on the average TOT in a single cluster events")
parser.add_argument("--countBS", action='store_true' , help="Also count bullshit frames. One frame = one ion")
parser.add_argument("--chess", action='store_true' , help="process run with the chessboard pixel config")
parser.add_argument("--ironrun", action='store_true', help="process Fe55 run")
parser.add_argument("--savecounts", action='store_true', help="save counting results to hardcoded json file")

#parser.add_argument("--TOA", action='store_true' , help="Process TOA run")
#parser.add_argument("--TOT", action='store_true' , help="Process TOT run")

args = parser.parse_args()

directory = args.dir
picname = args.name
SR = args.srbit
ST = args.stbit
nframes = args.nframes
nbsframes = args.bsframes
fChess = args.chess
fIron = args.ironrun
fSaveCounts = args.savecounts

t_run = args.runtime

if(not os.path.isdir(directory)):
    print(f"Directory {directory} does not exist, nahuy....")
    exit(0)

counting_results = "RunData-JSON/runIonCounts.json"

chessPixConfig = np.loadtxt("/media/vlad/NitroBeam/from-tpc23/RUNPARAMS/RunParams_006844_250826_12-57-42/chip_1_board_0_fec_0_matrix.txt",dtype=int)
mode_matrix = np.zeros((256,256),dtype=int)
toa_pixels, tot_pixels = None, None
if(fChess):
    for ipix in range(256):
        for jpix in range(256):
            regval = str(np.binary_repr(chessPixConfig[i,j],width=14))
            mode_bit = 8
            if(regval[mode_bit]=="1"):
                np.add.at(mode_matrix, (ipx,jpix), 1)

    toa_pixels = np.where(mode_matrix>0)
    tot_pixels = np.where(mode_matrix==0)

fPlotGlobalHists = args.plotglobal
fCutNoise = args.cutnoise
fMakeDir = args.makedir
fPlotHitRate = args.hitrate
fCountBS = args.countBS
fCIT = args.countTOT
fSingle = False

fPlotFrames = True if nframes > 0 else False
nSingleIons = 0
print("Flag check:")
print(f"Global Plots: {fPlotGlobalHists}")
print(f"Cut Noise: {fCutNoise}")
print(f"Count Bullshit Frames: {fCountBS}")
print(f"Make New Directory: {fMakeDir}")

if(fPlotFrames):
    print(f"PlotFrames: {fPlotFrames} : NEED {nframes}")
else:
    print(f"PlotFrames: {fPlotFrames}")

print("\n")

outdir = None
if (fMakeDir):
    outdir = f"NITRO-TOT-{picname}/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

filelist = glob.glob(directory+"*.txt")
nfiles = len(filelist)
print(f"Directory: {directory} has <{nfiles}> files.")

###################
# pulling data from DB json for current run 
run_base_dir_name = directory.split("/")[-2]
print(run_base_dir_name)
run_numb = run_base_dir_name.split("_")[1]
run_label = "Run_"+run_numb
print(f"Run number is: {MC.BLUE} {run_numb} {MC.RST}")

fsr_data = None
#fsrfile = "RunData-JSON/fsr_parameters.json"
fsrfile = "RunData-JSON/fsr_parameters_v3.json"

with open(fsrfile,'r') as jfile:
    runDB = json.load(jfile)
    try:
        fsr_data = runDB[run_label]
    except Exception as ex:
        print(f"Could not load json for {run_numb}: {ex}")
        #exit(0)

jfile.close()
jfile=None

#for key, val in fsr_data.items():
#    print(f"{key}-->{val}")
#
#exit(0)
##################

matrix = initMatrix(256,int)
totmatrix = initMatrix(256,int)

t_dead = 0.048 # deadtime in seconds
t_frame = getFrameTime(ST,SR)
fEstimatedTdead = False

fTimeFromInput = False

if(t_run > 0):
    fEstimatedTdead = True
    fTimeFromInput = True
    t_dead = estimateDeadTime(t_frame, nfiles, t_run*60) # since my input times are in minutes...
    print(f"Run time specified--> using estimated t_dead={t_dead:.4f} [s]")
else:
    t_run = fsr_data["t_run"]
    if(t_run != -1):
        fEstimatedTdead = True
        t_dead = estimateDeadTime(t_frame, nfiles, t_run)
    print(f"Run time based on JSON data is {t_run:.2f} [s] --> using estimated t_dead={t_dead:.4f} [s]")

# lists for first step in gathering global characteristics of frames

runtime = []
hitrate = []
fucked_hitrate = []

dtimes = []
part_trun = 0
full_time = 0

all_frame_hits = []
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
nIonBlobs = 0
total_blob_tot = 0
avg_blob_tots = []

occ_matrix = initMatrix(256,int)
occ_matrix_nonoise = initMatrix(256,int)
occ_matrix_noise = initMatrix(256,int)
occ_matrix_shite = initMatrix(256,int)
maxout_matrix = initMatrix(256,int)

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
low_nhits, high_nhits = 0, 0
npics = 0
nbspics = 0

for file in filelist:

    ifile+=1
    tmp_matrix = np.loadtxt(file,dtype=int)

    # i-th frame hits and sum TOT
    nhits = np.count_nonzero(tmp_matrix)
    sumTOT = np.sum(tmp_matrix)
    
    iframe_dt = estimateFrameDeadTime(nhits)
    dtimes.append(iframe_dt)

    all_frame_hits.append(nhits)
    time_iframe = None
    if(fPlotHitRate):
        time_iframe = ifile*0.048
        #time_iframe = ifile*iframe_dt

    # for now lets see full picture
    # if have more than 30k hits : probably spark and bullshit frame - discard for first loop
    if(fCutNoise and (nhits > 30000 or nhits < 500)):

        nzpixels = np.nonzero(tmp_matrix)

        if(nbsframes>0 and nbspics<nbsframes and nhits>30000):
            print(f"{MC.YELLOW} --> Saving BS frame {nbspics} with {nhits} hits... {MC.RST}")
            MP.plotMatrix(tmp_matrix,
                          f"BS-FRAME-{ifile}-{picname}",
                          labels=[f"High Occupancy Event {ifile}", "x,[pixel]","y,[pixel]"],
                          #infopos=[-90,25],
                          outdir=outdir,
                          figsize=(8,7),
                          #figtype=ipictype,
                          #geomshapes=[circ_sig1,circ_sig2],
                          #plotMarker=[meany,meanx],
                          fLognorm=True,
                          fDebug=True)
            nbspics+=1
        if(nhits>30000 and fPlotHitRate):
            fucked_hitrate.append([time_iframe, nhits])
        if(nhits>3e4):
            high_nhits+=1
            #nzpixels = np.nonzero(tmp_matrix)
            for nzx,nzy in zip(*nzpixels):
                np.add.at(occ_matrix_shite, (nzx,nzy), 1)
        if(nhits<500):
            low_nhits+=1
            #nzpixels = np.nonzero(tmp_matrix)
            for nzx,nzy in zip(*nzpixels):
                np.add.at(occ_matrix_noise, (nzx,nzy), 1)
        nskipped+=1
        continue
    if(fPlotHitRate):
        fucked_hitrate.append([time_iframe,-999]) 
        hitrate.append([time_iframe, nhits]) 

    maxout = tmp_matrix.astype(float)/11810.0 # checking if pixel value approaches/exceeds certain value
    maxout_matrix += maxout >= 1.0

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
    try:
        rfTOT_onesig = onesigmaTOT/sumTOT
    except ZeroDivisionError:
        rfTOT_onesig = -999
    try:
        rfTOT_twosig = twosigmaTOT/sumTOT
    except ZeroDivisionError:
        rfTOT_twosig = -999

    frame_FI1STD.append(rfTOT_onesig)
    frame_FI2STD.append(rfTOT_twosig)
    # calculating relative annoumt of charge inside area between sigma1 and sigma2 lines
    frac_QBsigmas = (twosigmaTOT - onesigmaTOT)/sumTOT
    frame_QBsigmas.append(frac_QBsigmas)

    # cheking average noise outside the 2sigma circle
    try:
        avg_noise = outTOT/outHits
    except ZeroDivisionError:
        acg_noise = -999
    #frame_avg_noise.append(outTOT/outHits)
    frame_avg_noise.append(avg_noise)
   
    # getting charge density of hits within 2*sigma_min
    try:
        qdensity = twosigmaTOT/twosigmaHits
    except ZeroDivisionError:
        qdensity = -999
    frame_qdensity.append(qdensity)
    
    # approximate, circular clsuter area of 2 sigmas
    cluster_area = getClusterCircleArea(minRadius*2)
    # area as a fraction of pixels covered within 2 sigmas / total number of pixels
    try:
        cluster_area_pixfrac = twosigmaHits/NPIXELS
    except ZeroDivisionError:
        cluster_area_pixfrac = -999

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

    # using blob-finder to count ions
    #blobs, labs, fSpark = detectBlobs(tmp_matrix, outdir, str(ifile), show_separation=False)
   
    testmat = np.zeros((256,256),dtype=int)
    goodpix = np.where((tmp_matrix>500) & (tmp_matrix<11810))
    goodtot = tmp_matrix[goodpix] 
    np.add.at(testmat, (goodpix), goodtot)
    nblobs_frame, nlabs_frame = simpleCountBlobs(testmat)
    #nblobs_frame = len(blobs)
    nIonBlobs+=nblobs_frame 

    testmat, goodpix, goodtot = None, None, None

    #nblob_tot = 0
    #for b in blobs:
    #    iblobtot = b.intensity_image[b.image].sum()
    #    nblob_tot += iblobtot
    #    total_blob_tot += iblobtot    
   
    #if(len(blobs) > 0):
    #    avg_blob_tots.append(nblob_tot/len(blobs))
    
    #if(fSpark):
    #    nSparks+=1 

    # checking for sparks
    spark, cols, runs = checkSparks(tmp_matrix, min_norm_height=0.8)
    fSpark = False
    if(spark):
        fSpark = True
        nSparks += 1
        print(f"Suspected spark - FRAME={ifile}, Columns={cols}, Runs={runs}")
        
    #if((fPlotFrames and npics <= nframes) or nblobs_frame>=4):
    if((fPlotFrames and npics <= nframes) or fSpark):
        #if(nhits>100 and nhits<10000):
        if(nhits>100):
            #print(f"IS SINGLE CLUSTER: [{fSingleCluster}]")
            comment = ""
            comment+=r"$N_{hits}$="+f"{nhits}\n:"
            comment+=r"$\Sigma$(TOT)="+f"{sumTOT}\n:"
            if(fSpark):
                comment+="SUS.SPARK\n:"
                comment+=f"ColNr={runs}\n:"
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

            ipictype = "png"
            snap_these = [101,103,104]
            if(ifile in snap_these):
                ipictype = "pdf"

            MP.plotMatrix(tmp_matrix,
                          f"FRAME-{ifile}-{picname}",
                          #labels=[f"EVENT-{ifile}", "x,[pixel]","y,[pixel]"],
                          labels=[f"", "x,[pixel]","y,[pixel]"],
                          info=comment,
                          infopos=[-65,25],
                          outdir=outdir,
                          figsize=(8,7),
                          figtype=ipictype,
                          sizeTextAxis=14,
                          sizeTextAxisLabels=14,
                          #geomshapes=[circ_sig1,circ_sig2],
                          #plotMarker=[meany,meanx],
                          fLognorm=True,
                          fDebug=True)

            ipname = str(ifile)

            blobs, _ , _ = detectBlobs(tmp_matrix, outdir, ipname, show_separation=True)
            print(f"{MC.BLUE}skimage found [ {len(blobs)} ] blobs{MC.RST}")
            blobs = None

            #print(f"{MC.YELLOW}Checking regions{MC.RST}")
            ##ndiv = 4 re-defined on very top as global param
            #active_pix = (tmp_matrix.T>150) & (tmp_matrix.T < 11810)
            #check_mat = np.zeros((256,256),dtype = int)
            #np.add.at(check_mat, active_pix, 1)

            ##reg_counts = np.round(checkRegionCounts(check_mat,ndiv)/npix_redreg, 4)*100.0
            #reg_counts = checkRegionCounts(check_mat,ndiv)
            #reg_counts_norm = np.round(reg_counts/np.sum(reg_counts),2)
            #print(f"Active pixels:\n{reg_counts_norm}" )


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
                                                    peak_width=PWIDTH,
                                                    fDebug=False)

    prelim_nxpeaks, prelim_nypeaks = len(peaksX), len(peaksY)
    if(fPlotFrames and npics <= nframes):
        if(nhits>100 and nhits<10000):
            print(f"\"find_peaks\" found [ {prelim_nxpeaks} / {prelim_nypeaks} ] peaks")
 
    nIonPeaks += countIons(len(peaksX),len(peaksY))

    for xp,xw in zip(propsX["prominences"],propsX['widths']):
        xprominences.append(int(xp))
        xwidths.append(float(xw))

    for yp,yw in zip(propsY["prominences"],propsY['widths']):
        yprominences.append(int(yp))
        ywidths.append(float(yw))
 
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
    #if((fPlotFrames and npics<=nframes) or nblobs_frame>7):
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

            blobs, _ , _ = detectBlobs(tmp_matrix, outdir, "MASKED-"+str(ifile))

            #iframe_coords = np.where((tmp_matrix.T>250) & (tmp_matrix.T<11800))
            #iframe_tots = tmp_matrix.T[iframe_coords]
            #iframe_density = computeChargeDensity(iframe_coords, iframe_tots)
            #plt.figure(figsize=(8,7))
            #plt.scatter(*iframe_coords, c=iframe_density, s=10, cmap='inferno')
            ##hlineset = np.linspace(0,256,ndiv)
            ##vlineset = np.linspace(0,256,ndiv)
            #for step in range(ndiv):
            #    plt.hlines(256-step*(256/ndiv),0,256,linestyles="--", color="red")
            #    plt.vlines(256-step*(256/ndiv),0,256,linestyles="--", color="red")
            #plt.xlabel("x,[pix]")
            #plt.ylabel("y,[pix]")
            #plt.xlim([0,256])
            #plt.ylim([0,256])
            #plt.title(f"Normalized Charge Density [frame {ifile}]")
            #plt.savefig(f"{outdir}TOT-DENSITY-{ifile}.png")
            #plt.close()

            #density_matrix = np.zeros((256,256),dtype=float)
            #np.add.at(density_matrix, (iframe_coords), iframe_density)
            #plotDetailedEvent(
            #        density_matrix,
            #        f"FRAME-DENSITY-{ifile}-{picname}",
            #        outdir=outdir
            #)

            #blobs, _ , _ = detectBlobs(density_matrix, outdir, "DENSITY-"+str(ifile))

            #iframe_coords = None
            #iframe_tots = None
            #iframe_density = None

            npics+=1

    #if(ifile==100):
    #    exit(0)
    
    MU.progress(nfiles, ifile)
    #MU.progress_bar(nfiles, ifile)
    # for loop ends here

print("============================")
print(f"FOUND {nIonPeaks} ION PEAKS")
print("============================")

print("============================")
print(f"FOUND {nIonBlobs} ION BLOBS")
print("============================")

print(f"--- Skipped {nskipped}/{nfiles} frames")
print(f"--- Found {nSingleIons} Single Ions")
print(f"--- Found {nSparks} sparks")

avg_tot_sions = None
try:
    avg_tot_sions = sum(single_sumTOT)/nSingleIons
except ZeroDivisionError:
    avg_tot_sions = 0

print(f"Found {nSingleIons} single ions")
print(f"average TOT per Ion = {avg_tot_sions:.2f} (PEAK COUNTING)")


#total_avg_blob_tot = None
#try:
#    total_avg_blob_tot = sum(avg_blob_tots)/len(avg_blob_tots)
#except ZeroDivisionError:
#    total_avg_blob_tot = 0

#print(f"average TOT per Ion = {total_avg_blob_tot:.2f} (BLOB counting)")

MP.plotMatrix(frame_charge_weights, 
              f"CHARGE_CENTERS-{picname}",
              labels=[f"Charge Weight Centers", "x, [pix]", "y, [pix]"],
              #cmap='viridis',
              cbarname=r"$N_{frames}$",
              cmap='coolwarm',
              outdir=outdir,
              fDebug=True)

MP.plotMatrix(occ_matrix,
                f"OCCUPANCY-{picname}",
                labels=["Occupancy Matrix", "x,[pix]", "y.[pix]"],
                cbarname=r'$N_{hits}$',
                cmap='coolwarm',
                outdir=outdir,
                #fLognorm=True,
                fDebug=True)

# plotting occupancy from events from BS frames...
if(np.sum(occ_matrix_shite)>0):
    MP.plotMatrix(occ_matrix_shite,
                    f"OCCUPANCY-BSFRAMES-{picname}",
                    labels=["Occupancy Matrix (High-Occupancy Frames)", "x,[pix]", "y.[pix]"],
                    cbarname=r'$N_{hits}$',
                    outdir=outdir,
                    cmap='coolwarm',
                    #fLognorm=True,
                    fDebug=True)
# plotting occupancy from events from BS frames...
if(np.sum(occ_matrix_noise)>0):
    MP.plotMatrix(occ_matrix_noise,
                    f"OCCUPANCY-NOISEFRAMES-{picname}",
                    labels=["Occupancy Matrix (Low-Occupancy Frames)", "x,[pix]", "y.[pix]"],
                    cbarname=r'$N_{hits}$',
                    cmap='coolwarm',
                    outdir=outdir,
                    #fLognorm=True,
                    fDebug=True)

MP.plotMatrix(maxout_matrix,
                f"OVERLY-ACTIVE-PIXELS-{picname}",
                labels=["Pixels Constantly Approaching Max TOT", "x,[pix]", "y.[pix]"],
                #cmap='viridis',
                cbarname=r'$N_{hits}$',
                cmap='coolwarm',
                outdir=outdir,
                #fLognorm=True,
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
plt.savefig(f"{outdir}Combined-Yaxis-DIFFs-{picname}.png")
plt.close()

MP.plotMatrix(no_response_pix,
                f"QUIET-PIXELS-{picname}",
                labels=["Pixels with 0 Activity over the Run Time", "x,[pix]", "y.[pix]"],
                figsize=(8,7),
                info=dcomment,
                infopos=[0,-20],
                outdir=outdir,
                cmap="Greens",#"gray",
                #cbarmax=1,
                cbarname="ON/OFF",
                #fLognorm=True,
                fDebug=True)

if(fPlotGlobalHists):

    MP.simpleHist(
            np.array(all_frame_hits), 
            100, 
            0, 
            np.max(all_frame_hits), 
            ["Global Hits per Frame (No Hit Cuts)", r"$N_{hits}$", r"$N_{frames}$"],
            f"ALL-HITS-{picname}",
            outdir=outdir,
            ylog=True
            )
 
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
    fig, (axg, axb) = plt.subplots(2,1,figsize=(10,6))
    #axg = fig.add_subplot(0,1)
    #axb = fig.add_subplot(1,1, sharex=axg)

    axg.plot([hitrate[i][0] for i in range(len(hitrate))],
              [hitrate[i][1] for i in range(len(hitrate))],
              color='darkblue', marker='s')
    
    axb.plot([fucked_hitrate[i][0] for i in range(len(fucked_hitrate))],
               [fucked_hitrate[i][1] for i in range(len(fucked_hitrate))],
               color='firebrick', marker='o')

    axg.set_title("Hits vs Time [Good frames]")
    axg.tick_params(labelbottom=False)
    axg.set_ylabel(r'$N_{hits}$')
    axg.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
    axg.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
    axg.minorticks_on()

    axb.set_title("Hits vs Time [Bad frames]")
    axb.set_xlabel("Time, [s]")
    axb.set_ylabel(r'$N_{hits}$')
    axb.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
    axb.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
    axb.minorticks_on()


    plt.tight_layout()
    plt.savefig(f"{outdir}HITRATE-{picname}.png")
    plt.close()

    # plotting averaged hitrate over time

    avg_step = 10
    time = [hitrate[i][0] for i in range(len(hitrate))]
    hits = [hitrate[i][1] for i in range(len(hitrate))]
    
    nbins = None
    if(nfiles<4000):
        nbins = len(time)//avg_step
    else:
        nbins = 100
        avg_step = len(time)//100

    median_hits = np.median(hits)
 
    means, edges, _ = binned_statistic(
        time,
        hits,
        statistic='mean',
        bins=nbins
    )

    std, _, _ = binned_statistic(
        time,
        hits,
        statistic='std',
        bins=nbins
    )

    centers = 0.5*(edges[:-1]+edges[1:])

    fig, ax = plt.subplots(figsize=(10,6))
    #ax.scatter(centers,means,color="forestgreen",marker="s")
    ax.errorbar(centers,means,yerr=std,color="forestgreen", ecolor="green",fmt="s")
    ax.hlines(median_hits,time[1],time[-2],colors='darkblue',linestyles="--", label=f"median rate = {median_hits}")
    ax.set_title("")
    ax.set_xlabel("Time, [s]")
    ax.set_ylabel(r"$\mathrm{N}_{\mathrm{hits}}$ averaged over"+f"{avg_step} consecutive frames, [N]")
    ax.grid(True)
    ax.legend()
    plt.savefig(f"{outdir}HITRATE-AVERAGED-{picname}.png")


print(f"Read {nfiles}, skipped {nskipped} (low={low_nhits},high={high_nhits})")

if(fCIT):

    N_Nitrogen = 0
   
    nCF_high = 0
    total_sumTOT = 0
    total_BSframes = 0

    n_nitro_blobs, n_nitro_objects = 0, 0
 
    nbad_frames = 0
    nfiles = len(filelist)
    ifile = 0
    for f in filelist:
    
        ifile+=1
        tmp_matrix = np.loadtxt(f,dtype=int)
        nhits = np.count_nonzero(tmp_matrix)# using total hits 
        iframe_dt = estimateFrameDeadTime(nhits)
        full_time += iframe_dt+t_frame

        if(nhits > 30000 or nhits < 500):
             total_BSframes += 1
             nbad_frames += 1
             if(fCountBS and nhits > 30000):
                N_Nitrogen += 1
                nbad_frames -= 1
                nCF_high += 1
             continue
   
        part_trun += iframe_dt+t_frame

        THR = 500 #250
        badIDX = np.where(tmp_matrix<THR)
        skippix = 5
        masked_matrix = np.ma.masked_array(tmp_matrix, np.zeros_like(tmp_matrix, dtype=int))
        masked_matrix[badIDX] = np.ma.masked # masking pix with TOT<THR
        masked_matrix[:skippix, :] = np.ma.masked # masking top Nskip
        masked_matrix[-skippix:,:] = np.ma.masked # bottom 
        masked_matrix[:, :skippix] = np.ma.masked # left
        masked_matrix[:, -skippix:] = np.ma.masked # right

        # using pixels with TOT above threshold
        cut_sumTOT = masked_matrix.filled(0).sum()
        total_sumTOT += cut_sumTOT

        # collecting data for occupancy matrix but with noise suppressed
        nonzero_pix = np.nonzero(masked_matrix.filled(0))
        for ix, iy in zip(*nonzero_pix):
            np.add.at(occ_matrix_nonoise, (ix,iy), 1)

        N_Nitrogen += np.floor(float(cut_sumTOT)/float(avg_tot_sions))

        nhits, cut_sumTOT = None, None
        tmp_matrix = None

        # counting blobs in masked matrix
        frame_nblobs, frame_nlabs = simpleCountBlobs(masked_matrix)
        #n_nitro_blobs += simpleCountBlobs(masked_matrix)
        n_nitro_blobs += frame_nblobs
        n_nitro_objects += frame_nlabs

        MU.progress(nfiles, ifile)

    print(f"\nSeparate Counting results in {N_Nitrogen} Ions in {ifile} frames")
    print(f"________\nFound {nCF_high} crap frames with N>3e4 (out of {nbad_frames} in total)\n~~~~~~~~~~")    

    # Extrapolating how many ions collected in total

    print(f"t_frame={t_frame} s")
    t_dead_msg = f"t_dead={t_dead:.4f}"
    if(fEstimatedTdead):
        t_dead_msg += " [ESTIMATED]!"
    print(t_dead_msg) 

    # ----------------------------------------------
    total_runtime = None
    if(t_run == -1):
        total_runtime = t_frame+t_dead*ifile
        print(f"Using [estimated] run time {total_runtime}")
    else:
        total_runtime = t_run # converting to seconds
        #total_runtime = t_run*60 # converting to seconds
        if(fTimeFromInput):
            print(f"Using [provided] run time {total_runtime}")
        else:
            print(f"Using [JSON] run time {total_runtime}")

    # Counting Ions by using "find_peaks" 
    #
    total_nitrogen = t_dead/t_frame*N_Nitrogen
    ionrate = total_nitrogen/total_runtime

    # Counting Ions based on average TOT per single Ion
    # (sumTOT(All_frames)/avgTOT(single ion))
    #
    N_Nitrogen_TOT = total_sumTOT/avg_tot_sions
    total_nitrogen_tot = t_dead/t_frame*N_Nitrogen_TOT
    ionrate_tot = total_nitrogen_tot/total_runtime

    # Counting Ions based on the recognized blobs over all data set
    #
    #N_blob_TOT = total_sumTOT/total_avg_blob_tot
    #total_blob_tot = (t_dead+t_frame)/t_frame*N_blob_TOT
    #ionrate_blob = total_blob_tot/total_runtime
    
    #total_nitro_blobs = (t_dead+t_frame)/t_frame*n_nitro_blobs
    #total_nitro_blobs = t_dead/t_frame*nIonBlobs
    total_nitro_blobs = t_dead/t_frame*n_nitro_blobs
    total_nitro_objects = t_dead/t_frame*n_nitro_objects
    ionrate_blobs = total_nitro_blobs/total_runtime
    ionrate_objects = total_nitro_objects/total_runtime

    # here counting the number of recognizable events 
    # & dividing by the number of frames they were in
    # Then extrapolating that rate over the whole run
    ngood_frames = nfiles - nbad_frames
    part_runtime = (t_frame+t_dead)*ngood_frames
    #part_runtime = part_trun 
    part_rate = N_Nitrogen/part_runtime
    extrap_ion_dose = part_rate*total_runtime
 
    print(f"{MC.BLUE}------ COUNTING (EXTRApolating) -------{MC.RST}")
    print(f"Accounting only for good frames: {ngood_frames} --> {part_runtime:.2f} [s] of good runtime")
    print(f"Total run time = {full_time:.4f} [s]")
    print(f"Partial run time (only good frames + estimated iframe dead time) = {part_trun:.4f} [s]")
    print(f"Partial run time (only good frames + estimated average dead time) = {part_runtime:.4f} [s]")
    print(f"Partial ion rate = {part_rate:.2f} [Hz]")
    print(f"{MC.GREEN_BGR}Corresponds to total exposure of <{extrap_ion_dose:.2f}> ions{MC.RST}")
    
    print(f"{MC.RED}------ COUNTING (Global TOT / avgTOT-per-ion) -------{MC.RST}")
    print(f"Averge TOT-per-single cluster = {avg_tot_sions:.2f}")
    print(f"Total collected TOT = {total_sumTOT:.2f}")
    print(f"Resulting in {N_Nitrogen_TOT:.2f} ions detected")
    print(f"Resulting in {ionrate_tot:.2f} [Hz]")
    print(f"{MC.GREEN_BGR}OVERALL DOSE: {total_nitrogen_tot:.2f} ions{MC.RST}\n")

    print(f"{MC.GREEN}------ COUNTING (BLOB-counting-based) -------{MC.RST}")
    #print(f"Frames have {n_nitro_blobs:.2f} valid BLOBS")
    print(f"Frames have {MC.CYAN}{n_nitro_blobs:.2f} valid BLOBS, {n_nitro_objects:.2f} objects{MC.RST} in total found")
    print(f"Resulting in {ionrate_blobs:.2f} [Hz] / {ionrate_objects:.2f} [Hz]")
    print(f"{MC.GREEN_BGR}OVERALL DOSE: {total_nitro_blobs:.2f} ions{MC.RST}, {MC.CYAN_BGR}{total_nitro_objects:.2f}{MC.RST}\n") 

    print(f"{MC.YELLOW}------ COUNTING (FRAME-based TOT counting) -------{MC.RST}")
    print(f"Ions observed = {N_Nitrogen:.2f}")
    print(f"Run time = {total_runtime/60.0:.2f} minutes")
    print(f"Dead-time scaling factor = {t_dead/t_frame:.2f}")
    print(f"rate={ionrate:.2f} [Hz]")
    print(f"{MC.GREEN_BGR}OVERALL DOSE: {total_nitrogen:.2f} ions{MC.RST}")
   
    if(fSaveCounts): 
        with open(counting_results, "r") as cr:
            iondata = json.load(cr)
        print("Got the counting json file")

        iondata["counts"][run_label] = {
            "runtype": "TOT",
            "extrap": extrap_ion_dose, 
            "avgTOT":total_nitrogen_tot, 
            "blobCTS":total_nitro_blobs, 
            "frameTOT": total_nitrogen,
            "NBSframes": total_BSframes
        }       

        print("Added data")
        with open(counting_results,"w") as cr:
            json.dump(iondata, cr, indent=4)

        print("Saved counting json file")


MP.plotMatrix(occ_matrix_nonoise,
                f"OCCUPANCY-PRUNED-{picname}",
                #labels=["Occupancy Matrix (PRUNED)", "x,[pix]", "y.[pix]"],
                labels=["", "x,[pix]", "y.[pix]"],
                cbarname=r'$N_{hits}$',
                outdir=outdir,
                #fLognorm=True,
                fDebug=True)

print("FINITO!")
