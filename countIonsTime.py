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


def initMatrix(size, dtype):
    return np.zeros((size,size),dtype=dtype)

def getSumTOT(matrix):
    return np.sum(matrix)

def getHits(matrix):
    return np.count_nonzero(matrix)

def getFrameTime(st,sr):

    return 256**sr * 46 * st / 4e7

def estimateDeadTime(tframe, nframes, trun):

    # lets use seconds here...
    tdead = trun/nframes - tframe
    return tdead
 
def getFramePeaks(matrix,        # matrix (original or pruned)
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

    return xpeaks, ypeaks, xprops, yprops

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

    #plt.tight_layout()
    if(outdir is not None):
        plt.savefig(f"{outdir}{picname}.png")
    else: 
        plt.savefig(f"{picname}.png")

    plt.close()

def plotToaClusters(clusterlist, clusterlabels, labels, outdir, picname):

    fig, ax = plt.subplots(figsize=(9,8))

    colors = plt.cm.viridis(np.linspace(0,1,len(clusterlist)))

    for clist, clab, clr in zip(clusterlist, clusterlabels, colors):
       ax.scatter(*clist, marker='s', color=clr, s=2, label=f"{clab}")
    ax.set_title(labels[0])
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[2])
    ax.set_xlim([0,256])
    ax.set_ylim([0,256])
        
    plt.legend(loc='upper left')
    plt.savefig(f"{outdir}TOA-CLUSTERS-{picname}.png")
    plt.close() 

def checkRegionCounts(matrix, nsplit=4):

    h,w = matrix.shape
    sh = h // nsplit
    sw = w // nsplit
    blocks = matrix.reshape(nsplit, sh, nsplit, sw)
    blocks = blocks.swapaxes(1,2)
    reg_counts = blocks.sum(axis=(2,3))
    return reg_counts
    
def filterNoise(coords, radius=2, min_neigh=2):
    
    y,x = coords
    points = np.column_stack((x,y))
    print(f"filtering {len(points)}")
    if(len(points)==0):
        return coords
    
    keep_mask = np.zeros(len(points), dtype=int)
    for i, p in enumerate(points):    

        dist2 = np.sum((points - p)**2, axis=1)
        neigh = np.sum((dist2<=radius**2) & (dist2>0))

        if(neigh >= min_neigh):
            keep_mask[i] = True

        #MU.progress_bar(i,len(points))
    return (y[keep_mask],x[keep_mask])


def filterNoiseKDTree(coords, radius=2, min_neigh=2):

    y,x = coords
    points = np.column_stack((x,y))
    if(len(points)==0):
        return coords

    tree = cKDTree(points)
    neigh = tree.query_ball_point(points, r=radius)
    keep_mask = np.array([len(n)-1 >= min_neigh for n in neigh])
    return (y[keep_mask],x[keep_mask])


#def computeDensity(coords, potential='gauss', sigma=3.0):
def computeDensity(coords, sigma=3.0):

    #fGauss = False
    #fExponent = False

    #if(potential=="gauss"):
    #    fGauss = True    
    #if(potential=="expo"):
    #    fExponent = True

    y,x = coords
    points = np.column_stack((x,y))
    N = len(points)
    density = np.zeros(N)
    for i, p in enumerate(points):
        #if(fGauss):
        dist2 = np.sum((points-p)**2, axis=1)
        density[i] = np.sum(np.exp(-dist2/(2*sigma**2)))
        #if(fExponent):
        #    dist = np.sum(points-p, axis=1)
        #    density[i] = np.sum(np.exp(-dist/(64)))

    return density

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
parser.add_argument("-SR", "--srbit", type=int, default=0, help="SR bits")
parser.add_argument("-ST", "--stbit", type=int, default=250, help="ST bits")
parser.add_argument("--runtime", type=int, default=-1, help="recorded run time of the measurement (enter n minutes,\ndefault:-1, in this case using t_dead=44ms)")
parser.add_argument("--nframes", type=int, default=-1, help="Number of frames to plot (Default=-1), Enter positive integer")
parser.add_argument("--cutnoise", action='store_true' , help="cut noisy frames out")
args = parser.parse_args()


directory = args.dir
picname = args.name
SR = args.srbit
ST = args.stbit
t_run = args.runtime
nframes = args.nframes

if(not os.path.isdir(directory)):
    print(f"Directory {directory} does not exist, nahuy....")
    exit(0)

fPlotFrames = True if nframes > 0 else False
fCutNoise = args.cutnoise


outdir = f"NITRO-CHARGE-{picname}/"
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

with open("RunData-JSON/fsr_parameters.json") as jfile:
    runDB = json.load(jfile)
    try:
        fsr_data = runDB[run_label]
    except Exception as ex:
        print(f"Could not load json for {run_numb}: {ex}")
        #exit(0)

jfile.close()
jfile=None

t_dead = 0.048 # deadtime in seconds (for ~20.83 Hz readout rate in full matrix readout)
t_frame = getFrameTime(ST,SR)
fEstimatedTdead = False

if(t_run > 0):
    fEstimatedTdead = True
    t_dead = estimateDeadTime(t_frame, nfiles, t_run*60) # since my input times are in minutes...
    print(f"Run time specified--> using estimated t_dead={t_dead:.4f} [s]")
else:
    t_run = fsr_data["t_run"]
    #fEstimatedTdead = True
    t_dead = estimateDeadTime(t_frame, nfiles, t_run)
    print(f"Run time based on JSON data --> using estimated t_dead={t_dead:.4f} [s]")

occ_matrix = initMatrix(256,int)

ifile, nskipped, npics = 0, 0, 0

nTOAclusters = 0

for file in filelist:

    ifile+=1
    tmp_matrix = np.loadtxt(file, dtype=int)

    nhits = np.count_nonzero(tmp_matrix)
    sumTOT = np.sum(tmp_matrix)
    if(fCutNoise and (nhits > 30000 or nhits < 500)):
        nskipped+=1
        continue

    ########### further happens only with normal frames ############# 
    nonzeropix = np.nonzero(tmp_matrix)

    cts, edges = np.histogram(np.ravel(tmp_matrix[nonzeropix]), 100, range=(0,11810))
    toa_peaks, _ = find_peaks(cts, 
                              rel_height=0.2, 
                              distance=3,
                              width=2)
    npeaks = len(toa_peaks)   

    if(fPlotFrames and npics <= nframes):
        MP.plotMatrix(tmp_matrix,
                        f"FRAME-TOA-{ifile}-{picname}",
                        labels=[f"EVENT-{ifile}", "x,[pixel]","y,[pixel]"],
                        infopos=[-90,25],
                        outdir=outdir,
                        figsize=(8,7),
                        cbarname="TOA",
                        #figtype=ipictype,
                        fLognorm=False,
                        fDebug=False)

        plotDetailedEvent(tmp_matrix,
                          f"FRAME-TOA-{ifile}-{picname}",
                         outdir=outdir)

        #print(f"Found peaks in TOA histogram => {npeaks}")

        bin_centers = (edges[:-1]+edges[1:])/2
        plt.figure(figsize=(8,8))
        plt.hist(bin_centers, weights=cts, bins=100, range=(0,11810),align='left',histtype='stepfilled',facecolor='darkblue')
        plt.xlabel("TOA")
        plt.ylabel(r"$N_{cts}$")
        plt.title(f"TOA histogram frame={ifile}")
        plt.yscale('log')
        plt.savefig(f"{outdir}TOA-HIST-{ifile}.png")
        plt.close()

        npics+=1

        #########################################################
        #########################################################

    for x,y in zip(*nonzeropix):
        np.add.at(occ_matrix, (x,y), 1)
    
    toa_clusters = []
    start = None
    acc = 0
    sumTHR, sumTHRmax = 50, 30000
    for i in range(len(cts)):
        if(cts[i]>= sumTHR):
            if start is None:
                start = i
            acc+=cts[i]
        else:
            if start is not None:
               #if(acc>=sumTHR and acc<sumTHRmax):
               if(acc>=sumTHR):
                    toa_clusters.append((edges[start],edges[i]))
               start = None
               acc = 0
    if start is not None and acc>=sumTHR:
        toa_clusters.append((edges[start],edges[-1]))
        
    toa_coords, toa_labels = [], []
    #print(f"{MC.BLUE}------FRAME--[{ifile}]------{MC.RST}")
    #print(f"HAVE FOLLOWING BOUNDS:\n{toa_clusters}")   

    fUniformCluster = False
    for tc_low, tc_high in toa_clusters:
        toa_pixels = np.where((tmp_matrix.T>=tc_low) & (tmp_matrix.T<=tc_high))
        imask = (tmp_matrix.T>=tc_low) & (tmp_matrix.T<=tc_high)
        test_matrix = np.zeros((256,256),dtype=int)
        np.add.at(test_matrix, imask, 1)
        reg_counts = checkRegionCounts(test_matrix,nsplit=4) # nsplit=2 and vc<0.4 works
        varcoeff = np.std(reg_counts)/(np.mean(reg_counts)+1e-9)
        #print(reg_counts)
        if(varcoeff < 0.7):
            #print(f"{MC.RED}vc = {varcoeff:.2f}; likely UNIFORM{MC.RST}")
            fUniformCluster = True
        else:
            fUniformCluster = False
            #print(f"{MC.GREEN}vc = {varcoeff:.2f}, likely HAS CLUSTERS{MC.RST}")

        if(len(toa_pixels[0])>=30000 or fUniformCluster):
            fUniformCluster = False
            continue

        #print(f"{MC.YELLOW}OLEG{MC.RST}")
        good_pixels = filterNoiseKDTree(toa_pixels, radius=1, min_neigh=2)
        ############################################################# 
        ############################################################# 
        ## tryna filter out bgr noise cluster
        #pix_stdx = np.std(toa_pixels[0])
        #pix_stdy = np.std(toa_pixels[1])
        #if(pix_stdx > 25.0 and pix_stdy > 25.0):
        #    print(f"large std(x/y), sumHits={len(toa_pixels)} between {tc_low} and {tc_high}")
        #    part_density = computeDensity(toa_pixels)
        #    plt.figure(figsize=(8,7))
        #    plt.scatter(*toa_pixels, c=part_density, s=10)
        #    plt.xlabel("x [pixel]")
        #    plt.ylabel("y [pixel]")
        #    plt.xlim([0,256])
        #    plt.ylim([0,256])
        #    plt.title("Pixel Densities")
        #    plt.savefig(f"{outdir}DENSITIES-LARGE-STDXY-{ifile}-{picname}.png")
        #    plt.close()
        #    toa_pixels = None
    toa_coords.append(good_pixels)
    #print(len(toa_coords[0]))
    toaMean = np.mean([tc_low,tc_high])
    nTOAclusters+=1
    toa_labels.append(f"TOA={toaMean:.2f}")
    #print(f"frame[{ifile}]: Mean cluster TOA = {toaMean:.2f}, stdx={pix_stdx:.2f}, stdy={pix_stdy:.2f}")
 
    if(fPlotFrames and npics <= nframes):
        plotToaClusters(toa_coords,
                        toa_labels,
                        [f"TOA Clusters, frame {ifile}","x,[pix]","y,[pix]"],
                        outdir,
                        f"{ifile}-{picname}")

        all_pixels = np.nonzero(tmp_matrix.T)
        all_densities = computeDensity(all_pixels, sigma=1.5)
        plt.figure(figsize=(8,7))
        plt.scatter(*all_pixels, c=all_densities, s=10)
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        plt.xlim([0,256])
        plt.ylim([0,256])
        plt.title("Pixel Densities")
        plt.savefig(f"{outdir}DENSITY-{ifile}-{picname}.png")
        plt.close()

    MU.progress_bar(ifile,nfiles)
    # end of FRAME loop
print("\n")
print(f"t_frame = {t_frame} [s]")
print(f"t_dead = {t_dead} [s]")
print(f"t_run = {t_run} [s]")


print(f"\n{MC.GREEN}Recognized {nTOAclusters} ions from TOA clustering{MC.RST}")
total_nitro_ions = t_dead/t_frame*nTOAclusters
total_rate = total_nitro_ions/t_run
print(f"{MC.GREEN}TOTAL dose = {total_nitro_ions:.2f} [ions]{MC.RST}")
print(f"{MC.GREEN}ion rate = {total_rate:.2f} [Hz]{MC.RST}")

MP.plotMatrix(occ_matrix,
              f"OCCUPANCY-TOA-{picname}",
              labels=[f"Matrix Occupancy", "x,[pixel]","y,[pixel]"],
              outdir = outdir,
              figsize=(10,9),
              figtype="png",
              cbarname="occurance",
              fLognorm=True,
              fDebug=False)


