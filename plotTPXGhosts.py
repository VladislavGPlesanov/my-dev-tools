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

def estimateFrameDeadTime(nhits):

    tdead_frame = td_slope*nhits + td_offset 

    return tdead_frame



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

args = parser.parse_args()

directory = args.dir
picname = args.name
SR = args.srbit
ST = args.stbit

if(not os.path.isdir(directory)):
    print(f"Directory {directory} does not exist, nahuy....")
    exit(0)

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

hits, sumtot, time = [], [], []
dead_times = []

occ_matrix = initMatrix(256,int)

ifile=0
t_frame = 0.048 #[s]
time_clock = 0.0
ncheck, icheck = 20, 0
for file in filelist:

    ifile+=1
    tmp_matrix = np.loadtxt(file,dtype=int)

    # i-th frame hits and sum TOT
    nhits = getHits(tmp_matrix)
    sumTOT = getSumTOT(tmp_matrix)
    hits.append(nhits)    
    sumtot.append(sumTOT)    

    if(icheck<ncheck and nhits> 300):
        MP.plotMatrix(tmp_matrix,
                      f"FRAME-{ifile}-{picname}",
                      labels=[f"SUS Event {ifile}", "x,[pixel]","y,[pixel]"],
                      #outdir=outdir,
                      figsize=(8,7),
                      fLognorm=True,
                      fDebug=True)
        icheck+=1


    iframe_dt = estimateFrameDeadTime(nhits)
    
    dead_times.append(iframe_dt)
    time_clock+=iframe_dt+t_frame
    time.append(time_clock)

    nonzeropix = np.nonzero(tmp_matrix)

    for ix, iy in zip(*nonzeropix):
        np.add.at(occ_matrix, (ix,iy), 1)

    MU.progress_bar(ifile, nfiles)


MP.plotMatrix(
    occ_matrix,
    f"OCCUPANCY-PRUNED-{picname}",
    labels=["Occupancy Matrix (PRUNED)", "x,[pix]", "y.[pix]"],
    cbarname=r'$N_{hits}$',
    #outdir=outdir,
    #fLognorm=True,
    fLognorm=True,
    fDebug=True
)

MP.plotScatter(
    time,
    hits,
    ["","Time,[s]",r"$\mathrm{N}_{\mathrm{Hits}}$"],
    f"HITS-{picname}",
    ufigsize=(10,6),
    fLogY=True
)

MP.plotScatter(
    time,
    sumtot,
    ["","Time,[s]",r"$\sum$(TOT)"],
    f"SUMTOT-{picname}",
    ufigsize=(10,6),
    fLogY=True
)

MP.simpleHist(
        np.array(dead_times), 
        100, 
        0, 
        np.max(dead_times), 
        ["Dead-Times of all Frames in a Run", r"$\mathrm{t}_{\mathrm{dead}}$", r"$N_{frames}$"],
        f"DEAD-TIMES-{picname}",
        ylog=True
        )
 
