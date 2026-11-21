import sys
import os
import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import tables as tb
from sklearn.cluster import DBSCAN 
from scipy.optimize import curve_fit
from scipy.special import erf
from lmfit import Model
from matplotlib.colors import LogNorm
import math
import glob
import datetime as dtime
from matplotlib.patches import Ellipse, Rectangle
import argparse as ap
import math

from MyPlotter import myPlotter
from MyPlotter import myUtils
from MyPlotter import mySymbols
from MyPlotter import myColors

import ROOT as rt 

import json

# for interactive 3d plotting 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

####################################################
MP = myPlotter()
MU = myUtils()
MC = myColors()
MS = mySymbols()
####################################################

parser = ap.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default="TEST", help="name for plots (suffix)")
parser.add_argument("-d", "--dir", type=str, default=None, help="Custom directory name")
parser.add_argument("-f", "--files", nargs="+", help="list of files ala ~/TPA/reco-weighted*\nCan be single file.")
parser.add_argument("--debug", action='store_true', help="Enable debug output")
parser.add_argument("--plotevents", action='store_true', help="Enable frame plotting")
parser.add_argument("--plotglobal", action='store_true', help="Enable global data plotting")
parser.add_argument("--npics", type=int, default=10, help="Number of frames to plot DEF:[10]" )

args = parser.parse_args()

picname = args.name
recofiles = args.files
custom_directory = args.dir
debug = args.debug
npics = args.npics

fGlobalPlots = args.plotglobal
fPlotTracks = args.plotevents

ncol, nrow = 256, 256

glob_matrix_tot = np.zeros((ncol,nrow),dtype=int)
glob_matrix_occ = np.zeros((ncol,nrow),dtype=int)

outdir = ""
if(custom_directory is not None):
    outdir += custom_directory+"/"
else:
    outdir += f"tmp-{plotname}/"

if not os.path.exists(outdir):
    os.makedirs(outdir)

ifile = 0 

comb_sumTOT, comb_hits, comb_redq = [], [], []

fIronRuns, fProtonRuns = False, False

for file in recofiles:

    if("Fe55" in file):
        fIronRuns = True
    elif("SCX" in file):
        fProtonRuns = True
    else:
        print(f"DAFUQ is run < {file} > about?")

    print(f"... processing <{file}> | Iron ?= {fIronRuns}, Protons ?= {fProtonRuns}")
    ifile+=1

    matrix_tot = np.zeros((ncol,nrow),dtype=int)
    matrix_occ = np.zeros((ncol,nrow),dtype=int)

    tmp_sumTOT ,tmp_nhits = [], []
    tmp_redUncQ = []

    with tb.open_file(file, 'r') as f:
    
        groups = f.walk_groups('/')
        grouplist = []
        for gr in groups:
            print(f'found {gr}')
            grouplist.append(gr)
        main_group = str(grouplist[len(grouplist)-1])
   
        grouplist = None 
    
        basewords = main_group.split('(')
   
        base_group_name = basewords[0][:-1]+'/'
        #                              ^ removes space at the end of 'run_xxx/chip0 '
        bgn_split = base_group_name.split('/')
        run_name = bgn_split[2]
        run_num = int(run_name[4:])

        groups = None
        grouplist = None
        main_group = None
        run_name, run_num, bgn_split = None, None, None 
        basewords = None

        # =========== gettin' shit ==================
        ToT = f.get_node(base_group_name+"ToT")
        #if(debug):
        #    print(f"found VLarray TOT of size {type(ToT)}")
        #centerX = f.get_node(base_group_name+"centerX")
        #if(debug):
        #    print(f"found centerX {type(centerX)}")
        #centerY = f.get_node(base_group_name+"centerY")
        #if(debug):
        #    print(f"found centerY {type(centerY)}")
        hits = f.get_node(base_group_name+"hits")
        #if(debug):
        #    print(f"found hits {type(hits)}")
        #excent = f.get_node(base_group_name+"eccentricity")
        #if(debug):
        #    print(f"found excentricity {type(excent)}")
        #length = f.get_node(base_group_name+"length")
        #if(debug):
        #    print(f"found length {type(length)}")
        #RMS_L = f.get_node(base_group_name+"rmsLongitudinal")
        #if(debug):
        #    print(f"found RMS_L {type(RMS_L)}")
        #RMS_T = f.get_node(base_group_name+"rmsTransverse")
        #if(debug):
        #    print(f"found RMS_T {type(RMS_T)}")
        #rotAng = f.get_node(base_group_name+"rotationAngle")
        #if(debug):
        #    print(f"found rotAng {type(rotAng)}")
        sumTOT = f.get_node(base_group_name+"sumTot")
        #if(debug):
        #    print(f"found sumTOT {type(sumTOT)}")
        #width = f.get_node(base_group_name+"width")
        #if(debug):
        #    print(f"found width {type(width)}")
        x = f.get_node(base_group_name+"x")
        #if(debug):
        #    print(f"found x {type(x)}")
        y = f.get_node(base_group_name+"y")
        #if(debug):
        #    print(f"found y {type(y)}")
        #EVENTNR = f.get_node(base_group_name+"eventNumber")
        #if(debug):
        #    print(f"found eventNumber {type(EVENTNR)}")

        # ---------- EVENT LOOP -----------------------------
        ipic = 0
        ievent = 0
        ntotal = ToT.shape[0]

        lo_sumTOT, hi_sumTOT = 0, 0
        lo_hits, hi_hits = 45, 4096 
        if(fIronRuns):
            lo_sumTOT, hi_sumTOT = 0, 1e5
            lo_hits = 45 
        elif(fProtonRuns):
            lo_sumTOT, hi_sumTOT = 0, 1e6
            lo_hits = 100
        else:
            lo_sumTOT, hi_sumTOT =  0, 1e9
            lo_hits = 25

        for totevent in ToT:

            goodEvent = sumTOT[ievent]<hi_sumTOT and hits[ievent]>lo_hits

            if(goodEvent):
   
                event_matrix = np.zeros((ncol,nrow),dtype=int)        

                for ix,iy,itot in zip(x[ievent],y[ievent],totevent):
                    np.add.at(matrix_tot, (ix,iy), itot)
                    np.add.at(matrix_occ, (ix,iy), 1)
                    if(fPlotTracks and ipic<npics):
                        np.add.at(event_matrix, (ix,iy), itot)
        
                tmp_sumTOT.append(sumTOT[ievent]) 
                tmp_nhits.append(hits[ievent])
                try:
                    tmp_redUncQ.append(sumTOT[ievent]/hits[ievent])
                except ZeroDivisionError:
                    tmp_redUncQ.append(-999)
                if(fPlotTracks and ipic<npics):
                    dataorigin = file.split("/")[-1].split(".")[0]
                    MP.plotMatrix(event_matrix,
                                f"FRAME-{ievent}-{ifile}-{picname}",
                                #labels=[f"{dataorigin}\nFrame {ievent}", "x,[pixel]","y,[pixel]"],
                                labels=["", "x,[pixel]","y,[pixel]"],
                                outdir = outdir,
                                figsize=(9,8),
                                figtype="png",
                                cbarname="TOT",
                                sizeTextAxis=14,
                                sizeTextAxisLabels=14,
                                #fLognorm=True,
                                fDebug=False)
                    ipic+=1
                event_matrix = None

            ievent+=1
            MU.progress_bar(ievent,ntotal)

    ## ----- loop over file ends HERE -------
    print("\n")
    MP.plotMatrix(matrix_occ,
                  f"OCCUPANCY-Protons-{ifile}-{picname}",
                  labels=[f"Matrix Occupancy", "x,[pixel]","y,[pixel]"],
                  outdir = outdir,
                  figsize=(10,9),
                  figtype="png",
                  cbarname="occupancy",
                  #fLognorm=True,
                  fDebug=False)

    MP.plotMatrix(matrix_tot,
                  f"TOT-Protons-{ifile}-{picname}",
                  labels=[f"Matrix TOT", "x,[pixel]","y,[pixel]"],
                  outdir = outdir,
                  figsize=(10,9),
                  figtype="png",
                  cbarname="TOT",
                  #fLognorm=True,
                  fDebug=False)

    maxSumTOT, maxhits, maxRedQ = None, None, None
    if("Fe55" in file):
        maxSumTOT = 8e4
        maxhits = 500
        maxRedQ = 800
    elif("SCX" in file):
        maxSumTOT = 1e6
        maxhits = 4100
        maxRedQ = 500
    else:
        maxSumTOT = np.max(tmp_sumTOT)
        maxhits = np.max(tmp_nhits)
        maxRedQ = np.max(tmp_redUncQ)
    MP.simpleHist(np.array(tmp_sumTOT),
                  100,
                  0,
                  maxSumTOT,
                  [r"$\sum$(TOT) over the data set (TOTAL)", r"$\sum$(TOT)",r"$N_{Clusters}$"],
                  f"TOTAL-SUMTOT-{ifile}-{picname}",
                  outdir=outdir#,
                  #ylog=True
                  )

    MP.simpleHist(np.array(tmp_nhits),
                  100,
                  0,
                  maxhits,
                  ["Hits per Cluster over the data set (TOTAL)", r"$\mathrm{N}_{\mathrm{hits}}$",r"$N_{Clusters}$"],
                  f"TOTAL-NHITS-{ifile}-{picname}",
                  outdir=outdir#,
                  #ylog=True
                  )

    MP.simpleHist(np.array(tmp_redUncQ),
                  100,
                  0,
                  maxRedQ,
                  ["Reduced Uncalibrated Charge per Hit", r"$\sum(\mathrm{TOT})/\mathrm{N}_{\mathrm{Hits}}$",r"$N_{Clusters}$"],
                  f"TOTAL-REDTOT-{ifile}-{picname}",
                  outdir=outdir#,
                  #ylog=True
                  )


    # list of coordinates for 10 noisiest pixels
    ban_x = [146,131,168,213,146,138,206,167,194,204]
    ban_y = [116,123,175,246,117,39,84,175,202,245]
    #badIDX = np.column_stack((ban_x, ban_y))
    masked_occup = np.ma.masked_array(matrix_occ, np.zeros_like(matrix_occ, dtype=int))
    masked_occup[ban_x, ban_y] = np.ma.masked

    MP.plotMatrix(masked_occup,
                  f"OCCUPANCY-MASKED-Protons-{ifile}-{picname}",
                  labels=[f"Matrix Occupancy (MASKED)", "x,[pixel]","y,[pixel]"],
                  outdir = outdir,
                  figsize=(10,9),
                  figtype="png",
                  cbarname="occupancy",
                  #fLognorm=True,
                  fDebug=False)

    fname, ithlabel = None, None
    if(fIronRuns):
        #fname = file.split('/')[-1].split('.')[0].split("-")[-2:]
        #ithlabel = ",".join(fname)
        if("23Jul" in file and "Morning" in file):
            ithlabel = "Before Irradiation"
        elif("24Jul" in file and "Morning" in file):
            ithlabel = "After Irradiation Day1"
        elif("24Jul" in file and "Evening" in file):
            ithlabel = "After Irradiation Day2"
        else:
            fname = file.split('/')[-1].split('.')[0].split("-")[-2:]
            ithlabel = ",".join(fname)

    elif(fProtonRuns):
        fname = file.split('/')[-1].split('.')[0].split("-")[-6:]
        ithlabel = ",".join(fname)
    else:
        fname = file.split('/')[-1].split('.')[0].split("-")
        ithlabel = ",".join(fname)
    comb_sumTOT.append([ithlabel,tmp_sumTOT])
    comb_hits.append([ithlabel,tmp_nhits])
    comb_redq.append([ithlabel,tmp_redUncQ])

    # file loop end HERE 

if(fIronRuns):
    maxSumTOT = 6e4
    maxhits = 500
    maxRedQ = 800
elif(fProtonRuns):
    maxSumTOT = 1e6
    maxhits = 4100
    maxRedQ = 500
else:
    maxSumTOT = np.max([np.max(data) for data in comb_sumTOT])
    maxhits = np.max([np.max(data) for data in comb_hits])
    maxRedQ = np.max([np.max(data) for data in comb_redq])


MP.simpleMultiHist(comb_sumTOT,
                   100,
                   0,
                   maxSumTOT,
                   ["Combined Fe55 Spectrum", "Uncalibrated Charge (TOT), [CTS]", r"$\mathrm{N}_{\mathrm{Frames}}$"],
                   f"SUMTOT-{picname}",
                   odir=outdir,
                   fNoTitle=True)

MP.simpleMultiHist(comb_hits,
                   100,
                   0,
                   maxhits,
                   ["Number of Hits per Cluster", ", [N]", r"$\mathrm{N}_{\mathrm{Frames}}$"],
                   f"NHITS-{picname}",
                   odir=outdir,
                   fNoTitle=True)

MP.simpleMultiHist(comb_redq,
                   100,
                   0,
                   maxRedQ,
                   ["Reduced Uncalibrated Charge per Cluster Hits", r"$\sum\mathrm{(TOT)}/\mathrm{N}_{\mathrm{Hits}}$, [CTS]", r"$\mathrm{N}_{\mathrm{Frames}}$"],
                   f"REDTOT-{picname}",
                   odir=outdir,
                   fNoTitle=True)





