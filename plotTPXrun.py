import sys
import os 
import numpy as np
import numpy.ma as ma
import argparse as ap
import glob

from time import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors  import LogNorm
from matplotlib import cm
from matplotlib.gridspec import GridSpec

from MyPlotter import myPlotter 
from MyPlotter import myUtils 
from MyPlotter import myColors

from scipy.ndimage import label
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def fancyTime(ts,te):

    runtime_s = te - ts
    msg = "Total runtime:"
    if(runtime_s/60.0<1):
        msg+=f"{runtime_s:.2f} [s]"
    if(runtime_s%60>=1):
        time_min = runtime_s//60
        msg+=f"{time_min:.2f} [min]"
    elif(runtime_s%3600>=1):
        time_hour = runtime_s/3600
        msg+=f"{time_hour:.2f} [h]"
    print(msg)


def getAxisRange(plot, axis):

    ax = plot.gca() 
    if(axis==0):
        return ax.get_xlim()   
    elif(axis==1):
        return ax.get_ylim()   
    else:
        print(f"ERROR: no axis with index {axis}")
        return -1,-1


def dynamic_dbscan(points, min_samples_scale = 0.1, eps_scale = 2.0, n_neighbors = 5):

    #neigh = NearestNeighbors(n_neighbors=5)
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(points)
    distances, _ = neigh.kneighbors(points)

    mean_nn_distance = np.median(distances[:,1]) # gettting characteristic distance between hits

    eps = eps_scale * mean_nn_distance

    min_samples = max(3, int( len(points) * min_samples_scale))

    db = DBSCAN(eps=eps, min_samples=min_samples)

    labels = db.fit_predict(points)

    return labels, eps, min_samples, mean_nn_distance

#####################################
# instantiating MyPlotter
MU = myUtils()
MP = myPlotter()
MC = myColors()
#######################


parser = ap.ArgumentParser()
parser.add_argument("-d", "--dir", nargs="+", type=str, default='', help="loaction of the data runs", required=True)
#parser.add_argument("-d2", "--dir2", type=str, default='', help="loaction ofr the data runs")
parser.add_argument('-n', "--name", type=str, default='EBALA', help="suffix of the plots")
parser.add_argument("--descr", nargs="+", type=str,default='', help="Description of the runs in option -d (exact same order)")
parser.add_argument("--plotframes", type=int,default=-1, help="Plot and save N frames")

args = parser.parse_args()

directories = args.dir
#directory2 = args.dir2
picname = args.name
description = args.descr
nframes = args.plotframes

tglob_start = time()

flen = []
filelist = []
for d in directories:
    if(not os.path.isdir(d)):
        print(f"Directory {d} does not exist, nahuy....")
        exit(1)

    flist = glob.glob(d+"*.txt")
    flen.append(len(flist))
    filelist.append(flist)

#print(len(filelist[1]))

filecut = int(np.floor(np.min(flen)*1.1))
print(f"directories have {flen} files, minimum is {filecut}")
ncol, nrow = 256,256

occupancy_mat = np.zeros((ncol,nrow), dtype=int)
matrix = np.zeros((ncol,nrow), dtype=int)
sumTOT, nhits = [], []
ncl_frame = []
hitspercluster, totpercluster = [], []

iframe = 0

comp_TOT, comp_hits= [], []

matrixlist = []

#datalabels = ["Before", "After"]
ifile, idata = 0,0

#for ilist in [filelist, filelist2]:

for ilist in filelist:
   
    ilistlen = len(ilist)
 
    nfailedfiles = 0
    
    time_avg = 0.0
    for file in ilist:
        
        ### DOING SOME TIMING ESTIMATES FOR LULZ 
        tstart = time() 

        ifile+=1

        tmp_matrix = np.loadtxt(file, dtype=int)
        total_hits = np.count_nonzero(tmp_matrix)
    
        if(total_hits<50):
            tmp_matrix = None
            nfailedfiles += 1
            ifile+=1
            continue
    
        matrix += tmp_matrix
    
        # clustering via DBSCAN from plotIonOccupancy.py
    
        tot_cut = 25
        min_samples_scale = 0.1
        eps_scale = 10.0
        n_neigh = 2
    
        y,x = np.where(tmp_matrix >= tot_cut) # using threshold
        points = np.column_stack((x,y))
        npoints_after_cut = len(points)
    
        labels, eps, min_samples, mean_neigh_dist = dynamic_dbscan(points, 
                                                                min_samples_scale, 
                                                                eps_scale, 
                                                                n_neigh)
 
        itot_matrix = tmp_matrix.T[x,y]  
    
        unq_labels = set(labels)
        nClusters = len(unq_labels)-1
    
        for ilab in unq_labels:
            if(ilab==-1):
                continue
    
            mask = (labels == ilab)
            cHits = np.sum(labels==ilab)
            cTOT = np.sum(itot_matrix[mask])
   
            #print(f"Frame {ifile}, Cluster {ilab}: sumHits={cHits} | sumTOT={cTOT}")
 
            matrix_icl = np.zeros((ncol,nrow),dtype=int)
    
            hitspercluster.append(cHits)
            totpercluster.append(cTOT)
            #SUKA.append(cTOT)

        if(nframes>0 and iframe<nframes):
 
            MP.plotMatrix(tmp_matrix,
                            f"OrigFrame-{ifile}-{description[idata]}",
                            labels=["Original Frame Hits","x, pix", "y,pix"],
                            figsize=(8,8),
                            figtype="png",
                            cmap="jet",
                            cbarname="TOT [CTS]",
                            fLognorm=True,
                            fDebug=False
            ) 
       
            colors = plt.cm.tab20(np.linspace(0,1,len(unq_labels)))
        
            nnoise_clust = 0
            plt.figure(figsize=(8,8))
            for lab, col in zip(unq_labels, colors):
                class_mask = labels == lab
                if(lab==-1):
                    col = 'k'
                    nnoise_clust+=1
                plt.scatter(points[class_mask,0], points[class_mask, 1],marker='.', s=1, c=[col], label=f'Clust={lab}')
        
            plt.legend()
            plt.title(f"DBSCAN - found {len(unq_labels)-1} clusters")
            plt.xlabel("x, [pix]")
            plt.ylabel("y, [pix]")
        
            minx, _ = getAxisRange(plt,0)
            _, maxy = getAxisRange(plt,1)
        
            plt.text(minx*1.1, maxy-20 , f"TOT threshold = {tot_cut}")
            plt.text(minx*1.1, maxy-25 , f"{npoints_after_cut} out of {nhits} survived cut")
            plt.text(minx*1.1, maxy-30 , f"EPS={eps}, mean.neigh.dist.={mean_neigh_dist}, minimal cluster size = {min_samples} ")
            
            plt.xlim([-10,260])
            plt.ylim([-10,260])
        
            plt.savefig(f"DBSCAN-{description[idata]}-FRAME-{ifile}.png")
            plt.close()
    
    
        sumTOT.append(np.sum(tmp_matrix))
        nhits.append(np.count_nonzero(tmp_matrix))
        activepix = (tmp_matrix > 0)
        occupancy_mat += activepix
    
        tmp_matrix = None
    
        tend = time()
        dt = tend-tstart
        MU.times.append(dt)
        if(ifile%50==0):
            time_avg = sum(MU.times)/len(MU.times)
 
        #MU.progress_bar(ifile,ilistlen,time_est=time_avg)
        MU.progress_bar(ifile,filecut,time_est=time_avg)
    
        iframe+=1
        if(ifile>=filecut):
            break 
  
    print(f"directory has {nfailedfiles} files with nhits<50")
    MP.plotMatrix(matrix,
                  f"TOT-{description[idata]}-{picname}",
                  labels=[f"Matrix, Total TOT", "x,[pixel]","y,[pixel]"],
                  #outdir = outdir,
                  figsize=(9,8),
                  figtype="png",
                  cbarname=r"$\sum$TOT",
                  fLognorm=True,
                  fDebug=False)
    
    MP.plotMatrix(occupancy_mat,
                  f"OCCUP-{description[idata]}-{picname}",
                  labels=[f"Matrix, Occupancy", "x,[pixel]","y,[pixel]"],
                  #outdir = outdir,
                  figsize=(9,8),
                  figtype="png",
                  cbarname=r"$\mathrm{N}_{\mathrm{Entries}}$",
                  fLognorm=True,
                  fDebug=False)
    matrixlist.append([f"{description[idata]}",occupancy_mat.copy()])    
    
    MP.simpleHist(np.array(nhits),
                  100,
                  0,
                  np.max(nhits),
                  ["Hits per Frame (TOTAL)", r"$N_{Hits}$",r"$N_{Frames}$"],
                  f"TOTAL-HITS-{description[idata]}-{picname}"#,
                  #outdir=outdir,
                  #ylog=True
                  )
    
    MP.simpleHist(np.array(sumTOT),
                  100,
                  0,
                  np.max(sumTOT),
                  [r"$\sum$(TOT) per Frame (TOTAL)", r"$\sum$(TOT)",r"$N_{Frames}$"],
                  f"TOTAL-sumTOT-{description[idata]}-{picname}"#,
                  #outdir=outdir,
                   #ylog=True
                  )
    occupancy_mat = np.zeros((ncol,nrow),dtype=int) 
    matrix = np.zeros((ncol,nrow),dtype=int) 
    sumTOT.clear()
    nhits.clear()

    ## -------- cluster info -----------
    MP.simpleHist(np.array(hitspercluster),
                  100,
                  0,
                  320, #np.max(hitspercluster),
                  [r"$\sum$(HITS,Cluster) per Cluster", r"$\sum$(HITS, cluster)",r"$N_{Frames}$"],
                  f"TOTAL-HITS-CLUSTER-{description[idata]}-{picname}"#,
                  #outdir=outdir,
                  #ylog=True
                  )
    
    MP.simpleHist(np.array(totpercluster),
                  100,
                  0,
                  6.6e4, #np.max(totpercluster),
                  [r"$\sum$(TOT,Cluster) per Cluster", r"$\sum$(TOT, cluster)",r"$N_{Frames}$"],
                  f"TOTAL-sumTOT-CLUSTER-{description[idata]}-{picname}"#,
                  #outdir=outdir,
                  #ylog=True
                  )


    #print(f"\nCollected {len(SUKA)} cluster data...")
    comp_TOT.append([description[idata], totpercluster.copy()])
    #comp_hits.append([description[idata], hitspercluster])

    print(f"comp_TOT = {len(comp_TOT)}")
    print(f"TOTPERCLUSTER={len(totpercluster)}")
    perc5 = np.percentile(totpercluster,5)
    perc95 = np.percentile(totpercluster,95)
    print(f"AVG-TOT = {np.mean(totpercluster):.2f}, 5%={perc5:.2f}, 95%={perc95:.2f}")
    print(f"Files scanned: {ifile}, data={idata} -> {description[idata]}")
    totpercluster.clear()
    hitspercluster.clear()
    idata+=1
    iframe= 0  
    ifile = 0
     

##### plotting comparisons ###################

MP.simpleMultiHist(
    comp_TOT,
    100,
    0, 
    6.6e4,
    #["","Uncalibrated Charge TOT, [CTS]", r"$\mathrm{N}_{\mathrm{Clusters}/\sum(\mathrm{clusters})}$"],
    ["","Uncalibrated Charge TOT, [CTS]", r"$\mathrm{N}_{\mathrm{Clusters}}$"],
    f"Combined-SUMTOT-{picname}",
    odir=None,
    legendLabels=description,
    sizeTextAxis=14,
    sizeTextAxisLabels=14,
    fNorm=True
)

if(len(matrixlist)==2):
    matrixOne, matrixTwo = matrixlist[0][1], matrixlist[1][1] 
    descrOne, descrTwo = matrixlist[0][0], matrixlist[1][0]

#    # definin offset from sides
#    offset = 125
#    
#    # definning ROIs
#
#    window = 64
#    row_end = matrixOne.shape[0]-offset
#    row_start = row_end-window
#    col_end = matrixOne.shape[1]-offset
#    col_start = col_end-window

    row_start, row_end = 10, 235
    col_start, col_end = 25, 240

    npixrow = row_end - row_start
    npixcol = col_end - col_start
    totpix = npixrow*npixcol

    mask = np.ones((256,256), dtype=bool)
    mask[row_start:row_end,col_start:col_end] = False

    bare_roiONE = matrixOne[row_start:row_end,col_start:col_end]
    bare_roiTWO = matrixTwo[row_start:row_end,col_start:col_end]

    roi_one = np.ma.array(matrixOne, mask=mask)
    roi_two = np.ma.array(matrixTwo, mask=mask)

    zerocoord_one = np.where(roi_one==0)
    zerocoord_two = np.where(roi_two==0)

    print(f"empty pixels ({descrOne}) = {zerocoord_one}")
    print(f"empty pixels ({descrTwo}) = {zerocoord_two}")

    nz_one = np.count_nonzero(bare_roiONE)
    nz_two = np.count_nonzero(bare_roiTWO)

    print(f"Number of empty pixels {descrOne}={nz_one}, {descrTwo}={nz_two}")

    norm_one = roi_one/matrixOne.sum()
    norm_two = roi_two/matrixTwo.sum()    
   
    #ratio = np.divide(
    #    roi_one,
    #    roi_two,
    #    out=np.zeros_like(roi_one,dtype=float),
    #    where=roi_two !=0
    #)

    norm_ratio = np.divide(
        norm_one,
        norm_two,
        out=np.zeros_like(norm_one,dtype=float),
        where=norm_two !=0
    )

    MP.plotMatrix(roi_one,
                  f"ROI-1-OCCUPANCY-{picname}",
                  labels=[f"Matrix, Occupancy ROI 1 {descrOne}", "x,[pixel]","y,[pixel]"],
                  #outdir = outdir,
                  figsize=(9,8),
                  figtype="png",
                  cmap="Blues",
                  cbarname="Norm. Occupancy",
                  netblock=16,
                  fLognorm=True,
                  fDebug=False)
 
    MP.plotMatrix(roi_two,
                  f"ROI-2-OCCUPANCY-{picname}",
                  labels=[f"Matrix, Occupancy ROI 2 {descrTwo}", "x,[pixel]","y,[pixel]"],
                  #outdir = outdir,
                  figsize=(9,8),
                  figtype="png",
                  cmap="Blues",
                  cbarname="Norm. Occupancy",
                  netblock=16,
                  fLognorm=True,
                  fDebug=False)

    comment = ""
    comment+=f"({descrOne}) = {totpix-nz_one} inactive:"
    comment+=f"({descrTwo}) = {totpix-nz_two} inactive:"
    comment+=f"Range x=[{col_start};{col_end}], y=[{row_start};{row_end}]" 

    #MP.plotMatrix(ratio,
    MP.plotMatrix(norm_ratio,
                  f"DIFF-OCCUPANCY-{picname}",
                  labels=[f"Matrix, Normalized Occupancy Difference", "x,[pixel]","y,[pixel]"],
                  #outdir = outdir,
                  figsize=(9,8),
                  figtype="png",
                  #plotMarker=[],
                  cmap="coolwarm",
                  cbarname="Norm. Occupancy",
                  #info=f"Active Pix ({descrOne}) = {nz_one}/{totpix}:Active Pix ({descrTwo}) = {nz_two}/{totpix}",
                  info=comment,
                  infopos=[25,225],
                  netblock=16,
                  #fLognorm=True,
                  fDebug=False)


##################################################################
# alterantive counting for whole matrix rather than ROI

block = 16
nx,ny = matrixOne.shape # matrixTwo has exact same shape
scaldown_one = np.zeros((ny//block, nx//block), dtype=int)
scaldown_two = np.zeros((ny//block, nx//block), dtype=int)

normOne = matrixOne/matrixOne.sum()
normTwo = matrixTwo/matrixTwo.sum()

for iy in range(scaldown_one.shape[0]):
    for ix in range(scaldown_one.shape[1]):
        roi=normOne[
            iy*block:(iy+1)*block,
            ix*block:(ix+1)*block
        ]
        
        scaldown_one[ix,iy]+=np.count_nonzero(roi==0)

for iy in range(scaldown_two.shape[0]):
    for ix in range(scaldown_two.shape[1]):
        roi=normTwo[
            iy*block:(iy+1)*block,
            ix*block:(ix+1)*block
        ]
        
        scaldown_two[ix,iy]+=np.count_nonzero(roi==0)

print(f"{descrOne}:")
print(scaldown_one)
print(f"{descrTwo}:")
print(scaldown_two)

diff_scaldown = scaldown_two - scaldown_one
print("Difference:")
print(diff_scaldown)

max_sway = np.max([np.abs(np.min(diff_scaldown)), 
                   np.abs(np.max(diff_scaldown))])

fig, ax = plt.subplots(figsize=(10,9))
im = ax.imshow(diff_scaldown.T, cmap="coolwarm", vmin=-max_sway, vmax=max_sway)
plt.colorbar(im, ax=ax, label=(r"$\Delta$ inactive pix"))
# adding text to the diff matrix
for iy in range(diff_scaldown.T.shape[1]):
    for ix in range(diff_scaldown.T.shape[0]):
        val = (diff_scaldown.T[iy,ix]/256)*100

        ax.text(
            ix,
            iy,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black"
        )


ax.set_xticks(np.arange(diff_scaldown.shape[1]))
ax.set_yticks(np.arange(diff_scaldown.shape[0]))
ax.set_xlabel("Pixel Block X "+r"[$\times$"+f"{block}]",fontsize=14)
ax.set_ylabel("Pixel Block Y "+r"[$\times$"+f"{block}]",fontsize=14)

ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

ax.invert_yaxis()
plt.savefig(f"DiffMapMatrix-nPixPerBlock-{block}-{picname}.png")

plt.close()

tglob_end = time() 
fancyTime(tglob_start,tglob_end)

    
    













