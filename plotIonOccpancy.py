import numpy as np
import numpy.ma as ma
import sys
import os
import glob
import tables as tb
import matplotlib
matplotlib.use("Agg")  # non-GUI backend (renders to files only)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from MyPlotter import myUtils  
from lmfit import Model
from matplotlib import cm

from sklearn.cluster import DBSCAN

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

def getFrameTime(SR,ST):

    return 256**SR * 46 * ST / 4e7

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)

#def plotSimpleHist(counts, edges, nbins, RNG, labels, pname):
#
#    plt.figure(figsize=(8,8))
#    plt.hist(edges[:-1], weights=counts, bins=nbins, range=RNG, align='left', histtype='stepfilled', facecolor='b')
#    plt.title(labels[0])
#    plt.xlabel(labels[1])
#    plt.ylabel(labels[2])
#    plt.yscale('log')
#    plt.savefig(f"HIST-{pname}.png")

def plotSimpleHist(data, nbins, odir, RNG=None, labels=None, pname="hist"):

    data = np.asarray(data)
    if data.size == 0:
        print(f"[WARN] Empty dataset for {pname}, skipping plot.")
        return

    # Auto-detect range if not provided
    if RNG is None:
        RNG = (data.min(), data.max())

    counts, edges = np.histogram(data, bins=nbins, range=RNG)

    if counts.sum() == 0:
        print(f"[WARN] Histogram for {pname} has no counts in range {RNG}, skipping plot.")
        return

    plt.figure(figsize=(8, 8))
    plt.hist(edges[:-1], weights=counts, bins=nbins, range=RNG,
             align='left', histtype='stepfilled', facecolor='b')

    if labels:
        plt.title(labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])

    if np.any(counts > 1e4):
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(f"{odir}HIST-{pname}.png")
    plt.close()


def checkHitSpread(data):

    xproj = data.sum(axis=0)
    yproj = data.sum(axis=1)

    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])

    xmean = np.average(x, weights=xproj)
    ymean = np.average(y, weights=yproj)

    stdx = np.sqrt(np.average((x-xmean)**2, weights=xproj))
    stdy = np.sqrt(np.average((y-ymean)**2, weights=yproj))

    xx, yy = np.indices(data.shape)
    total = data.sum()

    centerx = (xx * data).sum() / total
    centery = (yy * data).sum() / total

    return centerx, centery, stdx, stdy

############################################################################

main_path = sys.argv[1]
picname = sys.argv[2]
SR = int(sys.argv[3])
ST = int(sys.argv[4])

outdir = f"tmp-{picname}/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

#t_dead = 2.5e-2
#t_dead = 25e-3 # true for ZS mode
t_dead = 0.048#full frame is slower
t_frame = getFrameTime(SR,ST)

inputlist = glob.glob(main_path+"*.txt")

nfiles = len(inputlist)

matrix = np.zeros((256,256), dtype=int)
tot_matrix = np.zeros((256,256), dtype=int)
empty_events = np.zeros((256,256), dtype=int)

nIons, nIons_sus = 0, 0

hHits = []
ionHits = []

nEmpty = 0

n_nitro = 0
n_nitro_tot = 0

cnt = 0
npics = 0

cluster_hits = []
cluster_tot = []

singleClust_tot, singleClust_nhits = [],[]

for file in inputlist:
    
    tmp_array = np.loadtxt(file, dtype=int)
    #print(tmp_array.shape)

    nhits = np.count_nonzero(tmp_array)

    hHits.append(nhits)

    sumTOT = np.sum(tmp_array)

    initro = 0
    #avg_nhits = 4617.38
    #avg_nhits = 388.80
    avg_nhits = 689.59
    #avg_TOT_ion = 2328861.65
    avg_TOT_ion = 786183.57 # 740630.45

    initro_tot = np.floor(float(sumTOT)/avg_TOT_ion)
    if(nhits<=20):
        empty_events += tmp_array

    if(nhits<=100):
    #if(nhits<=5): # enable for Fe
        cnt+=1
        nEmpty+=1
        continue

    if(nhits <= 30000 and sumTOT >= avg_TOT_ion/2.0):
        n_nitro_tot += initro_tot

    initro_tot = None

    if(nhits>=avg_nhits/2.0):
        initro = np.floor(float(nhits) / avg_nhits)
        n_nitro += initro

    #if(nhits < 30000 and nhits > 220):
    if(nhits < 30000):
    #if(nhits < 2500 and nhits > 5): # enable for Fe
        ionHits.append(nhits)
        tot_matrix += tmp_array
        matrix += (tmp_array != 0).astype(int)

    ## tryna' DBSCAN
    masked_array = np.ma.masked_array(tmp_array, np.zeros_like(tmp_array, dtype=bool))
    masked_array[0:10, 249:256] = np.ma.masked  # mask region

    # definin' some parameters
    tot_cut = 20
    min_distance = 10 # "eps" parameter
    min_cluster_size = 100 # "min_samples" parameter    

    trans_masked_array = masked_array.T
    #nz_y, nz_x = np.nonzero(trans_masked_array) # using all nonzero entries
    y,x = np.where(trans_masked_array >= tot_cut) # using threshold
    points = np.column_stack((x,y))

    db = DBSCAN(eps=min_distance, min_samples=min_cluster_size).fit(points)
    labels = db.labels_ # -1=noise, 0...N-1 clusters  
    unq_labels = set(labels)

    # filtering out single cluster events (ROUGHLY)

    itot_matrix = tmp_array.T[y,x]  

    fSingle = False
    xcenter, ycenter, xdiv, ydiv = checkHitSpread(trans_masked_array) 

    if(xdiv < 7 and ydiv < 7 and xcenter > 15 and ycenter > 40 and ycenter < 220):
        fSingle = True

    for ilab in unq_labels:
        if(ilab==-1):
            continue
        else:
            msk = labels == ilab
            #
            cHits = np.sum(labels==ilab)
            cTOT = np.sum(itot_matrix[msk])
            cluster_hits.append(cHits)
            cluster_tot.append(cTOT)
                        
            if(fSingle):
                singleClust_tot.append(cTOT)
                singleClust_nhits.append(cHits)

    tot_iclust = None

    #if(nhits> 2000 and nhits < 10000 and npics < 70):
    if(nhits > 40000 and npics < 30):
    #if(fSingle and npics < 20):

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
        plt.title(f"DBSCAN - found {len(labels)-nnoise_clust} clusters")
        plt.xlabel("x, [pix]")
        plt.ylabel("y, [pix]")

        plt.xlim([-10,260])
        plt.ylim([-10,260])

        plt.savefig(f"{outdir}DBSCAN-{cnt}.png")
        plt.close()

        # ===============================================
        fig,ax = plt.subplots(figsize=(8,8))
        cax = fig.add_axes([0.86,0.1,0.05,0.8])
        #ms = ax.matshow(masked_array, cmap='viridis')
        ms = ax.matshow(trans_masked_array, cmap='viridis', vmin=1)

        fig.colorbar(ms,cax=cax, orientation='vertical', label='occupancy')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        comment = f"nhits={nhits}\nsumTOT={sumTOT},\n {initro} ions"
        if(xdiv is not None):
            comment+=f"\nxdiv={xdiv:.2f}, ydiv={ydiv:.2f}"
        ax.text(10, -25, comment, fontsize=10,color='black' )
        ax.invert_yaxis()
        plt.savefig(f"{outdir}FRAME-{cnt}-{picname}.png")
        plt.close()
        fig, ax = None, None

        # ===============================================
        CTS, EDG = np.histogram(np.ravel(masked_array), bins=100, range=(0,1000)) 
        plt.figure()
        plt.hist(EDG[:-1],weights=CTS, bins=100, range=(0,1000),align='left',histtype='stepfilled', facecolor='red' )
        plt.title(f"Event {cnt}")
        plt.xlabel("TOA")
        plt.ylabel(r"$N_{Hits}$")
        plt.yscale('log') 
        plt.savefig(f"{outdir}TOAHIST-FRAME-{cnt}.png")
        plt.close()

        npics+=1

    cnt+=1

    progress(nfiles, cnt)

    tmp_array = None
    fSingle = False


############## PLOTTING #######################
print(f"Single cluster events found = {len(singleClust_nhits)}")

avg_hits_per_ion = np.mean(singleClust_nhits)
avg_tot_per_ion = np.mean(singleClust_tot)

print("-----------------------------------------------")
print(f"Average nhits per ion = {avg_hits_per_ion:.2f}")
print(f"Average TOT per ion = {avg_tot_per_ion:.2f}")
print("-----------------------------------------------")
plotSimpleHist(cluster_hits, 100, outdir, (0,4000),  ["Hits per cluster", r"$N_{hits}$", r"$N_{clusters}$"], "ClusterHits-"+picname)
plotSimpleHist(cluster_tot, 100, outdir, (0,2.4e7),  ["TOT per cluster", r"$\sum{TOT}$", r"$N_{clusters}$"], "ClusterTOT-"+picname)

plotSimpleHist(singleClust_nhits, 100, outdir, (0,1300),  ["Hits per Single cluster", r"$N_{hits}$", r"$N_{clusters}$"], "SingleClusterHits-"+picname)
plotSimpleHist(singleClust_tot, 100, outdir, (0,2.2e6),  ["TOT per Single cluster", r"$\sum{TOT}$", r"$N_{clusters}$"], "SingleClusterTOT-"+picname)

# plotting pixel occpancuy
#masked_matrix = np.ma.masked_array(matrix, np.zeros_like(matrix, dtype=bool))
#masked_matrix[0:256, 252:256] = np.ma.masked  # mask region

hit_threshold = 6000
tmp_matrix = matrix.copy().astype(float)
tmp_matrix[tmp_matrix > hit_threshold] = np.nan

fig, ax = plt.subplots(figsize=(8,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
ms = ax.matshow(tmp_matrix, cmap='gist_earth_r')
ax.set_title("Pixel occupancy (Beam profile)")
ax.set_xlabel("Pixel x")
ax.set_ylabel("Pixel y")
fig.colorbar(ms,cax=cax,orientation='vertical')
ax.invert_yaxis()
fig.savefig(f"{outdir}TOTAL-HITS-{picname}.png")
fig.savefig(f"{outdir}TOTAL-HITS-{picname}.pdf")
ms = None
plt.close()

# mesin' around with occupancy as 3d surface
#
#xpos, ypos, zpos = [],[],[]
#for i in range(256):
#    for j in range(256):
#        xpos.append(i)
#        ypos.append(j)
#        zpos.append(tmp_matrix[i][j]) # this shite should be 2d....
#
#fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
#surf = ax.plot_surface(xpos,ypos,zpos, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#ax.zaxis.set_major_formatter('{x:.02f}')
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.savefig("SURF-{picname}.png")
#plt.close()
#
#plotting pixel accmulated TOT matrix

#masked_tot_matrix = np.ma.masked_array(tot_matrix, np.zeros_like(tot_matrix, dtype=bool))
#masked_tot_matrix[0:256, 252:256] = np.ma.masked  # mask region

integral_tot = np.sum(tot_matrix)
norm_tot_matrix = tot_matrix.astype(float)/integral_tot

#threshold = 5e6
#tmp_tot_matrix = tot_matrix.copy().astype(float)
#tmp_tot_matrix[tmp_tot_matrix > threshold] = np.nan

fig, ax = plt.subplots(figsize=(8,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
#ms = ax.matshow(tot_matrix.T, cmap='gist_earth_r')
#ms = ax.matshow(tot_matrix.T, cmap='gist_earth_r', norm=LogNorm(vmin=1, vmax=tot_matrix.max()))
#ms = ax.matshow(masked_tot_matrix.T, cmap='gist_earth_r', norm=LogNorm(vmin=1, vmax=tot_matrix.max()))
#ms = ax.matshow(tmp_tot_matrix, cmap='gist_earth_r', norm=LogNorm(vmin=1, vmax=np.nanmax(tmp_tot_matrix)))
ms = ax.matshow(norm_tot_matrix, cmap='gist_earth_r', norm=LogNorm(vmin=1e-8, vmax=1.0))
#ms = ax.matshow(tmp_tot_matrix.T, cmap='gist_earth_r')
ax.set_title(r"Accumulated TOT")
ax.set_xlabel("Pixel x")
ax.set_ylabel("Pixel y")
ax.invert_yaxis()
#ax.invert_xaxis()
fig.colorbar(ms,cax=cax,orientation='vertical')
#plt.plot()
fig.savefig(f"{outdir}TOTAL-TOT-{picname}.png")
fig.savefig(f"{outdir}TOTAL-TOT-{picname}.pdf")
plt.close()

# plot empty event frames to get noisiest channels
print(f"\nNoisy pixel matrix has {len(np.nonzero(empty_events))} entries")

fig, ax = plt.subplots(figsize=(8,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
ms = ax.matshow(empty_events, cmap='gist_earth_r')
ax.set_title("Noisiest channels")
ax.set_xlabel("pixel, [x]")
ax.set_xlabel("pixel, [y]")
ax.invert_yaxis()
fig.savefig(f"{outdir}Noisiest-motafuckas.png")
plt.close()
fig, ax = None, None

# rate calc
t_meas = (t_frame+t_dead)*nfiles #nfiles=nframes
total_nitro = t_dead/t_frame*n_nitro
total_nitro_tot = t_dead/t_frame*n_nitro_tot

print(f"\n\ntotal t_meas={t_meas:.2f} [s] -> {float(t_meas)/60.0:.2f} [min]")
# plotting all hits histogram

plt.figure(figsize=(8,8))
counts, edges = np.histogram(np.asarray(hHits), bins=100, range=(0,66000))
#counts, edges = np.histogram(np.asarray(hHits), bins=100, range=(0,2000)) # for Iron runs
plt.hist(edges[:-1], weights=counts, bins=100, range=(0,66000), align='left', histtype='stepfilled', facecolor='b')
plt.title('Total Hits per frame')
plt.xlabel(r"$N_{hits}$")
plt.ylabel(r"$N_{frames}$")
plt.yscale('log')
plt.savefig(f"{outdir}HITS_per_FRAME-{picname}.png")
plt.close()

#print(f"nIOns_sus={nIons_sus} in 5 min")

# plotting actual ion hits

plt.figure(figsize=(8,8))
nitcounts, nitedges = np.histogram(np.asarray(ionHits), bins=100, range=(220,10000))
#nitcounts, nitedges = np.histogram(np.asarray(ionHits), bins=100, range=(0,1000))
plt.hist(nitedges[:-1], weights=nitcounts, bins=100, range=(220,10000), align='left', histtype='stepfilled', facecolor='b')
plt.title('Ion Hits per frame')
plt.xlabel(r"$N_{hits}$")
plt.ylabel(r"$N_{frames}$")
plt.yscale('log')
plt.savefig(f"{outdir}ION_HITS_per_FRAME-{picname}.png")
plt.close()

print(f"Hit based counting = {n_nitro} ions colected in frames")
print(f"total ions passed = {total_nitro:.2f}")
print(f"t_meas = {t_meas:.2f} [s]")
print(f"Rate = {float(total_nitro)/t_meas:.2f} Hz")

print(f"\n{nEmpty} frames with nhits<=200, {float(nEmpty)/float(nfiles)*100.0:.2f}%\n")

print(f"ON average = {float(total_nitro_tot)/float(nfiles-nEmpty):.2f} ion/frame")
print(f"total ions passed (TOTbased) = {total_nitro_tot:.2f}")
print(f"Rate (TOTbased) = {float(total_nitro_tot)/t_meas:.2f} Hz")

t_meas2 = (t_dead+t_frame)*(t_dead/t_frame)*nfiles

print("------------------ timing estimates ---------------------------")
print(f"t_dead={t_dead}")
print(f"t_frame={t_frame}")
print(f"scale factor = t_dead/t_frame = {t_dead/t_frame:.2f}")
print(f"nfiles (nframes) ={nfiles}")
print(f"t_meas_1 = {(t_dead+t_frame)*nfiles:.2f} or {t_meas/60:.2f} min ")
print("---------------------------------------------------------------")
