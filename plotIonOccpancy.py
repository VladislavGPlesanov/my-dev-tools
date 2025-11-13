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
import matplotlib.patches as patch
from matplotlib.patches import Ellipse

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
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

def getChargeCenter(matrix):

    x, y = mp.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), indexing='ij')
    sumtot = matrix.sum()
    mean_x = (matrix * x).sum()/sumtot
    mean_y = (matrix * y).sum()/sumtot
    return mean_x, mean_y

def getPatchStartPoint(center, unit, nsteps):

    start_point = center - nsteps * unit 

    if(start_point < 0):
        return 0
    elif(start_point > 256):
        return 256
    else:
        return start_point

def getPatchSides(wstart, hstart, wstep, hstep, wunit, hunit):

    print(f"\nPatch at x={wstart:.2f} y={hstart:.2f} ,sides are")
    wend = wstart + wstep*wunit    
    print(f"width= {wstart:.2f} + {wstep*wunit:.2f}")
    print(f"WEND={wend:.2f}")

    hend = hstart + hstep*hunit    
    print(f"height= {hstart:.2f} + {hstep*hunit:.2f}")
    print(f"HEND={hend:.2f}")

    if(hend > 256):
        hend = 256
        print("OOPS - hend is outside matrix")

    if(wend > 256):
        print("OOPS - wend is outside matrix")
        wend = 256

    print(f"resulting {wend:.2f} and {hend:.2f}")
    return wend, hend

def getRectangleParams(matrix, THR):

    projX = matrix.sum(axis=0)
    projY = matrix.sum(axis=1)
    
    xstart, ystart, xend, yend = 0,0,0,0

    for i in range(len(projX)):
        if(projX[i]>THR):
            xstart = i
            break

    for i in range(len(projX)):
        if(projX[255-i]>THR):
            xend = 255-i
            break

    for i in range(len(projY)):
        if(projY[i]>THR):
            ystart = i
            break

    for i in range(len(projY)):
        if(projY[255-i]>THR):
            yend = 255-i
            break

    width = xend - xstart
    height = yend - ystart    

    return xstart, ystart, width, height


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

def getAxisRange(plot, axis):

    ax = plot.gca() 
    if(axis==0):
        return ax.get_xlim()   
    elif(axis==1):
        return ax.get_ylim()   
    else:
        print(f"ERROR: no axis with index {axis}")
        return -1,-1


def SearchEllipse(matrix, centerX, centerY, stdevX, stdevY, nsigma=2):

    yy, xx = np.indices(matrix.shape)

    elipse_mask = ((xx - centerX)**2 / (nsigma*stdevX)**2 + (yy - centerY)**2 / (nsigma*stdevY)**2) <= 1 

    n_pixels = np.count_nonzero(elipse_mask)

    sumQ = np.sum(matrix[elipse_mask])

    return n_pixels, sumQ, elipse_mask


############################################################################

main_path = sys.argv[1]
picname = sys.argv[2]
SR = int(sys.argv[3])
ST = int(sys.argv[4])

outdir = f"NITRO-{picname}/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

#t_dead = 2.5e-2
#t_dead = 25e-3 # true for ZS mode
t_dead = 0.048#full frame is slower
t_frame = getFrameTime(SR,ST)

inputlist = glob.glob(main_path+"*.txt")

nfiles = len(inputlist)

print(f"Directory {main_path} contains {nfiles}")
matrix = np.zeros((256,256), dtype=int)
tot_matrix = np.zeros((256,256), dtype=int)
empty_events = np.zeros((256,256), dtype=int)

nIons, nIons_sus = 0, 0
nIons_dbscan = 0

hHits = []
ionHits = []
totalTOT = []

nEmpty = 0

n_nitro = 0
n_nitro_tot = 0

cnt = 0
npics = 0

cluster_hits = []
cluster_tot = []

singleClust_tot, singleClust_nhits = [],[]

insta_occupancy, insta_time = [], []

for file in inputlist:
    
    tmp_array = np.loadtxt(file, dtype=int)
    #print(tmp_array.shape)

    nhits = np.count_nonzero(tmp_array)

    hHits.append(nhits)

    sumTOT = np.sum(tmp_array)

    totalTOT.append(sumTOT)

    initro = 0
    #avg_nhits = 4617.38
    #avg_nhits = 388.80
    #avg_nhits = 689.59 # unknown run
    #avg_TOT_ion = 2328861.65
    #avg_TOT_ion = 786183.57 # 740630.45

    # Run 6817
    avg_nhits = 675.16
    avg_TOT_ion = 730925.82
    
    if(nhits<=300):
    #if(nhits<=5): # enable for Fe
        #print("Not enough hits")
        cnt+=1
        nEmpty+=1
        insta_occupancy.append(float(nhits)/float(256**2))
        insta_time.append(cnt*t_dead+t_frame)
        continue

    #initro_tot = np.floor(float(sumTOT)/avg_TOT_ion)
    initro_tot = np.ceil(float(sumTOT)/avg_TOT_ion)
    if(nhits<=20):
        empty_events += tmp_array

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
    #masked_array[0:10, 249:256] = np.ma.masked  # mask region
    #masked_array[0:256, 249:256] = np.ma.masked  # mask region
    border = 5 #pixels
    masked_array[:border, :]  = np.ma.masked # masking top
    masked_array[-border: , :]= np.ma.masked # masking bottom
    masked_array[:, :border]  = np.ma.masked # masking left
    masked_array[:, -border:] = np.ma.masked # maskign right

    # in the masked array need to find out-of-cluster pixels that are maxed-out
    #
    #

    # definin' some parameters

    iframe_occupancy = float(nhits)/float(256**2)
    insta_occupancy.append(iframe_occupancy)
    iframe_time = cnt*t_dead+t_frame
    insta_time.append(iframe_time)

    #incident_area = 2*xdiv * 2*ydiv # taking area of 2 sigmas of data spread in matrix

    #rect_area = patch.Rectangle((xcenter-2*xdiv, ycenter-2*ydiv), 2*xdiv, 2*ydiv, linewidth=1.5, edgecolor='r', facecolor='none')
  
    #start_x = getPatchStartPoint(xcenter, xdiv, 2)
    #start_y = getPatchStartPoint(ycenter, ydiv, 2)

    #xend, yend = getPatchSides(start_x, start_y, xdiv, ydiv, 4, 4)
    #print(f"\nFrame: {cnt}")
    #xend, yend = getPatchSides(start_y, start_x, ydiv, xdiv, 4, 4)
    #xend, yend = getPatchSides(start_x, start_y, xdiv, ydiv, 3, 3)

    #start_x, start_y, width, height = getRectangleParams(tmp_array, 1500)
    THR = iframe_occupancy * np.nanmax(masked_array)
    #if(iframe_occupancy>0.25):
    #    THR = 3000
    start_x, start_y, width, height = getRectangleParams(masked_array, THR)

    #rect_area = patch.Rectangle((start_y, start_x), 4*ydiv, 4*xdiv, linewidth=1.5, edgecolor='r', facecolor='none')
    #rect_area = patch.Rectangle((start_y, start_x), 4*xdiv, 4*ydiv, linewidth=1.5, edgecolor='r', facecolor='none')
    rect_area = patch.Rectangle((start_y, start_x), height, width,  linewidth=1.5, edgecolor='r', facecolor='none')
    #rect_area = patch.Rectangle((start_x, start_y), start_x - xend, start_y - yend, linewidth=1.5, edgecolor='r', facecolor='none')

    incident_area = height * width

    charge_density_per_hit = float(sumTOT)/float(nhits)
    charge_density_per_area = float(sumTOT)/float(incident_area)

    tot_cut = None     

    tot_cut = 1200
    #tot_cut = 100
    #min_distance = 9 # "eps" parameter
    #min_cluster_size = 99 # "min_samples" parameter    

    min_samples_scale = 0.02
    eps_scale = 3.0

    trans_masked_array = masked_array.T
    #nz_y, nz_x = np.nonzero(trans_masked_array) # using all nonzero entries

    y,x = np.where(trans_masked_array >= tot_cut) # using threshold

    points = np.column_stack((x,y))

    npoints_after_cut = len(points)

    n_neigh = 2
    labels, eps, min_samples, mean_neigh_dist = dynamic_dbscan(points, min_samples_scale, eps_scale, n_neigh)

    #db = DBSCAN(eps=min_distance, min_samples=min_cluster_size).fit(points)
    #labels = db.labels_ # -2=noise, 0...N-1 clusters  
    ##############################################################################

    unq_labels = set(labels)

    nIons_dbscan += len(unq_labels) - 1

    fSingle = False
    xcenter, ycenter, xdiv, ydiv = checkHitSpread(trans_masked_array) 

   # filtering out single cluster events (ROUGHLY)

    itot_matrix = tmp_array.T[y,x]  

    ###############################################################################
    ## tyin' out spectral clustering
    ##
    #sy,sx = np.where(trans_masked_array >= 100) # using threshold

    #spoints = np.column_stack((sx,sy))

   
    #spoints = points /256.0
    #
    #spec = SpectralClustering(

    #    n_clusters = int(initro),
    #    eigen_solver = 'arpack',
    #    affinity = 'nearest_neighbors',
    #    n_neighbors = 10,
    #    assign_labels = 'kmeans',
    #    random_state = 42

    #)

    #spec_labels = spec.fit_predict(spoints)

    ################################################################################
    # tryna' count stuff inside ellipse

    nell_hits, sumq, elmask = SearchEllipse(trans_masked_array, xcenter, ycenter, xdiv, ydiv)

    nsig =2    
    ellipse = Ellipse(
        #(xcenter, ycenter),          # center coordinates
        (ycenter, xcenter),          # center coordinates
        width=2 * nsig * xdiv,       # total width (2a)
        height=2 * nsig * ydiv,      # total height (2b)
        edgecolor='green',            # ellipse outline color
        facecolor='none',            # transparent fill
        linewidth=2,
        linestyle='--'
    )

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

    #if(fSingle and nhits < 30000 and npics < 100):
    if(nhits < 30000 and npics < 100):
    #if(nhits > 40000 and npics < 30):
    #if(fSingle and npics < 20):
        ##################################################################

        #plt.figure(figsize=(8,8))
        #for lbl in np.unique(spec_labels):
        #    m = labels == lbl
        #    plt.scatter(spoints[m,0]*256, spoints[m,1]*256, s=4, label=f"Cluster {lbl}")
        #plt.legend()
        #plt.title("Spectral Clustering Results")
        #plt.xlabel("x [pix]")
        #plt.ylabel("y [pix]")
        #plt.xlim([-10,260])
        #plt.ylim([-10,260])
        #plt.savefig(f"{outdir}SpectralScan-{cnt}.png")
        #plt.close()

        ##################################################################
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

        if(fSingle):
            plt.text(minx*1.1, maxy-35, "Single Cluster")

        plt.xlim([-10,260])
        plt.ylim([-10,260])

        plt.savefig(f"{outdir}DBSCAN-{cnt}.png")
        plt.close()

        # ===============================================
        fig,ax = plt.subplots(figsize=(8,8))
        cax = fig.add_axes([0.86,0.1,0.05,0.8])
        #ms = ax.matshow(masked_array, cmap='viridis')
        #ms = ax.matshow(trans_masked_array, cmap='viridis', vmin=1)
        ms = ax.matshow(trans_masked_array, cmap='viridis', norm=LogNorm(vmin=1,vmax=12000))

        fig.colorbar(ms,cax=cax, orientation='vertical', label='TOT counts')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        comment = f"nhits={nhits}, sumTOT={sumTOT}, {initro} ions, TOTcut>{tot_cut} cts, THR={THR:.2f}"
        comment += f"\n[occupancy={iframe_occupancy:.2}]"
        #comment += f"Q_dens={charge_density_per_hit:.2f} (per hit), {charge_density_per_area:.2f}(per area)"
        comment += f"Hits io ellipse = {nell_hits:.2f}, sumq = {sumq:.2f}"
        if(xdiv is not None):
        #if(fSingle):
            comment+=f", xdiv={xdiv:.2f}, ydiv={ydiv:.2f}"
        ax.text(10, -23, comment, fontsize=10,color='black' )
        ax.add_patch(rect_area)
        ax.add_patch(ellipse)
        # this ain't work cuz for matshow x,y  are inverted
        #ax.scatter(xcenter,ycenter, c='m', marker='*')
        # this works
        ax.scatter(ycenter, xcenter, c='m', marker='*')

        ax.invert_yaxis()
        plt.savefig(f"{outdir}FRAME-{cnt}-{picname}.png")
        plt.close()
        fig, ax = None, None
        rect_area = None

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


print(f"Stoppped at {cnt} frames")

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
sum_hits = np.sum(counts[0:50]) 
bwidth = np.diff(edges)
integral_hits = np.sum(counts[0:50] * bwidth[0:50]) 

plt.hist(edges[:-1], weights=counts, bins=100, range=(0,66000), align='left', histtype='stepfilled', facecolor='b')
plt.title('Total Hits per frame')
plt.xlabel(r"$N_{hits}$")
plt.ylabel(r"$N_{frames}$")
_ , maxy = getAxisRange(plt,1)

plt.text(10000, maxy*0.85, f"Equivalent Ions (first 50 bins) = {integral_hits/avg_nhits:.2f}", fontsize=13)
plt.text(10000, maxy*0.65, f"Total exposure = {(integral_hits/avg_nhits)*t_dead/t_frame:.2f}", fontsize=13)
plt.yscale('log')
plt.savefig(f"{outdir}HITS_per_FRAME-{picname}.png")
plt.close()

print("=================UVAGA!==================")
print(counts[0:50])
print(np.sum(counts[0:50]))
print(avg_nhits)
print(f"{sum_hits/avg_nhits:.2}")

counts, edges, bwidth = None, None, None
#print(f"nIOns_sus={nIons_sus} in 5 min")


# plotting sum of TOT over a frame

plt.figure(figsize=(8,8))
counts, edges = np.histogram(np.asarray(totalTOT), bins=100, range=(0,1e8))
#sum_tot = np.sum(counts[0:50])
bwidth = np.diff(edges)
integral_tot = np.sum(counts[0:50] * bwidth[0:50]) 
plt.hist(edges[:-1], weights=counts, bins=100, range=(0,1e8), align='left', histtype='stepfilled', facecolor='b')
plt.title('Total Hits per frame')
plt.xlabel(r"$N_{hits}$")
plt.ylabel(r"$N_{frames}$")
_ , maxy = getAxisRange(plt,1)
plt.text(1e7, maxy*0.85, f"Equivalent Ions (Integral over 50 bins) = {integral_tot/avg_TOT_ion:.2f}", fontsize=13)
plt.text(1e7, maxy*0.65, f"Total exposure = {(integral_tot/avg_TOT_ion)*t_dead/t_frame:.2f}", fontsize=13)
plt.yscale('log')
plt.savefig(f"{outdir}TOT_per_FRAME-{picname}.png")
plt.close()

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

print(f"\nDBSCAN recognized {nIons_dbscan} clusters/ions in {cnt} frames\n")

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



arr_insta_occ = np.array(insta_occupancy)
arr_insta_time = np.array(insta_time)
mask_crap = (arr_insta_occ > 0.4)
mask_good = (arr_insta_occ <= 0.4)

plt.figure(figsize=(14,8))
#plt.scatter(arr_insta_time[mask_crap], arr_insta_occ[mask_crap], c="red", marker='.')
#plt.scatter(arr_insta_time[mask_good], arr_insta_occ[mask_good], c="blue", marker='.')
#plt.plot(arr_insta_time[mask_crap], arr_insta_occ[mask_crap], c="red", marker='.')
plt.plot(insta_time, insta_occupancy, c="red", marker='.')
plt.plot(arr_insta_time[mask_good], arr_insta_occ[mask_good], c="blue", marker='.')
#plt.plot(insta_time, insta_occupancy, c="red")
#plt.plot(arr_insta_time[mask_crap], arr_insta_occ[mask_crap], c="blue")
plt.xlabel("Time, [s]")
plt.ylabel("Occupancy, [#hits]")
plt.title("Relative Pixel Matrix Occupancy vs Time")
plt.grid(True)
plt.savefig(f"{outdir}RelOcc-{picname}.png")
plt.close()

