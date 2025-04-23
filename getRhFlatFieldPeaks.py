import sys
import glob
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from time import sleep
import matplotlib.patches as patch


###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

def gauss(x, A, mu, sigma):

    return A*np.exp(-((x-mu)**2)/(2*sigma**2))

def progress(ntotal, ith, other):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done, {other[0]} = {round(other[1],2)}", end="",flush=True)

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

def simpleHist(nuarray, nbins, minbin, maxbin, labels, picname):

    plt.figure()

    fit_pars = None

    counts, bin_edges = np.histogram(nuarray, bins=nbins, range=(minbin,maxbin))

    #plt.hist(nuarray, nbins, range=(minbin,maxbin), histtype='stepfilled', facecolor='b')
    plt.hist(bin_edges[:-1], weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', facecolor='b')
    plt.figsize=(8,8)
    maxbin_cnt = np.max(counts)
    minbin_cnt = np.min(counts)
    #minval, maxval = max, counts[len(counts)-1]
    val_ibin = (maxbin-minbin)/nbins
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    print(f"Range= {minbin} -> {maxbin}, with {nbins} bins gives {val_ibin} per bin")

    ########################################################
    model = None

    peakbin = getMaxBin(counts)

    print(f"Found maximum bin at {peakbin} = {counts[peakbin]}")
    model = Model(gauss)
    
    pars = model.make_params(A=maxbin_cnt, mu=peakbin*val_ibin, sigma=np.std(counts))

    print(f"Gauss Fit: Setting: A={maxbin_cnt}, mu={peakbin*val_ibin}, sigma={np.std(counts)}")

    print(f"\n\ncounts={counts} \n\n")
    print(maxbin_cnt)
    print(minbin_cnt)
    print(f"max_amplitude={maxbin_cnt - minbin_cnt}")

    print("########## FITTING GASUSS FUNCTION #############")
    result = model.fit(counts[:-1], pars, x=bin_centers[:-1], scale_covar=False)
    print(result.fit_report()) 
    if(result.params['mu']<=0):

        #pars['A'].min = maxbin_cnt*0.8
        #pars['A'].max = maxbin_cnt*1.2

        pars['mu'].min = peakbin*0.8*val_ibin
        pars['mu'].max = peakbin*1.2*val_ibin
 
        #pars['sigma'].min = np.std(counts)*0.8
        #pars['sigma'].max = np.std(counts)*1.2
        if(pars['mu'].min == pars['mu'].max):
            pars['mu'].max = pars['mu'].min+np.std(counts)

        print("FIT FAILED: restricting fit parameters and re-fitting")
        result = model.fit(counts[:-1], pars, x=bin_centers[:-1], scale_covar=False)
        #result = model.fit(counts[peakbin-10:peakbin+10], pars, x=bin_centers[peakbin-10:peakbin+10])

        print(result.fit_report()) 
    ########################################################            

    fitlab = ""
    miny,maxy = 0, 0

    #print("########## FITTING GAUS FUNCTION #############")
    A = round(result.params["A"].value,2)
    err_A = round(result.params["A"].stderr,2)
    mu = round(result.params["mu"].value,2)
    err_mu = round(result.params["mu"].stderr,2)
    sigma = round(result.params["sigma"].value,2)
    err_sigma = round(result.params["sigma"].stderr,2)
    chisq_red = round(result.redchi,2)
       
    fit_pars = [A, err_A,  mu, err_mu, sigma, err_sigma]

    ax = plt.gca()
    _ , xmax = ax.get_xlim()
    _ , ymax = ax.get_ylim()

    chired_sym = r"$\chi^{2}_{red.}$"
    pm = r"$\pm$"
    tstart = xmax*0.7
    plt.text(tstart, ymax*0.95, f"A = {A} {pm} {err_A}")
    plt.text(tstart, ymax*0.90, f"{G_mu} = {mu} {pm} {err_mu}")
    plt.text(tstart, ymax*0.85, f"{G_sigma} = {sigma} {pm} {err_sigma}")
    plt.text(tstart, ymax*0.80, f"{chired_sym} = {chisq_red}")

    ax = plt.gca()
    plt.plot(bin_centers[:-1], result.best_fit, '--r')

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    #plt.yscale('log')
    plt.grid()
    plt.savefig(f"Rh-FF-run-{picname}.png")

    return fit_pars

########################################################################

location = sys.argv[1]
picname = sys.argv[2]

dir_files = glob.glob(location+"*.h5")

for i in dir_files:
    print(i)
inputlist = sorted(dir_files)

peak_amp = []
peak_amp_err = []
peak_mu = []
peak_mu_err = []
peak_sigma = []
peak_sigma_err = []
positions = []

fcnt = 0

#totHistList = [ [] for i in range(3)]
totHistList = [ [] for i in range(6)]
#totHist = []

#densities = []

#nbins = 51
#binrange = (0,2048)
#edges = np.linspace(binrange[0],binrange[1],nbins+1)
#cnt_square1 = np.zeros(nbins, dtype=np.int64)

matrix = np.zeros((256,256),dtype=np.uint16)
slicedMat = np.zeros((256,256),dtype=np.uint16)

run_grid = np.zeros((2,3), dtype=np.float64)

spotNames = ["BL", "BM", "BR", "TL", "TM", "TR"]

ifile = 0
for file in inputlist:

    #x, y = None, None
    #TOT = []

    #matrix = np.zeros((256,256), dtype=np.uint16)
    with tb.open_file(file) as f:

        print(f"Reading {f}")
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
        #                              ^ removes space at the end of 'run_xxx/chip0 '
        print(f'base group name is : <{base_group_name}>')
        bgn_split = base_group_name.split('/')
        print(bgn_split)
        run_name = bgn_split[2]
        print(f"<{run_name}>")
        run_num = int(run_name[4:])
        print(f'run number is {run_num}')
        basewords = None

        ###############################################
        sumTOT = f.get_node(base_group_name+"sumTot")
        cluster_x = f.get_node(base_group_name+"x")
        cluster_y = f.get_node(base_group_name+"y")
        ToT = f.get_node(base_group_name+"ToT")

        print(f"Contains {len(cluster_x)} events/hits")
        print(f"Contains {len(cluster_y)} events/hits")
        print(f"Contains {len(ToT)} events/hits")

        nclus = len(cluster_x)
        iclus = 0

        maskRanges_xmin = [75 , 130, 180, 75 , 130 ,180]
        maskRanges_xmax = [85 , 140, 190, 85 , 140 ,190]
        maskRanges_ymin = [95 , 95 , 95 , 150, 150 ,150]
        maskRanges_ymax = [105, 105, 105, 160, 160 ,160]
        
        for xpos, ypos, tot in zip(cluster_x, cluster_y, ToT):
             
            #xmin, xmax = np.nanmin(xpos), np.nanmax(xpos)
            #ymin, ymax = np.nanmin(ypos), np.nanmax(ypos)
            #area = (xmin+xmax)*(ymin+ymax)
            #QperArea = None
            #try:            
            #    QperArea = float(np.sum(tot))/float(area)
            #except ZeroDivisionError:
            #    QperArea = -1.

            #densities.append(QperArea)

            np.add.at(matrix, (xpos,ypos), 1)            

            # region loop
            iregion = 0
            for x_min, x_max, y_min, y_max in zip(maskRanges_xmin, maskRanges_xmax, maskRanges_ymin, maskRanges_ymax):

                #xmask = np.logical_and(xpos>=75, xpos<=85)
                #ymask = np.logical_and(ypos>=95, ypos<=105)

                xmask = np.logical_and(xpos>=x_min, xpos<=x_max)
                ymask = np.logical_and(ypos>=y_min, ypos<=y_max)
                combMask = np.logical_and(xmask,ymask)

                for xm, ym in zip(xpos[combMask],ypos[combMask]):
                    np.add.at(slicedMat, (xm,ym), 1)             

                sumtot = np.sum(np.array(tot)[combMask])
                nonZero = np.count_nonzero(np.array(tot)[combMask])

                if(sumtot>0):
                    totHistList[iregion].append(sumtot/nonZero)
                iregion+=1

            progress(nclus,iclus, ["totHistList[1]",np.mean(totHistList[1])])
            iclus+=1
    
    ifile+=1
    if(ifile==5):
        break
    print(f"--------------------DONE----------------------")


print("\n=========== COMPLETED with all files ===========\n")
#sleep(1)

cnth = 0
plt.figure(figsize=(8,8))
for h in totHistList:
    counts, edges = np.histogram(np.array(h), bins=51, range=(0,512))
    plt.hist(edges[:-1],weights=counts, bins = 51, range=(0,512),align='left',alpha=0.2,histtype='stepfilled',label=f"region={spotNames[cnth]}")
    cnth+=1

plt.title("TOT summed over 6 regions of matrix")
plt.xlabel("TOT sum, n CYCLES")
plt.ylabel("N")
plt.legend()
plt.savefig("HUYA-totPeaksCombined-"+picname+".png")

######################################################################


peak_info = None
pcnt=0
ppos = []
for h in totHistList:

    print(np.mean(h))
    
    peak_info = simpleHist(h, 51 ,np.nanmin(h), np.nanmax(h), [f"TOT per event (position {spotNames[pcnt]})", "TOT cycles", "N"], f"position-{spotNames[pcnt]}-"+picname)

    #peak_amp.append(peak_info[0])
    #peak_amp_err.append(peak_info[1])
    peak_mu.append(peak_info[2])
    #peak_mu_err.append(peak_info[3])
    #peak_sigma.append(peak_info[4])
    #peak_sigma_err.append(peak_info[5])

    if(pcnt==0):
        run_grid[1][0]+=round(peak_info[2])
    elif(pcnt==1):
        run_grid[1][1]+=round(peak_info[2])
    elif(pcnt==2):
        run_grid[1][2]+=round(peak_info[2])
    elif(pcnt==3):
        run_grid[0][0]+=round(peak_info[2])
    elif(pcnt==4):
        run_grid[0][1]+=round(peak_info[2])
    elif(pcnt==5):
        run_grid[0][2]+=round(peak_info[2])

    peak_info = None
    ppos.append(pcnt)
    pcnt+=1

plt.figure(figsize=(8,8))

#plt.scatter(ppos, peak_mu)

#plt.savefig("Rh-FF-run-peaks.png")

fig,ax = plt.subplots(figsize=(8,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
ms = ax.matshow(matrix.T, cmap='viridis')
fig.colorbar(ms,cax=cax, orientation='vertical', label='occupancy')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.invert_yaxis()

### addding square

#bottom row
rect = patch.Rectangle((75,95),10,10, linewidth=1.5, edgecolor='r', facecolor='none')
rect2 = patch.Rectangle((130,95),10,10, linewidth=1.5, edgecolor='r', facecolor='none')
rect3 = patch.Rectangle((180,95),10,10, linewidth=1.5, edgecolor='r', facecolor='none')
#top row
rect4 = patch.Rectangle((75,150),10,10, linewidth=1.5, edgecolor='r', facecolor='none')
rect5 = patch.Rectangle((130,150),10,10, linewidth=1.5, edgecolor='r', facecolor='none')
rect6 = patch.Rectangle((180,150),10,10, linewidth=1.5, edgecolor='r', facecolor='none')


ax.add_patch(rect)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)

###

plt.savefig("FullMatrix-"+picname+".png")
########### sliced matrix ########################## 
fig,ax = plt.subplots(figsize=(8,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
ms = ax.matshow(slicedMat.T, cmap='viridis')
fig.colorbar(ms,cax=cax, orientation='vertical', label='occupancy')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.invert_yaxis()
plt.savefig("SlicedMatrix-"+picname+".png")

#########################################################
# run grid plot

#run_grid = run_grid.T
plt.figure()
plt.imshow(run_grid, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Fit Mean TOT')
plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
for k in range(run_grid.shape[0]):
    for l in range(run_grid.shape[1]):
        plt.text(k,l,f"{run_grid[k,l]}", ha='center',va='center', color='white')
plt.xticks(range(run_grid.shape[0]))
plt.yticks(range(run_grid.shape[1]))
plt.xlabel("column")
plt.ylabel("row")
#plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("RunGrid-FitMeans-"+picname+".png")




