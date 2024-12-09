import sys
import glob
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'


def gauss(x, A, mu, sigma):

    return A*np.exp(-((x-mu)**2)/(2*sigma**2))

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)

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

        print("FIT FAILED: restricting fit parameters and re-fitting")
        result = model.fit(counts[:-1], pars, x=bin_centers[:-1], scale_covar=False)
        #result = model.fit(counts[peakbin-10:peakbin+10], pars, x=bin_centers[peakbin-10:peakbin+10])

        print(result.fit_report()) 
    ########################################################            

    fitlab = ""
    miny,maxy = 0, 0

    #print("########## FITTING GAUS FUNCTION #############")
    A = result.params["A"].value
    err_A = result.params["A"].stderr
    mu = result.params["mu"].value
    err_mu = result.params["mu"].stderr
    sigma = result.params["sigma"].value
    err_sigma = result.params["sigma"].stderr

    fit_pars = [A, err_A,  mu, err_mu, sigma, err_sigma]

    ax = plt.gca()
    miny,maxy = ax.get_ylim()
    #plt.yscale('log')
    plt.plot(bin_centers[:-1], result.best_fit, '--r')
    plt.text(minbin*1.1, maxy*0.9, f"A={round(A,2)}, {G_mu}={round(mu,2)}, {G_sigma}={round(sigma,2)}")

    #plt.legend(loc='upper right')

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    #plt.yscale('log')
    plt.grid()
    plt.savefig(f"gridScan-runs-{picname}.png")

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

run_grid = np.zeros((3,3))

totHistList = []
Vgrid_list = []

fcnt = 0

matOfMatices=[]

for file in inputlist:

    x, y = None, None
    TOT = []
    #position = None
    matrix = np.zeros((256,256), dtype=np.uint16)
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

        fname = str(inputlist[fcnt])
        words = fname.split("-")
        #print(words)
        row = words[len(words)-2]
        column = words[len(words)-1][:-3]# removin .h5 from column name
        if(column == "left"):
            x = 0
        elif(column=="center"):
            x = 1
        elif(column=="right"):
            x = 2
    
        if(row=="bottom"):
            y = 0
        elif(row=="center"):
            y = 1
        elif(row=="top"):
            y = 2

        print(row)
        print(column)
        positions.append(row+"-"+column)

        #exit(0)

        ###############################################
        sumTOT = f.get_node(base_group_name+"sumTot")
        cluster_x = f.get_node(base_group_name+"x")
        cluster_y = f.get_node(base_group_name+"y")
        ToT = f.get_node(base_group_name+"ToT")
        print(f"found sumTOT {type(sumTOT)}")
        print(f"Contains {len(sumTOT)} events/hits")
        print(f"Contains {len(cluster_x)} events/hits")
        print(f"Contains {len(cluster_y)} events/hits")
        print(f"Contains {len(ToT)} events/hits")

        for totsum in sumTOT:
            TOT.append(totsum)

        #print(cluster_x.shape)
        #print(type(cluster_x))
        #print(cluster_x[:10])
        #print(sumTOT.shape)
        #print(type(sumTOT))
        #print(sumTOT[:10])

        iclus = 0
        #for xpos, ypos in zip(cluster_x, cluster_y):
        #for xpos, ypos, tot in zip(cluster_x, cluster_y, ToT):
        for xpos, ypos in zip(cluster_x, cluster_y):

            #print(xpos)
            #print(len(xpos))
            #print(type(xpos[0]))
            #print(ypos)
            #print(len(ypos))
            #print(type(ypos[0]))
            #print(tot)
            #print(len(tot))
            #print(type(tot[0]))

            #for i,j,t in zip(xpos,ypos,tot):
            #    print(f"x={i} y={j} => tot={t}")
            #    matrix[i,j] += t

            ##exit(0)
            #try:
            #    assert len(xpos) == len(ypos) == len(tot), "xpos, ypos, and tot arrays must have the same length."
            #except:
            #    print("Error SUKA!")
            np.add.at(matrix, (xpos, ypos), 1)

            iclus+=1

    print(f"--------------------DONE----------------------")
   
    print("tryna pot full event")
    
    fig, ax = plt.subplots()
    #fig.figsize(8,8)

    cax = fig.add_axes([0.86,0.1,0.05,0.8])

    ms = ax.matshow(matrix, cmap='hot')
    fig.colorbar(ms,cax=cax,orientation='vertical')

    plt.title("Total event matrix TOT")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(f"matrix-{positions[fcnt]}-{picname}.png")

    matOfMatices.append(matrix)
    matrix = None

    print("===============================================")
    peak_info = simpleHist(TOT, 101,0, 40000, [f"TOT per event (positions[fcnt])", "TOT cycles", "N"], f"{positions[fcnt]}-"+picname)

    totHistList.append([TOT,row+"-"+column])

    peak_amp.append(peak_info[0])
    peak_mu.append(peak_info[2])
    peak_sigma.append(peak_info[4])
    
    #print(x)
    #print(y)
    print(run_grid)
    run_grid[x][y] += round(peak_info[2],2)

    x, y = None, None
    peak_info = None
    TOT = None

    fcnt+=1

plt.figure(figsize=(8,8))
for data in totHistList:
    counts, bin_edges = np.histogram(data[0], bins=101, range=(0,11000))
    plt.hist(bin_edges[:-1], weights=counts, bins=101, range=(0,11000), align='left', histtype='stepfilled', alpha=0.2, label=f"{data[1]}")
    
plt.title("Energy spectrum, all runs")
plt.xlabel("TOT cycles")
plt.ylabel("N")
plt.legend(loc='upper left')
plt.savefig(f"TOT-GridScan-Combined-{picname}.png")

print("\nPotting peaks\n")

combinedMat = np.sum( matOfMatices, axis=0)

fig, ax = plt.subplots(figsize=(12,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
#ms = ax.matshow(combinedMat, cmap='hot')
ms = ax.matshow(combinedMat.T, cmap='viridis')
fig.colorbar(ms,cax=cax,orientation='vertical', label="occupancy")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.xaxis.set_label_position('top') 
#ax.invert_xaxis()
ax.invert_yaxis()

#plt.xlabel("x")
#plt.ylabel("y")

plt.savefig(f"matrix-combined-{picname}.png", dpi=400)

plt.figure()
plt.plot(positions, peak_mu)
plt.xticks(rotation=45) 
plt.savefig(f"allPeakMu-{picname}.png")

##########################################################

run_grid = run_grid.T
plt.figure()
plt.figure(figsize=(8, 8))  # Adjust size as needed
plt.imshow(run_grid, cmap='viridis', interpolation='nearest')

# Add a color bar to interpret the scale
plt.colorbar(label='Peak sumTOT')

# Add gridlines
plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

# Label each cell with the number
for i in range(run_grid.shape[0]):
    for j in range(run_grid.shape[1]):
        plt.text(j, i, f'{run_grid[i, j]}', ha='center', va='center', color='white')

# Configure ticks to match grid layout
plt.xticks(range(run_grid.shape[1]))
plt.yticks(range(run_grid.shape[0]))
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.title("3x3 Grid Scan")
plt.gca().invert_yaxis()
#plt.invert_xaxis()

# Show the plot
plt.tight_layout()
plt.savefig(f"RunGrid-peak-means-{picname}.png")



