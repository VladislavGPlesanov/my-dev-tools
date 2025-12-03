import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import tables as tb


def pixelMap2d(nuArray, plotname, title):

    stats = None # list of floats with [mean,median,stdev]
    pixelmask = None # 2D numpy array later
    #these = ["occMap-", "checkin-event"]
    #if(these in plotname):
    if("occMap-" in plotname):
        _ , stats = findNoisyPixels(nuArray)       
     
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.86, 0.1, 0.05, 0.8])
    ms = ax.matshow(nuArray)
    fig.colorbar(ms, cax=cax, orientation='vertical')
    if(stats is not None):
        ax.text(-85,270,r'$occ_{mean}$'+f' = {stats[0]:.2f}',fontsize=10)
        ax.text(-85,280,r'$occ_{med}$'+f' = {stats[1]:.2f}',fontsize=10)
        ax.text(-85,290,r'$\sigma_{occ}$'+f' = {stats[2]:.2f}',fontsize=10)
    ax.set_title(title)

    plt.plot()
    fig.savefig("IMG-2d-"+plotname+".png")

##################################################################################

maskfile_this = sys.argv[1]
maskfile_that = sys.argv[2]
picname = sys.argv[3]

filename_this = maskfile_this.split(".")[0]
filename_that = maskfile_that.split(".")[0]

mask_this, mask_that = None, None

fThisDataFile = False
fThatDataFile = False

if("DataTake" in maskfile_this):
    fThisDataFile = True
if("DataTake" in maskfile_that):
    fThatDataFile = True

with tb.open_file(maskfile_this, 'r') as fthis:

    if(not fThisDataFile):
        mask_this = fthis.root.mask_matrix[:].T 
    else:
        mask_this = fthis.root.configuration.mask_matrix[:].T 

with tb.open_file(maskfile_that, 'r') as fthat:

    if(not fThatDataFile):
        mask_that = fthat.root.mask_matrix[:].T 
    else:
        mask_that = fthat.root.configuration.mask_matrix[:].T 

#print(mask_this)
#print(mask_that)

ndyke = (3*256*2)+(3*250*2)

n_masked_this = np.count_nonzero(mask_this)-ndyke
n_masked_that = np.count_nonzero(mask_that)-ndyke
n_masked_between = n_masked_that - n_masked_this

perc_this = (n_masked_this/(256*256))*100.0
perc_that = (n_masked_that/(256*256))*100.0

print(f"MASK [{filename_this}] has ({n_masked_this}) channels active outside dyke mask {perc_this:.2f} %")
print(f"MASK [{filename_that}] has ({n_masked_that}) channels active outside dyke mask {perc_that:.2f} %")
print(f"DIFFERENCE={n_masked_between}")


pixelMap2d(mask_this, f"this-{picname}", filename_this)
pixelMap2d(mask_that, f"that-{picname}", filename_that)

mask_diff = mask_this^mask_that

pixelMap2d(mask_diff, f"diff-{picname}", "Difference")



