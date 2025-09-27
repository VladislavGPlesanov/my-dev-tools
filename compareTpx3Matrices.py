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

with tb.open_file(maskfile_this, 'r') as fthis:

    mask_this = fthis.root.mask_matrix[:].T 

with tb.open_file(maskfile_that, 'r') as fthat:

    mask_that = fthat.root.mask_matrix[:].T 

print(mask_this)
print(mask_that)

pixelMap2d(mask_this, f"this-{picname}", filename_this)
pixelMap2d(mask_that, f"that-{picname}", filename_that)

mask_diff = mask_this^mask_that

pixelMap2d(mask_diff, f"diff-{picname}", "Difference")



