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

main_path = sys.argv[1]
nframes = int(sys.argv[2])
picname = sys.argv[3]

outdir = f"FRAMES-{picname}/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

inputlist = glob.glob(main_path+"*.txt")

nfiles = len(inputlist)

print(f"Directory {main_path} contains {nfiles}")

cnt = 0

total_mat = np.zeros((256,256),dtype=int)

npics = 0
for file in inputlist:
    
    tmp_array = np.loadtxt(file, dtype=int)
    print(tmp_array.shape)
    nhits = np.count_nonzero(tmp_array)
    sumTOT = np.sum(tmp_array)
    if(nhits < 20000):
        total_mat += tmp_array
    #if(cnt < nframes): 
    if(npics < 50 and nhits > 100): 
        # ===============================================
        fig,ax = plt.subplots(figsize=(8,8))
        cax = fig.add_axes([0.86,0.1,0.05,0.8])
        ms = ax.matshow(tmp_array.T, cmap='viridis', norm=LogNorm(vmin=1,vmax=12000))
        fig.colorbar(ms,cax=cax, orientation='vertical', label='TOT counts')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.text(20,240, f"{nhits} hits")
        ax.text(20,220, f"sum(TOT)={sumTOT:.2f} ")
        ax.invert_yaxis()
        plt.savefig(f"{outdir}EVENT-{cnt}-{picname}.png")
        plt.close()
        fig, ax = None, None
        
        npics+=1

    cnt+=1

fig,ax = plt.subplots(figsize=(8,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
ms = ax.matshow(total_mat.T, cmap='viridis', norm=LogNorm(vmin=1,vmax=12000))
fig.colorbar(ms,cax=cax, orientation='vertical', label='TOT counts')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.invert_yaxis()
plt.savefig(f"{outdir}TOTAL-{picname}.png")
plt.close()
fig, ax = None, None



