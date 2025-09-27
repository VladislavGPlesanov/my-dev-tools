import sys
import glob
import tables as tb
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend (renders to files only)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)

location = sys.argv[1]
picname = sys.argv[2]

dir_files = glob.glob(location+"*.txt")

nframes = len(dir_files)

#occ_matrix = []
occ_matrix = np.zeros((256,256), dtype=int)
tot_matrix = np.zeros((256,256), dtype=int)

iframe = 0
for file in dir_files:

    f = open(file, 'r')

    nhits = None

    for line in f:
       
        if("FEC" in line or "Board" in line):
            continue
        if("Chip" in line):
            ihits = int(line.split(":")[1])
            if(ihits>4000):
                continue
            else:
                nhits = ihits
                continue
        if(nhits is not None):

            words = line.split(" ")
            x = int(words[0])
            y = int(words[1])
            tot = int(words[2])

            np.add.at(occ_matrix, (x,y), 1)
            np.add.at(tot_matrix, (x,y), tot)

            x, y, tot = None, None, None
            words = None

    progress(nframes, iframe)
    iframe+=1

print("\nplotting...")
#plotting occupancy
n = 10  # number of hottest pixels to mask

msk_occ_matrix = occ_matrix.copy().astype(float)

# flatten, get indices of n largest values
flat = msk_occ_matrix.ravel()
top_idx = np.argpartition(flat, -n)[-n:]   # indices of n largest values

# mask them
flat[top_idx] = np.nan

# reshape back
msk_occ_matrix = flat.reshape(msk_occ_matrix.shape)

# plotting
vmax = np.nanmax(msk_occ_matrix)
#threshold = 250
#msk_occ_matrix = tot_matrix.copy().astype(float)
#msk_occ_matrix[msk_occ_matrix > threshold] = np.nan

fig, ax = plt.subplots(figsize=(12,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
#ms = ax.matshow(occ_matrix.T, cmap='viridis')
ms = ax.matshow(msk_occ_matrix, cmap='gist_earth_r', norm=LogNorm(vmin=1, vmax=np.nanmax(msk_occ_matrix)))
fig.colorbar(ms,cax=cax,orientation='vertical', label="occupancy")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.xaxis.set_label_position('top') 
ax.invert_yaxis()
plt.savefig(f"occupancy_matrix-{picname}.png", dpi=400)
ms = None
plt.close()

# plotting tot
fig, ax = plt.subplots(figsize=(12,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
ms = ax.matshow(tot_matrix.T, cmap='gist_earth_r')
fig.colorbar(ms,cax=cax,orientation='vertical', label="TOT")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.xaxis.set_label_position('top') 
ax.invert_yaxis()
plt.savefig(f"tot_matrix-{picname}.png", dpi=400)
ms = None
plt.close()

