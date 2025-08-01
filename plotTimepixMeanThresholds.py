import numpy as np
import matplotlib.pyplot as plt 
import sys

infile = sys.argv[1]
infile_after = sys.argv[2]
picname = sys.argv[3]

#### declaring some data containers

xpix, ypix = [], []

THL, THL_after = [], []
THL_0, THL_0_after = [], []
THL_15, THL_15_after = [], []
trimmers, trimmers_after = [], []

THL_matrix=np.zeros((256,256), dtype=int)
THL_matrix_after=np.zeros((256,256), dtype=int)

### opening files, reading data

f = open(infile,'r')

for line in f:

    words = line.split("\t")

    x = int(words[0])
    y = int(words[1])
    thl0 = int(words[2])
    thl15 = int(words[3])
    trim = int(words[4])
    thl_result = int(words[5])

    np.add.at(THL_matrix, (x,y), thl_result)

    xpix.append(x)
    ypix.append(y)

    THL.append(thl_result)
    THL_0.append(thl0)
    THL_15.append(thl15)
    trimmers.append(trim)

    words=None

f.close()
f_after = open(infile_after,'r')

for line in f_after:

    words = line.split("\t")

    x = int(words[0])
    y = int(words[1])
    thl0 = int(words[2])
    thl15 = int(words[3])
    trim = int(words[4])
    thl_result = int(words[5])

    np.add.at(THL_matrix_after, (x,y), thl_result)

    THL_after.append(thl_result)
    THL_0_after.append(thl0)
    THL_15_after.append(thl15)
    trimmers_after.append(trim)

    words=None

f_after.close()

### plotting trimmers

plt.figure(figsize=(8,8))
plt.hist(np.array(trimmers), 15, range=(0,15), alpha=0.5, label="Before Irradiation")
plt.hist(np.array(trimmers_after), 15, range=(0,15), alpha=0.25, label="After Irradiation")
plt.title("Distriubution of Equalisation Bits")
plt.xlabel("Trimmer value")
plt.ylabel("# channels")
plt.legend()
plt.grid()
plt.savefig(f"TrimmerDistr-{picname}.png")
plt.savefig(f"TrimmerDistr-{picname}.pdf")
plt.close()

### plotting Equalised Thresholds

plt.figure(figsize=(8,8))
plt.hist(np.array(THL), 100, range=(250,450), alpha=0.5, label="Before Irradiation")
plt.hist(np.array(THL_after), 100, range=(250,450), alpha=0.25, label="After Irradiation")
plt.title("Distriubution of Equalised Thresholds")
plt.xlabel("DAC counts")
plt.ylabel("# channels")
plt.legend()
plt.grid()
plt.savefig(f"EqualisedTHLDistr-{picname}.png")
plt.savefig(f"EqualisedTHLDistr-{picname}.pdf")
plt.close()

### plotting changes in matrix

THL_matrix_diff = np.subtract(THL_matrix,THL_matrix_after)

print(THL_matrix_diff)

fig, ax = plt.subplots()
cax = fig.add_axes([0.86,0.1,0.05,0.8])
ms = ax.matshow(THL_matrix_diff.T, cmap='viridis')
ax.set_title("Change in Absolute Threshold of Pixels")
ax.set_xlabel("Pixel x")
ax.set_ylabel("Pixel y")
fig.colorbar(ms,cax=cax,orientation='vertical')
plt.plot()
fig.savefig(f"THL-matrix-diff-{picname}.png")
fig.savefig(f"THL-matrix-diff-{picname}.pdf")

#fig, ax = plt.subplots()
#cax = fig.add_axes([0.86,0.1,0.05,0.8])
#ms = ax.matshow(THL_matrix_after.T, cmap='viridis')
#ax.set_title("THL of Each pixel after Irrad.")
#ax.set_xlabel("Pixel x")
#ax.set_ylabel("Pixel y")
#fig.colorbar(ms,cax=cax,orientation='vertical')
#plt.plot()
#fig.savefig(f"THL-matrix-after-{picname}.png")







