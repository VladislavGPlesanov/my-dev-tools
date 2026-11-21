import numpy as np
import matplotlib.pyplot as plt 
import sys

from MyPlotter import myPlotter

infile = sys.argv[1]
infile_after = sys.argv[2]
picname = sys.argv[3]

MP = myPlotter()
#### declaring some data containers

xpix, ypix = [], []

THL, THL_after = [], []
THL_0, THL_0_after = [], []
THL_15, THL_15_after = [], []
trimmers, trimmers_after = [], []

THL_matrix=np.zeros((256,256), dtype=int)
THL_matrix_after=np.zeros((256,256), dtype=int)

matrix_btrim = np.zeros((256,256), dtype=int)
matrix_atrim = np.zeros((256,256), dtype=int)

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
    np.add.at(matrix_btrim, (x,y), trim)

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
    np.add.at(matrix_atrim, (x,y), trim)

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

MP.plotMatrix(
    matrix_btrim.T,
    f"TRIMMERS-BEFORE-{picname}",
    figsize=(9,8),
    cmap='jet',
    cbarname="Trimmers [DAC]",
    fInvertYback=True
    #fLognorm=True
)

MP.plotMatrix(
    matrix_atrim.T,
    f"TRIMMERS-AFTER-{picname}",
    figsize=(9,8),
    cmap='jet',
    cbarname="Trimmers [DAC]",
    fInvertYback=True
    #fLognorm=True
)

trim_diff = matrix_atrim - matrix_btrim

MP.plotMatrix(
    trim_diff.T,
    f"TRIMMERS-DIFF-{picname}",
    figsize=(9,8),
    cmap='coolwarm',
    cbarname=r"$\Delta$ Trimmers [DAC]",
    fInvertYback=True
    #fLognorm=True
)

### plotting Equalised Thresholds

thlstd_before = np.std(THL)
meanthl_before = np.mean(THL)
thlstd_after = np.std(THL_after)
meanthl_after = np.mean(THL_after)

xmin, xmax = 280, 400
#xmin = int(np.floor(np.min([np.percentile(THL,5),np.percentile(THL_after,5)])*0.9))
#xmax = int(np.floor(np.max([np.percentile(THL,95),np.percentile(THL_after,95)])*1.1))

plt.figure(figsize=(8,8))
plt.hist(np.array(THL), 
            xmax-xmin, 
            range=(xmin,xmax), 
            alpha=0.5, 
            label="Before Irradiation\n"+r"$\mu$="+f"{meanthl_before:.2f}\n"+r"$\sigma$="+f"{thlstd_before:.2f}")
plt.hist(np.array(THL_after), 
            xmax-xmin, 
            range=(xmin,xmax), 
            alpha=0.5, 
            label="After Irradiation\n"+r"$\mu$="+f"{meanthl_after:.2f}\n"+r"$\sigma$="+f"{thlstd_after:.2f}")
plt.title("Distriubution of Equalised Thresholds")
plt.xlabel("DAC counts")
plt.ylabel("# channels")
plt.legend()
plt.grid()
plt.savefig(f"EqualisedTHLDistr-{picname}.png")
plt.savefig(f"EqualisedTHLDistr-{picname}.pdf")
plt.close()

### plotting changes in matrix

MP.plotMatrix(
    THL_matrix.T,
    f"THL-BEFORE-{picname}",
    figsize=(9,8),
    uTitle="Pixel Threshods (Before)",
    cmap='coolwarm',#'jet',
    cbarname="DAC,[CTS]",
    cbarmax = 600,
    fInvertYback=True,
    fLognorm=True
)

MP.plotMatrix(
    THL_matrix_after.T,
    f"THL-AFTER-{picname}",
    figsize=(9,8),
    uTitle="Pixel Threshods (After)",
    cmap='coolwarm',#'jet',
    cbarname="DAC,[CTS]",
    cbarmax = 600,
    fInvertYback=True,
    fLognorm=True
)

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

# tryna' plot direction change in trimmer values
updown_mat = np.zeros((256,256), dtype=int)

for i in range(256):
    for j in range(256):

        change = THL_matrix[i][j] - THL_matrix_after[i][j]
        updown_mat[i][j] = change

MP.plotMatrix(
    updown_mat.T,
    f"THL-CHANGE-{picname}",
    figsize=(9,8),
    cmap='jet',
    cbarname=r"$\Delta$(DAC)",
    fInvertYback=True#,
    #fLognorm=True
)



