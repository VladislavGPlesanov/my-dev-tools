import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)


dirpath = sys.argv[1]
npixels_active = int(sys.argv[2])
picname = sys.argv[3]

matrix = np.zeros((256,256), dtype=int)

filelist = glob.glob(dirpath+"/*.txt")
flist = sorted(filelist)

nfiles = len(flist)
n_lowcount = 0

#total_hits = []

ifile = 0
for file in flist:

    tmp_array = np.loadtxt(file, dtype=int)
    nhits = np.count_nonzero(tmp_array)
    if(nhits <= npixels_active):
        ifile+=1
        n_lowcount+=1
        matrix += tmp_array    
    else:
        ifile+=1
        #continue
    progress(nfiles, ifile)


nonzero = np.nonzero(matrix)
print("for this dataset the most noisy channest are:")
for i,j in zip(nonzero[0],nonzero[1]):
    print(f"x={i}, y={j} with {matrix[i][j]} times triggered")

fig, ax = plt.subplots()
cax = fig.add_axes([0.86, 0.1, 0.05, 0.8])
#ms = ax.matshow(matrix.T, cmap='gist_earth_r')
ms = ax.matshow(matrix.T, cmap='viridis')
ax.set_title("Noisy channels")
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.plot()
fig.savefig(f"Noisy-channels-FullMatrix-{picname}.png")

ind = np.unravel_index(np.argsort(matrix, axis=None), matrix.shape)
print(ind)
print(matrix[ind])
print("-------- highest occupancy pixels -------------------------")
cnt = 0

Nchannels = 10

huya = ind[0][len(ind[0])-Nchannels:len(ind[0])]
suka = ind[1][len(ind[1])-Nchannels:len(ind[1])]

xstring, ystring = "", ""

for k,l in zip(huya,suka):
    print(f"[{k}] [{l}] ---> {matrix[k][l]}")
    xstring+= str(k)+","
    ystring+= str(l)+","
    cnt+=1
    if(cnt==10):
        break

print(xstring)
print(ystring)

#print("+++++++++++++++++++++++++++++++++++++++++++++\n")
#
#counts, edges = np.histogram(total_hits, bins=100, range=(0,12000))
#bin_centers = (edges[:-1] + edges[1:])/2
#bin_width = edges[1] - edges[0]
#
#bin_cut = 7
#integral = float(bin_width*sum(counts[bin_cut+1:len(counts)]))
#
#plt.figure(figsize=(8,8))
#plt.hist(bin_centers, weights=counts, bins=100,  range=(0,12000), histtype='stepfilled', facecolor='b')
#plt.plot([],label=f"Total hits = {integral} above bin {bin_cut}")
#plt.plot([],label=f"AVG hits per proton track = 1403.5")
#plt.plot([],label=r"$N_{protons}$ = "+f"{(integral/1403.5):.2f}")
#plt.title("Total hits on matrix")
#plt.xlabel("N pixels activated")
#plt.ylabel("CNT")
#plt.grid(True)
#plt.yscale('log')
#plt.legend(loc='upper right')
#plt.savefig(f"Total-hits-recorded-{picname}.png")
#plt.savefig(f"Total-hits-recorded-{picname}.pdf")
#plt.close()

print(f'For this run we have {nfiles} frames recorded')
print(f'{n_lowcount} are with {npixels_active} pixels active or less')




