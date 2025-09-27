import numpy as np
import matplotlib.pyplot as plt
import sys
import tables as tb

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)


infile = sys.argv[1]
picname = sys.argv[2]

matrix = np.zeros((256,256), dtype=np.int32)
xskip = [96,206,183,56,101,128,103,106,194,204, -9999]
yskip = [161,84,123,64,103,125,112,127,202,245, -9999]

with tb.open_file(infile,'r') as f:

    xdata = f.root.data.x[:]
    ydata = f.root.data.y[:]
    #TOTdata = f.root.data.TOT[:]
    TOTdata = f.root.data.TOA[:]
  
    nframes = len(xdata)
    iframe = 0
    for xhits, yhits, tots in zip(xdata, ydata, TOTdata):
        for x,y,t in zip(xhits,yhits,tots):
            if(x in xskip and y in yskip):
                continue
            else:
                np.add.at(matrix, (x,y), t)
        iframe+=1
        progress(nframes, iframe)


plt.figure(figsize=(8,8))
plt.imshow(matrix, origin='lower', cmap='inferno')
#plt.colorbar(label='Sum TOT')
plt.colorbar(label='Sum TOA')
plt.xlabel('pixel x')
plt.ylabel('pixel y')
#plt.title('Total Charge For Each Pixel')
plt.title('Timing (TOA) For Each Pixel')
plt.tight_layout()
#plt.savefig(f'SunTOTmatrix-{picname}.png')
plt.savefig(f'SumTOAmatrix-{picname}.png')


