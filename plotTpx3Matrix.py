import numpy as np                                                                                                                                                                                      
import numpy.ma as ma  
import sys
import os
#import glob                                                                                       
#import tables as tb
import matplotlib
matplotlib.use("Agg")  # non-GUI backend (renders to files only)                                  
import matplotlib.pyplot as plt                                                                   
from matplotlib.colors import LogNorm                                                             

file = sys.argv[1]                                                                           
picname = sys.argv[2]
limit = int(sys.argv[3])

tmp_array = np.loadtxt(file)

mask = (tmp_array > limit)

masked_array = np.ma.masked_array(tmp_array, mask)


fig,ax = plt.subplots(figsize=(8,8))
cax = fig.add_axes([0.86,0.1,0.05,0.8])
#ms = ax.matshow(tmp_array.T, cmap='viridis', norm=LogNorm(vmin=1,vmax=np.nanmax(tmp_array)))
#ms = ax.matshow(tmp_array[mask].T, cmap='jet')
ms = ax.matshow(masked_array, cmap='jet')
fig.colorbar(ms,cax=cax, orientation='vertical')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.invert_yaxis()
plt.savefig(f"MATRIX-{picname}.png")
plt.close()
fig, ax = None, None
 


