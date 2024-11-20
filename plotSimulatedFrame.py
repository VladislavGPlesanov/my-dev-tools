import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def plot2dEvent(nuarray, picname):
 
    # ---- matrix 2d hist ----
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.86,0.1,0.05,0.8])
    #ms = ax.matshow(nuarray, cmap='plasma')
    ms = ax.matshow(nuarray, cmap='hot')
    fig.colorbar(ms,cax=cax,orientation='vertical')

    #xmin = np.min(nuarray, axis=0)
    #ymin = np.min(nuarray, axis=1)

    #xmax = np.max(nuarray, axis=0)
    #ymax = np.max(nuarray, axis=1)

    #print(f"Plotting 2D event for {info}, limits -> {xmin}, {xmax}, {ymin}, {ymax}")

    #start = 120
    #ax.text(-90, start, info, fontsize=10,color='black' )    

    #plt.xlim(xmin,xmax)
    #plt.ylim(ymin,ymax)

    plt.plot()
    fig.savefig(f"{picname}.png")


##################################################################
##################################################################
##################################################################

pname = sys.argv[1]
txt_list = list(sys.argv[2:])

wskip = ['Chip','FEC','Board']

nframes = 0
for txt in txt_list:

    print(f"Looking at file: {txt}")

    plt.figure()
    matrix = np.zeros((256,256),dtype=int)

    f = open(txt)
    nlines = 0

    for line in f:
    
        words = line.split()
        
        if(words[0] in wskip):
            nlines+=1
            continue

        posx = int(words[0])
        posy = int(words[1])
        nElec = int(words[2])

        matrix[posx,posy] = nElec

        nlines+=1
        

    nz = np.count_nonzero(matrix)
    print(f"matrix has {nz} position entries")
    nlines = None
    #if(nframes%10 == 0):
    ebala = f"RAWevent-{nframes}-{pname}"
    print(f"plotting: {ebala}")
    plot2dEvent(matrix, ebala) 
    if(nframes==30):
        break

    nframes+=1

