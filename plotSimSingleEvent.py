import numpy as np
import matplotlib.pyplot as plt
from MyPlotter import myPlotter
from MyPlotter import myUtils 
import sys
import glob
import os
from operator import add


inputfile =sys.argv[1]

pixelX, pixelY, Charge = [], [], []

matrixEvent = np.zeros((256,256),dtype=int)

cnt=0 # n total hits
with open(inputfile,'r') as f:
    
    for line in f:

        if('FEC' in line or 'Board' in line or 'Chip' in line):
            continue
        
        words = line.split()
        px = int(words[0])     
        py = int(words[1])     
        q = int(words[2])     

        if(cnt<20):
            print(f"Found x={px} y={py} q={q} on line {cnt}")

        cnt+=1

        np.add.at(matrixEvent, (px,py), q)


# event number

fullline = inputfile.split('/')
print(fullline)
filename = fullline[len(fullline)-1]
print(filename)
runnr = filename.split('_')[3]
print(runnr)

mp = myPlotter()

mp.pixelMap(matrixEvent,f"Event-{runnr}")





