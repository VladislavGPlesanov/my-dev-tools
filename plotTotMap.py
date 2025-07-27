import numpy as np
import matplotlib.pyplot as plt
import sys
from MyPlotter import myPlotter

file = sys.argv[1]
 
#rawtot = np.loadtxt(file, delimiter=' ')
rawtot = np.genfromtxt(file, delimiter='\t')
print(rawtot)
clearTOT = np.nan_to_num(rawtot, nan=0.0)

print(clearTOT)

print(len(clearTOT))
print(len(clearTOT[0]))

meanTOT = np.mean(clearTOT)
std = np.std(clearTOT)

print(np.floor(meanTOT), np.ceil(std))

mp = myPlotter()

mp.pixelMap(clearTOT, "EBALA")

