import numpy as np
import matplotlib.pyplot as plt
import sys

infiles = sys.argv[1].split(',')
picname = sys.argv[2]

labels = []

for name in infiles:
    half = name.split('-')
    hole_size = half[1].split('.')
    labels.append(hole_size[0][:-4])

rates, positions = [],[]

plt.figure(figsize=(8,8))

f_cnt = 0
for file in infiles:

    f = open(file)


    for line in f:

        words = line.split(',')

        positions.append(float(words[0]))    
        rates.append(float(words[1]))    
        
    f.close()
    plt.scatter(positions, rates, s=20, marker='o', label = f"diam={labels[f_cnt]}",zorder=2)

    positions,rates = [],[]
    f_cnt+=1

#plt.yscale('log')
plt.legend()
plt.title("Total event countrate based on the absolute beam position")
plt.xlabel('Coordinate x,[cm]')
plt.ylabel('Rate,[Hz]')
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)


plt.savefig(f'combined-{picname}.png')




