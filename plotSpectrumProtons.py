import numpy as np
import matplotlib.pyplot as plt
import sys
#
#works for files with one column of numbers

#infile = ~/Downloads/build5_PrimaryParticles_GPix_gasvol.csv
infile = sys.argv[1]
nbins = 30

f = open(infile,'r')

n_zero , n_int = 0, 0
clean_data = None
shmata = []

for line in f:
    
    if("Energy" in line):
        continue
    
    shmata.append(float(line))

shmata = np.array(shmata, dtype=float)

counts, edges = np.histogram(shmata, bins=nbins) 

plt.figure()
plt.hist(edges[:-1], weights=counts, bins=nbins, range=(0,15))
plt.title('Proton energy spectrum for 13,609 MeV beam (after shielding, in air)')
plt.xlabel(r'$E_{p^{+}}$, [MeV]')
plt.ylabel('CTS')
plt.yscale('log')
plt.grid(True)
plt.savefig("OLOLO-spectrum.png")
