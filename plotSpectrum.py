import numpy as np
import matplotlib.pyplot as plt
import sys

def count_zero_and_nonzero_groups(data):
    i = 0
    n = len(data)

    while i < n:
        # Count consecutive elements starting with '0'
        n_zeros = 0
        while i < n and data[i].startswith('0'):
            n_zeros += 1
            i += 1

        # If no zeros found, skip to the next
        if n_zeros == 0:
            i += 1
            continue

        # Count consecutive elements that do NOT start with '0'
        n_integers = 0
        while i < n and not data[i].startswith('0'):
            n_integers += 1
            i += 1

        print(f"group of {n_zeros} have {n_integers} afterwards")

infile = sys.argv[1]
nbins = int(sys.argv[2])

f = open(infile,'r')

n_zero , n_int = 0, 0
ibins, counts, edges = [],[],[]
#nbins = 1024
clean_data = None
shmata = []

for line in f:
    
    words = line.split(',')

    clean_data = words[2:]
    cnt = 0
    for w in words:
        if(w[0]=='B' or w[0]=='C'):
           
            continue
        elif(w[0]=='0'):
            n_zero+=1
            ibins.append(int(w))
            shmata.append(int(w))
        else:
            n_int+=1
            counts.append(int(w))
            shmata.append(int(w))
            edges.append(float(cnt)-0.5)
            cnt+=1
edges.append(edges[len(edges)-1]+1)

#
#print(f"N_zero={n_zero} and N_int={n_int}")
#print('\n BINS:')
#print(ibins)
#print('\n COUNTS:')
#print(counts)
#print('\n')
#for i,j in zip(ibins,counts):
#    print(f"{i} , {j}")
#
#print(f"{len(ibins)} vs {len(counts)}")
#
print("================================================")
print(len(edges))
print(len(counts))
count_zero_and_nonzero_groups(clean_data)
print("================================================")

#
plt.figure()
plt.hist(edges[:-1], weights=counts, bins=nbins, range=(0,nbins))
#plt.hist(shmata, bins=4096, range=(0,4096))
plt.savefig("OLOLO.png")
