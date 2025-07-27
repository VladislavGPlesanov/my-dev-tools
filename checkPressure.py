import numpy as np
import matplotlib.pyplot as plt
import sys

pfile = open(sys.argv[1])

time, relP = [],[]

picname = sys.argv[1].split(".")[0]

runNumber = picname.split('_')[1]

cnt = 0

for line in pfile:

    if("#" in line):
        continue

    words = line.split()
    #tstamp.append(int(words[0]))
    relP.append(float(words[1]))
    time.append(cnt)
    cnt+=1    


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
#ax.scatter(tstamp, relP)
ax.scatter(time, relP)
ax.set_xlabel("Time, [a.u.]")
ax.set_ylabel("Overpressure, [bar]")
#plt.xlim([0.1, 5])
#plt.ylim([0, 5])
ax.set_title('Pressure during measurement '+runNumber)
ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
ax.grid(which='minor', color='grey', linestyle='--', linewidth=0.25)
ax.minorticks_on()
fig.savefig("EBALA"+picname+".png")






