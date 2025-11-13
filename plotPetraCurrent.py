import numpy as np
import sys
import matplotlib.pyplot as plt

logfile = sys.argv[1]
picname = sys.argv[2]

Ibeam, time = [],[]

nWords = []

f = open(logfile,'r')

cnt = 0
dfound = False
for line in f:
    cnt+=1
    #print(f"{line}")
    if(cnt<149 or "!" in line):
        #print("SKIP")
        continue
    else:
        #print("USING")
        words = line.split(" ")
        #print(len(words))
        nWords.append(len(words))
        Ibeam.append(float(words[8]))
        time.append(float(words[17]))

runnr = logfile.split("_")[1].split(".")[0]

#print(nWords)
avg_current = np.mean(Ibeam)
std_current = np.std(Ibeam)

plt.figure(figsize=(8,6))
plt.plot(time,Ibeam, marker=".", c="red")
plt.scatter([],[],label=r"$\overline{I_{beam}}$="+f"{avg_current:.2f}")
plt.xlabel("time, [s]")
plt.ylabel(r"$I_{beam}$")
plt.title(f"Petra Beam Current (run {runnr})")
plt.ylim([avg_current-4*std_current , avg_current+4*std_current])
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(f"PetraCurrent-{runnr}-{picname}.png",dpi=400)


