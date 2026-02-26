import numpy as np
import sys
import matplotlib.pyplot as plt

picname = sys.argv[1]

# charge peak, rotatin' runs

logfiles = ['gridpix_00080.fio',
            'gridpix_00082.fio',
            'gridpix_00083.fio',
            'gridpix_00086.fio']

basepath = '/home/vlad/Documents/P09-aux-files/'

Ibeamlist, timelist = [],[]
run_labels = []

runtypes = {"00046":"CP main 2h", 
            "00047":"CP aux 1h",
            "00087":"CP1",
            "00088":"CP2",
            "00083":"CP3",
            "00082":"CP4",
            "00081":"CP5 central",
            "00080":"CP5"
            #"":"",
            #"":"",
            #"":"",
}

#nWords = []

for file in logfiles:
    f = open(basepath+file,'r')
    
    Ibeam, time = [],[]
    cnt = 0
    dfound = False
    for line in f:
        cnt+=1
        #print(f"{line}")
        if(cnt<149 or "!" in line):
            print("SKIP")
            continue
        else:
            print("USING")
            words = line.split(" ")
            print(len(words))
            #nWords.append(len(words))
            Ibeam.append(float(words[8]))
            time.append(float(words[17]))
    
    runnr = file.split("_")[1].split(".")[0]
    if(runnr in runtypes.keys()):
        run_labels.append(runtypes[runnr])
    else:
        run_labels.append(runnr)
    Ibeamlist.append(Ibeam)
    timelist.append(time)
    
    #Ibeam.clear()
    #time.clear()


print(len(timelist))
print(timelist[0][0:20])
print(len(Ibeamlist))
print(Ibeamlist[0][0:20])
#print(nWords)
#avg_current = np.mean(Ibeam)
#std_current = np.std(Ibeam)

plt.figure(figsize=(10,6))
for icur, itime, ilab in zip(Ibeamlist, timelist, run_labels):
   plt.plot(itime,icur, marker=".", label=ilab)
#plt.scatter([],[],label=r"$\overline{I_{beam}}$="+f"{avg_current:.2f}")
plt.xlabel("time, [s]")
plt.ylabel(r"$I_{beam}$")
plt.title(f"Petra Beam Current")
#plt.ylim([avg_current-4*std_current , avg_current+4*std_current])
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(f"PetraMultipleCurrent-{picname}.png",dpi=200)

#total_runtime = max(time)
#print(f"Total runtime = {total_runtime:.4f} [s]")

