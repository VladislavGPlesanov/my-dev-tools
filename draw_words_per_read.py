import numpy as np 
import matplotlib.pyplot as plt
import sys


def plotScatter(listx,listy,axislist,labellist,title,picname):
    plt.figure(0,figsize=(10,10))

    i = 0;
    #clrs = ['blue','orange','green','yellow','cyan','magenta','red','black']
 
    for ilist in listx:
        if(len(ilist)!=len(listy[i])):
            print("incompatible list sizes at iteration {}!".format(i))
        #plt.scatter(ilist,listy[i],c=clrs[i],label=labellist[i])
        plt.scatter(ilist,listy[i],label=labellist[i])
        i+=1

    plt.title("words per read")
    plt.xlabel(axislist[0])
    plt.ylabel(axislist[1])
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(picname+".png")

def getUniqueDacs(daclist):

    # straight iteration through list
    #unique_dacs = []
    #for dac in daclist:
    #    if dac not in unique_dacs:
    #        unique_dacs.append(dac)
    #
    # or use set func.
    
    return list(set(daclist))

########################################################

pyfile = sys.argv[1]
cppfile = sys.argv[2]
picname = sys.argv[3]

words_cpp = []
words_py = []
tot_words_py = []
tot_words_cpp = []

words_cpp_nozero = []
words_py_nozero = []
scan_ids_py_nozero = []
scan_ids_cpp_nozero = []

scan_ids_py = []
scan_ids_cpp = []

tries_per_id_cpp = {}
tries_per_id_py = {}

fpy = open(pyfile)

prev_id = None
this_d = None

py_data = []
cpp_data = []

##### readin' py word txt
cnt = 0
nzero_py = 0
for line in fpy:
    words = line.split(",")
    if(cnt<10):
        print(words)
    totwords = float(words[0])
    nwords = float(words[1])
    dac = float(words[2])
    words_py.append(nwords)
    if(nwords==0):
        nzero_py+=1
    if(nwords>0):
        words_py_nozero.append(nwords)    
        scan_ids_py_nozero.append(dac)
    scan_ids_py.append(dac)
    tot_words_py.append(totwords)
    py_data.append([nwords,dac])
    cnt+=1

print(f"Went through {cnt} lines")
fpy.close()
##### readin' cpp word txt
fcpp = open(cppfile)

cnt2 = 0
nzero_cpp = 0
for line in fcpp:
    words = line.split(",")
    if(cnt2<20):
        print(words)
    totwords = float(words[0])
    nwords = float(words[1])
    dac = float(words[2])
    words_cpp.append(nwords)
    if(nwords==1):
        nzero_cpp+=1
    if(nwords>1):
        words_cpp_nozero.append(nwords)    
        scan_ids_cpp_nozero.append(dac)
    scan_ids_cpp.append(dac)
    tot_words_cpp.append(totwords)
    cpp_data.append([nwords,dac])
    cnt2+=1

print(f"Went through {cnt2} lines")
fcpp.close()
####### calc avg words per readout cycle ########

unique_py_dacs = sorted(getUniqueDacs(scan_ids_py))
unique_cpp_dacs = sorted(getUniqueDacs(scan_ids_cpp))
for i in range(len(unique_py_dacs)):
    if(i%10==0):
        print(unique_py_dacs[i])

avg_reads_py = []
avg_reads_cpp = []
py_nreads = []
cpp_nreads = []
###for py file

nreads = 0
words_acc = 0

for dac in unique_py_dacs:
    for lst in py_data:
        if lst[1] == dac and lst[0]>0:
            words_acc+=lst[0]
            nreads+=1
        else:
            continue
    avg = 0
    try: 
        avg = words_acc/float(nreads)
    except ZeroDivisionError:
      print("tried to divide: {} by {} for DAC={}".format(words_acc, float(nreads), dac))
    avg_reads_py.append(avg)
    py_nreads.append(nreads)

    words_acc = 0
    nreads = 0

print("unique_py_dacs={}".format(len(unique_py_dacs)))
print("avg_reads_py={}".format(len(avg_reads_py)))

###for cpp file
for dac in unique_cpp_dacs:
    for lst in cpp_data:
        if lst[1] == dac and lst[0]>0:
              words_acc+=lst[0]
              nreads+=1
        else:
             continue 
    avg = 0
    try:
      avg = words_acc/float(nreads)
    except ZeroDivisionError:
      print("tried to divide: {} by {} for DAC={}".format(words_acc, float(nreads), dac))
    avg_reads_cpp.append(avg)
    cpp_nreads.append(nreads)

    words_acc = 0
    nreads = 0 
        

print("unique_cpp_dacs={}".format(len(unique_cpp_dacs)))
print("avg_reads_cpp={}".format(len(avg_reads_cpp)))

####### plotting #####
plotScatter([scan_ids_py, scan_ids_cpp],
            [words_py, words_cpp],
            ["Scan_ids, [DAC]", "N words per interval"],
            ["python", "C++"],
            "Words Per Read",
            picname)

print(len(words_py))
print(len(words_cpp))

plt.figure(1,figsize=(10,10))
plt.scatter(scan_ids_py_nozero,words_py_nozero,c='blue',label="python")
plt.scatter(scan_ids_cpp_nozero,words_cpp_nozero,c='orange', label="c++")
plt.title("words per read, no zero")
plt.xlabel("SCAN_IDS")
plt.ylabel("N_WORDS_PER INTERVAL_nozero")
plt.yscale('log')
plt.grid(True)
plt.legend(loc="upper left")
plt.savefig(picname+"-nozero.png")

print(len(words_py_nozero))
print(len(words_cpp_nozero))

plt.figure(2,figsize=(10,10))
plt.scatter(scan_ids_py,tot_words_py,c='blue', label="python")
plt.scatter(scan_ids_cpp,tot_words_cpp,c='orange', label="c++")
plt.title("Words recorded by \"scan_base::handle_data()\"")
plt.xlabel("Scan_ids, [DAC]")
plt.ylabel("Total words recorded (x10^6) [N]")
#plt.yscale('log')
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig(picname+"-total.png")

print("py data has {} packets of len=0".format(cnt - nzero_py))
print("cpp data has {} packets of len=0".format(cnt2 - nzero_cpp))

##########################################################

plt.figure(3,figsize=(10,10))
#plt.hist(Elist, bins=100, range=(0,20000),alpha=0.25,label=f'{Ej} Gev')
countspy, edgespy, barspy = plt.hist(words_py, bins=100, range=(0,1000), alpha=0.5, label="python")
countscpp, edgescpp, barscpp = plt.hist(words_cpp, bins=100, range=(0,1000), alpha=0.5, label="c++")
for i in range(99): # nbins-1 
    if(countspy[i]>0):
        plt.text(edgespy[i]+2.5, countspy[i], str(int(countspy[i])), color='blue', fontsize=13)

for i in range(99): # nbins-1 
    if(countscpp[i]>0):
        plt.text(edgescpp[i]+10.5, countscpp[i], str(int(countscpp[i])), color='orange',fontsize=13)

plt.hist([], color='white', label=f'N(Py, total)={len(words_py)}')
plt.hist([], color='white', label=f'N(C++, total)={len(words_cpp)}')

plt.yscale('log')
plt.title("Package lengths (low end)")
plt.xlabel("Data length, [# words]")
plt.ylabel("N packages")
plt.legend(loc="upper right",prop={'size':14})
plt.grid(True)
plt.savefig(picname+"-hist-nwords")

##################################################################
##print("histo for py = {}, {}, {}".format(xpy,ypy,conpy))
#print("histo for py = {}\n{}\n".format(countspy,edgespy))
##print("histo for py = {}, {}, {}".format(xcpp,ycpp,concpp))
#print("histo for cpp = {}\n{}\n".format(countscpp,edgescpp))

plt.figure(4,figsize=(10,10))
idcountspy, idedgespy, idbarspy = plt.hist(scan_ids_py, bins=60, range=(1199.5,1260.5), alpha=0.5, label="python")
idcountscpp, idedgescpp, idbarscpp = plt.hist(scan_ids_cpp, bins=60, range=(1199.5,1260.5), alpha=0.5, label="c++")
for i in range(59): # nbins-1 
    if(idcountspy[i]>0 and i%2==0):
        plt.text(idedgespy[i], idcountspy[i], str(int(idcountspy[i])), color='blue')

for i in range(59): # nbins-1 
    if(idcountscpp[i]>0 and i%2==0):
        plt.text(idedgescpp[i], idcountscpp[i], str(int(idcountscpp[i])), color='red')

plt.title("Received packages - including zero length")
plt.xlabel("Scan ID, [DAC]")
plt.ylabel("Number of packages received, [N]")
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig(picname+"-hist-scanids-all")

plt.figure(5,figsize=(10,10))
nzidcountspy, nzidedgespy, nzidbarspy = plt.hist(scan_ids_py_nozero, bins=60, range=(1199.5,1260.5), alpha=0.5, label="python")
nzidcountscpp, nzidedgescpp, nzidbarscpp = plt.hist(scan_ids_cpp_nozero, bins=60, range=(1199.5,1260.5), alpha=0.5, label="c++")
for i in range(59): # nbins-1 
    if(nzidcountspy[i]>0 and i%2==0):
        plt.text(nzidedgespy[i]+0.5, nzidcountspy[i], str(int(nzidcountspy[i])), color='blue')

for i in range(59): # nbins-1 
    if(idcountscpp[i]>0 and i%2==0):
        plt.text(nzidedgescpp[i], nzidcountscpp[i], str(int(nzidcountscpp[i])), color='red')

#plt.xscale('log')
#plt.yscale('log')
plt.title("Received packages - Non-zero length")
plt.xlabel("Scan ID, [DAC]")
plt.ylabel("Number of packages received, [N]")
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig(picname+"-hist-scanids-nozero")


################################################################

plt.figure(6,figsize=(10,10))
plt.scatter(unique_py_dacs,avg_reads_py,c='blue', label="python")
plt.scatter(unique_cpp_dacs,avg_reads_cpp,c='orange', label="c++")
plt.title("Averaged words per readout cycle")
plt.xlabel("Scan_ids, [DAC]")
plt.ylabel("Words, [N]")
#plt.yscale('log')
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig(picname+"-words-avg.png")


plt.figure(7,figsize=(10,10))
plt.scatter(unique_py_dacs,py_nreads,c='blue', label="python")
plt.scatter(unique_cpp_dacs,cpp_nreads,c='orange', label="c++")
plt.title("Averaged words per readout cycle")
plt.xlabel("Scan_ids, [DAC]")
plt.ylabel("number of readout tries (non-rezo), [N]")
#plt.yscale('log')
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig(picname+"-words-tries.png")





















