import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

#accepts lists as input
def getStats(datapoints):
    
    maximum = max(datapoints)
    avg = sum(datapoints)/len(datapoints)   
    minimum = min(datapoints) 

    return maximum, avg, minimum

def checkIndexAndPut(index,data,threadlist):
    if(index in range(len(data))):
        threadlist.append(float(data[index][:-4]))
    else:
        threadlist.append(float(0))   

file_in = None
picname = None
file_in = sys.argv[1]
try:
    picname = sys.argv[2]
except IndexError:
    print("You forgot the picname! Please enter it\n")
    try:
        picname = input()
    except KeyboardInterrupt:
        print("ok, you had your chance -the resulting pic will be named with a \"default\" + some randon integer in the name")
        sleep(2)
        some_nr = np.random.randint(0,1000,1)
        picname = "default"+str(some_nr)

if(file_in==None):
    print("please specify input textfile")
    exit(0)

#if(picname == None):
#    if(file_in != None):
#        fname = file_in.split(".")
#        if("/" in fname):
#            fname_noslash = fname.split("/")
#            picname = fname_noslash[len(fname_noslash)]
#        else:
#            picname = fname[0]

fullPicName = "memusg_"+picname+".png"

f = open(file_in)

mem_reader = []
mem_main = []
mem_trans = []
mem_plotter = []

main_index = 0
reader_index = 2
trans_index = 4
plotter_index = 6

cnt=0
nthreads = 0
nthreads_now = 0
# to be commented out
#for line in f:
#    words = line.split(',')
#    clean_word0 = None
#    clean_word2 = None
#    clean_word4 = None
#    clean_word6 = None
#    if(len(words)>=4):
#        clean_word0 = words[main_index][:-4]
#        clean_word2 = words[reader_index][:-4]
#        mem_main.append(float(clean_word0))    
#        mem_reader.append(float(clean_word2))    
#        if(plotter_index in range(len(words))):
#            clean_word4 = words[transc_index][:-4]
#            clean_word6 = words[plotter_index][:-4]
#            mem_trans.append(float(clean_word4))
#            mem_plotter.append(float(clean_word6))
#
#    if(cnt==200):
#        break   
#
#    if(cnt<40):
#        print("len={} : {} ".format(len(words),words))
#        if(cnt==0):
#            nthreads = len(words)/2
#
#    cnt+=1
###############################

xaxis = []
for line in f:
    words = line.split(',')
    #clean_word0 = None
    #clean_word2 = None
    #clean_word4 = None
    #clean_word6 = None
    #n_columns = len(words)
    checkIndexAndPut(main_index, words, mem_main)
    checkIndexAndPut(reader_index, words, mem_reader)
    checkIndexAndPut(trans_index, words, mem_trans)
    checkIndexAndPut(plotter_index, words, mem_plotter)
 
    if(cnt<40):
        print("len={} : {} ".format(len(words),words))
        if(cnt==0):
            nthreads = len(words)/2

        nthreads_now = len(words)/2
        if(nthreads_now > nthreads):
            nthreads = nthreads_now

    cnt+=1
    xaxis.append(cnt*0.5)

maxmem_reader = max(mem_reader)
avgmem_reader = sum(mem_reader)/len(mem_reader)
minmem_reader = min(mem_reader)

maxmem_main = max(mem_main)
avgmem_main = sum(mem_main)/len(mem_main)
minmem_main = min(mem_main)

maxmem_trans = max(mem_trans)
maxmem_plotter = max(mem_plotter)

reader_leak = max(mem_reader[5:len(mem_reader)-5]) - min(mem_reader[5:len(mem_reader)-5])
main_leak = max(mem_main[5:len(mem_main)-5]) - min(mem_main[5:len(mem_main)-5])
plotter_leak = max(mem_plotter[5:len(mem_plotter)-5]) - min(mem_plotter[5:len(mem_plotter)-5])

maxy = max([maxmem_reader, maxmem_plotter, maxmem_trans, maxmem_main])

####### plotting 
#
plt.figure(0,figsize=(10,10))
plt.plot(xaxis, mem_main,c='red',label='tpx3_gui main thread')
plt.plot(xaxis, mem_reader,c='blue',label='fifo_readout::reader')
if(nthreads>2):
    plt.plot(xaxis, mem_trans,c='orange',label='UI::GUI::converter::transceiver')
    plt.plot(xaxis, mem_plotter,c='green',label='plotter')
    plt.scatter(-1,maxmem_trans,c='white',label='max(transceiver)={} [Mb]'.format(maxmem_trans))
    plt.scatter(-1,maxmem_plotter,c='white',label='max(plotter)={} [Mb]'.format(maxmem_plotter))

    
plt.scatter(-1,maxmem_main,c='white',label='max(main)={} [Mb]'.format(maxmem_main))
plt.scatter(-1,maxmem_reader,c='white',label='max(reader)={} [Mb]'.format(maxmem_reader))    
plt.scatter(-6,main_leak,c='red',label=f'main leak = {main_leak:.3f} [Mb]')
plt.scatter(-6,reader_leak,c='blue',label=f'reader leak = {reader_leak:.3f} [Mb]')
if(nthreads>2):
    plt.scatter(-6,plotter_leak,c='green',label=f'plotter leak = {plotter_leak:.3f} [Mb]')

plt.ylabel("memory usage [Mb]")
plt.xlabel("sampling time [s]")
plt.xlim(-1, xaxis[len(xaxis)-1]+2)
plt.title(picname)
#plt.legend(loc='upper left')
plt.legend(fontsize=14)
#plt.ylim(120,280)
plt.ylim(0,maxy*1.5)
plt.grid(True)
plt.show()
plt.savefig(fullPicName)




