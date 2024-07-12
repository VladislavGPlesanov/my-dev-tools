import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

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

main_index = 0
reader_index = 2

cnt=0
for line in f:
    words = line.split(',')
    clean_word0 = None
    clean_word2 = None
    if(len(words)==4):
        clean_word0 = words[main_index][:-4]
        clean_word2 = words[reader_index][:-4]
        mem_main.append(float(clean_word0))    
        mem_reader.append(float(clean_word2))    
    if(cnt<40):
        print("len={} : {} ".format(len(words),words))
        #print(clean_word0) 
  
    #mem_main.append(float(clean_word0))    
    #mem_reader.append(float(clean_word2))    

    cnt+=1


maxmem_reader = max(mem_reader)
avgmem_reader = sum(mem_reader)/len(mem_reader)
minmem_reader = min(mem_reader)

maxmem_main = max(mem_main)
avgmem_main = sum(mem_main)/len(mem_main)
minmem_main = min(mem_main)

reader_leak = max(mem_reader[5:len(mem_reader)-5]) - min(mem_reader[5:len(mem_reader)-5])

####### plotting 
#
plt.figure(0,figsize=(10,10))
plt.plot(mem_main,c='red',label='main_thread')
plt.plot(mem_reader,c='blue',label='reader_thread')
#plt.plot(maxmem_main,label='max(main)')
plt.scatter(-1,maxmem_main,c='white',label='max(main)={} [Mb]'.format(maxmem_main))
plt.scatter(-1,maxmem_reader,c='white',label='max(reader)={} [Mb]'.format(maxmem_reader))
plt.scatter(-1,reader_leak,c='white',label=f'reader leak in ~60s = {reader_leak:.3f} [Mb]')
plt.ylabel("memory usage [Mb]")
plt.xlabel("sampling itreation")
plt.title(picname)
#plt.text(0.1,0.8,f"reader_max={maxmem_reader}", fontsize=12)
#plt.text(0.1,0.75,f"main_max={maxmem_main}", fontsize=12)
plt.legend(loc='upper left')
#plt.ylim(120,280)
plt.ylim(0,400)
plt.grid(True)
plt.savefig(fullPicName)




