import sys
import time
import psutil
import os
import subprocess
from datetime import datetime as dt
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np

def findProc(name):
    
    pidlist = []
    out = subprocess.check_output(["pgrep",name])
    separatePIDs = out.split()
    for pid in separatePIDs:
        yoba = int(pid)
        pidlist.append(yoba)
            
    return pidlist

def monitor(procname, freq=0.5, fplot=False):

    prev_mem = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    date = dt.now()
    myd = str(date.year)+"-"+str(date.month)+"-"+str(date.day)+"_"+str(date.hour)+"-"+str(date.minute)+"-"+str(date.second)
    print(date)
    fname = "memusg-"+myd+".txt"
    ofile = open(fname,"a")
    timeunit = freq

    ofhead = "procs:"
    for p in findProc(procname):
        ofhead+=f"{p},"
    ofhead+="\n"
    ofile.write(ofhead)
    ofhead = None

    while(True):
        try:
            cnt = 0
            proclist = findProc(procname)
            outstr = ""
            filestring = ""

            for proc in proclist:
                mem = psutil.Process(proc).memory_info().rss / 1e6
                if(prev_mem is None):
                    tempstr = "[{}] {:10.3f}[Mb]".format(proc, mem)
                    tempfilestr = "{:10.3f}".format(mem)

                    if(cnt!=0):
                        tempstr = "\t" + tempstr
                        tempfilestr = ", " + tempfilestr
                    outstr+=tempstr
                    filestring += tempfilestr
                else:
                    tempstr = "[{}] {:10.3f}[Mb] {:+10.3f}[Mb]".format(proc, mem, mem - prev_mem[cnt])
                    tempfilestr = "{:10.3f}, {:+10.3f}".format(mem, mem - prev_mem[cnt])
                    if(cnt!=0):
                        tempstr = "\t" + tempstr
                        tempfilestr = ", " + tempfilestr

                    outstr+=tempstr    
                    filestring += tempfilestr

                prev_mem[cnt] = mem
                cnt+=1
        
            print(outstr)

            filestring+="\n"            

            time.sleep(freq)

            ofile.write(filestring)
        except KeyboardInterrupt:
            try:
                input('pausing monitor, <cr> to continue, ^C to end..')
            except KeyboardInterrupt:
                print('\n')
                ofile.close()
                return

if __name__ == '__main__':

    parser = ap.ArgumentParser()    
    parser.add_argument('-n','--name', type=str, default="")
    parser.add_argument('-f', '--freq', type=float, default=0.5)
    args = parser.parse_args()

    monitor(args.name, args.freq)
    sys.exit(0)


    

