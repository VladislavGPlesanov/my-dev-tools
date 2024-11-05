import sys
import time
import psutil
import os
import subprocess
from datetime import datetime as dt


def findProc(name):
    
    pidlist = []
    out = subprocess.check_output(["pgrep",name])
    #print("[1]=> {}".format(out))
    separatePIDs = out.split()
    #print("[2]=> {}".format(separatePIDs))
    for pid in separatePIDs:
        yoba = int(pid)
        pidlist.append(yoba)
            
    return pidlist

#def monitor(pid, freq=0.5, procname):
def monitor(procname, freq=0.5):

    prev_mem = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    date = dt.now()
    myd = str(date.year)+"-"+str(date.month)+"-"+str(date.day)+"_"+str(date.hour)+"-"+str(date.minute)+"-"+str(date.second)
    print(date)
    fname = "memusg-"+myd+".txt"
    ofile = open(fname,"a")

    while(True):
        try:
            cnt = 0
            #proclist = findProc("tpx3_gui")
            proclist = findProc(procname)
            #proclist = findProc("tpx3_cli")
            outstr = ""
            filestring = ""

            for proc in proclist:
                #mem = proc.memory_info().rss / 1e6
                mem = psutil.Process(proc).memory_info().rss / 1e6
                if(prev_mem is None):
                    tempstr = "[{}] {:10.3f}[Mb]".format(proc, mem)
                    tempfilestr = "{:10.3f}[Mb]".format(mem)

                    if(cnt!=0):
                        tempstr = "\t" + tempstr
                        tempfilestr = ", " + tempfilestr
                    outstr+=tempstr
                    filestring += tempfilestr
                else:
                    tempstr = "[{}] {:10.3f}[Mb] {:+10.3f}[Mb]".format(proc, mem, mem - prev_mem[cnt])
                    tempfilestr = "{:10.3f}[Mb], {:+10.3f}[Mb]".format(mem, mem - prev_mem[cnt])
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
    if(len(sys.argv)<2):
        print("usege: python3 <thisexe.py> <PID>")
        sys.exit(1)
    #pid = int(sys.argv[1])
    procname = sys.argv[1] 
    monitor(procname)
    #monitor(pid, procname)
    sys.exit(0)


