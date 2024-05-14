import sys
import time
import psutil
import os
import subprocess

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

def monitor(pid, freq=0.5):

    prev_mem = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    while(True):
        try:
            cnt = 0
            proclist = findProc("tpx3_gui")
            outstr = ""

            for proc in proclist:
                #mem = proc.memory_info().rss / 1e6
                mem = psutil.Process(proc).memory_info().rss / 1e6
                if(prev_mem is None):
                    tempstr = "[{}] {:10.3f} [Mb]".format(proc, mem)
                    if(cnt!=0):
                        tempstr = "\t" + tempstr
                    outstr+=tempstr    
                else:
                    tempstr = "[{}] {:10.3f} [Mb] {:+10.3f}[Mb]".format(proc, mem, mem - prev_mem[cnt])
                    if(cnt!=0):
                        tempstr = "\t" + tempstr
                    outstr+=tempstr    
                prev_mem[cnt] = mem
                cnt+=1

            print(outstr)
            
            time.sleep(freq)
        except KeyboardInterrupt:
            try:
                input('pausing monitor, <cr> to continue, ^C to end..')
            except KeyboardInterrupt:
                print('\n')
                return

if __name__ == '__main__':
    if(len(sys.argv)<2):
        print("usege: python3 <thisexe.py> <PID>")
        sys.exit(1)
    pid = int(sys.argv[1])
    monitor(pid)
    sys.exit(0)


