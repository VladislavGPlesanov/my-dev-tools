import sys
import time
import psutil
import os
import subprocess
from datetime import datetime as dt


def findproc(name):

    pidlist = []
    out = subprocess.check_output(["pgrep",name])
    separatePIDs = out.split()
    for pid in separatePIDs:
        yoba = int(pid)
        pidlist.append(yoba)
            
    return pidlist

def makefilename():

    date = dt.now()
    myd = str(date.year)+"-"+str(date.month)+"-"+str(date.day)+"_"+str(date.hour)+"-"+str(date.minute)+"-"+str(date.second)
    print(date)
    fname = "cpuusg-"+myd+".txt"

    return fname

def monitorCPU(freq, scriptname, write):

    ofile = None
    outstr = ""
    line = ""

    while True:
        try:
            proclist = None
            cnt=0
            ###############################################33
            try:
               proclist = findproc(str(scriptname))
            except subprocess.CalledProcessError:
               try:
                   input("CAN NOT FIND {}".format(scriptname))
                   proclist = findproc(str(scriptname))
               except KeyboardInterrupt:
                   #if ofile is not None:
                   #     ofile.close()
                   return

            ###############################################
            #print("\r{}".format(proclist))
            outstr=""
            for proc in proclist:
                usg_cpu = None
                try:
                    if(psutil.pid_exists(proc)):
                        usg_cpu = psutil.Process(proc).cpu_percent(interval=freq)
                    else:
                        print(f"Process {proc} was closed\n")
                        usg_cpu = 0.0
                except FileNotFoundError or psutil.NoSuchProcess:
                    pass
                finally:
                    usg_cpu = 0.0

                tempstr = "[{}], {:.2f}\%, ".format(proc,usg_cpu)
                if cnt!=0:
                    tempstr = ",\t" + tempstr
                outstr += tempstr

            print("\r{}".format(outstr), end=" ",flush=True)
            cnt+=1

            time.sleep(freq)
        except KeyboardInterrupt:
            try:
                input("ON PAUSE, press ENTER to CONTINUE, or ctrl+c to STOP")
            except KeyboardInterrupt:
                print('\n')
                return

if __name__ == '__main__':
    print("Got {} args".format(len(sys.argv)))
    if len(sys.argv)<=3:
        print("missing argument:\n use python3 cpuMon.py <procname> <time interval> <write flag>")

    process = str(sys.argv[1])
    time_interval = float(sys.argv[2])
    writing_flag = sys.argv[3].lower() == True

    print("Running with {}, {}, {}".format(process, time_interval, writing_flag))

    monitorCPU(time_interval, process, writing_flag)

    sys.exit(0)





