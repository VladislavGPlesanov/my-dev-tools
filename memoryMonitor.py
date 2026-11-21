import sys
import time
import psutil
import os
import subprocess
from datetime import datetime as dt
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np

#g_outputfile = ""


def plotLastData(fname):
    # did'n give a shite to waste time so vibecoded this func.

    with open(fname, "r") as f:
        lines = f.readlines()

    # ----------------------------------------
    # determine number of processes from header
    # ----------------------------------------
    header = lines[0].strip()

    if not header.startswith("procs:"):
        raise ValueError("Invalid memlog header")

    proc_ids = [
        p for p in header.replace("procs:", "").split(",")
        if p.strip()
    ]

    nproc = len(proc_ids)

    print(f"Detected {nproc} processes")

    # ----------------------------------------
    # storage
    # ----------------------------------------
    memdata = [[] for _ in range(nproc)]
    deltadata = [[] for _ in range(nproc)]

    expected_cols = 2 * nproc

    # ----------------------------------------
    # parse body
    # ----------------------------------------
    for line in lines[1:]:

        line = line.strip()

        if not line:
            continue

        vals = [v.strip() for v in line.split(",") if v.strip()]

        # stop once one process disappeared
        if len(vals) < expected_cols:
            print("Detected process exit -> stopping parse")
            break

        try:

            for iproc in range(nproc):

                mem_idx = 2 * iproc
                delta_idx = mem_idx + 1

                memval = float(vals[mem_idx])
                deltaval = float(vals[delta_idx])

                memdata[iproc].append(memval)
                deltadata[iproc].append(deltaval)

        except ValueError:
            continue

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True
    )

    ax_mem = axes[0]
    ax_delta = axes[1]

    nsamples = len(memdata[0])

    x = np.arange(nsamples)

    # ----------------------------------------
    # absolute memory
    # ----------------------------------------
    for iproc in range(nproc):

        ax_mem.plot(
            x,
            memdata[iproc],
            label=f"PID {proc_ids[iproc]}"
        )

    ax_mem.set_ylabel("Memory [MB]")
    ax_mem.set_title("Absolute memory usage")

    ax_mem.grid(True)
    ax_mem.legend()

    # ----------------------------------------
    # delta memory as bipolar bars
    # ----------------------------------------
    barwidth = 0.8 / nproc

    for iproc in range(nproc):

        xshift = x + iproc * barwidth

        ax_delta.bar(
            xshift,
            deltadata[iproc],
            width=barwidth,
            label=f"PID {proc_ids[iproc]}"
        )

    ax_delta.axhline(0, linestyle='--')

    ax_delta.set_xlabel("Sample")
    ax_delta.set_ylabel("Δ Memory [MB]")

    ax_delta.set_title(
        "Memory increase/decrease per sample"
    )

    ax_delta.grid(True) 
    ax_delta.legend()

    plotname = fname[:-4]
    plt.savefig(f"{plotname}.png")
    plt.close()

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
    #g_outputfile += fname
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
                return fname

if __name__ == '__main__':

    parser = ap.ArgumentParser()    
    parser.add_argument('-n','--name', type=str, default="")
    parser.add_argument('-f', '--freq', type=float, default=0.5)
    parser.add_argument('--plot',action='store_true')
    args = parser.parse_args()

    fname = monitor(args.name, args.freq)
    if(args.plot):
        print("Plotting obtained data...")
        plotLastData(fname)

    sys.exit(0)


    

