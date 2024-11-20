import numpy as np
import sys
import matplotlib.pyplot as plt
import subprocess as sp
import argparse
import glob
import os

###################################################################
parser = argparse.ArgumentParser()
#parser.add_argument('--wfoil', action='store_true')
parser.add_argument('-wf', '--wfoil', type=str, default='home/ebala/')
parser.add_argument('-f', '--files',nargs='+')
parser.add_argument('-p', '--picname', type=str)
args = parser.parse_args()

files = []
for patt in args.files:
    files.extend(glob.glob(patt))

plotname = args.picname
#wfoil = args.wfoil
wfoil = False
foilpath = args.wfoil
if not os.path.exists(foilpath):
    wfoil = False
    print(f"{foilpath} is not valid path - using wfoil={wfoil}")
else:
    wfoil = True
    print(f"using foil data: {foilpath}")

print("Using argparse:")
print(f"arg1-{plotname}\narg2-{wfoil}\narg3-{files}")

##############################################################
#plotname = sys.argv[1]
#wfoil = sys.argv[2]
#files = sys.argv[3:]

###############################################################
print(f"Account for foil? -> {wfoil}")
foilEfficiency = []
foilEgamma = []
if(wfoil):
 
    words = foilpath.split('/')
    foilfilename = words[len(words)-1]
    foilType = foilfilename.split()[0]   
    #foilpath = "/home/vlad/readoutSW/tools/GasTransmissions/Foils/mylarFoil-500um.dat"
    ff = open(foilpath)
    iline = 0
    for line in ff:
        #print(line)
        if iline < 2:
            iline+=1 
            continue
        
        entries = line.split()         
        #foilEfficiency.append(1.0-float(entries[1]))
        foilEfficiency.append(float(entries[1]))
        foilEgamma.append(float(entries[0]))

        iline+=1

    plt.figure(figsize=(16,9))
    plt.scatter(foilEgamma,foilEfficiency,marker='+',color='red')
    plt.xlabel("photon energy [eV]")
    plt.ylabel("Absorption [frac]")
    plt.savefig(f"foilEfficiency-{foilType}.png")

foilEgamma = None
###############################################################
print(f"looking for files {files}")

plt.figure(figsize=(16,9))

ifile = 0

coeffList = [] 

for file in files:

    x, y = [],[]

    xname, yname = None, None
    ilegend = None 
    best_abs, worst_abs = None, None

    print(f"=== checking file {file} ===")
    cnt=0
    f = open(file)
    for line in f:

        #if(cnt < 3 ):
        if(cnt < 2 ):
            if(cnt==0):
                ilegend = line[:-1]
                #print(ilegend)
            if(cnt==1 and xname is None):
                #print(line)
                words = line.split()
                #print(words)
                xname = f"{words[0]} {words[1]} {words[2][:-1]}"
                yname = words[3]
                
            cnt+=1
            continue
        else:
            #print(f"file={ifile}, line={cnt}, text-> {line}")
            words = line.split()
            #print(f"words={words}")
            Egamma = float(words[0])
            Trans = float(words[1])

            epsilon = 1.0 - Trans
            
            if(wfoil):
                y.append(epsilon*foilEfficiency[len(x)])
            else:
                y.append(epsilon)
            x.append(Egamma)

            #if(Egamma==7000.0):
            if(Egamma>6800.0 and Egamma<7200.0 and best_abs is None):
                best_abs = round(1.0 - Trans,2)
                #print("--------------------------------")
                #print(f"best case absorption={best_abs}")
            #if(Egamma==8000.0):
            if(Egamma>7800.0 and Egamma<8200.0 and worst_abs is None):
                worst_abs = round(1.0 - Trans,2)
                #print(f"worst case absorption={worst_abs}")
                #print("--------------------------------")
            #print("len(x)={len(x)}, len(y)={len(y)}")
           
        cnt+=1 
    #if("ArCO2" in ilegend):
    #    plt.scatter(x,y,marker='+',label=ilegend)
    #    #plt.plot(x,y,label=ilegend)
    #elif("ArCH4" in ilegend):
    #    plt.scatter(x,y,marker='o',label=ilegend)
    #    #plt.plot(x,y,label=ilegend)
    #elif("ArC4H10" in ilegend):
    #    plt.scatter(x,y,marker='v',label=ilegend)
    #    #plt.plot(x,y,label=ilegend)
    #elif("SF6" in ilegend):
    #    plt.scatter(x,y,marker='*',label=ilegend)
    #    #plt.plot(x,y,label=ilegend)
    #elif("Ne" in ilegend):
    #    plt.scatter(x,y,marker='s',label=ilegend)
    #    #plt.plot(x,y,label=ilegend)
    #elif("Kr" in ilegend):
    #    plt.scatter(x,y,marker='d',label=ilegend)
    #elif("Xe" in ilegend):
    #    plt.scatter(x,y,marker='1',label=ilegend)
    #else:
    #    plt.scatter(x,y,marker='x',label=ilegend)
        #plt.plot(x,y,label=ilegend)
    # setting markers based on pressure
    #if("Pressure=760" in ilegend):
    #    plt.scatter(x,y,marker='+',label=ilegend)
    #elif("Pressure=1140" in ilegend):
    #    plt.scatter(x,y,marker='x',label=ilegend)
    #else:
    #    plt.scatter(x,y,marker='v',label=ilegend)
    #
    # for foils and other media
    #
    if("Fe" in ilegend):
        plt.scatter(x,y,marker='+',label=ilegend)
    elif("Al" in ilegend):
        plt.scatter(x,y,marker='x',label=ilegend)
    elif("Cu" in ilegend):
        plt.scatter(x,y,marker='^',label=ilegend)
    elif("C10" in ilegend):
        plt.scatter(x,y,marker='v',label=ilegend)
    elif("Si3N4" in ilegend):
        plt.scatter(x,y,marker='*',label=ilegend)
    else:
        plt.scatter(x,y,marker='s',label=ilegend)


    coeffList.append([ilegend, worst_abs, best_abs])

    cnt=0
    if(ifile == 0):
        print("set plot axis labels")
        plt.xlabel(xname)
        plt.ylabel(yname)
    print(f"=== finished with file {file} ===")
    ifile+=1 

plt.legend()
if("1k-10k" in files[0]):
    plt.xlim(1005,10005)
    plt.ylim(-0.05,1.05)
else:
    plt.xlim(6400,9300)
    plt.ylim(-0.05,1.05)
#if("crazy" in plotname):
#    plt.ylim(0.45,1.05)
plt.title("EBALA!")
#plt.grid(True, which='major', linestyle='-')
plt.grid(which='major', linestyle='-')
#plt.grid(True, which='minor', linestyle='--')
plt.grid(which='minor', linestyle='--')
plt.minorticks_on()
plt.vlines(7000,0,1,linestyles='dashed')
plt.vlines(8000,0,1,linestyles='dashed')
if(wfoil):
    plt.savefig(plotname+"-transmission-wFoil.png")
else:
    plt.savefig(plotname+"-transmission.png")

for gasres in coeffList:
    print(f"{gasres[0]} - {gasres[1]} -> {gasres[2]} mean={round((gasres[1]+gasres[2])/2,2)}")

# ------------------------------------------
# correcting for the absorption of the foil
# ------------------------------------------



