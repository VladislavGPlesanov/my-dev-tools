#import pydoc

import sys
import os
import argparse
import h5py
import numpy as np
import copy
from matplotlib import colors, cm 
#import pandas as pd
import tables as tb
from time import sleep
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt 
from collections import Counter
import statistics as stat
#from scipy import optimize as opt
from sklearn.cluster import DBSCAN

##################################################################
import sys
sys.path.insert(0,'/home/vlad/readoutSW/tpx3-daq-fifo-readout-cpp')
import tpx3.analysis as analysis

"""
   Code loops through the HDF files in a quick and simple way.
   Params(input converted to string by default):
      filename: name of the file
      option: name of the branch/leaf
   Returns(for each option):
      mask: 2d hist
      thr: 2d hist
      links: link status of RX links int 0 to 7
      config: local chip config
      genconfig: general run configuration
      dacs: DAC settings
      mdata: meta_data 5 lines with a graph of discard and decode errors per chunk
      rdata: raw_data length
"""

TITLE_COLOR = '#07529a'
GREEK_MU = '\u03BC'
GREEK_SIGMA = '\u03C3'
GREEK_CHI = '\u03C7'
GREEK_DELTA = '\u0394'

#os.environ["QT_QPA_PLATFORM"] = "wayland"

def testIfHasInterpreted(file):
    if '/interpreted/' not in f:
        print("NAW INTERPRETED DATA MATE!")
        return False
    else:
        #file.list_nodes('/interpreted/')
        print("YISS, its here!")
        return True

def testIfHasOccupancy(file):
    if('/HistOcc/' not in file):
        print("Occupancy plots not included in this file")
        return False
    else:
        return True

def hasInterpretedBranch(file, branch):
    bname = '/interpreted/'+str(branch)+'/' 
    if(bname in file):
        return True
    else:
        return False

#def checkNearPix(seed_x,seed_y,meanTOT):
#    #
#    # get matrix mean TOT and use it as starting par
#    # 256x256 mat
#    # 
#    
#    
#
#    return np.array(neighbors)
#    
#
#def checkXnoise(xarray):
#    #
#    # how to check for hits that can be a cluster but are sequences of pixel in a row/column?
#    #
#
#    
#
#

def countNodes(file, opt, dgroup):

    cnt = 0 
    node = None

    optlist = {
               "dac":"dac", 
               "run_config":"run_config",
               "thr_map":"ThresholdMap",
               "scurves":"HistSCurve",
               "mu2D":"mu2D"
               }

    if(opt in optlist.keys()):
        node = optlist[opt]
        print("found option {} for key {} -> {}".format(optlist[opt],opt, type(optlist[opt])))

    else:
        print("No option for node {}".format(opt))
        return -1

    group = None

    if(dgroup == "intp"):
        group = file.root.interpreted
    elif(dgroup == "cfg"):
        group = file.root.configuration
    else:
        print(f"countNodes: no option for group {dgroup}")
        return -1

    for br in group:
        if(node in str(br)):
            print("HUYA-{}".format(br))
            cnt+=1
    return cnt    


def getDictList(file,n_nodes,node):

    dictList = []
    keys = []
    for i in range(n_nodes):
        nodename = f"/configuration/{node}_{i}"
        nd = dict(f.get_node(nodename)[:])
        dictList.append(nd)
        if(i==0):
            keys = list(nd.keys())

    return dictList, keys


def compareRegs(reglist, keylist):

    reglistlen = len(reglist)
    print("found keys: {}".format(keylist))

    reg_difs = 0
    diff_keys = []
    for key in keylist:
        if ("maskfile" in str(key)):
            continue
        parlist = []
        for j in range(reglistlen):
            parlist.append(reglist[j][key])
        if(len(set(parlist))>1):
            print("key={}, parlist={}.".format(key,parlist))
            reg_difs += 1
            diff_keys.append(key)
         
    if(reg_difs == 0):
        print("++++++++++++++++++++++++++++++")
        print("all configs are identical!")
        print("++++++++++++++++++++++++++++++")
    else:
        print("++++++++++++++++++++++++++++++")
        print("Found differences in {}".format(diff_keys))
        print("++++++++++++++++++++++++++++++")

def countInterpretedBranches(file, branch):

    cnt = 0
    nbranches = 0 
    for b in file.root.interpreted:
        print(b)
        if(branch in str(b)):
            print("HUYA!")
            cnt+=1

    return cnt

def removePath(string):
    string_split = string.split('/')
    clean_name = string_split[len(string_split)-1]
    clean_name = clean_name[:-3]
    return clean_name

def getRunType(string):

    splitSlash = string.split('/')
    run_file = splitSlash[len(splitSlash)-1].split('_')
    return run_file[0] 

def getDerivativeNonZero(xlist, ylist):
    if(len(xlist) != len(ylist)):
        print("x and y lists are different lengths {} and {}".format(len(xlist), len(ylist)))
        return -1

    posDiscard = True
    if(max(ylist) == 0 and min(ylist) < 0):
        posDiscard = False
        print("discard errors are positive numbers")

    plato_high = None
    kink = None

    maxrange = len(xlist)//4*4
    print("usable range = 0->{}".format(maxrange))

    if(posDiscard):
        clist = ['r','g','b','y','m']
        for i in range(0,maxrange):
            print("y={} @ i=[{}]".format(ylist[i],i))
            if(ylist[i]>0):
                print("Found slope at i=[{}]".format(i))
                kink = i        
                break

        # looking for high disc err plato
        print("tryna' find start of plato ") 
        for i in reversed(range(0,len(ylist))):
            print("y={} @ i=[{}]".format(ylist[i],i))             
            if(ylist[i]<2000):
                print("found plato at i={}".format(plato_high))
                plato_high = i+1
                break
                
    if(plato_high == None):
        print("Could not find the plato")
        plato_high = 0

    if(kink == None):
        print("Could not find the place where parameters deviate")
        kink = 0

    return kink, plato_high

def getCoordFromNumber(genpos):
    # based on the routine of assembly of s curve numbering
    # as n = x*256 + y
    # basically, reversing it...
    nrows = 256
    x = genpos // nrows
    column = x*nrows
    y = genpos - column
    return x,y

def getFirstErrorPos(scanIDList, discList, option):
    
    if(len(scanIDList) != len(discList)):
        print("getFirstErrorPos::List sizes are incompatible!")
    else:
        print("getFirstErrorPos::List sizes compatible")
    step = 40
    firstpos = 0
    lastpos = 0
    for i in range(0,len(discList),step):
        mean = sum(discList[i:i+step])/step
        if(option=="disc"):
            if(mean<-100):
                firstpos = i
                lastpos = i+step
                break
        else:
            if(mean>100):
                firstpos = i
                lastpos = i+step
                break

    print("Found range = {} to {} ({} DAC to {} DAC)".format(firstpos,lastpos, scanIDList[firstpos],scanIDList[lastpos]))
    thispos = 0
    checkval = 50
    if(option=="disc"):
        checkval = -50

    for i in range(firstpos,lastpos,1): 
        print("position range:[{}] of {}-{}".format(i,firstpos,lastpos))
        diff = discList[i-1]-discList[i]
        print("difference: ({}) - ({}) = ({})".format(discList[i-1], discList[i], diff))
        if(diff<checkval and option=="disc"):
            thispos = i
            break
        if(diff>checkval and option!="disc"):
            thispos = i
            break
    if(thispos==0):
        thispos = firstpos

    return thispos

def getUniqueList(inputlist):
    unique = list(set(inputlist))
    return sorted(unique)


def getAvgList(unique_list, data_list):

    # data_list  must be at list of lists, with len(val) >= 2  
    # where val[0] is data to sort and val[1] the unique ids

    avg_list = []
    total_list = []
    accum = 0
    nentries = 0
   
    cntr = 0  
    for unq in unique_list:
        for val in data_list:
            cntr+=1
            #if(val[0]!=0 and cntr % 100==0):            
            #    print("val[0]={}, unq={}".format(val[0],unq),flush=True)
            if(val[1] == unq and val[0]>0):            
               accum += val[0] 
               nentries += 1
            else:
               continue
        avg = 0
        total_list.append(accum)
        try:
            avg = float(accum)/float(nentries)
            
        except ZeroDivisionError:
            print("[ERROR] tried to divide {} by {} for id={}".format(accum, nentries, unq))
            avg = 0

        #print("avg = {}, accum = {}, nentries = {}".format(avg, accum, nentries))
        avg_list.append(avg)

        nentries = 0
        accum = 0

    return avg_list, total_list


def fillArray(inputData, data_index):

    dataList = []

    pos = 0
    for i in range(0,len(inputData)):
       for j in range(len(inputData[i])):
           if(j == data_index):
              dataList.append(inputData[i][j])
              pos+=1
           else:
              continue
    numArray = np.asarray(dataList)
    return numArray

def simpleHist(data, nbins, minbin, maxbin, labels, picname, ylog):

    plt.figure()

    plt.hist(data, nbins, range=(minbin,maxbin), histtype='step', facecolor='b')

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])

    if(ylog):
        plt.yscale('log')

    plt.savefig("simpleHist-"+picname+".png")


def quickPlot(data, axis, picname):

    plt.figure()

    plt.plot(data, "bo")
    plt.title(axis[0])
    plt.xlabel(axis[1])
    plt.ylabel(axis[2])

    plt.savefig("quick_plot-"+picname+".png")

def plot_masked_scat(xdata,ydata,criterion,title,plotname,label,axisnames):

    print("assemblying plot for - [{}]".format(title)) 
 
    #if(type(xdata) is not numpy.array):
    #    print("masked data works on np.arrays")

    mask_below, mask_above = None, None
    mask_below = (xdata < criterion)
    mask_above = (xdata > criterion)
    #mask_between = ((ydata > 100) & (ydata < 300))
    #mask_special = ((xdata > criterion) & (ydata > 200))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(xdata[mask_below],ydata[mask_below], s=10,c='b',marker="s",label=f"below {criterion} {len(xdata[mask_below])}")
    ax1.scatter(xdata[mask_above],ydata[mask_above], s=10,c='g',marker="s",label=f"above {criterion} {len(xdata[mask_above])}")
    #ax1.scatter(xdata[mask_special],ydata[mask_special], s=10,c='y',marker="s",label=f"special {len(xdata[mask_special])}")
    #ax1.scatter(xdata[mask_between],ydata[mask_between], s=10,c='k',marker="s",label=f"between {len(xdata[mask_between])}")
    plt.legend()
    if(max(ydata)>5000):
       ax1.set_yscale("log")
    if max(xdata) > 5000:
       ax1.set_xscale("log")
    fig.suptitle(title)
    ax1.set_xlabel(axisnames[0])
    ax1.set_ylabel(axisnames[1])
    if("Data Length"in title):
       ax1.set_ylim(min(ydata)*0.9,max(ydata)*1.1)
    elif("timestamp" in title and "iteration" not in title):
       ax1.set_xlim(min(xdata[1:len(xdata)]),max(xdata))
    elif("iteration" and "timestamp" in title):
       avgy = sum(ydata)/len(ydata)
       ax1.set_ylim(min(ydata[1:len(ydata)])*0.999,max(ydata)*1.001)
       ax1.set_yscale('linear')
       ax1.set_xscale('linear')
    else: 
       ax1.set_ylim(min(ydata),max(ydata))
    if("sigma" in title[2]):
       ax1.set_ylim(min(ydata)-1,max(ydata)+1)
    if("sigma" in title[1]):
       ax1.set_xlim(min(xdata)-1,max(xdata)+1)
    ax1.grid(True)
    ax1.plot()
    fig.savefig(plotname+"-masked.png")



def plot_scat(xdata, ydata, title, plotname, label, axisnames):
 
    print("assemblying plot for - [{}]".format(title)) 
 
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(xdata,ydata, s=10,c='b',marker="s",label=label)
    plt.legend()
    if(max(ydata)>5000):
       ax1.set_yscale("log")
    if max(xdata) > 5000:
       ax1.set_xscale("log")
    fig.suptitle(title)
    ax1.set_xlabel(axisnames[0])
    ax1.set_ylabel(axisnames[1])
    if("Data Length"in title):
       ax1.set_ylim(min(ydata)*0.9,max(ydata)*1.1)
    elif("timestamp" in title and "iteration" not in title):
       ax1.set_xlim(min(xdata[1:len(xdata)]),max(xdata))
    elif("iteration" and "timestamp" in title):
       avgy = sum(ydata)/len(ydata)
       ax1.set_ylim(min(ydata[1:len(ydata)])*0.999,max(ydata)*1.001)
       ax1.set_yscale('linear')
       ax1.set_xscale('linear')
    else: 
       ax1.set_ylim(min(ydata)*0.8,max(ydata)*1.2)
    ax1.grid(True)
    ax1.plot()
    fig.savefig(plotname+'.png')

def fitPolyFunc(xdata,ydata,power):

    fitfunc = np.poly1d(np.polyfit(xdata, ydata ,power))

    print("Fit returns: {}".format(fitfunc))

    return fitfunc


def plot_scat_stack(xdata, ydatalist, plotname, lablist, axisnames):

    nUniqueDACs = set(xdata)

    if(len(ydatalist)<6):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        cnt = 0
        clist = ['r','g','b','y','m']
        clist_rate = [(0.0,0.0,1.0),(0.12,0.8,0.0)]
        markList = ['+','x','o','v','D']
        for ydata in ydatalist:
           if("-rate-vs-discardErr" in plotname):
               ax1.scatter(xdata,ydata, s=10, c=clist_rate[cnt],marker=markList[cnt],label=lablist[cnt])
           else:
               ax1.scatter(xdata,ydata, s=10, c=clist[cnt],marker=markList[cnt],label=lablist[cnt])
        
           cnt+=1
        if max(ydatalist[0]) > 5000:
            ax1.set_yscale("log")
        if max(xdata) > 5000:
            ax1.set_xscale("log")

        if("-rate-vs-discardErr" in plotname):
            kink, plato = getDerivativeNonZero(xdata, ydatalist[1])       
            ax1.axvline(xdata[kink], color='c',linestyle='--')
            ax1.scatter([],[],marker='',label=f'start @ {xdata[kink]}[DAC]')
            #ax1.axvline(xdata[plato], color='y',linestyle='--')
            #fitline = fitPolyFunc(xdata[kink:plato], ydatalist[1][kink:plato],2)
            #ax1.plot(xdata[kink:plato], fitline(xdata[kink:plato]), color='r')


        ax1.set_xlabel(axisnames[0])
        ax1.set_ylabel(axisnames[1])
        ax1.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
        ax1.grid(which='minor', color='grey', linestyle='-', linewidth=0.125)
        ax1.minorticks_on()

        plt.legend(loc='upper left')
        ax1.plot()
        fig.savefig(plotname+'.png', dpi=300)
    else:
        print("plot_scatstack:Can not stack more than 5 plots together!")

def ploth2d(xdata, ydata, plotname):

    xmin = xdata.min()
    xmax = xdata.max()
    ymin = ydata.min()
    ymax = ydata.max()
  
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.hist2d(xdata, 
               ydata, 
               range= np.array([(xmin,xmax),(xmin,xmax)]),
               bins=100)
               #cmap="Greens")
    ax1.plot()
    fig.savefig(plotname+".png")

# grand theft code from tpx3/plotting.py
def plot_scurves(scurves, scan_parameters, scan_parameter_name=None, title='S-curves', ylabel='Occupancy', max_occ=None, plotname=None):

    if max_occ is None:
        max_occ = np.max(scurves) + 5

    x_bins = np.arange(min(scan_parameters) - 1, max(scan_parameters) + 2)
    y_bins = np.arange(-0.5, max_occ + 0.5)

    param_count = scurves.shape[0]
    hist = np.empty([param_count, max_occ], dtype=np.uint32)

    scurves[scurves>=max_occ] = max_occ-1

    for param in range(param_count):
        hist[param] = np.bincount(scurves[param, :], minlength=max_occ)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    fig.patch.set_facecolor('white')
    cmap = copy.copy(cm.get_cmap('cool'))
    if np.allclose(hist, 0.0) or hist.max() <= 1:
        z_max = 1.0
    else:
        z_max = hist.max()
    # for small z or if coosen use linear scale, otherwise log scale
    if z_max <= 10.0: #or non_log:
        bounds = np.linspace(start=0.0, stop=z_max, num=255, endpoint=True)
        norm = colors.BoundaryNorm(bounds, cmap.N)
    else:
        bounds = np.linspace(start=1.0, stop=z_max, num=255, endpoint=True)
        norm = colors.LogNorm()

    im = ax.pcolormesh(x_bins, y_bins, hist.T, norm=norm, rasterized=True)

    if z_max <= 10.0:
        cb = fig.colorbar(im, ticks=np.linspace(start=0.0, stop=z_max, num=min(
            11, math.ceil(z_max) + 1), endpoint=True), fraction=0.04, pad=0.05)
    else:
        cb = fig.colorbar(im, fraction=0.04, pad=0.05)
    cb.set_label("# of pixels")
    ax.set_title(title + ' for %d pixel(s)' % (scurves.shape[1]), color=TITLE_COLOR)
    if scan_parameter_name is None:
        ax.set_xlabel('Scan parameter')
    else:
        ax.set_xlabel(scan_parameter_name)
    ax.set_ylabel(ylabel)

    if(plotname is None):
        plt.savefig("scurves-ebala.png") 
    else:
        plt.savefig("scurve-"+plotname+".png") 


def plot_dist(data, fit=True, plot_range=None, x_axis_title=None, electron_axis=False, use_electron_offset=True, y_axis_title='# of hits', title=None, suffix=None):

   if plot_range is None:
       diff = np.amax(data) - np.amin(data)
       if (np.amax(data)) > np.median(data) * 5:
           plot_range = np.arange(
               np.amin(data), np.median(data) * 5, diff / 100.)
       else:
           plot_range = np.arange(np.amin(data), np.amax(
               data) + diff / 100., diff / 100.)

   tick_size = np.diff(plot_range)[0]

   if type(data) == np.ma.core.MaskedArray:
       hist, bins = np.histogram(data.compressed(), bins=plot_range)
   else:
       hist, bins = np.histogram(np.ravel(data), bins=plot_range)

   fig = plt.figure() 
   ax = fig.add_subplot(111)

   ax.bar(bins[:-1], hist, width=tick_size, align='edge')

   ax.set_xlim((min(plot_range), max(plot_range)))
   ax.set_title(title, color=TITLE_COLOR)
   if x_axis_title is not None:
       ax.set_xlabel(x_axis_title)
   if y_axis_title is not None:
       ax.set_ylabel(y_axis_title)
   ax.set_yscale('log')
   ax.grid(True)
   plt.savefig("distro-"+clean_filename+".png")
      
###########################################################

def plot_2dim(nuarray, parlist, title, axlabels, plotname):


    x_bins = np.arange(min(parlist) - 1, max(parlist) + 2)
    parcount = nuarray.shape[0]
    maxocc = 0
    

    for par in range(parcount):
        this_max = np.max(nuarray[par,:])
        if(this_max>maxocc):
            maxocc = this_max


    fig = plt.figure() 

    im = plt.imshow(nuarray)
    cb = fig.colorbar(im, fraction=0.01, pad=0.05)
    plt.title(title)
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1])
    plt.savefig("2dimshow-{}.png".format(plotname))

def getToaClusters(toalist, nskip):

    tmp = []
    clust = []
    cnt = 0 
    for i in toalist:
        #print(f'iter[{cnt}] - tmp={tmp} at beginning')
        if(len(tmp)==0):
            tmp.append(i)
            #print(f"iter[{cnt}] - list empty, adding {i}")
        elif(cnt==len(toalist)-1):
            #print(f"iter[{cnt}] - last number")
            if(i-tmp[len(tmp)-1]<nskip):
                tmp.append(i)
            clust.append(tmp)
            break
        else:
            if(len(tmp)==1):
                if(i-tmp[0]<nskip):
                    tmp.append(i)
                    #print(f"iter[{cnt}] - list len=1, i-tmp[0]{tmp[0]}<2 adding {i} to tmp={tmp}")
                else:
                    clust.append(tmp)
                    #print(f"iter[{cnt}] - list len=1, i-tmp[0]{tmp[0]}>=2 adding tmp to clusters, tmp=[]")
                    tmp=[]
            else:
                if(i-tmp[len(tmp)-1]<nskip):
                    tmp.append(i)
                    #print(f"iter[{cnt}] - list len>1, i-{tmp[len(tmp)-1]}<2 adding {i} to tmp={tmp}")
                else:
                    clust.append(tmp)
                    #print(f"iter[{cnt}] - list len>1, i-{tmp[len(tmp)-1]}>=2 adding tmp to clusters, tmp=[]")
                    tmp=[]
        #print(f"iter[{cnt}] - tmp={tmp} at THE END")
        cnt+=1 

    print("found TOA clusters:")
    for c in clust:
        print(c)
    
    return clust

#def clusterSpread()


def findNoisyPixels(nuArray):
   
    median = np.median(nuArray)
    mean = np.mean(nuArray)
    stdev = np.std(nuArray)
    
    thisTHR = median + stdev*5 
    altTHR = mean + stdev*5

    print("Looking for pixels with occ above {} + {}*mult = {}".format(median,stdev,thisTHR))
    print("alternative {} + {}*mult = {}".format(mean,stdev,altTHR))

    badPixels = (nuArray > thisTHR).astype(int)

    print("found {} bad pixels".format(len(np.argwhere(badPixels>0))))

    stats = [mean, median, stdev]    

    return badPixels, stats

def redo_vths(scurves, param_range, Vstart):

     vths_test = np.zeros((256, 256), dtype=np.uint16)                                                                                                                                                       
     cnt = 0
     n_zero, n_nega = 0, 0
                                            
     for x in range(256):                                                                         
         for y in range(256):                                                                     
             sum_of_hits = 0                                                                      
             weighted_sum_of_hits = 0                                                             
             for i in range(param_range):                                                    
                 sum_of_hits += scurves[x*256+y, i]                                               
                 weighted_sum_of_hits += scurves[x*256+y, i] * i                                  
             if(sum_of_hits > 0):                                                                 
                 vths_test[x, y] = Vstart + weighted_sum_of_hits / (sum_of_hits * 1.0)       
                 if(vths_test[x,y] > Vstart+param_range):                               
                     print("tpx3::analysis:vths_test: High Threshold: \u03A3 = {}, weighted \u03A3= {}".format(sum_of_hits, weighted_sum_of_hits))

             if(sum_of_hits == 0):
                n_zero+=1
                if(cnt%100==0):
                    print(f"channel {x},{y}->{cnt} has 0 sum of hits ")
                    print(f"total 0 sums up to now = {n_zero}")
             if(sum_of_hits < 0):              
                n_nega+=1
                if(cnt%100==0):
                    print(f"channel {x},{y}->{cnt} has negative sum of hits {sum_of_hits}")
                    print(f"Check: {weighted_sum_of_hits}")
             cnt+=1


     n_valid = 65535 - n_zero - n_nega
     print(f"\n\nfor this data we have - negative={n_nega}, zero={n_zero}, valid={n_valid}")    

     return vths_test                                                                                  

def countHits(thrmap, Vstop):
    
    hist = np.zeros(Vstop+1,dtype=np.uint16)
    cnt=0
    negative_thresholds = 0
    for x in range(0,256):
        for y in range(0,256):
            if(cnt%1000==0 or int(thrmap[x,y]<0)):
                print("pixel[{},{}] -> thr={}, Vstop={} int(thr[x,y])={}".format(x,y,thrmap[x,y],Vstop,int(thrmap[x,y])))
                negative_thresholds+=1
            if(int(thrmap[x,y])<Vstop and int(thrmap[x,y]>=0)):
                hist[int(thrmap[x,y])]+=1
            cnt+=1
    print("[WARNING] Found {} negative thresholds!".format(negative_thresholds))
    return hist

def findOutliers(nuarray):

    tot_dist = 8 - nuarray

    overflow = (tot_dist > 8).astype(np.uint8)
    underflow = (tot_dist < -8).astype(np.uint8)
    badPixels = ((tot_dist > 8 ) | (tot_dist < -8)).astype(np.uint8)

    return badPixels, underflow, overflow


def plotSurf(data, plotname):
    
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

    x = np.arange(0,256,1)
    y = np.arange(0,256,1)
    x, y = np.meshgrid(x,y)
    z = data

    surf = ax.plot_surface(x,y,z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plt.savefig(plotname+".png")    


def pixelMap2dPlasma(nuArray, plotname):

    print("-----------------------------------------------------------------")
       
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.86, 0.1, 0.05, 0.8])
    ms = ax.matshow(nuArray, cmap='plasma')
    fig.colorbar(ms, cax=cax, orientation='vertical')

#    ax.text(-85, 300, '-2: 0 in mask and mu', fontsize=10, color='black')
#    ax.text(-85, 310, '-1: 1 in mask not mu', fontsize=10, color='black')
#    ax.text(-85, 320, ' 0: else', fontsize=10, color='black')
#    ax.text(-85, 330, ' 1: 1 in mu not mask', fontsize=10, color='black')
#    ax.text(-85, 340, ' 2: 1 in mu and mask', fontsize=10, color='black')

#        ax.text(0.05,0.9,r'$occ_{mean}$'+f' = {stats[0]:.2f}',fontsize=10)
#        ax.text(0.05,0.85,r'$occ_{med}$'+f' = {stats[1]:.2f}',fontsize=10)
#        ax.text(0.05,0.8,r'$\sigma_{occ}$'+f' = {stats[2]:.2f}',fontsize=10)
#
    start = 120
    step = 15
    ax.text(-90,start,        '-2: 0(mask&mu) ', fontsize=10, color='black')
    ax.text(-90,start+step,   '-1: mask=1,mu=0', fontsize=10, color='black')
    ax.text(-90,start+step*2, ' 0: else'       , fontsize=10, color='black')
    ax.text(-90,start+step*3, ' 1: mask=0,mu=1', fontsize=10, color='black')
    ax.text(-90,start+step*4, ' 2: 1(mask&mu)' , fontsize=10, color='black')


    plt.plot()
    fig.savefig("IMG-2d-"+plotname+".png")
    print("-----------------------------------------------------------------")

def pixelMap2d(nuArray, plotname):

    stats = None # list of floats with [mean,median,stdev]
    pixelmask = None # 2D numpy array later
    #these = ["occMap-", "checkin-event"]
    #if(these in plotname):
    if("occMap-" in plotname):
        _ , stats = findNoisyPixels(nuArray)       
     
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.86, 0.1, 0.05, 0.8])
    ms = ax.matshow(nuArray)
    fig.colorbar(ms, cax=cax, orientation='vertical')
    if(stats is not None):
        ax.text(-85,270,r'$occ_{mean}$'+f' = {stats[0]:.2f}',fontsize=10)
        ax.text(-85,280,r'$occ_{med}$'+f' = {stats[1]:.2f}',fontsize=10)
        ax.text(-85,290,r'$\sigma_{occ}$'+f' = {stats[2]:.2f}',fontsize=10)

    plt.plot()
    fig.savefig("IMG-2d-"+plotname+".png")


def plotHist(xlist, nbins, minrange, maxrange, transparency, label, title, axislabel, plotname):

    plt.figure()

    overflow = 0
    for i in range(0,len(xlist)):
        if(xlist[i]>maxrange):
            overflow+=1

    val, cnt = np.unique(xlist, return_counts=True)
    ind = np.argmax(cnt)

    #print("Checking xlist:")
    #print("len(xlist)={},\nval={},\ncnt={},\nind={}".format(len(xlist), val,cnt,ind))

    maxy = np.max(cnt)

    counts, edges, bars = plt.hist(xlist, 
                                   bins=nbins, 
                                   range=(minrange,maxrange),
                                   alpha=transparency,
                                   label=label)

    plt.text(minrange+10, maxy+10, "len(xlist)={}, overflow={}".format(len(xlist), overflow))

    plt.title(title)
    plt.xlabel(axislabel[0])
    plt.ylabel(axislabel[1])
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(plotname+"-hist.png")

def getMinMaxFromList(datalist):

    minlist = []
    maxlist = []
    for dlist in datalist:
        minlist.append(min(dlist))
        maxlist.append(max(dlist))

    #return list(min(minlist), max(maxlist))
    return [min(minlist), max(maxlist)]

def plotHistStack(datalist, nbins, binrange, labels, histlabels, picname, logscale):

    plt.figure()

    cnt = 0
    for data in datalist:

        plt.hist(data,nbins, range=(binrange[0],binrange[1]), alpha=0.25, label=histlabels[cnt])
        cnt+=1

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    if(logscale):
        plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.savefig("histStack-"+picname+".png")

    print(f"Createdstacked histogram plot for {picname}")


def comparePlots(datalist, title, lablist, axislab, plotname):

    cnt = 0
    maxnum = 0
    plt.figure(figsize=(10,10))
    for data in datalist:
        i_max = np.max(data)
        if(i_max >= maxnum):
            maxnum = i_max
        plt.plot(data, label=lablist[cnt])
        cnt+=1
    if(maxnum > 5000):
        plt.yscale('log')
    plt.title(title)
    plt.xlabel(axislab[0])
    plt.xlabel(axislab[1])
    plt.legend(loc='upper left')
    plt.savefig("compare-plots-"+plotname+".png")

######### funcs end here! ###########

filename = str(sys.argv[1])
option = str(sys.argv[2])

clean_filename = removePath(filename)

with tb.open_file(filename, 'r') as f:

     if(option=="mask"):
        mask = None

        if("_mask_" in filename):
            mask= f.root.mask_matrix[:].T 
        else:
            mask = f.root.configuration.mask_matrix[:].T

        pixelMap2d(mask,"mask_matrix-"+clean_filename) 
     elif(option=="thr"):
        if("_equal_" in filename):
            thr = f.root.thr_matrix[:].T
            print(thr.shape) 
            pixelMap2d(thr,"thresholdMap-"+clean_filename)
            plotHist(thr.reshape(-1), 16, -0.5, 16.5, 1, "Thr. dist", "", ["THR", "N_thr"], "threshols-1d-"+clean_filename) 
        else:
            thr = f.root.configuration.thr_matrix[:].T
            print(thr.shape) 
            pixelMap2d(thr,"thresholdMap-"+clean_filename)

            minthr = -0.5
            maxthr = 15.5           

            simpleHist(np.ravel(thr), 16, minthr, maxthr,["thr hist","THR", "N_THR"], "THR-simpleHist-"+clean_filename,False)
            
     elif(option=="links"):
        links = f.root.configuration.links[:].T
        for i in range(0,7):
           print("{}={}".format(links[i][7],links[i][6]))
     elif(option=="config"):
        conf = None
        if("PixelDAC" in filename):
            conf = f.root.configuration.run_config_0[:]
        else:
            conf = f.root.configuration.run_config[:]
            
        for i in range(0,len(conf)):
          print(conf[i])
     elif(option=="genconfig"):
        genconf = f.root.configuration.generalConfig[:]
        for i in range(0,len(genconf)):
          print(genconf[i])
     elif(option=="dacs"):
        dacs = None
        if("PixelDAC" in filename):
            dacs = f.root.configuration.dacs_0[:]
        else:
            dacs = f.root.configuration.dacs[:]
        for i in range(0,len(dacs)):
           print(dacs[i])
     elif(option=="mdata"):
        mdata = None
        conf = None
        Vstart = None
        Vstop = None
        if("PixelDAC" in filename):
            mdata = f.root.meta_data_0[:].T 
            conf = dict(f.root.configuration.run_config_0[:])
        else:
            conf = dict(f.root.configuration.run_config[:])
            mdata = f.root.meta_data[:].T 

        Vstart = int(conf[b'Vthreshold_start'])
        Vstop = int(conf[b'Vthreshold_stop'])
        scurves = None
        if("ThresholdScan" in filename and testIfHasInterpreted(f)):
            conf = dict(f.root.configuration.run_config[:])
            scurves = None
            ##### shit for scurves

            n_pulses = int(conf[b'n_injections'])
            if(hasInterpretedBranch(f,'HistSCurve')):
           
                scurves = f.root.interpreted.HistSCurve[:].T
                plot_scurves(scurves,list(range(Vstart,Vstop)),scan_parameter_name="Vthreshold",max_occ=n_pulses*3)

            #############
            if(hasInterpretedBranch(f,'HistOcc')):
                histocc = f.root.interpreted.HistOcc[:].T
                pixelMap2d(histocc,"THL-scan-2d-histocc-"+clean_filename)

            if(hasInterpretedBranch(f,'NoiseMap')):
                noisemap = f.root.interpreted.NoiseMap[:].T
                pixelMap2d(noisemap,"THL-scan-2d-noisemap-"+clean_filename)

        if(testIfHasOccupancy(f)):
           histOcc = f.root.interpreted.HistOcc[:].T
           pixelMap2d(histOcc,"NS-histOcc-2D-"+clean_filename)

        scanid, idx_start, idx_stop, dec, dis, datalen, tstamp_0, tstamp_end, rx_fifo = ([] for i in range(9))

        dac_n_fifo_comb = []
        dac_n_hits = []
        dac_n_discard = []
        dac_n_bitrate = []

        testcnt = 0

        print(clean_filename)

        arr_scanid = fillArray(mdata, 5)
        arr_x = fillArray(mdata, 1)
        arr_idx_start = fillArray(mdata, 0)
        arr_disc = fillArray(mdata, 7)
        arr_deco = fillArray(mdata, 6)
        arr_dlen = fillArray(mdata, 2)

        n_chunks = 0

        #t_interval = 0.05 #s
        t_interval = 2 #s

        scurvex = []
        scurvey = []

        if(scurves is not None):
            for i in range(0,10):
                print(scurves[i][1])

        n_datacolumns = None

        flag_old_scanid = True
        print(f"Using vstart={Vstart}, vstop={Vstop}")
        vth_range = Vstop-Vstart

        for i in range(0,len(mdata)):
           if(i==0):
                n_datacolumns = len(mdata[i])
                print("Data table has {} columns".format(n_datacolumns))
           idx_start.append(mdata[i][0])
           idx_stop.append(mdata[i][1])
           dec.append(mdata[i][7])
           if("DataTake" in filename):
               dis.append(mdata[i][6])
           else:
               dis.append(mdata[i][6]*(-1))
           if("DataTake" in filename):
               datalen.append(mdata[i][2]/2/t_interval)
           else:
               datalen.append(mdata[i][2])
           if("DataTake" in filename):
               time_s = np.float64(mdata[i][3])
               time_e = np.float64(mdata[i][4])
               tstamp_0.append(time_s/1e9)
               tstamp_end.append(time_e/1e9)
           else:
               tstamp_0.append(np.float64(mdata[i][3]))
               tstamp_end.append(np.float64(mdata[i][4]))
           #scanid.append(mdata[i][5])

          # if("PixelDAC" in filename):
          #     if(i<=len(mdata)/2):
          #          scanid.append(Vstart+mdata[i][5])
          # else:
          #          scanid.append(Vstart+mdata[i][5]-vth_range)
           if(i==0 and mdata[i][5]==Vstart):
                flag_old_scanid = False
           if(flag_old_scanid):
               scanid.append(Vstart+mdata[i][5])
           else:
               scanid.append(mdata[i][5])

           if(n_datacolumns>9):
               rx_fifo.append(mdata[i][9])

           bitrate = mdata[i][2]*32/t_interval

           dac_n_bitrate.append([bitrate, mdata[i][5]])
           dac_n_discard.append([mdata[i][6], mdata[i][5]])
           if(n_datacolumns>9): 
               dac_n_fifo_comb.append([mdata[i][9], mdata[i][5]])
           dac_n_hits.append([mdata[i][2]/2/t_interval, mdata[i][5]])
           #if(i==0):
           #  t_interval = mdata[i][10]/1000
           #  print("Found data interval to be : {} [s]".format(t_interval),flush=True)
           n_chunks+=1
           #############################
    
        unq_dac = []
        avg_fifo = None
        avg_hits = None
        avg_discard = None
        tot_unq_hits = None

        mindatalen = min(datalen)
        maxdatalen = max(datalen)

        if("DataTake" not in filename):
    
            if("NoiseScan" in filename):

                unq_dac = getUniqueList(scanid)
                print(len(unq_dac)) 
                if(len(dac_n_fifo_comb)>0):
                    avg_fifo, _ = getAvgList(unq_dac, dac_n_fifo_comb)
                    print(len(avg_fifo))
                print("Obtaining average n hits")
                avg_hits, tot_unq_hits = getAvgList(unq_dac, dac_n_hits)
                avg_bitrate, _= getAvgList(unq_dac, dac_n_bitrate)

                print("Obtaining average n discard")
                avg_discard, _ = getAvgList(unq_dac, dac_n_discard)                

                print(unq_dac[:10])
    
                if(dac_n_fifo_comb is not None):
                    print(dac_n_fifo_comb[:10])
                if(avg_fifo is not None):
                    print(avg_fifo[:10])

                print("Plotting")

                plot_scat_stack(unq_dac,[avg_hits, avg_discard],
                                clean_filename+"-rate-vs-discardErr",
                                ["average hits","average discard errors"],
                                ["DAC", "Average value [N_hits & N_error]"])

                plot_scat(unq_dac, avg_bitrate, "Average bitrate vs Threshold DAC",
                          clean_filename+"avg-bitrate",
                          "len(datawords)*32bit/t_readout",
                          ["Threshold DAC [cnt]","Bit rate [bits]"]
                         )

                plot_scat(unq_dac, tot_unq_hits, "total hits per DAC",
                            clean_filename+"-tothits-vs-DAC",
                            "total hits",
                            ["DAC","total hits"])

                if(avg_fifo is not None):
                    plot_scat(unq_dac,
                              avg_fifo,
                              "TPX3 RX FIFO Size vs Threshold DAC",
                              clean_filename+"_DAC_vs_rx_fifo_size",
                              "average fifo size",
                              ["DAC","TPX3 RX fifo size"])
        
                plot_scat(unq_dac,
                          avg_hits,
                          "Avg. Hitrate vs Threshold DAC",
                          clean_filename+"_DAC_vs_hitrate",
                          "average readout rate [bits]",
                          ["DAC","len(data)\*32(bits)/t_readout"])

            print("Number of chunks => {}".format(n_chunks))

        ##########
            plot_scat_stack(scanid,
                           [dec,dis],
                           clean_filename+"_DecDiscErrors",
                           ["decoding","discard"],
                           ["scan parameter ID","Errors,[N]"])

            #########

            plot_scat(scanid,
                      idx_stop,
                      "Total words recorded vs scan ID",
                       clean_filename+"recordedData_vs_scanId",
                       "total words recorded = {}".format(max(idx_stop)),
                       ["scanID, [DAC]","scan_bas::handle_data::total_words, [N]"])
   
            #ploth2d(np.asarray(scanid), np.asarrdistro-EqualisationCharge_2024-09-27_16-23-46.pngay(rx_fifo),clean_filename+"-fifo-size") 
            if(len(rx_fifo)>0):
                plotHist(rx_fifo, 
                         41, 
                         1040, 
                         1080,
                         1,
                         "RX_fifo_size", 
                         "RX FIFO SIZE", 
                         ["DAC", "Nentries"],
                         clean_filename+"-fifo-size")        
            
            plot_scat(scanid,
                      datalen,
                      "Data Length vs Threshold DAC",
                      clean_filename+"_DataLength-vs-DAC",
                      "Data length",
                      ["scan parmeter ID","Data length"])

            print(len(tstamp_0))
            print(len(tstamp_end))
            print(len(scanid))

            plot_scat_stack(scanid,
                            [tstamp_0,tstamp_end],
                            clean_filename+"_timestamps",
                            ["stamp_start","stamp_end"],
                            ["scanid","t_stamps"])

    
            plotHist(datalen,
                     100,
                     mindatalen,
                     maxdatalen,
                     1,
                     "data length per frame",
                     "ebala",
                     ["length","n counts"],
                     clean_filename+"-hist-datalen")

        else:
            #plot_scat(idx_start,
            #plot_scat(x,#tstamp_0,

            if(len(rx_fifo)>0):
                plot_scat(tstamp_0,
                          rx_fifo,
                          "timestamp vs RX fifo size",
                          clean_filename+"_tstamp_vs_rx_fifo_size",
                          "RX FIFO size",
                          ["timestamp","RX fifo size"])
    
            plot_scat(tstamp_0,
                      datalen,
                      "timestamp vs data length",
                      clean_filename+"_DataLength-vs-iterationIndex",
                      "rate [Hz]",
                      ["timestamp","Hits per second"])

            #plot_scat(tstamp_0,
           #           avg_hits,
           #           "Avg. Hitrate vs Timestamp",
           #           clean_filename+"_hitrate",
           #           "average hitrate Hz",
           #           ["timestamp","len(data)/2/t_readout"])

            #plot_scat(idx_start,
            #plot_scat(x,#tstamp_0,
            plot_scat(tstamp_0,
                      dec,
                      "decoding errors vs timestamp",
                      clean_filename+"_DecodingErr-vs-Timestamp",
                      "Decoding Err.",
                      ["timestamp","Decoding Errors [N]"])

            plot_scat(tstamp_0,
                      dis,
                      "discard errors vs timestamp",
                      clean_filename+"_DiscardErr-vs-Timestamp",
                      "Discard Err.",
                      ["timestamp","Discard Errors [N]"])

            plot_scat(idx_start,
                      tstamp_0,
                      "iteration vs timestamp",
                      clean_filename+"_idx_vs_tstamp",
                      "timestamp",
                      ["Start Index","timestamp"])


     elif(option=="rdata"):
        rdata = f.root.raw_data[:]
        print("raw data length:{}".format(len(rdata)))

        print(type(rdata))
        print(f'data shape: {rdata.shape}')
        print(f'data type: {rdata.dtype}')
        
        mdata = f.root.meta_data[:].T
        gencfg = f.root.configuration.generalConfig[:]
        
        op_mode = [row[1] for row in gencfg if row[0]==b'Op_mode'][0]
        vco = [row[1] for row in gencfg if row[0]==b'Fast_Io_en'][0]
 
        hit_data = analysis.interpret_raw_data(rdata[:500000], op_mode, vco, mdata, progress = None)
        print(type(hit_data))

        hit_data = hit_data[hit_data['data_header']==1]

        print(hit_data.dtype)

        # -----------------------------------------
        hit_data_tot = hit_data['TOT']     

        hit_data_tot_1d = hit_data_tot.flatten()

        #simpleHist(hit_data_tot_1d, 51, 0, 200, ["TOT", "TOT", "#"], "from_raw_to_intp-TOTplot2" , False) 
        simpleHist(hit_data_tot_1d*26, 51, 0, 2600, ["TOT", "e-", "#"], "from_raw_to_intp-TOTplot2" , False) 
        hit_data_tot_1d = None

        # -----------------------------------------
        totVal = []
        #totVal = np.array([],dtype=np.uint16)
        #xpos = np.array((256,256),dtype=np.uint16)
        #ypos = np.array((256,256),dtype=np.uint16)
        cnt_mask = np.array((256,256), dtype=np.uint16)

        tempTOT = 0
        tempTOA = 0
        totPerFrame = []
        toaPerFrame = []
        curr_combTOA = 0
        combTOAlist = []
        cnt = 0
        nchunks = hit_data.shape[0]
#            if(i // 1000):
#                print('\r progress => {:.2f}%'.format(float(i)/float(nchunks)*100),end= " ", flush=True)
        
        allCombTOA = hit_data['TOA_Combined']
        unq_comTOA = np.unique(allCombTOA)

        TOTeventMatrix = np.zeros((256,256),dtype=np.uint16)
        TOAeventMatrix = np.zeros((256,256),dtype=np.uint16)

        these_events = [661786260,736194492,736194495,736194496,736194497, 710270671, 573885847]

        cnt=0
        for ctoa in unq_comTOA:

            print(f"----------- Coombined TOA {ctoa} ------------")
            mask = (allCombTOA == ctoa)
            #print(f"ctoa = {ctoa}, mask = {mask}")
            x = hit_data['x'][mask]
            y = hit_data['y'][mask]
            TOA = hit_data['TOA'][mask]
            TOT = hit_data['TOT'][mask]

            TOTfilterSum = np.sum(TOT)

            #if(diff_TOT > 50 and len(np.unique(TOA))>2):
            #if(diff_x <200 and diff_y < 200 and diff_TOT > 50 and len(np.unique(TOA))>2):
            #if(len(x)>5 and len(x)<256 and len(np.unique(TOT))>2):
                #print(f"{len(x)}:x={x}")
                #print(f"{len(y)}:y={y}")
                #print(f"{len(TOT)}:tot={TOT}")
                #print(f"{len(TOA)}:toa={TOA}")
                #print(f"dTOT={diff_TOT}, dx={diff_x}, dy={diff_y}")

            #if(len(x)>5 and len(x)<256 and len(np.unique(TOT))>2):
            #if(len(x)>5 and len(x)<4096):

            TOTsmeared = TOT.reshape(64,8,64,8).mean(axis=(1,3))
            if(TOTfilterSum < 500):
            #if(ctoa in these_events):
                # --- tryna do db scan try2 ---
                nz_points = []
                i_cnt = 0
                for i,j in zip(x,y):
                    if(TOT[i_cnt]>0):
                        nz_points.append([i,j])
                    i_cnt+=1
   
                nz_points = np.array(nz_points)
    
                #print(type(nz_points))
                #n_holes = 10
                n_holes = 20
                min_hits = 10
                #labels, nclusters_found = None, None
                labels, nclusters_found = None, -999
                if(len(nz_points)>0):
                    db = DBSCAN(eps=n_holes, min_samples=min_hits).fit(nz_points)
                
                    labels = db.labels_
                    nclusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    
                TOTeventMatrix[x,y] = TOT
                TOAeventMatrix[x,y] = TOA
 
                #if(ctoa==534986354):
                if(ctoa in these_events):

                    nonzero = (TOTeventMatrix > 0)
                    print("---------- pizda! ----------")
                    print(len(TOT))
                    simpleHist(np.ravel(TOTeventMatrix[nonzero]), 51, 0, 201, [f"event-{ctoa}-allTOT", "TOT", "#"], f"{ctoa}-TOT", False)
                    print(len(TOA))
                    simpleHist(np.ravel(TOAeventMatrix[nonzero]), 51, 0, 16384, [f"event-{ctoa}-allTOA", "TOA", "#"], f"{ctoa}-TOA", False)

                    UniqueTOA = np.unique(np.ravel(TOAeventMatrix))
                    print(f"found {len(UniqueTOA)} unique TOAs")

                    clustTOA = getToaClusters(UniqueTOA, 50)

                    #cntr = 0
                    #for ntoa in clustTOA:

                    #    these_hits = hit_data[ctoa]['TOA']
                    #    local_mask = (these_hits == ntoa[1])
                    #   
                    #    iTOA = hit_data['TOA'][local_mask] 
                    #    ix = hit_data['x'][local_mask]
                    #    iy = hit_data['y'][local_mask]

                    #    iclust = np.zeros((256,256),dtype=np.uint16)
                    #    iclust[ix,iy] = iTOA

                    #    pixelMap2d(iclust, f"test-iclust")
                    #    iclust = np.zeros((256,256),dtype=np.uint16)
                    #    
                    #    #print(f"going through TOA cluster {ntoa}")
                    #    #singleTOACluster = np.zeros((256,256), dtype=np.uint16)

                    #    #print(f"TOA-list={TOA}")

                    #    #cntr2 = 0
                    #    #for i,j in zip(x,y):
                    #    #    print(f"check 1[{TOAeventMatrix[x,y]}] 2[{ntoa}]")
                    #    #    if(TOAeventMatrix[x,y] in ntoa):
                    #    #        singleTOACluster[i,j] = TOTeventMatrix[x,y]
                    #    #        print("{} in {} list - adding {} to matrix {}".format(TOA[cntr2], ntoa, TOT[cntr2], np.count_nonzero(singleTOACluster)))
                    #    #    cntr2+=1    

                    #    #if(cntr//4):
                    #    #    pixelMap2d(singleTOACluster, f"ctoa-{ctoa}-toa-{ntoa[0]}-TOTcluster")                    

                    #    cntr+=1

                #if((len(TOT)<50 and len(TOT)>5 and np.sum(TOT)>50 and len(unqTOA)>1) or (ctoa in these_events)):
                #if(ctoa in these_events):
                #haveClusters = nclusters is not None
                
                if(ctoa in these_events or nclusters_found > 1):
                #if(ctoa in these_events or nclusters_found>1):
                #if(nclusters_found is not None and nclusters_found > 1):
    
                    pixelMap2d(TOTeventMatrix, f"chekin-event-matrix-TOT-comTOA-{ctoa}-{nclusters_found}")
                    pixelMap2d(TOAeventMatrix, f"chekin-event-matrix-TOA-comTOA-{ctoa}-{nclusters_found}")
                    # ---------- tryin' smearin' --------------------------------
                    #lowResMat = TOTeventMatrix.reshape(64,4, 64,4).mean(axis=(1,3))
                    #pixelMap2d(lowResMat, f"lowRes-TOT-comTOA-{ctoa}-{nclusters_found}")
                    # -----------------------------------------------
                    # tryna; DBSCAN
     
                    #points = []
                    #for i in range(256):
                    #    for j in range(256):
                    #        if(TOTeventMatrix[i,j]>5):
                    #            points.append([i,j])
    
                    #points = np.array(points)
    
                    #n_holes = 10
                    #min_hits = 5
                    #db=DBSCAN(eps=n_holes, min_samples=min_hits).fit(points)
                    #labels = db.labels_

                    diff_x = np.max(x) - np.min(x)
                    diff_y = np.max(y) - np.min(y)
                    diff_tot = np.max(TOT) - np.min(TOT)
                    diff_toa = np.max(TOA) - np.min(TOA)
                    totSum = np.sum(TOT)

                    n2dtoa = np.count_nonzero(TOAeventMatrix)
                    print(f'EVENT STATS:')                    
                    print("##################################################################")
                        
                    print(f"DIFFS: dx = {diff_x}, dy = {diff_y}, dtot = {diff_tot}, dtoa = {diff_toa}")
                    print(f"TOT_SUM={totSum} , len(TOA)={len(TOA)}, n2Dpoints={n2dtoa}, len(nz_points)={len(nz_points)}")
                    #print(f"")       

                    print("##################################################################")

                    # - plotting this shit
                    plt.figure(figsize=(8,6))
                    #plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='plasma', s=10)
                    plt.scatter(nz_points[:, 0], nz_points[:, 1]*(-1), c=labels, cmap='plasma', s=10) 
                    plt.title('DBSCAN Clustering of Charge Matrix')
                    plt.xlabel('X Position')
                    plt.ylabel('Y Position')
                    plt.xlim(0,256)
                    plt.ylim(-256,0)
                    plt.text(0, 260, f"n hits = {len(x)}")
                    plt.colorbar(label='Cluster Label')
                    plt.savefig(f"DBSCAN-ctoa-{ctoa}.png")
       
                    # -----------------------------------------------
                    TOTeventMatrix = np.zeros((256,256),dtype=np.uint16)
                    TOAeventMatrix = np.zeros((256,256),dtype=np.uint16)
                    cnt+=1
    
    
                if(cnt==20):
                    break
            
            else:
                continue

        #print(len(totVal))
        #print(totVal[0])

        print(f"counts={cnt}")
        print(len(set(combTOAlist)))

        #print(np.unique(np.array(combTOAlist),return_counts=True))

        #simpleHist(totVal, 51,0,1025, ["TOT plot", "ToT", "#"], "from-raw_to-intp-TOTplot",False)

        #simpleHist(toaPerFrame, 51, 0, 4097, ["TOA per frame", "avg TOA", "#"], "intp-TOA",False)
        #simpleHist(totPerFrame, 51, 0, 1025, ["TOT per frame", "avg TOT", "#"], "intp-TOT",False)

        totVal = None
        hit_data = None
        

     elif(option=="idata"):
        if(testIfHasInterpreted(f)):
            if("Equalisation" in filename):
                ##### shit for scurves
                conf = dict(f.root.configuration.run_config[:])
                Vstart = int(conf[b'Vthreshold_start'])
                Vstop = int(conf[b'Vthreshold_stop'])
                n_pulses = int(conf[b'n_injections'])
                #############
                eqdist = None
                eqmap = None 
                badpix, uflow, oflow = None, None, None
                mumap0, mumap15 = None, None

                vth0, vth15, sig2d, sig2d15, chi2, chi215 = None, None, None,None, None, None

                maskmap = f.root.configuration.mask_matrix[:].T

                thrmap0, thrmap15 = None, None

                if(hasInterpretedBranch(f,"ThresholdMap_th0")):
                    thrmap0 = f.root.interpreted.ThresholdMap_th0[:]
                    thrmap15 = f.root.interpreted.ThresholdMap_th15[:]

                    print("COUNTING THRESHOLDS:")
                    h_th0 = countHits(thrmap0,Vstop)

                    pixelMap2d(thrmap0, "ThresholdMap-th0-"+clean_filename)
                    pixelMap2d(thrmap15, "ThresholdMap-th15-"+clean_filename)
                    pixelMap2d(thrmap15-thrmap0, "ThresholdMapDifference"+clean_filename)

                    #thrmaps = [np.ravel(thrmap0), np.ravel(thrmap15)]

                    #comparePlots(thrmaps,"threshold maps",["thr0","thr15"],["DAC","Nentries"],"bothThresholdsMaps-"+clean_filename)

                    print("thrmap0.shape={}".format(thrmap0.shape))
                    thrdata0 = np.ma.masked_array(thrmap0,maskmap)
                    plot_dist(thrdata0, plot_range=np.arange(Vstart-0.5,Vstop+0.5,1), x_axis_title="Vthreshold", title="THR dist pixDAC 0")

                if(countNodes(f,"mu2D", "intp")>0):
                    mumap0 = f.root.interpreted.mu2D_th0[:].T
                    mumap15 = f.root.interpreted.mu2D_th15[:].T

                if(hasInterpretedBranch(f,"EqualisatioMap")):
                    eqmap = f.root.interpreted.EqualisationMap[:].T

                if(hasInterpretedBranch(f,"EqualisationDistancesMap")):
                    eqdist = f.root.interpreted.EqualisationDistancesMap[:].T

                    badpix, uflow, oflow = findOutliers(eqdist)

                    pixelMap2d(eqdist, "DistanceMap-"+clean_filename)
                    pixelMap2d(badpix, "DistanceMap-BP-"+clean_filename)
                    pixelMap2d(uflow, "DistanceMap-UF-"+clean_filename)
                    pixelMap2d(oflow, "DistanceMap-OF-"+clean_filename)

                if(hasInterpretedBranch(f,"vth_histTh0")):
                    vth0 = f.root.interpreted.vth_histTh0[:]
                    vth15 = f.root.interpreted.vth_histTh15[:]
                    sig2d = f.root.interpreted.sig2D_th0[:].T.astype(np.int64)
                    sig2d15 = f.root.interpreted.sig2D_th15[:].T.astype(np.int64)
                    chi2 = f.root.interpreted.chi2ndf2D_th0[:].T
                    chi215 = f.root.interpreted.chi2ndf2D_th15[:].T

                scurves_0 = f.root.interpreted.HistSCurve_th0[:].T
                scurves_15 = f.root.interpreted.HistSCurve_th15[:].T

                print("\ntest re-run of tpx3::analysis::vths\n")
                #eblo = redo_vths(scurves_0, Vstop-Vstart-1, Vstop)
                #eblo = redo_vths(scurves_0, Vstop-Vstart+1, Vstop)

                print("scurves_0.shape = {}, {}".format(scurves_0.shape, type(scurves_0)))
                print("scurves_15.shape = {}".format(scurves_15.shape))

                ##########################################################################
                plot_scurves(scurves_0, list(range(Vstart,Vstop)), title="scurves-at-th0", scan_parameter_name="Vthreshold",max_occ=n_pulses*5,plotname="huemorgen-"+clean_filename)
                ## ----- looking how many entries per histogram there is for scurves
                #rand_pixels = np.random.randint(0,65535,size=20)
            
                scurve_par_entries = []
                for i in range(65536):
        
                    this_curve = scurves_0[:,i]
                    hitsinscurve = np.count_nonzero(np.asarray(this_curve))
                    scurve_par_entries.append(hitsinscurve)

                #    if(i in rand_pixels):
                #        quickPlot(this_curve, [f"scurve THR=0 chan {i}", "scan_id", "N_entries"] , f"quickScurve-{i}-"+clean_filename)                

                simpleHist(scurve_par_entries,100,0,800,["Counting entries per channel histogram","n_entries", "#"],"counting-par-entries-in-scurves-"+clean_filename, True)
                ##########################################################################

                plot_scurves(scurves_15, list(range(Vstart,Vstop)),title="scurves-at-th15", scan_parameter_name="Vthreshold",max_occ=n_pulses*5,plotname="huemorgen-th15-"+clean_filename)
                ## -------- same but at THR=15
                
                scurve_par_entries_15 = []
                for i in range(65536):
        
                    this_curve = scurves_15[:,i]
                    hitsinscurve = np.count_nonzero(np.asarray(this_curve))
                    scurve_par_entries_15.append(hitsinscurve)
   
                #    if(i in rand_pixels):
                #        quickPlot(this_curve, [f"scurve THR=15 chan {i}", "scan_id", "N_entries"] ,f"quickScurve15-{i}"+clean_filename)                

                simpleHist(scurve_par_entries_15,100,0,800,["Counting entries per channel histogram","n_entries", "#"],"counting-par-entries-in-scurves15-"+clean_filename, True)

                # ---------- lalka ------------------------- 

                plt.figure()
                plt.plot(scurves_0[:,36000],"bo",label="THR=0")
                plt.plot(scurves_15[:,36000],"ro",label="THR=15")
                plt.legend()
                plt.savefig("lalka-chan36000-"+clean_filename+".png")

                # ------------------------------------------

                if(vth0 is not None):
                    print("vth0.shape = {}, {}".format(vth0.shape,type(vth0)))
                    print("vth15.shape = {}".format(vth15.shape))
    
                    print("sigmas shape = {}, {}".format(sig2d.shape, type(sig2d)))
                    print(type(sig2d))
                    print(type(sig2d[1][66]))
                    print("chisq shape = {}".format(chi2.shape))

                    plot_2dim(sig2d,list(range(Vstart,Vstop)), "sig2D", ["params","pixels"],"HIST-sig2d-{}".format(clean_filename))
    
                    #nonil_indices_0 = np.nonzero(vth0)
                    #nonil_indices_15 = np.nonzero(vth15)

                    #print("chcking vth0:")
                    #for index in nonil_indices_0:
                    #    print("[{},{}]".format(vth0[index],index))
                              
                    binrange = Vstop - Vstart
    
                    plotHist(Vstart+vth0,binrange,Vstart,Vstop ,1, "vth0","Threshold_0",["THR","Counts"],"vth0-"+clean_filename)
                    plotHist(Vstart+vth15,binrange, Vstart,Vstop,1,"vth15","Threshold_15",["THR","Counts"],"vth15-"+clean_filename)

                    comparePlots([vth0,vth15],
                                 "Comparing thersholds @ 0 and 15 Pixel DAC",
                                 ["THR=0","THR=15"], 
                                 ["DAC","N entries"],
                                 "THR0-THR15")

                    ###############################
                    sigmas = []
                    sigmas15 = []
                    thrs = [] 
                    thrs15 = []
                    chisq = []
                    chisq15 = []
                    mus = []
                    mus15 = []
                    pixel_dist = None
                    distances = []
                    cntr = 0
                    ##############################

                    med_0 = np.median(np.ravel(thrmap0))
                    stdev_0 = np.std(np.ravel(thrmap0))

                    # plotting threshold outliers ~1200 to 1300 DAC
                    out_x = []
                    out_x15 = []
                    out_y = []
                    out_y15 = []
                    out_sigmas = []
                    out_sigmas15 = []
                    out_chi, out_chi15 = [],[]
                    out_mu = []
                    out_thr, out_thr15 = [], []
                    out_eqdist = []

                    testmask = np.zeros((256,256),dtype=int)                    
                    testmask2 = np.zeros((256,256),dtype=int)

                    mask_diff = np.zeros((256,256),dtype=int)

                    mu_out = np.where(mumap0==0,1,0)
                    mu_out15 = np.where(mumap15==0,1,0)

                    actual_mask = f.root.configuration.mask_matrix[:].T

                    pixelMap2d(actual_mask,"actually-masked-pixels")
                    pixelMap2d(mu_out, "pixels-with-mu0-guess-zero") 
                    pixelMap2d(mu_out15, "pixels-with-mu15-guess-zero") 

                    eblo = [0,0,0,0]                   
 
                    n_pixels_missed = 0
                    for i in range(256):
                        for j in range(256):
                            
                            if(int(mu_out[i,j])==1 and int(actual_mask[i,j])==0):
                                mask_diff[i,j] = 1
                                eblo[2]+=1

                            if(int(mu_out[i,j])==0 and int(actual_mask[i,j])==1):
                                mask_diff[i,j] = -1 
                                eblo[1]+=1
                                   
                            if(int(mu_out[i,j])==1 and int(actual_mask[i,j])==1):
                                mask_diff[i,j] = 2 
                                eblo[3]+=1

                            if(int(mu_out[i,j])==0 and int(actual_mask[i,j])==0):
                                mask_diff[i,j] = -2 
                                eblo[0]+=1
             
                    print(f"cathegories found:{eblo}")
                    pixelMap2dPlasma(mask_diff,"mask-and-mu0-difference")
                    
                    # 0) mu0 = 0
                    # 1) mu15 = 0
                    # 2) chi2 > 50
                    # 3) chi215 > 50
                    # 4) sigma0 = 0
                    # 5) sigma15 = 0
                    defect_pixels = [0,0,0,0,0,0]

                    if(sig2d is not None and thrmap0 is not None):
                        for x in range(0,256):
                            for y in range(0,256):
                                pixel_sigma = sig2d[x,y]
                                pixel_sigma15 = sig2d15[x,y]
                                pixel_thr = thrmap0[x,y]
                                pixel_thr15 = thrmap15[x,y]
                                pixel_chi2 = chi2[x,y]
                                pixel_chi215 = chi215[x,y]
                                pixel_mu = mumap0[x,y]
                                pixel_mu15 = mumap15[x,y]
                                #if(thrmap0[x,y] < med_0-(2*stdev_0) or thrmap15[x,y] < med_0 - (2*stdev_0)):
                                #if(pixel_mu == 0):
                                #    out_x.append(x)
                                #    out_y.append(y)
                                #    out_sigmas.append(pixel_sigma)
                                #    out_mu.append(pixel_mu)
                                #    out_chi.append(pixel_chi2)
                                #    out_thr.append(pixel_thr)

                                #if(pixel_mu15 == 0):
                                #    out_x15.append(x)
                                #    out_y15.append(y)
                                #    out_sigmas15.append(pixel_sigma15)
                                #    #out_mu15.append(pixel_mu15)
                                #    out_chi15.append(pixel_chi215)
                                #    out_thr15.append(pixel_thr15)

                                #if(thrmap0[x,y] < med_0 - (2*stdev_0) or thrmap15[x,y] < med_0 - (2*stdev_0)):
                                if(pixel_mu == 0 or pixel_mu15 == 0):
                                    testmask[x,y]=1
                                else:
                                    testmask[x,y]=0

                                sigmas.append(pixel_sigma)
                                sigmas15.append(pixel_sigma15)
                                thrs.append(pixel_thr)
                                thrs15.append(pixel_thr15)
                                chisq.append(pixel_chi2)
                                chisq15.append(pixel_chi215)
                                mus.append(pixel_mu)
                                mus15.append(pixel_mu15)
                                if(eqdist is not None):
                                    pixel_dist = eqdist[x,y]
                                    distances.append(pixel_dist)
                                   
                                    out_eqdist.append(pixel_dist) 

                                    #test masking try2 
                                    if(pixel_dist > 8 or pixel_dist < -8):
                                        testmask2[x,y] = 1
                                    else:
                                        testmask2[x,y] = 0

                                    #--- tryna' plot unequalizable pixel data ----------------
                                    if(pixel_dist > 8 or pixel_dist < -8):

                                        out_x.append(x)
                                        out_y.append(y)
                                        out_sigmas.append(pixel_sigma)
                                        out_mu.append(pixel_mu)
                                        out_chi.append(pixel_chi2)
                                        out_thr.append(pixel_thr)
                                        out_x15.append(x)
                                        out_y15.append(y)
                                        out_sigmas15.append(pixel_sigma15)
                                        #out_mu15.append(pixel_mu15)
                                        out_chi15.append(pixel_chi215)
                                        out_thr15.append(pixel_thr15)

                              
                                #----------------------------------------------
                                
                                if(pixel_mu == 0):
                                    defect_pixels[0] += 1
                                if(pixel_mu15 == 0): 
                                    defect_pixels[1] += 1
                                if(pixel_chi2 > 50):
                                    defect_pixels[2] += 1
                                if(pixel_chi215 > 50):
                                    defect_pixels[3] += 1
                                if(pixel_sigma == 0):
                                    defect_pixels[4] += 1
                                if(pixel_sigma15 == 0):
                                    defect_pixels[5] += 1

                                #----------------------------------------------
                                cntr+=1

                    n_testmasked = np.count_nonzero(testmask)
                    print(f"Masking based on {GREEK_MU}=0 yields {n_testmasked} pixels masked!")
                    print(f"found:\n mu=0 {defect_pixels[0]},\nmu15=0 {defect_pixels[1]},\nchi2>50 {defect_pixels[2]},\nchi215 > 0 {defect_pixels[3]},\nsigma=0 {defect_pixels[4]},\nsigma15=0 {defect_pixels[5]}")
 
                    plot_scat(np.asarray(thrs),np.asarray(chisq),"Pixel THR vs Pixel chi2",
                                "thr-vs-chisq-"+clean_filename,"",["THR","chisq"])                   
                    plot_scat(np.asarray(sigmas),np.asarray(chisq),"Pixel sigma vs Pixel chi2",
                                "sigma-vs-chisq-"+clean_filename,"",["sigma","chisq"])
                    plot_masked_scat(np.asarray(thrs),np.asarray(sigmas),2000,"Pixel TRH vs Pixel sigma",
                             "thr-vs-sigma-"+clean_filename,"", ["thr0","sigma"] )


                    # -----------------------------------------------------------------
                    # stacking histgrams
                    plotHistStack(
                        [np.asarray(thrs),np.asarray(thrs15)],
                        100,
                        [min([min(thrs),min(thrs15)]),max([max(thrs),max(thrs15)])],
                        ["THL at pdac=0 and pdac-15", "THL", "Nentries"],
                        ["THR=0", "THR=15"],
                        "comparing-thresholds-"+clean_filename,
                        True
                    )
 
                    # -----------------------------------------------------------------
                    # comparing dat between THR=0 & THR15

                    plotHistStack(
                        [np.asarray(mus),np.asarray(mus15)],
                        100,
                        getMinMaxFromList([mus,mus15]),
                        ["Mus @ pdac=0 and pdac=15", "\u03BC, [DAC]", "Nentries"],
                        ["THR=0", "THR=15"],
                        "comparing-fit-mu-bwin-0n15-"+clean_filename,
                        False
                    )
                   
                    plotHistStack(
                        [np.asarray(sigmas),np.asarray(sigmas15)],
                        100,
                        getMinMaxFromList([sigmas,sigmas15]),
                        ["Sigmas @ pdac=0 and pdac=15", "\u03C3, [DAC]", "Nentries"],
                        ["THR=0", "THR=15"],
                        "comparing-fit-sigma-bwin-0n15-"+clean_filename,
                        True
                    )

                    plotHistStack(
                        [np.asarray(chisq),np.asarray(chisq15)],
                        100,
                        getMinMaxFromList([chisq,chisq15]),
                        ["CHi2 @ pdac=0 and pdac=15", "\u03C7^2, [const float]", "Nentries"],
                        ["THR=0", "THR=15"],
                        "comparing-fit-chi2-bwin-0n15-"+clean_filename,
                        True
                    )

                    # -----------------------------------------------------------------
                                    
                    plotHistStack(
                        [np.asarray(sigmas),np.asarray(out_sigmas)],
                        100,
                        [min(sigmas),max(sigmas)],
                        ["Fit sigmas", "Fit \u03C3", "Nentries"],
                        ["All", "outliers"],
                        "comparing-sigmas-"+clean_filename,
                        True
                    )

                    # ------ outliers at thr=0 ----
                    simpleHist(np.asarray(out_x), 256, 0, 256,["outlier x positions", "pixel x", "#"], "-outlier-xpos-"+clean_filename, False)
                    simpleHist(np.asarray(out_y), 256, 0, 256,["outlier y positions", "pixel y", "#"], "-outlier-ypos-"+clean_filename, False)
                    #simpleHist(np.asarray(out_sigmas),50, min(out_sigmas)+1, max(out_sigmas)-1,["outlier fit sigmas", f"fit {GREEK_SIGMA}", "#"], "-outlier-sigmas"+clean_filename, False)
                    simpleHist(np.asarray(out_sigmas),13, -1, 10,["outlier fit sigmas", f"fit {GREEK_SIGMA}", "#"], "-outlier-sigmas-"+clean_filename, False)
                    simpleHist(np.asarray(out_mu), 50, min(out_mu)-1, max(out_mu)+1,[f"outlier estimated {GREEK_MU}", f"estimated {GREEK_MU}", "#"], "-outlier-mus-"+clean_filename, False)
                    simpleHist(np.asarray(out_thr), 50, min(out_thr)-1, max(out_thr)+1,[f"outlier measured THL @ {GREEK_MU}=0", "THL", "#"], "-outlier-thl-"+clean_filename, False)
                    simpleHist(np.asarray(out_chi), 50, min(out_chi)-1, max(out_chi)+1,[f"outlier estimated {GREEK_CHI}", f"estimated {GREEK_CHI}", "#"], "-outlier-chis-"+clean_filename, False)
                    simpleHist(np.asarray(out_eqdist), 50, min(out_eqdist)-1, max(out_eqdist)+1,[f"outlier equalisation distances", f"estimated {GREEK_DELTA}THR", "#"], "-outlier-deltaTHR-"+clean_filename, False)
                    # ------ outliers at thr=15 ----
                    simpleHist(np.asarray(out_x15), 256, 0, 256,["outlier x positions", "pixel x", "#"], "-outlier-xpos15-"+clean_filename, False)
                    simpleHist(np.asarray(out_y15), 256, 0, 256,["outlier y positions", "pixel y", "#"], "-outlier-ypos15-"+clean_filename, False)
                    simpleHist(np.asarray(out_sigmas15),13, -1, 10,["outlier fit sigmas", f"fit {GREEK_SIGMA}", "#"], "-outlier-sigmas15-"+clean_filename, False)
                    #simpleHist(np.asarray(out_mu15), 50, min(out_mu15)-1, max(out_mu15)+1,[f"outlier estimated {GREEK_MU}", f"estimated {GREEK_MU}", "#"], "-outlier-mus15-"+clean_filename, False)
                    simpleHist(np.asarray(out_thr15), 50, min(out_thr15)-1, max(out_thr15)+1,[f"outlier measured THL @ {GREEK_MU}=0", "THL", "#"], "-outlier-thl15-"+clean_filename, False)
                    simpleHist(np.asarray(out_chi15), 50, min(out_chi15)-1, max(out_chi15)+1,[f"outlier estimated {GREEK_CHI}", f"estimated {GREEK_CHI}", "#"], "-outlier-chis15-"+clean_filename, False)
                    # -----------------------------------------------------------------
 

                    plot_scat(np.asarray(mus),np.asarray(thrs),"Pixel MU vs Pixel THR",
                                "mu-vs-thr-"+clean_filename,"",["mu","THR"])                   
                    if(len(distances)>0):
                        plot_scat(np.asarray(distances),np.asarray(mus),"Pixel equalisation distance vs mu",
                                    "eqdist-vs-mu-"+clean_filename,"",["eq dist","mu"])                   
    
                        plot_scat(np.asarray(distances),np.asarray(sigmas),"Pixel equalisation distance vs fit sigma",
                                    "eqdist-vs-sigma-"+clean_filename,"",["eq dist","sigma"])                   

                    pixelMap2d(testmask,"testmask-BasedOn-mu-guess-zero")
                    pixelMap2d(testmask2,"testmaskBasedOn-distances")
                    pixelMap2d(eqdist,"distances")

                    ########### fooling around
                    arr_thr = np.asarray(thrs)
                    arr_sig = np.asarray(sigmas)
                    arr_chi = np.asarray(chisq)
                    mask_chi = (arr_chi> 100.0)
                    mask_rest = (arr_chi <= 100.0)
                    fig = plt.figure(figsize=(10,10))
                    ax = fig.add_subplot(111,projection='3d')                   
                    ax.scatter(arr_thr[mask_chi],
                               arr_sig[mask_chi],
                               arr_chi[mask_chi],s=10,c='red')

                    ax.scatter(arr_thr[mask_rest],
                               arr_sig[mask_rest],
                               arr_chi[mask_rest],s=10,c='blue')

                    ax.set_xlabel('threshold')
                    ax.set_ylabel('sigma')
                    ax.set_zlabel('chi2')
                    ax.view_init(15,135,0)
                    plt.savefig("3D-thr-sig-chi.png") 

                if(eqmap is not None):
                    pixelMap2d(eqmap, "EqualisationMap-"+clean_filename)

            elif("PixelDAC" in filename):

                if(hasInterpretedBranch(f,"ThresholdMap_th0_0")):

                    n_maps = countInterpretedBranches(f,"ThresholdMap") 
                    #print(f"found {n_maps} threshold maps")
                    ndacs = countNodes(f,"dac","cfg")
                    #print(f"found {ndacs} DAC configs")

                    daclist, keys = getDictList(f,ndacs,"dacs")
                    daclistlen = len(daclist)
                    #print("found keys: {}".format(keys))

                    print("________ checking DACs __________")
                    compareRegs(daclist, keys)
                    n_rc = countNodes(f,"run_config","cfg")
                    rc_list, rc_keys = getDictList(f,n_rc,"run_config")                    

                    #print(f"found {n_rc} run configs!")
                    print("________ checking run configs __________")
                    compareRegs(rc_list,rc_keys)

                    conf = dict(f.root.configuration.run_config_0[:])
                    Vstart = int(conf[b'Vthreshold_start'])
                    Vstop = int(conf[b'Vthreshold_stop'])
                    n_pulses = int(conf[b'n_injections'])
                    for i in range(n_maps // 2):# because half is for THR=0, and half is for THR=15
                        map0name = f"/interpreted/ThresholdMap_th0_{i}"
                        map15name = f"/interpreted/ThresholdMap_th0_{i}"

                        thrmap0 = f.get_node(map0name)[:]
                        thrmap15 = f.get_node(map15name)[:]
                        
                        pltname0 = f"PDAC-THRmap-0-it{i}-"   
                        pltname15 = f"PDAC-THRmap-15-it{i}-"   
 
                        pixelMap2d(thrmap0, pltname0+clean_filename)
                        pixelMap2d(thrmap15, pltname15+clean_filename)

                
                if(hasInterpretedBranch(f,"HistSCurve_th0_0")):
                    
                    n_scurves = countNodes(f,"scurves", "intp")
                    print(f"\nfound {n_scurves} in the file\n")

                    for curve in range(n_scurves // 2):

                        curve_0 = f.get_node(f"/interpreted/HistSCurve_th0_{curve}")[:].T
                        curve_15 = f.get_node(f"/interpreted/HistSCurve_th15_{curve}")[:].T

                        print(curve_0[:,256])

                        plt.figure()
                        plt.plot(curve_0[:,120])
                        plt.savefig("testScurve.png")

                        plot_scurves(curve_0, list(range(Vstart,Vstop)), title=f"scurve-at-th0-{curve}", scan_parameter_name="Vthreshold",max_occ=n_pulses*5,plotname=f"PDAC-scurves-0-{curve}-"+clean_filename)
                        plot_scurves(curve_15, list(range(Vstart,Vstop)), title=f"scurve-at-th15-{curve}", scan_parameter_name="Vthreshold",max_occ=n_pulses*5,plotname=f"PDAC-scurves-15-{curve}-"+clean_filename)


            elif("ToTCalib" in filename):

                if(hasInterpretedBranch(f,'HistToTCurve')):

                    these_pixels = np.zeros((256,256), dtype=np.uint16)

                    totCurves = f.root.interpreted.HistToTCurve[:].T
                    rand_pixels = np.random.randint(0,65535,size=20)
                    for i in rand_pixels:
                        
                        nonneg = np.count_nonzero(totCurves[:,i])
                        if(nonneg > 0):       
                            quickPlot(totCurves[:,i],[f"TOT curve, pixel-{i}", "VTP(fine)", "N TOT CTS [x25 ns]"], f"-testTOT-{i}-"+clean_filename)
                        print(f"Pixel {i} TOT curve - {nonneg} positive entries")
                                       


                #if(hasInterpretedBranch(f,'HistToTCurve_Count')):

                #if(hasInterpretedBranch(f,'HistToTCurve_Full')):

            else:


                occupancy = f.root.interpreted.HistOcc[:].T 
                mask = f.root.configuration.mask_matrix[:].T
              
                

                #print(f"\nOccupancy stats: mean={mean_occ}, median={med_occ}, stdev={std_occ}\n")

                pixelMap2d(mask, "InfileMaskMatrixMap-"+clean_filename)
                pixelMap2d(occupancy, "occMap-"+clean_filename)

                plotSurf(occupancy, "Surf-Occupancy-"+clean_filename)
        
                THR = 100;
        
                #noiseMask, _, _ = findNoisyPixels(occupancy, THR)
                noiseMask, occ_stats = findNoisyPixels(occupancy)
        
                pixelMap2d(noiseMask, "TempMaskMap-"+clean_filename)
    
                plotHist(occupancy.reshape(-1), 
                         100, 
                         np.min(occupancy),
                         np.max(occupancy),
                         1,
                         "pixel occupancy",
                         "Pixel occupancy",
                         ["n hits","n pixels"],
                         "Occupancy-1d-hist-"+clean_filename) 
 
                ##### shit for scurves
                n_pulses = None
                conf = dict(f.root.configuration.run_config[:])
                Vstart, Vstop = None, None
                if("TestpulseScan" in filename):
                    Vstart = int(conf[b'VTP_fine_start'])
                    Vstop = int(conf[b'VTP_fine_stop'])
                else:
                    Vstart = int(conf[b'Vthreshold_start'])
                    Vstop = int(conf[b'Vthreshold_stop'])
                if("NoiseScan" in filename):
                    n_pulses = 100
                else:
                    n_pulses = int(conf[b'n_injections'])
                #############
                    
                if(hasInterpretedBranch(f,"HistSCurve")):
                    #scurve = f.get_node(f"/interpreted/HistSCurve")[:].T
                    scurve = f.get_node(f"/interpreted/HistSCurve")[:]
                    print(scurve.shape) 
                    plot_scurves(scurve, list(range(Vstart,Vstop)), title=f"scurve", scan_parameter_name="Vthreshold",max_occ=n_pulses*20,plotname="THL-Scan-"+clean_filename)
                    

                if(hasInterpretedBranch(f,"NoiseCurveHits")):
                    
                    ncurve = f.get_node("/interpreted/NoiseCurveHits")[:].T
                    nz = np.count_nonzero(ncurve)                   
                    print("NoiseCurveHits - Found {} non zero entries".format(nz))
                    if(nz != 0):
                        quickPlot(ncurve, ["NoiseCurveiHits", "x","y"], "-NoiseCurveHits-"+clean_filename) 
                    
                if(hasInterpretedBranch(f,"NoiseCurvePixel")):

                    pcurve = f.get_node("/interpreted/NoiseCurvePixel")[:].T
                    nz = np.count_nonzero(pcurve)
                    print("NoiseCurvePixel - Found {} non zero entries".format(nz))
                    if(nz != 0):
                        quickPlot(pcurve, ["NoiseCurvePixel", "x","y"], "-NoiseCurvePixel-"+clean_filename)                     

        else:

            print("Did not find interpreted data in this dataset")

     elif(option=="hits"):

        if(testIfHasInterpreted(f) and hasInterpretedBranch(f,"hit_data")):

            hit_data = f.get_node('/interpreted/hit_data')
            print(type(hit_data))
            # probing info about the table:

            cols = hit_data.colnames
            nrows = hit_data.nrows
            descr = hit_data.description
            print(f"hit data structure:\n # column names ={cols},\n # rows={nrows},\n description={descr}")

            #print(cols)
            #for i in range(10):
            #    print(hit_data[:i])
            #    print("\n")

            hit_data = None

        else:
            print(f"File {filename} either has no /interpreted/ data branch or has no /interpreted/hit_data/ node!")

     else:
        print("------- File Structure --------")
        for group in f.walk_groups("/"):
            if "interpreted" in str(group):
               print("FOUND INTERPRETED DATA GROUP!")
            if("reconstruction" in str(group)):
               print("FOUND RECONSTRUCTION DATA GROUP!")
               if("Timepix" in str(group)):
                    print("Found Timepix!")
            print(group)
            #print(vars(group))
            print("======= Arrays in a group ===========")
            for array in f.list_nodes(group):
                print(array) 
        print("----------------------------------")
        print("You either did not give any valid option \n or did not type in anything")

f.close()
