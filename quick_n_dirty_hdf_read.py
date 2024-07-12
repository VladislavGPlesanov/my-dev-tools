#import pydoc

import sys
import argparse
import h5py
import numpy as np
#import pandas as pd
import tables as tb
from time import sleep
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt 
from collections import Counter
import statistics as stat
#from scipy import optimize as opt

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

def testIfHasInterpreted(file):
    if '/interpreted/' not in f:
        print("NAW INTERPRETED DATA MATE!")
        return False
    else:
        #file.list_nodes('/interpreted/')
        print("YISS, its here!")
        return True

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
       ax1.set_ylim(min(ydata),max(ydata))
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

def pixelMap2d(nuArray, plotname):
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.86, 0.1, 0.05, 0.8])
    ms = ax.matshow(nuArray)
    fig.colorbar(ms, cax=cax, orientation='vertical')
    plt.plot()
    fig.savefig("IMG-2d-"+plotname+".png")

def findNoisyPixels(nuArray, threshold):
    
    badPixels = (nuArray > threshold).astype(int) * 2 - 1

    return badPixels

def plotHist(xlist, nbins, minrange, maxrange, transparency, label, title, axislabel, plotname):
    #plt.figure(2,figsize=(10,10))
    plt.figure()

    counts, edges, bars = plt. hist(xlist, 
                                    bins=nbins, 
                                    range=(minrange,maxrange),
                                    alpha=transparency,
                                    label=label)
    plt.title(title)
    plt.xlabel(axislabel[0])
    plt.ylabel(axislabel[1])
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(plotname+"-hist.png")
 

######### funcs end here! ###########

filename = str(sys.argv[1])
option = str(sys.argv[2])

clean_filename = removePath(filename)

with tb.open_file(filename, 'r') as f:

     if(option=="mask"):
        mask = f.root.configuration.mask_matrix[:].T
        pixelMap2d(mask,"maskMap-"+clean_filename) 
     elif(option=="thr"):
        if("_equal_" in filename):
            thr = f.root.thr_matrix[:].T
            print(thr.shape) 
            pixelMap2d(thr,"thresholdMap-"+clean_filename) 
        else:
            thr = f.root.configuration.thr_matrix[:].T
            print(thr.shape) 
            pixelMap2d(thr,"thresholdMap-"+clean_filename) 
     elif(option=="links"):
        links = f.root.configuration.links[:].T
        for i in range(0,7):
           print("{}={}".format(links[i][7],links[i][6]))
     elif(option=="config"):
        conf = f.root.configuration.run_config[:]
        for i in range(0,len(conf)):
          print(conf[i])
     elif(option=="genconfig"):
        genconf = f.root.configuration.generalConfig[:]
        for i in range(0,len(genconf)):
          print(genconf[i])
     elif(option=="dacs"):
        dacs = f.root.configuration.dacs[:]
        for i in range(0,len(dacs)):
           print(dacs[i])
     elif(option=="mdata"):
        mdata = f.root.meta_data[:].T 
        scurves = None
        if("ThresholdScan" in filename):
            scurves = f.root.interpreted.HistSCurve[:].T
            print("found scurve data - {}, {}".format(type(scurves), scurves.shape))
            print(scurves.shape[1])
            #sleep(2)
        print(len(mdata))
        for i in range(0,20):
           print("DL={},\tscanParID={}".format(mdata[i][2],mdata[i][5]))


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
        
        #sleep(2)
        for i in range(0,len(mdata)):
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
           scanid.append(mdata[i][5])
           rx_fifo.append(mdata[i][9])

           bitrate = mdata[i][2]*32/t_interval

           dac_n_bitrate.append([bitrate, mdata[i][5]])
           dac_n_discard.append([mdata[i][6], mdata[i][5]])
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
                avg_fifo, _ = getAvgList(unq_dac, dac_n_fifo_comb)
                print(len(avg_fifo))

                print("Obtaining average n hits")
                avg_hits, tot_unq_hits = getAvgList(unq_dac, dac_n_hits)
                avg_bitrate, _= getAvgList(unq_dac, dac_n_bitrate)

                print("Obtaining average n discard")
                avg_discard, _ = getAvgList(unq_dac, dac_n_discard)                

                print(unq_dac[:10])
                print(dac_n_fifo_comb[:10])
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
   
            #ploth2d(np.asarray(scanid), np.asarray(rx_fifo),clean_filename+"-fifo-size") 
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
        rdata = f.root.raw_data[:].T 
        print("raw data length:{}".format(len(rdata)))

     elif(option=="idata"):
        testIfHasInterpreted(f)
        occupancy = f.root.interpreted.HistOcc[:].T
        pixelMap2d(occupancy, "occMap-"+clean_filename)

        THR = 100;

        noiseMask = findNoisyPixels(occupancy, THR)

        pixelMap2d(noiseMask, "TempMaskMap-"+clean_filename)

     else:
        print("------- File Structure --------")
        for group in f.walk_groups("/"):
            if "interpreted" in str(group):
               print("FOUND INTERPRETED DATA GROUP!")
            print(group)
            print("======= Arrays in a group ===========")
            for array in f.list_nodes(group):
                print(array) 
        print("----------------------------------")
        print("You either did not give any valid option \n or did not type in anything")

f.close()
