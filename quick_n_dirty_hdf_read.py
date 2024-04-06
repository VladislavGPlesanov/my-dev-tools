#import pydoc

import sys
import argparse
import h5py
import numpy as np
#import pandas as pd
import tables as tb
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt 
from collections import Counter


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

def fillArray(inputData, data_index):

    dataList = []

    pos = 0
    for i in range(0,len(inputData)):
       for j in range(len(inputData[i])):
           if(j == data_index):
              #np.put(dataArray, inputData[i][j], pos)
              dataList.append(inputData[i][j])
              pos+=1
              #if(i<15):
              #    print("put {} from {}".format(inputData[i][j],inputData[i]))
           else:
              continue
    numArray = np.asarray(dataList)
    return numArray

def plot_scat(xdata, ydata, plotname, label, axisnames):
   
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(xdata,ydata, s=10,c='b',marker="s",label=label)
    plt.legend(loc='upper center')
    if(max(ydata)>5000):
       ax1.set_yscale("log")
    if max(xdata) > 5000:
       ax1.set_xscale("log")
    ax1.set_xlabel(axisnames[0])
    ax1.set_ylabel(axisnames[1])
    ax1.plot()
    fig.savefig(plotname+'.png')

def plot_scat_stack(xdata, ydatalist, plotname, lablist, axisnames):

    nUniqueDACs = set(xdata)
    #print("OLO:{}".format(nUnique))
    #print("OLO:{}".format(len(nUnique)))
    #ntrials = len(ydatalist[1])/len(nUnique)
           
    if(len(ydatalist)<6):
        fig = plt.figure()
        #plt.rcParams.update({"text.usetex":True})
        ax1 = fig.add_subplot(111)
        cnt = 0
        clist = ['r','g','b','y','m']
        markList = ['+','x','o','v','D']
        for ydata in ydatalist:
           #ax1.scatter(xdata,ydata, s=10, c=clist[cnt],marker='.',label=lablist[cnt])
           ax1.scatter(xdata,ydata, s=10, c=clist[cnt],marker=markList[cnt],label=lablist[cnt])
           cnt+=1
        if max(ydatalist[0]) > 5000:
            ax1.set_yscale("log")
        if max(xdata) > 5000:
            ax1.set_xscale("log")
        ax1.set_xlabel(axisnames[0])
        ax1.set_ylabel(axisnames[1])
        #ax1.set_ylim(-1000,2100)
        #if("timestamps" in plotname):
        #    ax1.set_ylim([0,20])

        #ax1.text(0.1,0.8,'YIPEEKAYEY \nfornicator of \nthy motherhood!',transform=ax1.transAxes)
        #ax1.axvline(x = 1275, 
        #           ymin = -250,
        #           ymax = 250) 
        #           #colors = 'magenta', 
        #           #label = 'more than 1RX is full')
        ax1.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper center')
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

 

######### funcs end here! ###########

#parser = argparse.ArgumentParser(description="argument parser")

filename = str(sys.argv[1])
option = str(sys.argv[2])

clean_filename = removePath(filename)


#runtype = getRunType(filename)

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
        print(len(mdata))
        for i in range(0,9):
           print(mdata[i])

        #scanid = []
        #idx_start = []
        #x = []
        #dec = []
        #dis = []
        #datalen = []
        #tstamp_0 = []       

        scanid, idx_start, x, dec, dis, datalen, tstamp_0, tstamp_end = ([] for i in range(8))

        testcnt = 0

        print(clean_filename)

        arr_scanid = fillArray(mdata, 5)
        arr_x = fillArray(mdata, 1)
        arr_idx_start = fillArray(mdata, 0)
        arr_disc = fillArray(mdata, 7)
        arr_deco = fillArray(mdata, 6)
        arr_dlen = fillArray(mdata, 2)

        #ploth2d(arr_dlen,arr_deco, "H2D-"+clean_filename)
        #ploth2d(arr_disc,arr_deco, "H2D-"+clean_filename)

        for i in range(0,len(mdata)): 
           idx_start.append(mdata[i][0])
           x.append(mdata[i][1])
           dec.append(mdata[i][7])
           dis.append(mdata[i][6]*(-1))
           datalen.append(mdata[i][2])
           #tstamp_0.append(mdata[i][3]*10e-9)
           #tstamp_0.append(np.float64(mdata[i][3])*10e-9)
           tstamp_0.append(np.float64(mdata[i][3]))
           #tstamp_end.append(np.float64(mdata[i][4])*10e-9)
           tstamp_end.append(np.float64(mdata[i][4]))
           scanid.append(mdata[i][5])
            
        ##########
        plot_scat_stack(x, #scanid,
                       [dec,dis],
                       clean_filename+"_DecDiscErrors",
                       ["decoding","discard"],
                       ["scan parameter ID","Errors,[N]"])
        #########

        plot_scat(x, #scanid,
                  datalen,
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

        if("DataTake" in filename):
            #plot_scat(idx_start,
            plot_scat(x, #tstamp_0,
                      datalen,
                      clean_filename+"_DataLength-vs-iterationIndex",
                      "len(Data)",
                      ["Start Index","Data length"])

            #plot_scat(idx_start,
            plot_scat(x,#tstamp_0,
                      dec,
                      clean_filename+"_DecodingErr-vs-iterationIndex",
                      "Decoding Err.",
                      ["Start Index","Decoding Errors [N]"])

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

#pydoc.writedoc('quickAndDirtyHDF')
