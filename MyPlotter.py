import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import colors, cm
from matplotlib.colors import LogNorm
import tables as tb 
import sys
import shutil
import json

## for moniotor ##
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np

from collections import deque

########################

class myPlotter:
    def __init__(self):

        self.default_color = 'r'
        self.default_figsize = (8,8)


    def quickPlot(self, data, axis, picname):

        plt.figure()

        plt.plot(data, "bo")
        plt.title(axis[0])
        plt.xlabel(axis[1])
        plt.ylabel(axis[2])

        plt.savefig("quick_plot-"+picname+".png")

    def simpleHist(self, 
                   data, 
                   nbins, 
                   minbin, 
                   maxbin, 
                   labels, 
                   picname,
                   outdir=None,
                   figsize=None,
                   histcolor=None,
                   histtype=None,
                   autobins=None,
                   getStats=False, 
                   savePDF=False,
                   normalize=False,
                   ylog=False):

        fig = None
        if(figsize is not None):
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure(figsize=(8,7))

        defHistColor = histcolor if histcolor is not None else 'b'
        defHistType = histtype if histtype is not None else 'step'
        nbins_auto, sumcts_auto = -1, -1

        if(autobins is not None):
            # checking if the method in autobins is supported by np.histogram
            # or typed in correctly. Else use 
            defmethods = ["fd","auto","sturges","scott"]
            method = autobins.lower() if autobins.lower() in defmethods else "auto"

            autoCTS, autoBINS, _ = plt.hist(data, 
                                            bins=method, 
                                            histtype=defHistType, 
                                            facecolor=defHistColor)
            nbins_auto = len(autoBINS)-1
            sumcts_auto = np.sum(autoCTS)
        else:
            cts, edges, _ = plt.hist(data, 
                                    nbins, 
                                    range=(minbin,maxbin), 
                                    histtype=defHistType, 
                                    facecolor=defHistColor,
                                    density=normalize)

        if(getStats):
            mean = np.mean(data)
            median = np.median(data)
            stdevx = np.std(data)
            ymin,ymax = plt.gca().get_ylim()
            xmin,xmax = plt.gca().get_xlim()
            if(nbins_auto>0):
                plt.scatter([],[],c='white',label=f"AutoBins={nbins_auto}")
                plt.scatter([],[],c='white',label=r"$\sum$N"+f"={sumcts_auto}")
            else:
                integral = None
                if(type(cts[0]) is float):
                    integral = int(np.round(np.sum(cts)))
                else:
                    integral = np.sum(cts)

                plt.scatter([],[],c='white',label=f"UserBins={nbins}") 
                plt.scatter([],[],c='white',label=r"$\sum$N"+f"={integral}") 
            plt.vlines(mean, 0, ymax*0.9, linestyles=":", colors="darkblue", label=r"$\bar{\mathrm{x}}$"+f"={mean:.2f}")
            plt.vlines(median, 0, ymax*0.9, linestyles=":", colors="darkgreen", label=r"$\tilde{\mathrm{x}}$"+f"={median:.2f}")
            plt.vlines(mean+stdevx, 0, ymax*0.9, linestyles=":", colors="orange",label=r"$\sigma_{\mathrm{x}}$"+f"={stdevx:.2f}")
            plt.vlines(mean-stdevx, 0, ymax*0.9, linestyles=":", colors="orange")
            plt.legend()            

        plt.title(labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
    
        if(ylog):
            plt.yscale('log')

        basename = f"simpleHist-{picname}"
        if(outdir is not None):
            basename = outdir+basename   
        
        fig.savefig(basename+".png")
        if(savePDF):
            fig.savefig(basename+".pdf")

        fig = None
        plt.close()

    def simpleMultiHist(self,
                        datalist, # list of lists -> [str, list(float/int)], label + list
                        nbins, # int 
                        minbin, # int 
                        maxbin, # int 
                        labels, # list of strings
                        picname, # string
                        odir=None, # string
                        debug=False, # bool
                        figsize=None,
                        yscale=None, 
                        fNorm=False,
                        fNoTitle=False,
                        legendLabels=None,
                        ffit=None,
                        fitrange=None,
                        sizeTextAxis=None,
                        sizeTextAxisLabels=None): # fitrange float -> nsigmas from peak

        if(debug): 
            print(f"{OUT_MAGEN}[simpleMultiHist]-->{picname}-->{labels[0]} ({len(datalist)}) {OUT_RST}")

        pFigsize = (10,8)
        if(figsize is not None):        
            pFigsize = figsize
        plt.figure(figsize=pFigsize)

        cnt = 0
        for data in datalist:

            weights, counts = None, None
            if(debug):
                print(f"[simpleMultiHist]: have data set {len(data[1])}")

            counts, bin_edges = np.histogram(data[1], bins=nbins, range=(minbin,maxbin))
            if(fNorm):
                weights = counts/sum(counts)
            bin_cent = (bin_edges[:-1] + bin_edges[1:])/2
            if(debug):
                print(f"counts={counts}\nedges={bin_edges}")
                print(f"bin_cent={bin_cent}")

            
            #pmean, pamp = None, None
            #e_mean, e_amp = None, None
            # pre-estimating stats for fits:
            #idat_mean = np.mean(data[1]) 
            #idat_std = np.std(data[1])

            #nsigma = 3.0 if fitrange is None else fitrange
            #fRestrict = True if fitrange is not None else False

            #fitmask = (bin_cent > idat_mean-idat_std*nsigma) & ( bin_cent < idat_mean+idat_std*nsigma) if fRestrict else None

            #if(ffit is not None):
            #    pars = None
            #    if(fNorm):
            #        weights = counts/sum(counts)
            #        peakbin=getMaxBin(weights)
            #        p0 = [weights[peakbin], idat_mean, idat_std]
            #        bdry = ([0.0, 0.0, 0.0], 
            #                [np.inf, np.inf, np.inf])

            #        if(fRestrict):
            #            pars, pcov = curve_fit(gauss, bin_cent[fitmask], weights[fitmask], p0=p0, bounds=bdry, maxfev=100000)
            #        else: 
            #            pars, pcov = curve_fit(gauss, bin_cent, weights, p0=p0, bounds=bdry, maxfev=100000)

            #    else:
            #        peakbin=getMaxBin(counts)
            #        p0 = [counts[peakbin], idat_mean, idat_std]
            #        bdry = ([0.0, 0.0, 0.0], 
            #                [np.inf, np.inf, np.inf])
            #        if(fRestrict):
            #            pars, pcov = curve_fit(gauss, bin_cent[fitmask], counts[fitmask], p0=p0, bounds=bdry, maxfev=100000)
            #        else: 
            #            pars, pcov = curve_fit(gauss, bin_cent, counts, p0=p0, bounds=bdry, maxfev=100000)

            #    pmean, pamp, psigma = pars[1], pars[0], pars[2]
            #    cond_num = np.linalg.cond(pcov)
            #    print(f"[simpleMultiHist]: Checking condition number= {cond_num:.4f}\ncovariances=\n{pcov}")
            #    e_amp, e_mean, e_sigma = np.sqrt(np.diag(pcov))
            #    #if(debug):
            #    print(f"[simpleMultiHist]: Gauss fit:")
            #    print(f"A={pamp:.4f} +- {e_amp:.4f},\n{G_mu}={pmean:.4f} +- {e_mean:.4f}\n{G_sigma}={psigma:.4f} +- {e_sigma:.4f}")

            #
            ilabel = label=f"{data[0]}"
            #if(ffit is not None):
            #    ilabel+=", "+r"$\mu$="+f"{pmean:.2f}"+r"$\pm$"+f"{e_amp:.2f}, A={pamp:.2f}"+r"$\pm$"+f"{e_amp:.2f}"

            if(fNorm):
                if(legendLabels is not None):
                    plt.hist(bin_cent, weights=weights, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', alpha=0.2, label=legendLabels[cnt])
                else:
                    plt.hist(bin_cent, weights=weights, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', alpha=0.2, label=ilabel)
            else:
                if(legendLabels is not None):
                    plt.hist(bin_cent, weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', alpha=0.2, label=legendLabels[cnt])
                else:
                    plt.hist(bin_cent, weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', alpha=0.2, label=ilabel)
            #if(ffit is not None):
            #    plt.errorbar([pmean],[pamp],yerr=[e_amp],xerr=[e_mean], fmt='*', ecolor='black', capsize=8)

            cnt+=1

        if(fNoTitle):
            plt.title("")
        else:
            plt.title(labels[0])
        if(sizeTextAxisLabels is not None):
            plt.xlabel(labels[1],fontsize=sizeTextAxisLabels)
            plt.ylabel(labels[2],fontsize=sizeTextAxisLabels)
        else:
            plt.xlabel(labels[1])
            plt.ylabel(labels[2])
        if(sizeTextAxis is not None):
            plt.tick_params(axis='x',labelsize=sizeTextAxis)
            plt.tick_params(axis='y',labelsize=sizeTextAxis)

        plt.legend(loc='upper right')
        plt.grid(True)
        if(yscale is not None):# yscale = 'log' for instance
            plt.yscale(yscale)

        basename = f"MultiHist-{picname}.png"
        if(odir is not None):
            basename = odir+basename        

        plt.savefig(basename)
        plt.close()


    def plotScatter(self, 
                    xdata, # list  
                    ydata, # list 
                    labels, # list of len >=3 of type <str>
                    picname, # str
                    outdir=None,
                    figtype=None,
                    ufigsize=None,
                    ucolor=None,
                    umsize=None,
                    umarker=None,
                    limX = None,
                    limY = None,
                    fLogX = False,
                    fLogY = False,
                    fDebug=False):



        if(fDebug):
            print("assemblying plot for - [{}]".format(title)) 

        deffigsize = (9,8) if ufigsize is None else ufigsize
 
        fig, ax = plt.subplots(figsize=(deffigsize))

        defcolor = "royalblue" if ucolor is None else ucolor
        defmarksize = 8 if umsize is None else umsize
        defmark = "s"if umarker is None else umarker
        if(len(labels)>3):
            ax.scatter(xdata,ydata,color=defcolor,marker=defmark,s=defmarksize,label=labels[3])
        else:
            ax.scatter(xdata,ydata,color=defcolor,marker=defmark,s=defmarksize)

        ax.set_title(labels[0])
        ax.set_xlabel(labels[1])
        ax.set_ylabel(labels[2])

        if(fLogY):
           ax.set_yscale("log")
        if(fLogX):
           ax.set_xscale("log")
        if(limX is not None):
           ax.set_xlim(limX)  
        if(limY is not None):
           ax.set_ylim(limY)  

        ax.grid(True)

        basename = f"SCATT-{picname}."
        pictype = None
        if(figtype is not None):
            pictype = figtype
        else:
            pictype = "png"

        if(outdir is not None):
            if("/" not in outdir):
                outdir+="/"
            basename = outdir+basename+pictype

        fig.savefig(f"{basename}")      
        plt.close()


    def simple2Dhist(self, xdata,ydata, pltorange, plotname):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.hist2d(xdata, 
                   ydata, 
                   range= np.array([plotrange[0],plotrange[1]]),
                   bins=100)
                   #cmap="Greens")
        ax1.plot()
        fig.savefig(plotname+".png")

    def plot2Dhist(self, 
                    x, # numpy array
                    y, # numpy array
                    nbins, # number of bins
                    labels, # list of strings
                    picname, # string
                    odir=None, # string
                    figsize=None,
                    cutx=None,
                    cuty=None,
                    cmap=None,
                    fCutXgeq=False,
                    fCutYgeq=False,
                    fLogNorm=False,
                    fDebug=False):
     
        nentries_x = x.shape[0] 
        nentries_y = y.shape[0]
    
        if(fDebug):
            print(f"[plot2Dhist]--> {picname} --> {labels[0]} ({nentries_x},{nentries_y})")
        if(nentries_x==0 or nentries_y==0):    
            print(f"[ERROR::EMPTY_DATA] --> {picname} MISSING DATA")
            return 0
    
        mismatch = 0
        idx_stop = None
        if(nentries_x > nentries_y):
            idx_stop = nentries_y-1
            mismatch = nentries_y/nentries_x
            print(f"  MISMATCH in {picname} x={nentries_x}, y={nentries_y} -> {mismatch*100:.2f}%")
        elif(nentries_y > nentries_x):
            idx_stop = nentries_x-1
            mismatch = nentries_x/nentries_y
            print(f"  MISMATCH in {picname} x={nentries_x}, y={nentries_y} -> {mismatch*100:.2f}%")
        else:
            if(fDebug):
                print(f"  Data arrays equal") 
    
        if(mismatch>0.1):
            print(f"[plot2Dhist]: mismatch of {mismatch*100:.2f}% of data sets for {labels[0]}")
   
        fig = None
        if(figsize is not None):
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure(figsize=(8,6))

        idx, idy = None, None    
        if(cutx is not None):
            if(fCutXgeq):
                idx = np.where(x>=cutx)
            else:
                idx = np.where(x<cutx)

        if(cuty is not None):
            if(fCutYgeq):
                idy = np.where(y>=cuty)
            else:
                idy = np.where(y<cuty)

        usecmap = 'jet'
        if(cmap is not None):
            usecmap = cmap

        if(cutx is not None and cuty is None):
            if(fLogNorm): 
                plt.hist2d(x[idx], y[idx], bins=nbins, norm=LogNorm(), cmap=usecmap)
            else:
                plt.hist2d(x[idx], y[idx], bins=nbins, cmap=usecmap)

        elif(cuty is not None and cutx is None):
            if(fLogNorm):
                plt.hist2d(x[idy], y[idy], bins=nbins, norm=LogNorm(), cmap=usecmap)
            else:
                plt.hist2d(x[idy], y[idy], bins=nbins, cmap=usecmap)

        elif(cutx is not None and cutx is not None):
            x, y = x[idx], y[idx]
            if(fLogNorm):
                plt.hist2d(x[idy], y[idy], bins=nbins, norm=LogNorm(), cmap=usecmap)
            else:
                plt.hist2d(x[idy], y[idy], bins=nbins, cmap=usecmap)
    
        else:
            if(fLogNorm):
                plt.hist2d(x[0:idx_stop], y[0:idx_stop], bins=nbins, norm=LogNorm(), cmap="jet")
            else:
                plt.hist2d(x[0:idx_stop], y[0:idx_stop], bins=nbins, cmap="jet")
            

        plt.colorbar(label=r"$N_{Entries}$")
    
        plt.title(labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])

        basename = f"2DHist-{picname}.png"
        if(odir is not None):
            basename = odir+basename        

        plt.savefig(basename)
        plt.close()



    def plot_2dim(self, nuarray, parlist, title, axlabels, plotname):
    
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

    def pixelMap(self, nuArray, plotname):
    
        stats = None # list of floats with [mean,median,stdev]
        pixelmask = None # 2D numpy array later
        #these = ["occMap-", "checkin-event"]
        #if(these in plotname):
        if("occMap-" in plotname):
            _ , stats = findNoisyPixels(nuArray)       
         
        fig, ax = plt.subplots(figsize=(10,8))
        cax = fig.add_axes([0.86, 0.1, 0.05, 0.8])
        #ms = ax.matshow(nuArray)
        ms = ax.matshow(nuArray, cmap='viridis', vmin=30, vmax=98)
        fig.colorbar(ms, cax=cax, orientation='vertical')
        if(stats is not None):
            ax.text(-85,270,r'$occ_{mean}$'+f' = {stats[0]:.2f}',fontsize=10)
            ax.text(-85,280,r'$occ_{med}$'+f' = {stats[1]:.2f}',fontsize=10)
            ax.text(-85,290,r'$\sigma_{occ}$'+f' = {stats[2]:.2f}',fontsize=10)
    
        plt.plot()
        fig.savefig("pixelMap-"+plotname+".png")

    def plotMatrix(self,
                    nuarray, # numpy array (should be SQUARE)
                    picname, # string
                    outdir=None, # string 
                    figtype=None, # string
                    plotMarker=None, # list
                    uTitle=None, # user plot title (str)
                    matrix_lim=None, # number of columns/rows, list [ncol,nrow]
                    info=None,      #string, separated by ":"
                    infopos=None,  # list with 2 elements -> [x_start,y_start] 
                    figsize=None,  # tuple (i,j)
                    cmap=None,    # string
                    cbarname=None, # string
                    cbarmax=None, # int 
                    labels=None, # list of strings [str,str,str]
                    netblock=None, # plot a net of blocks with size nxn pixels
                    geomshapes=None, # list of matplotlib patches
                    sizeTextAxis=None,
                    sizeTextAxisLabels=None,
                    fLognorm=False,
                    fInvertYback=False,
                    fGrid=False,
                    fDebug=False):  
        
        if(fDebug):
            print(f"[plotMatrix]--> {picname} ({len(nuarray)})")
     
        if(np.sum(nuarray)==0):
            print(f"[ERROR::EMPTY_DATA] {picname} matrix all zeros!")
            return 0
    
        # ---- matrix 2d hist ----
        fig, ax  = None, None
        if(figsize is not None):
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=(8,7))
        cax = fig.add_axes([0.86,0.1,0.05,0.8])
        # ---------------------------------

        ms = None
            
        deftitle = "" if uTitle is None else uTitle
        deflabelx, deflabely = "x, [pixel]", "y, [pixel]"
        
        defcmap = 'jet'
        if(cmap is not None):
            defcmap = cmap
        
        defcbarname = 'TOT'

        if(fLognorm):
            cbarmin = 1
            cbarmax = np.nanmax(nuarray) if cbarmax is None else cbarmax
            ms = ax.matshow(nuarray, cmap=defcmap, norm=LogNorm(vmin=cbarmin,vmax=cbarmax))
        else:
            ms = ax.matshow(nuarray, cmap=defcmap)
        if(cbarname is not None):
            fig.colorbar(ms,cax=cax,orientation='vertical',label=cbarname)
        else:
            fig.colorbar(ms,cax=cax,orientation='vertical',label=defcbarname)

        if(netblock is not None):

            nblocks = nuarray.shape[0]//netblock
            xmax, ymax = nuarray.shape[0], nuarray.shape[1]
            if(xmax==ymax):
                for i in range(nblocks):
                    ax.hlines(i*netblock,0,ymax,color='indigo',ls="--",lw=1)
                    ax.vlines(i*netblock,0,xmax,color='indigo',ls='--',lw=1)

        if(labels is not None):
            ax.set_title(labels[0])
            ax.set_xlabel(labels[1])
            ax.set_ylabel(labels[2])
        else:
            ax.set_title(deftitle)
            ax.set_xlabel(deflabelx)
            ax.set_ylabel(deflabely)
        if(fGrid):
            ax.grid(c='black', ls=':',lw=0.4)
           
        if(sizeTextAxis is not None):
            ax.tick_params(axis='x', labelsize=sizeTextAxis)
            ax.tick_params(axis='y', labelsize=sizeTextAxis)

        if(sizeTextAxisLabels is not None): 
            ax.xaxis.label.set_fontsize(sizeTextAxisLabels)
            ax.yaxis.label.set_fontsize(sizeTextAxisLabels)

        startx, starty = -90, 123
        if(infopos is not None):
            startx, starty = infopos[0],infopos[1]

        if(info is not None):
            comments = info.split(":")
            iline = 0
            hspace = 8
            for com in comments:
                ax.text(startx, starty+iline*hspace, com, fontsize=9,color='black' )    
                iline+=1
        if(plotMarker is not None):
            markx = plotMarker[0]
            marky = plotMarker[1]
            ax.scatter([markx],[marky],c='red', marker='+',s=80)
        if(geomshapes is not None):
            for shape in geomshapes:    
                ax.add_patch(shape)


        ax.invert_yaxis()
        if(fInvertYback):           
            ax.invert_yaxis()

        defAxisLim = [0,256]
        if(matrix_lim is not None):
            ax.set_xlim([0,matrix_lim[0]])
            ax.set_ylim([0,matrix_lim[1]])
        else:
            ax.set_xlim([0,256])
            ax.set_ylim([0,256])

        basename = f"MATRIX-{picname}."
        pictype = None
        if(figtype is not None):
            pictype = figtype
        else:
            pictype = "png"

        if(outdir is not None):
            if("/" not in outdir):
                outdir+="/"
            basename = outdir+basename+pictype

        fig.savefig(f"{basename}")
      
        plt.close()



class myUtils:


    def __init__(self):
        self.times = deque(maxlen=(100))

    def safe_call(self,func,*args,**kwargs):
    
        try:
            func(*args,**kwargs)
        except Exception as ex:
            print(f"[myUtils::safe_call] Error: {ex}")

    def safe_division(self, num, denom, failreturn=-999):
        
        yoba = None
        try:
            yoba = num/denom
        except ZeroDivisionError:
            yoba = failreturn
        return yoba

    def progress(self, total, n):

        try:
            perc = round(float(n)/float(total)*100.0,2)
        except ZeroDivisionError:
            perc = 0.0
        finally:
            if(n==total):
                print(f"\r{perc}% done\n", end="",flush=True)
            else:
                print(f"\r{perc}% done", end="",flush=True)

    def progress_bar(self, iteration, total, time_est=None):

        # detecting terminal width
        term_width = shutil.get_terminal_size().columns

        # reserving space for text
        #barwidth = term_width - 40
        barwidth = term_width - 60
        barwidth = max(10,barwidth)

        frac = iteration/total
        filled = int(barwidth*frac)

        # unicode chars
        fchar = "\u2588"
        echar = "\u2591"

        bar = fchar*filled + echar*(barwidth-filled)
        perc = frac*100 

        msg = f"\r|{bar}| {perc:6.2f}% ({iteration}/{total})"
        #sys.stdout.write(f"\r|{bar}| {perc:6.2f}% ({iteration}/{total})")
        if(time_est is not None):
            estTimeLeft = time_est*(total-iteration)
            msg+=f",<tleft={estTimeLeft/60.0:.2f}[min]>"
        sys.stdout.write(msg)
        sys.stdout.flush()
        if(iteration==total):
            print("\n")

        

    def removePath(self,string):
        string_split = string.split('/')                                                                                                                                                                   
        clean_name = string_split[len(string_split)-1]
        clean_name = clean_name[:-3]
        return clean_name

    def torrToAtm(self, p_torr):

        return round(p_torr/760,2)

    def atmToTorr(self,p_atm):

        return round(p_atm*760,2)

    def getHistoRange(self, data, nsigma=5):
        # data must be 1D np.array([(int/float)])
        
        mean = np.mean(data)
        left = data[data<mean]
        right = data[data>=mean]
        sigmaleft = np.std(left)
        sigmaright = np.std(right)
        xmin = mean - nsigma*sigmaleft
        xmax = mean + nsigma*sigmaright

        return xmin, xmax

    def getBaseGroupName(self, f, debug=False):
   
        infile = tb.open_file(f,'r')
 
        groups = infile.walk_groups('/')

        infile.close()
        grouplist = []
        for gr in groups:
            print(f'found {gr}')
            grouplist.append(gr)
        main_group = str(grouplist[len(grouplist)-1])
        if(debug):
            print(f"last entry in walk_groups = \n{main_group}")
    
        grouplist = None 
    
        basewords = main_group.split('(')
        if(debug):
            print(basewords)
    
        base_group_name = basewords[0][:-1]+'/'
        #                              ^ removes space at the end of 'run_xxx/chip0 '
        if(debug):
            print(f'base group name is : <{base_group_name}>')
        #bgn_split = base_group_name.split('/')
        #if(debug):
        #    print(bgn_split)
        #run_name = bgn_split[2]
        #if(debug):
        #    print(f"<{run_name}>")
        #run_num = int(run_name[4:])
        #if(debug):
        #    print(f'run number is {run_num}')
    
        basewords = None
     
        return base_group_name

    def getJsonData(self, path):
    
        data = None
        try:
            content = open(path,'r')
            data = json.load(content)
        except Exception as ex:
            print(f"Failed to load json due to : {ex}")
            sys.exit(0)
    
        return data


       
class myColors:

    RST = '\033[0m'                                                                                                                                                                                    
    RED = '\033[1;31m'
    BLUE = '\033[1;34m'
    GREEN = '\033[1;32m'
    GRAY = '\033[1;37m'
    YELLOW = '\033[1;33m'                                                                        
    CYAN = '\033[1;36m'                                                                          
    MAGEN = '\033[1;35m'                                                                         
    RED_BGR = '\x1b[41m'                                                                         
    BLUE_BGR = '\x1b[44m'                                                                        
    GREEN_BGR = '\x1b[42m'                                                                       
    YEL_BGR = '\x1b[43m'                                                                         
    CYAN_BGR = '\x1b[46m'                                                                        
    WHITE_BGR = '\x1b[47m'                                                                       
    MAG_BGR = '\x1b[45m'       

class mySymbols:

    # =========================
    # Greek lowercase
    # =========================
    alpha = "\u03B1"
    beta = "\u03B2"
    gamma = "\u03B3"
    delta = "\u03B4"
    epsilon = "\u03B5"
    zeta = "\u03B6"
    eta = "\u03B7"
    theta = "\u03B8"
    iota = "\u03B9"
    kappa = "\u03BA"
    lambda_ = "\u03BB"
    mu = "\u03BC"
    nu = "\u03BD"
    xi = "\u03BE"
    pi = "\u03C0"
    rho = "\u03C1"
    sigma = "\u03C3"
    tau = "\u03C4"
    phi = "\u03C6"
    chi = "\u03C7"
    psi = "\u03C8"
    omega = "\u03C9"

    # =========================
    # Greek uppercase
    # =========================
    Gamma = "\u0393"
    Delta = "\u0394"
    Theta = "\u0398"
    Lambda = "\u039B"
    Xi = "\u039E"
    Pi = "\u03A0"
    Sigma = "\u03A3"
    Phi = "\u03A6"
    Psi = "\u03A8"
    Omega = "\u03A9"

    # =========================
    # Math operators
    # =========================
    plusminus = "\u00B1"
    times = "\u00D7"
    divide = "\u00F7"
    approx = "\u2248"
    neq = "\u2260"
    leq = "\u2264"
    geq = "\u2265"
    infinity = "\u221E"
    partial = "\u2202"
    nabla = "\u2207"
    integral = "\u222B"
    #sum = "\u2211"
    product = "\u220F"
    sqrt = "\u221A"

    # =========================
    # Arrows
    # =========================
    left = "\u2190"
    right = "\u2192"
    up = "\u2191"
    down = "\u2193"
    leftright = "\u2194"

    # =========================
    # Misc physics symbols
    # =========================
    degree = "\u00B0"
    angstrom = "\u212B"
    micro = "\u00B5"

class myMonitor:

    def __init__(self):

        #self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.resize(400,300)
        self.win.setWindowTitle("EBALA")
        self.win.show()

        # plot window/canvas
        self.plot = self.win.addPlot(title="Test Plot")
        self.plot.setLabel('left','Gain')
        self.plot.setLabel('bottom','Vgrid')
        self.plot.showGrid(x=True,y=True,alpha=0.3)
        #self.plot.setBackground('k')

        # scatter plot 
        self.scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255,100,100,200),
            size=10,
            symbol='s'
        )
        self.plot.addItem(self.scatter)

        # flag
        self._dirty = False

        # timer

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._refresh)
        self.timer.start(1000)

        # data containers
        self.xpoints = []
        self.ypoints = []
#        self.plot.show()

    def updateScatter(self,x,y):

        self.xpoints.append(x)
        self.ypoints.append(y)
        self._dirty = True
        

    def _refresh(self):
        if(self._dirty):
            self.scatter.setData(self.xpoints,self,ypoints)
            self._dirty = False


