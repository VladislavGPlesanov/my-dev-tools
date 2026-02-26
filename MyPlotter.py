import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import colors, cm
from matplotlib.colors import LogNorm
import tables as tb 


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
                   getStats=False, 
                   savePDF=False,
                   ylog=False):

        fig = None
        if(figsize is not None):
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure(figsize=(8,7))


        plt.hist(data, nbins, range=(minbin,maxbin), histtype='step', facecolor='b')
        if(getStats):
            mean = np.mean(data)
            median = np.median(data)
            stdevx = np.std(data)
            ymin,ymax = plt.gca().get_ylim()
            xmin,xmax = plt.gca().get_xlim()
            plt.vlines(mean, 0, ymax*0.9, linestyles=":", colors="darkblue", label=f"mean={mean:.2f}")
            plt.vlines(median, 0, ymax*0.9, linestyles=":", colors="darkgreen", label=f"median={median:.2f}")
            plt.vlines(mean+stdevx, 0, ymax*0.9, linestyles=":", colors="orange",label=r"$\sigma=$"+f"{stdevx:.2f}")
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


    def plotScatter(self, xdata, ydata, title, plotname, label, axisnames):

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
                    info=None,      #string, separated by ":"
                    infopos=None,  # list with 2 elements -> [x_start,y_start] 
                    figsize=None,  # tuple (i,j)
                    cmap=None,    # string
                    cbarname=None, # string
                    labels=None, # list of strings [str,str,str]
                    geomshapes=None, # list of matplotlib patches
                    fLognorm=False, 
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
            
        deftitle, deflabelx, deflabely = "Matrix", "x, [pixel]", "y, [pixel]"
        
        defcmap = 'jet'
        if(cmap is not None):
            defcmap = cmap
        
        defcbarname = 'TOT'

        if(fLognorm):
            #ms = ax.matshow(nuarray.T, cmap=defcmap, norm=LogNorm(vmin=1,vmax=np.nanmax(nuarray)))
            ms = ax.matshow(nuarray, cmap=defcmap, norm=LogNorm(vmin=1,vmax=np.nanmax(nuarray)))
        else:
            #ms = ax.matshow(nuarray.T, cmap=defcmap)
            ms = ax.matshow(nuarray, cmap=defcmap)
            ax.set_title("Pixel occupancy (Beam profile)")
        if(cbarname is not None):
            fig.colorbar(ms,cax=cax,orientation='vertical',label=cbarname)
        else:
            fig.colorbar(ms,cax=cax,orientation='vertical',label=defcbarname)

        if(labels is not None):
            ax.set_title(labels[0])
            ax.set_xlabel(labels[1])
            ax.set_ylabel(labels[2])
        else:
            ax.set_title(deftitle)
            ax.set_xlabel(defxlabel)
            ax.set_ylabel(defylabel)
        if(fGrid):
            ax.grid(c='black', ls=':',lw=0.4)
            
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
        ax.set_xlim([0,256])
        ax.set_ylim([0,256])

        basename = f"MATRIX-{picname}."
        pictype = None
        if(figtype is not None):
            pictype = figtype
        else:
            pictype = "png"

        if(outdir is not None):
            basename = outdir+basename+pictype

        fig.savefig(f"{basename}")
      
        plt.close()



class myUtils:

    def progress(self, total, n):

        try:
            perc = round(float(n)/float(total)*100.0,2)
        except ZeroDivisionError:
            perc = 0.0
        finally:
            print(f"\r{perc}% done", end="",flush=True)

    def removePath(self,string):
        string_split = string.split('/')                                                                                                                                                                   
        clean_name = string_split[len(string_split)-1]
        clean_name = clean_name[:-3]
        return clean_name

    def torrToAtm(p_torr):

        return round(p_torr/760,2)

    def atmToTorr(p_atm):

        return round(p_atm*760,2)

    def getBaseGroupName(f, debug=None):
   
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

#class myColors:
#
#    def _init_(self):
#
#        self.bgr_col = None
#        self.text_col = None
#        self.colordict = {
#            
#            
#        }
#
#    def set
#
#    # put these into a helper class
#    #"\033[1;37;40m \033[2;37:40m TextColour BlackBackground  
#    #"\033[1;30;40m Dark Gray      \033[0m 1;30;40m           
#    #"\033[1;31;40m Bright Red     \033[0m 1;31;40m           
#    #"\033[1;32;40m Bright Green   \033[0m 1;32;40m           
#    #"\033[1;33;40m Yellow         \033[0m 1;33;40m           
#    #"\033[1;34;40m Bright Blue    \033[0m 1;34;40m           
#    #"\033[1;35;40m Bright Magenta \033[0m 1;35;40m           
#    #"\033[1;36;40m Bright Cyan    \033[0m 1;36;40m           
#    #"\033[1;37;40m White          \033[0m 1;37;40m           
#    #
#    #        TextColour GreyBackground                
#    # \033[0;30;47m Black      \033[0m 0;30;47m       
#    # \033[0;31;47m Red        \033[0m 0;31;47m       
#    # \033[0;32;47m Green      \033[0m 0;32;47m       
#    # \033[0;33;47m Brown      \033[0m 0;33;47m       
#    # \033[0;34;47m Blue       \033[0m 0;34;47m       
#    # \033[0;35;47m Magenta    \033[0m 0;35;47m       
#    # \033[0;36;47m Cyan       \033[0m 0;36;47m       
#    # \033[0;37;40m Light Grey \033[0m 0;37;40m       
#    #
#    #WhiteText ColouredBackground\033[0;37;40m\n"
#    #\033[0;37;41m Black      \033[0m 0;37;41m"
#    #\033[0;37;42m Black      \033[0m 0;37;42m"
#    #\033[0;37;43m Black      \033[0m 0;37;43m"
#    #\033[0;37;44m Black      \033[0m 0;37;44m"
#    #\033[0;37;45m Black      \033[0m 0;37;45m"
#    #\033[0;37;46m Black      \033[0m 0;37;46m"
#    #\033[0;37;47m Black      \033[0m 0;37;47m"
#    #\033[0;37;48m Black      \033[0m 0;37;48m"
#

# other helper classes later....
#class hdfHelper(object):
#
#    def __init__():
#  
