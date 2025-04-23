import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import colors, cm

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
                   ylog=False):

        plt.figure()
    
        plt.hist(data, nbins, range=(minbin,maxbin), histtype='step', facecolor='b')
    
        plt.title(labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
    
        if(ylog):
            plt.yscale('log')
    
        plt.savefig("simpleHist-"+picname+".png")

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

def getBaseGroupName(f, debug=None):

    groups = f.walk_groups('/')
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



# other helper classes later....
#class hdfHelper(object):
#
#    def __init__():
#  
