import numpy
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



# other helper classes later....
#class hdfHelper(object):
#
#    def __init__():
#  
