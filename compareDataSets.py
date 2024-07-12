import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tables as tb
import h5py

############################################################
def plotMultiScat(xll, yll, picname,xlab,ylab,datalab):

    fig = plt.figure(figsize=(8,6))
    sub = fig.add_subplot(111)
    cntr = 0
    mrkr = None
    for l in dac_list:
        print(len(l))
    
    for l in disc_list:
        print(len(l))
    
    for x in xll:
        if((i+1)%2==0):
            mrkr = "+"
        else:
            mrkr = "X"
        print("iteration [{}], mrkr={}".format(cntr,mrkr))
    
        sub.scatter(x, yll[cntr], marker=mrkr, label=datalab[cntr])
    
        cntr += 1
        sub.set_xlabel(xlab)
        sub.set_ylabel(ylab)
        sub.set_yscale('log')
    
        
    sub.grid(which='major', color='grey', linestyle='-', linewidth=0.25)
    sub.grid(which='minor', color='grey', linestyle='--', linewidth=0.125)
    sub.minorticks_on()
    plt.legend()
    sub.plot()
    
    fig.savefig("comparison-"+picname+".png")

########################################################

picname = sys.argv[1]
filelist = sys.argv[2:]

if(len(filelist)<1):
    print("ARGUMENT LIST TOO SHORT")
    print(filelist)

else:
    print("ARGUMENT LIST of acceptable len")
    print(filelist)


dac_list = [ [] for i in range(len(filelist)) ]
unq_dac_list = [ [] for i in range(len(filelist)) ]

rxfifo_list = [ [] for i in range(len(filelist)) ]
disc_list = [ [] for i in range(len(filelist)) ]
avg_disc_list = [ [] for i in range(len(filelist)) ]
avg_rxfifo_list = [ [] for i in range(len(filelist)) ]
rate_list = [ [] for i in range(len(filelist)) ]
dec_list = [ [] for i in range(len(filelist)) ]
recwords_stop = [ [] for i in range(len(filelist)) ] 

runlabels = []

filenr = 0
for file in filelist:
    
    with tb.open_file(file) as f:

        rconfig = f.root.configuration.run_config[:]
        gencfg = f.root.configuration.generalConfig[:]       

        runname = str(rconfig[1][1])
        runname = runname.split('\'')[1]
        runlabels.append(runname[13:])

        for i in range(0, len(rconfig)):
            print(rconfig[i])
        print("------------------")

        for i in range(0, len(gencfg)):
            print(gencfg[i])
        print("------------------\n\n")

        mdata = f.root.meta_data[:].T 

        prev_dac = None
        ndacs = 0
        disc_sum = 0
        datasum = 0
        fifosum = 0
        dt = 0.0
        t0, tend = 0,0 

        for i in range(0,len(mdata)):
            dac = mdata[i][5]
            disc = mdata[i][6]
            dec = mdata[i][7]
            nwords = mdata[i][1]
            datalen = mdata[i][2]
            t_start = mdata[i][3]
            t_stop = mdata[i][4]
            nrxfifo = mdata[i][9]

            #if(filenr==0 and dac==1050):
            #   print("nw={},dlen={},t0={},tend={},".format(nwords,datalen,t_start,t_stop))

            if(prev_dac == 0):
                prev_dac = dac
                #t0 = t_start
            if(prev_dac == dac):
                disc_sum += disc
                datasum += datalen
                fifosum += nrxfifo
                ndacs += 1
            if(prev_dac != dac):
                #tstart = mdata[i-ndacs-1][3]
                #tend = mdata[i-1][4]
                # calculate shit
                avg_disc = 0
                avg_fifo = 0
                try:
                    avg_disc = round(float(disc_sum)/float(ndacs),2)
                except ZeroDivisionError:
                    pass
                try:
                    avg_fifo = round(float(fifosum)/float(ndacs),2)
                except ZeroDivisionError:
                    pass

                # set calc values to lists

                #print("lookin\' at times {} - {} => dt={}".format(tend, tstart, tend-tstart))
                #print("lookin\' at datasize {}".format(datasum))
                #print("________________________________________")    
                avg_disc_list[filenr].append(avg_disc)
                avg_rxfifo_list[filenr].append(avg_fifo)
                unq_dac_list[filenr].append(prev_dac)
                rate_list[filenr].append((datasum/2.0)*32)
                disc_sum = 0
                datasum = 0
                fifosum = 0
                ndacs = 0
                # reset summation
                prev_dac = dac
                disc_sum += disc
                datasum += datalen
                fifosum += nrxfifo

            rxfifo_list[filenr].append(nrxfifo)
            dac_list[filenr].append(dac)
            disc_list[filenr].append(disc)
            dec_list[filenr].append(dec)
            recwords_stop[filenr].append(nwords)

    filenr+=1

######################################################

plotMultiScat(dac_list,disc_list, "discarErr-"+picname, "DAC","N discard errors",runlabels)
plotMultiScat(dac_list,dec_list, "decodeErr-"+picname, "DAC","N decode errors",runlabels)
plotMultiScat(dac_list,recwords_stop, "wordsRecStop"+picname, "DAC","stop index (words recorded)",runlabels)
plotMultiScat(unq_dac_list,avg_disc_list,"avg-discard-"+picname,"unique DAC","N discard errors (AVG)",runlabels)
plotMultiScat(unq_dac_list,rate_list,"rate-"+picname,"unique DAC","hitrate [Mbit]",runlabels)
plotMultiScat(dac_list,rxfifo_list,"RX-FIFO-sizes-"+picname,"DAC","RX FIFO size [cts]",runlabels)
plotMultiScat(unq_dac_list,avg_rxfifo_list,"AVG-RX-FIFO-sizes-"+picname,"DAC","average RX FIFO size [cts]",runlabels)

