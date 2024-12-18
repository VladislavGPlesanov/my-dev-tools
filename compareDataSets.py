import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tables as tb
import h5py

import os.path


############################################################

asci_red = "\033[1;31m"
asci_reset = "\033[0m"

def getCleanName(full_name):

    namelist = str(full_name).split('/')
    clean_name = namelist[len(namelist)-1]

    return clean_name

def addSpaces(string,mean_length):

    for i in range(mean_length):
        string+=" "    
    return string    


def checkDictionaries(keylist, dictlist):

    for key in keylist:
         params = []
         outstring = "{}:".format(key)
         if(len(outstring)<30):
            outstring = addSpaces(outstring,30-len(outstring))
         for i in range(nfiles):
            ival = dictlist[i][key]
            params.append(ival)
            if(key==b'maskfile'):
                splitword = str(ival).split('/')
                fname=splitword[len(splitword)-1]
                outstring+="\t{}\t".format(fname)
            else:
                outstring+="\t{}\t".format(ival)

         numbers = set(params)
         #print(params)
         #print(numbers)
         if(len(numbers)>1):
            outstring = asci_red+outstring+asci_reset
         print(outstring)
         params = None



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

action = str(sys.argv[1])
picname = sys.argv[2]
filelist = sys.argv[3:]

print(f"Got:  \naction={action}, \npicname={picname}, \nlist of files={filelist}")

actions = ["plot", "settings", "mask"]


if(os.path.isfile(picname) is True):
    print("Second argument must be <picname>! ")
    exit(0)

if(action not in actions):
    print("No procedure for action={}".format(action))
    exit(0)

if(len(filelist)==0):
    print("Did not find file list")

for file in filelist:
    if(os.path.isfile(file) is not True):
        print(f"could not find {file}")

if(action=="plot"):
    if(len(sys.argv) < 4):
        print("INCMPLETE list of arguments")
        print("USAGE: python3 compareDatasets.py <action> <picname> <file_0> ... <file_n>")
    
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

if(action=="settings"):

    #path = "home/vlad/Timepix3/scans/hdf/"

#    asci_red = "\033[1;31m"
#    asci_reset = "\033[0m"

    settings=[
        b'Vfbk',
        b'n_injections',
        b'Ibias_PixelDAC',
        b'Vthreshold_start',
        b'Vthreshold_stop',
        b'mask_step',
        b'tp_period',
        b'Ibias_Preamp_ON'
    ]
        
    configlist = []
    daclist = []
    genconfiglist = []
    fnames = []
    
    dac_keys = []
    config_keys = []    
    gconf_keys = []

    cnt = 0 
    for file in filelist:
        with tb.open_file(file) as f:
            config = None
            if("PixelDAC" in str(file)):
                 config = dict(f.root.configuration.run_config_0[:])
            else:
                 config = dict(f.root.configuration.run_config[:])
            genconf = dict(f.root.configuration.generalConfig[:])
            dacs = None
            if("PixelDAC" in str(file)):
                dacs = dict(f.root.configuration.dacs_0[:])
            else:
                dacs = dict(f.root.configuration.dacs[:])
            daclist.append(dacs)
            configlist.append(config)
            genconfiglist.append(genconf)
            if(cnt==0):
                dac_keys = list(dacs.keys())
                config_keys = list(config.keys())
                gconf_keys = list(genconf.keys())

        cnt+=1    

    clean_fnames = []
    for name in filelist:
       clean_fnames.append(getCleanName(str(name)))
    nfiles = len(filelist)
    print("checking DACS for:")
    # getting largest string in all dictionaries
    key_maxlen = 22 #characters

    i_name=0
    header = ""
    for name in clean_fnames:
        print("[{}]({})".format(i_name, name))
        header+="\t\t\t[{}]".format(i_name)
        i_name+=1
    print(header)

    checkDictionaries(dac_keys, daclist)

    print("checking config for:")
    checkDictionaries(config_keys, configlist)
    print("checking GenConfig for:")
    checkDictionaries(gconf_keys, genconfiglist)


if(action=="mask"):

    masklist = []
    filenames = []
    occupancies = []

    for file in filelist:    

        filenames.append(getCleanName(file))
        with tb.open_file(file) as f:

            if('/interpreted/HistOcc/' in f):
                occupancies.append(f.root.interpreted.HistOcc[:].T)
            else:
                print(f"did not find occupancy histograms for {file}")

            maskmat = f.root.configuration.mask_matrix[:].T
            masklist.append(maskmat)

        f.close()

    # ----------------------------------------

    nmasked = []
    for mask in masklist:

        nmasked.append(np.count_nonzero(np.ravel(mask)))

    occ_means, occ_stdevs = [],[]
    for occ in occupancies:

        occ_means.append(np.mean(occ))
        occ_stdevs.append(np.std(occ))
    

    # ----------------------------------------
    fig = plt.figure(figsize=(8,6))

    cnt = 0
    for ma in nmasked:
        
        plt.scatter(filenames[cnt], (float(ma)/65535)*100 , label=f"{filenames[cnt]}")
        cnt+=1

    cnt=0
    #plt.title("Number of masked channels")
    plt.title("number of masked channels")
    plt.xticks([])
    #plt.xlabel("files")
    plt.ylabel("% of chan masked")
    #plt.ylim(min(nmasked)-2, max(nmasked)+2)
    #plt.xticks(rotation=5)

    plt.legend()
    plt.grid(True)
    fig.savefig("Comparing-masked-channel-numbers-"+picname+".png")

    # ------------------------------------------
    # occupancies
    if(len(occupancies)!=0):
        # ------ occupancy means -----------
        fig = plt.figure(figsize=(8,6))
       
        for om in occ_means:
            plt.scatter(filenames[cnt], om, label=f"{filenames[cnt]}")
            cnt+=1
       
        cnt=0 
        plt.xticks([])
        plt.ylabel("mean occupancy")
        plt.legend()
        plt.grid(True)
    
        fig.savefig("comparing-Occupancies-"+picname+".png")
 
        # ------ occupancy stdevs ----------   
        fig = plt.figure(figsize=(8,6))
       
        for ostd in occ_stdevs:
            plt.scatter(filenames[cnt], ostd, label=f"{filenames[cnt]}")
            cnt+=1
        
        cnt=0 
        plt.xticks([])
        plt.ylabel("occupancy stdevs")
        plt.legend()
        plt.grid(True)
    
        fig.savefig("comparing-Occupancies-stdevs-"+picname+".png")
    
   


