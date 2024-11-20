import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tables as tb
from sklearn.cluster import DBSCAN 
from scipy.optimize import curve_fit
from lmfit import Model


###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

######
def cosfunc(x, Ap, Bp, phi0):

    return Ap + Bp*np.cos(x - phi0)**2

def modFac(Ap,Bp):

    return Bp/(2*Ap+Bp)

def polDeg(Ap, Bp, mu):

    return (1/mu)*(Bp/((2*Ap)+Bp))

def check_node(gr_path, file):

    return gr_path in file

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)

def multiHist(nuarray_list, nbins, binbounds, labels, legends, logy ,picname, odir):

    plt.figure()

    cnt = 0 
    for arr in nuarray_list:
        plt.hist(arr, nbins, range=(binbounds[0],binbounds[1]), alpha=0.125, label=f"{legends[cnt]}")
        cnt+=1

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    if(logy):
        plt.yscale('log')
    plt.legend()
    plt.savefig(f"{odir}/multiHist-{picname}.png")

   

def simpleHist(nuarray, nbins, minbin, maxbin, labels, picname, odir,fit=None):

    plt.figure()

    if(fit):
        plt.figsize=(8,8)

    counts, bin_edges = np.histogram(nuarray,bins=nbins,range=(minbin,maxbin))

    #plt.hist(nuarray, nbins, range=(minbin,maxbin), histtype='stepfilled', facecolor='b')
    plt.hist(bin_edges[:-1], weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', facecolor='b')
    if(fit):
        maxbin_cnt = np.max(counts)
        minbin_cnt = np.min(counts)
        bin_centers = (bin_edges[:-1]+ bin_edges[1:])/2

        ########################################################            
        model = Model(cosfunc)
        pars = model.make_params(Ap=minbin_cnt*0.9,Bp=maxbin_cnt*1.1,phi0=1.5)
        pars['Ap'].min = minbin_cnt*0.8
        #pars['Ap'].max = minbin*1.5
        #pars['Bp'].min = minbin_cnt*1.1
        #pars['Bp'].max = maxbin*1.5
        pars['phi0'].min = 0.0
        pars['phi0'].max = 3.14

        print(f"\n\ncounts={counts} \n\n")
        print(maxbin_cnt)
        print(minbin_cnt)
        print(f"max_amplitude={maxbin_cnt - minbin_cnt}")

        result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
        print(result.fit_report()) 
        ########################################################            

        fitlab = ""
        Ap = result.params['Ap'].value
        Bp = Ap + result.params['Bp'].value
        phi = result.params['phi0'].value
        fitlab+=f"Ap={round(Ap,2)}, Bp={round(Bp,2)}, {G_phi}={round(phi,2)}"
        plt.plot(bin_centers[:-1], result.best_fit, '--r')
        plt.hlines(Ap, -0.1, 3.15, colors='g', label=f"Ap={round(Ap,2)}")
        plt.hlines(Bp, -0.1, 3.15, colors='y', label=f"Bp={round(Bp,2)}")
        plt.vlines(phi, 0, np.max(counts)*1.2, colors='m',label=f"{G_phi}={round(phi,2)}")
        ax = plt.gca()
        miny,maxy = ax.get_ylim()
        mu = modFac(Ap,Bp)
        P = polDeg(Ap,Bp,mu)
        plt.text(0.05, maxy*0.9, f"{G_mu}={round(mu*100,2)}%, P={round(P,2)}")
        plt.text(0.05, maxy*0.95, r"$N(\phi) = A_{P} + B_{P}\cdot cos^2(\phi-\phi_{0})$")
        plt.legend()

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.grid()
    plt.savefig(f"{odir}/1DHist-{picname}.png")


def plot2Dhist(x,y,labels,picname,odir):
    
    plt.figure()

    #hist,xedges,yedges = np.histogram2d(x,y,bins=100)

    #plt.imshow(hist.T, 
    #            origin="lower", 
    #            cmap="plasma", 
    #            extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    #plt.colorbar(label="cnt")
    plt.hist2d(x,y,bins=(51,51),cmap="plasma")

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.xlim(0,2)
    plt.ylim(0,1)
    plt.plot()
    plt.savefig(f"{odir}/2Dhist-{picname}.png")


def plotDbscan(index, dblabels, ievent, nfound, odir):

    #plt.figure(figsize=(8,8))
    plt.figure()
    plt.scatter(index[:, 0], index[:, 1], c=labels, cmap='plasma', s=10)
    plt.gca().invert_yaxis() 
    plt.gca().invert_xaxis() 
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0,64)
    plt.ylim(0,64)
    plt.title(f"DBSCAN on smeared matrix event {ievent}")
    plt.text(2,55 , f"n hits = {len(index)},\n ncluster={nfound}", fontsize=10, color="black")
    plt.colorbar(label='Cluster Label')
    plt.savefig(f"{odir}/DBSCAN-{ievent}.png")


def plot2dEvent(nuarray, info, picname, odir, plotMarker=None):
 
    # ---- matrix 2d hist ----
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.86,0.1,0.05,0.8])
    #ms = ax.matshow(nuarray, cmap='plasma')
    ms = ax.matshow(nuarray, cmap='hot')
    fig.colorbar(ms,cax=cax,orientation='vertical')

    start = 120
    ax.text(-90, start, info, fontsize=10,color='black' )    
    if(plotMarker is not None):
        ax.scatter(plotMarker[0],plotMarker[1],c='blue', marker='*')
        ax.text(-90,200, f"redX={plotMarker[0]}\nredY={plotMarker[1]}", fontsize=10,color='black')

    #plt.xlim(xmin,xmax)
    #plt.ylim(ymin,ymax)

    plt.plot()
    fig.savefig(f"{odir}/reco-event-{picname}.png")
    plt.close()


def other2Dplot(nuarray, picname, odir):

    non_zero_index = np.nonzero(nuarray)

    xmin, xmax = non_zero_index[0].min(), non_zero_index[0].max()
    ymin, ymax = non_zero_index[1].min(), non_zero_index[1].max()

    ROI = nuarray[xmin-10:xmax+10, ymin-10:ymax+10]

    plt.figure(figsize=(8,8))

    plt.imshow(ROI, cmap='hot', origin='lower', extent=(ymin,ymax,xmin,xmax))
    plt.gca().invert_yaxis() 
    plt.colorbar(orientation='vertical', label='Counts')

    plt.title('Zoom-in')    
    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot()
    plt.savefig(f"{odir}/reco-zoom-event-{picname}.png")

def simpleSimRecoHist(data, nbins, minbin, maxbin, labels, picname, ylog, odir):

    plt.figure() 

    plt.hist(data, nbins, range=(minbin,maxbin), histtype='step', facecolor='b')
 
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
 
    if(ylog):                                                                                       
        plt.yscale('log')
                
    plt.savefig(f"{odir}/simpleSimRecoHist-{picname}.png")                                   


#####################################################################
recofile = sys.argv[1]
plotname = sys.argv[2]

outdir = f"tmp-{plotname}/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

# valid for NeCO2-45deg run only
#empty_ev = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
#prop_ev =[41, 42, 43, 44, 45, 46, 47, 183, 184, 185, 186, 193, 205, 206, 207, 322, 352, 366, 383, 384, 385, 386, 387, 583, 584, 585, 586, 587, 747, 748] 

#good_events = []

with tb.open_file(recofile, 'r') as f:
   
     
    groups = f.walk_groups('/')
    grouplist = []
    for gr in groups:
        print(f'found {gr}')
        grouplist.append(gr)
    main_group = str(grouplist[len(grouplist)-1])
    print(f"last entry in walk_groups = \n{main_group}")
    #print(grouplist)
    #groupname_parts = grouplist[0].split('/')
    #print(groupname_parts)

    grouplist = None 

    basewords = main_group.split('(')
    print(basewords)

    base_group_name = basewords[0][:-1]+'/'
    #                              ^ removes space at the end of 'run_xxx/chip0 '
    print(f'base group name is : <{base_group_name}>')
    bgn_split = base_group_name.split('/')
    print(bgn_split)
    run_name = bgn_split[2]
    print(f"<{run_name}>")
    run_num = int(run_name[4:])
    print(f'run number is {run_num}')

    #exit(0)
    basewords = None
    # =========== gettin' shit ==================
    #base_group_name = '/reconstruction/run_9999/chip_0/'
    #base_group_name = '/reconstruction/run_1868/chip_0/'
    
    ToT = f.get_node(base_group_name+"ToT")
    print(f"found VLarray TOT of size {type(ToT)}")
    centerX = f.get_node(base_group_name+"centerX")
    centerY = f.get_node(base_group_name+"centerY")
    hits = f.get_node(base_group_name+"hits")
    excent = f.get_node(base_group_name+"eccentricity")
    FIRT = f.get_node(base_group_name+"fractionInTransverseRms")
    kurtosisL = f.get_node(base_group_name+"kurtosisLongitudinal")
    kurtosisT = f.get_node(base_group_name+"kurtosisTransverse")
    length = f.get_node(base_group_name+"length")
    LDRT = f.get_node(base_group_name+"lengthDivRmsTrans")
    RMS_L = f.get_node(base_group_name+"rmsLongitudinal")
    RMS_T = f.get_node(base_group_name+"rmsTransverse")
    rotAng = f.get_node(base_group_name+"rotationAngle")
    skewL = f.get_node(base_group_name+"skewnessLongitudinal")
    skewT = f.get_node(base_group_name+"skewnessTransverse")
    sumTOT = f.get_node(base_group_name+"sumTot")
    width = f.get_node(base_group_name+"width")
    x = f.get_node(base_group_name+"x")
    y = f.get_node(base_group_name+"y")
#    kurtosisL = f.get_node(base_group_name+"")

    #print(f"centerX is {type(centerX)}, centerY is {type(centerY)}")    
    #print(f"centerX shape={centerX.shape}")
    #print(f"first ten centerX={centerX[0:10]}")
    #print(f"centerY shape={centerY.shape}")
    #print(f"first ten centerY={centerY[0:10]}")

    #print(f"ToT is {type(ToT)}")
    #print(f"x is {type(x)}")
    #print(f"y is {type(y)}")

    #print("cehcking some variable length arrays:")
    #for i in range(5):
    #    print(f"x={x[i]}, y={y[i]}, TOT={ToT[i]}")

    # =========== plottin' shit =================

    simpleSimRecoHist(centerX, 50, 6, 8, ['centerX','x','cnt'], plotname+"centerX", False, outdir)
    simpleSimRecoHist(centerY, 50, 6, 8, ['centerY','Y','cnt'], plotname+"centerY", False, outdir)
    # - -------------------------------------------------------------------------
    simpleSimRecoHist(excent, 50, 0.9, 10 , ['Excentricity','','cnt'], plotname+"excent", True, outdir)
    simpleSimRecoHist(FIRT, 50, np.min(FIRT), np.max(FIRT)+0.5, ['frac_In_rms_T','','cnt'], plotname+"fracInRmsTrans", True, outdir)
    simpleSimRecoHist(kurtosisL, 50, np.min(kurtosisL), np.max(kurtosisL), ['kurtosisL','','cnt'], plotname+"kurtL", True, outdir)
    simpleSimRecoHist(kurtosisT, 50, np.min(kurtosisT), np.max(kurtosisT), ['kurtosisT','','cnt'], plotname+"kurtT", True, outdir)
    #simpleSimRecoHist(length, 50, np.min(length), np.max(length), ['length','','cnt'], plotname+"length", False, outdir)
    simpleSimRecoHist(length, 50, np.min(length), 6, ['length','','cnt'], plotname+"length", False, outdir)

    #simpleSimRecoHist(hits, 50, np.min(hits), np.max(hits), ['hits','','cnt'], plotname+"hits", True, outdir)
    simpleSimRecoHist(hits, 50, np.min(hits), 500, ['hits','','cnt'], plotname+"hits", True, outdir)

    simpleSimRecoHist(LDRT, 50,0,50, ['lenDivRmsTrans','','cnt'], plotname+"LDRT", True, outdir)
    simpleSimRecoHist(RMS_L, 50,np.min(RMS_L), 2.0, ['RMS_L','',''], plotname+"RMS_L", False, outdir)

    simpleSimRecoHist(RMS_T, 50,np.min(RMS_T), 1.5, ['RMS_T','',''], plotname+"RMS_T", False, outdir)

    #simpleSimRecoHist(rotAng*180/3.14159, 50,0,360, ['rotAng','',''], plotname+"rotAng", False, outdir)
    simpleSimRecoHist(rotAng, 50,np.min(rotAng),np.max(rotAng), ['rotAng','',''], plotname+"rotAng", False, outdir)
    simpleSimRecoHist(skewL, 50,np.min(skewL),np.max(skewL), ['skewL','',''], plotname+"skewL", True, outdir)
    simpleSimRecoHist(skewT, 50,np.min(skewT),np.max(skewT), ['skewT','',''], plotname+"skewT", True, outdir)
    simpleSimRecoHist(sumTOT, 50,np.min(sumTOT),np.max(sumTOT), ['sumTOT','',''], plotname+"sumTOT", True, outdir)
    simpleSimRecoHist(width, 50,np.min(width), 2, ['width','',''], plotname+"width", False, outdir)
    # -----------------------------------------------------------------------------

    mean_glob_x = np.mean(centerX)   
    mean_glob_y = np.mean(centerY)   
    std_centerX = np.std(centerX)
    std_centerY = np.std(centerY)

    print(f"Mean position of the \"beam\" x={mean_glob_x},y={mean_glob_y}")
    print(f"sigmmas of the \"beamspot\" {round(std_centerX,2)}, {round(std_centerY,2)}")
    #print()
    #plot2Dhist(excent.reshape(-1),FIRT.reshape(-1), ["\u03B5 vs FIRT","excentricity","FIRT"], "epsilon-vs-FIRT", outdir)    

    alt_rotang = []    

    #exit(0)
    ################################################################
    ################################################################
    ################################################################

    if(check_node(base_group_name+"angle_fiststage", f)):
        print("Found reconstruction data!")
        
        firstangles = f.get_node(base_group_name+"angle_fiststage")[:].T
        secondangles = f.get_node(base_group_name+"angle_secondstage")[:].T

        multiHist([firstangles,secondangles], 
                    62, 
                    [0.0,3.15],
                    ['xray angles','Angles [radian]','#'],
                    ['stage1','stage2'],
                    False,
                    f"RecoAngles-{plotname}",
                    outdir) 

        simpleHist(firstangles, 
                    31, 
                    0.0,
                    3.15,
                    ["Reco angles first stage", "Angle,[radian]", "N"],
                    "XPolFirstStage",
                    outdir,
                    fit="Yes")

        simpleHist(secondangles, 
                    31, 
                    0.0,
                    3.15,
                    ["Reco angles second stage", "Angle,[radian]", "N"],
                    "XPolSecondStage",
                    outdir,
                    fit="Yes")


        #exit(0)


    #matrix = np.zeros((256,256),dtype=np.uint16)
    matrixTotal = np.zeros((256,256),dtype=np.uint16)

    selMatrix = np.zeros((256,256),dtype=np.uint16)

    eventString = ""

    #multiClusterEvents = []

    ntotal = ToT.shape[0]

    #print(sumTOT.shape)
    #print(sumTOT.shape[0])

    #print(f"sumTOT={sumTOT[0]}")
    #print(f"x[0]={x[0]}")
    #print(f"y[0]={y[0]}")
    #print(f"ToT[0]={ToT[0]}")

    position_bin_edges = np.linspace(0,256,257)
    x_bin_cnt = np.zeros(256,dtype=np.int64)
    y_bin_cnt = np.zeros(256,dtype=np.int64)

    print(ntotal)
    ievent, npics, mcevents = 0, 0, 0
    n_good = 0
    for event in ToT:
    
        #if( (centerX[ievent] < mean_glob_x-(2*std_centerX) or 
        #    centerX[ievent] > mean_glob_x + (2*std_centerX) ) and 
        #    (centerY[ievent] > mean_glob_y + (2*std_centerY) or 
        #    centerY[ievent] > mean_glob_y + (2*std_centerY) ) 
        #    ):
        #    ievent+=1 
        #    continue

        #if(len(event) < 10 or length[ievent] > 2.0 or LDRT[ievent]>20.0 or LDRT[ievent] < 2):
        #    ievent+=1
        #    continue

        matrix = np.zeros((256,256),dtype=np.uint16)
        nhits = len(event)
        #npos = 0
        #if(ievent==0):
        #    print(f"x={x[ievent]},\ny={y[ievent]},\nTOT={event[ievent]}")
        Sum_Qx, Sum_Qy = 0.0, 0.0
        for i in range(nhits):
            matrix[x[ievent][i],y[ievent][i]] = event[i]
            matrixTotal[x[ievent][i], y[ievent][i]] += 1
            Sum_Qx += float(event[i])*float(x[ievent][i])
            Sum_Qy += float(event[i])*float(y[ievent][i])

            #if(ievent==0):
            #    #print(f"x={x[ievent][i]}")
            #    #print(f"y={y[ievent][i]}")
            #    #print(f"{event[i]}")
            #    #print(f"SumQx={Sum_Qx}")
            #    #print(f"SumQy={Sum_Qy}")
            ##npos+=1

        ith_upd_x, _ = np.histogram(x[ievent],bins=position_bin_edges)
        ith_upd_y, _ = np.histogram(y[ievent],bins=position_bin_edges)
        x_bin_cnt += ith_upd_x
        y_bin_cnt += ith_upd_y
           
        redX = int(Sum_Qx/float(sumTOT[ievent]))
        redY = int(Sum_Qy/float(sumTOT[ievent]))
        #if(ievent==0):
            #print(f"sumTOT[ievent]={sumTOT[ievent]}")
            #print(f"")

        Sum_Qx = 0.0
        Sum_Qy = 0.0

        # tryna' dbscan (fuck it slows down things...)
        #
        #smearedMat = matrix.reshape(64,4,64,4).mean(axis=(1,3))
        #nz_index = np.transpose(np.nonzero(smearedMat))
        #nholes = 1
        #min_hits = 2
        #db = DBSCAN(eps=nholes, min_samples=min_hits).fit(nz_index)
 
        #labels = db.labels_
        #nfound = len(set(labels)) - (1 if -1 in labels else 0)

        #if(nfound>1):
        #    multiClusterEvents.append(ievent)
        #    mcevents+=1
       
        rotAngDeg = round(rotAng[ievent]*180/3.1415,2)  
        characs = f"cX={round(centerX[ievent],2)}\ncY={round(centerY[ievent],2)}\n\u03B5={round(excent[ievent],2)}\nrotAng={rotAngDeg}\nnhits={nhits}\nlen={round(length[ievent],2)}\nwidth={round(width[ievent],2)}\nRMS:\n(T={round(RMS_T[ievent],2)},L={round(RMS_L[ievent],2)})\nFIRT={round(FIRT[ievent],2)}\nkurt:\nT={round(kurtosisT[ievent],2)}\n,L={round(kurtosisL[ievent],2)}\nLDRT={round(LDRT[ievent],2)}"        

        #---- working below -------
        if(LDRT[ievent] > 10 and nhits>10 and FIRT[ievent]>0.01):
        #--------------------------
        #if(ievent in prop_ev or ievent in empty_ev):
            #suffix = None
            #if(ievent in prop_ev):
            #    suffix = "PROP"
            #else:
            #    suffix = "EMPTY"
            #good_events.append(ievent)

            alt_rotang.append(rotAngDeg)

            n_good+=1
            for i in range(nhits):
                #selMatrix[x[ievent][i], y[ievent][i]] += 1
                selMatrix[x[ievent][i], y[ievent][i]] += event[i]

            eventString+=f"{ievent},"
            if(npics < 50):
                plot2dEvent(matrix, characs, f"cluster-{ievent}-{plotname}", outdir, [redX,redY]) 
                #plot2dEvent(matrix, characs, f"cluster-{ievent}-{suffix}-{plotname}", outdir) 
                #other2Dplot(matrix, f"cluster-{ievent}-{plotname}",outdir)
                #plotDbscan(nz_index, labels, ievent, nfound, outdir)
                npics+=1
        matrix = np.zeros((256,256),dtype=np.uint16)
        progress(ntotal, ievent)
        ievent+=1
        
        if("RomeRun" in recofile and ievent==30000):
            break

        #exit(0)

    print(f"FOUND <{n_good}> events")

    plot2dEvent(matrixTotal, "", "total-run",outdir)
    other2Dplot(selMatrix, "selected-run",outdir)

    simpleHist(alt_rotang, 51, 0.0, 180.0, ["Selected Cluster inclination","angle[deg]","N"], "ALT_ROTANG", outdir) 

    #mat_xproj = np.sum(matrixTotal,axis=0)
    #mat_yproj = np.sum(matrixTotal,axis=1)
   
    #print(mat_xproj.shape)
 
    stdxproj = round(np.std(x_bin_cnt),2)
    stdyproj = round(np.std(y_bin_cnt),2)

    epsilon = round(stdxproj/stdyproj,2)

    #simpleHist(mat_xproj,51,0,256,["x projection", "xpos", "N"], "YOBA-X", outdir)
    #simpleHist(mat_yproj,51,0,256,["y projection", "ypos", "N"], "YOBA-Y", outdir)

    plt.figure()

    plt.hist(position_bin_edges[:-1], weights=x_bin_cnt, range=(0,255),bins=256, align='left',histtype='step',edgecolor='red')
    plt.savefig(f"{outdir}/EBALA.png")


    print(f"TOTAL matrix projections: \nstdev_x={stdxproj},\nstdev_y={stdyproj},\nexccentriccity={epsilon}")

    print(f"found {mcevents} multi cluster events")

    print(f"Interesting events have indices:{eventString[:-1]}")

    listFile = open(f"{plotname}-indices.txt", "w")
    listFile.write(eventString[:-1])
    #listFile.write(eventString[:-1])
    listFile.close()



