import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tables as tb
from sklearn.cluster import DBSCAN 
from scipy.optimize import curve_fit
from lmfit import Model

# for interactive 3d plotting 
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

###GREEK LETTERS###

G_mu = '\u03bc'
G_sigma = '\u03c3'
G_chi = '\u03c7'
G_delta = '\u0394'
G_phi = '\u03C6'

######

def cosfunc(x, Ap, Bp, phi0):

    return Ap + Bp*np.cos(x - phi0)**2
    #return Bp + Ap*np.cos(x - phi0)**2

def gauss(x, A, mu, sigma):

    return A*np.exp(-((x-mu)**2)/(2*sigma**2))

def modFac(Ap,Bp):

    return Bp/(2*Ap+Bp)

def calcIntensity(A, B):

    return np.round(A + B/2.0, 2)

def deltaMu(mu, Ap, Bp, Ap_err, Bp_err):

    return mu * np.sqrt( (Bp_err / Bp)**2  + (2 * Ap_err / (Bp + 2*Ap) )**2 )

def polDeg(Ap, Bp, mu):

    return (1/mu)*(Bp/((2*Ap)+Bp))

def check_node(gr_path, file):

    return gr_path in file

def toDegrees(angle):

    return angle*180.0/3.14159 

def toRadian(angle):

    return angle*3.14159/180

def getMaxBin(numbers):

    maxcounts = 0
    maxbin = 0
    cnt = 0
    for n in numbers:
        if(n>maxcounts):
            maxcounts = n
            maxbin = cnt
            cnt+=1
        else:
            cnt+=1

    return maxbin

def getMinBin(numbers):

    mincounts = 0
    minbin = 0
    cnt = 0
    for n in numbers:
        if(n<mincounts):
            mincounts = n
            minbin = cnt
            cnt+=1
        else:
            cnt+=1

    return minbin

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)

def plotInteractive3D(xlist, ylist, zlist, labels, picname, odir):

    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=xlist, y=ylist, z=zlist,
        mode="markers",
        marker=dict(size=3,color="blue"),
        name=f"Track {labels}"
 
    )) 
    print("SUKA")   
 
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x [pixel]", range=[0,256]),
            yaxis=dict(title="y [pixel]", range=[0,256]),
            zaxis=dict(title="z [TOA combined]")
        ),
        margin=dict(l=0,r=0,b=0,t=0)
    )
    print("PADLA")   
    fig.write_html(f"{odir}/Track-{picname}.html")


def runAngleReco(positions, charges):

    xpos, ypos = positions
    center = np.average(positions, axis=1, weights=charges)
    X = np.vstack((xpos - center[0], ypos - center[1]))
    
    # Covariance matrix 
    M = np.dot(X*charges, X.T)

    # getin' eigen val's n vectors for covariance matrix
    eigenVal, eigenVect = np.linalg.eig(M) 

    # get axis which maximizes second moment - eigenvector w biggest eigenvalue
    prime_axis = eigenVect[:, np.argmax(eigenVal)] 

    # projectin' new axis on x-y and calc its angle      
    projection_xy = np.array([prime_axis[0],prime_axis[1]])
    angle = np.arctan2(projection_xy[1],projection_xy[0])

    # projectin' data points onto new axis plane
    projection_xy_fit = np.dot(X[:2].T, prime_axis[:2])

    # calculatin' skewness
    skew_xy = np.sum(charges * projection_xy_fit**3)/np.sum(charges)

    if skew_xy > 0:
        if angle > 0: 
            angle = -np.pi + angle
        else:
            angle = np.pi + angle
    else:
        angle = angle

    return angle

#def scanForPeaks(nuarray, thr):
#
#    cnt_above_thr = 0
#    n_peaks = 0
#    counter = 0
#
#    for num in nuarray:
#
#        if num > thr:
#            cnt_above_thr+=1
#        else:
#            continue
#
#        if nuarray[counter+1] < thr:
#            n_peaks+=1
#            cnt_above_thr = 0
#        counter+=1
#    
#    return n_peaks
#
#def countDiscont(matrix):
# 
#    xproj, yproj = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), indexing='ij')
#
#    npeaks_x = scanForPeaks(xproj, max(xproj)*0.5)
#    npeaks_y = scanForPeaks(yproj, max(yproj)*0.5), 
#    
#    return npeaks_x, npeaks_y

def getClusterCenter(matrix):

    x,y = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), indexing='ij')
    
    norm_factor = matrix.sum()

    mean_x = (matrix * x).sum() / norm_factor
    mean_y = (matrix * y).sum() / norm_factor

    std_x = np.sqrt( (matrix * (x - mean_x)**2).sum() / norm_factor)
    std_y = np.sqrt( (matrix * (y - mean_y)**2).sum() / norm_factor)

    return mean_x, mean_y, std_x, std_y


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


def fitAndPlotModulation(nuarray, nbins, minbin, maxbin, labels, picname, odir):

    # faithfully stolen from Markus's plot_angles.py
    # fitting 
    clear_data = nuarray[~np.isnan(nuarray)]
    counts, bin_edges = np.histogram(clear_data, bins = nbins)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    
    count_errors = np.sqrt(counts)

    params, covariance = curve_fit(cosfunc, bin_centers, counts, bounds=([0,0,-np.pi],[np.inf,np.inf,np.pi]),maxfev=1000000)

    Ap, Bp, phi = params

    expected_cts = cosfunc(bin_centers, *params)
    
    chisquare = np.sum(((counts - expected_cts)**2 )/ count_errors**2)
    dof = len(counts) - len(params)
    chired = chisquare / dof

    Ap_err, Bp_err, phi_err = np.sqrt(np.diag(covariance))

    # plotting 
    plt.figure(figsize=(8,6))
    plt.hist(bin_edges[:-1], 
             weights=counts, 
             bins=nbins, 
             range=(minbin,maxbin), 
             align='left', 
             histtype='stepfilled', 
             facecolor='forestgreen')

    plt.plot(bin_centers[:-1], cosfunc(bin_centers[:-1], Ap, Bp, phi), '--r')
    plt.vlines(phi, 
                0, 
                np.max(counts)*1.05,
                colors='darkviolet',
                linestyles='dashed',
                label=f"{G_phi}={phi:.2f}"+r"$\pm$"+f"{phi_err:.2f}\n({toDegrees(phi):.2f} deg)")
    ax = plt.gca()
    miny,maxy = ax.get_ylim()
    print(f"Getting miny/maxy for histogram: {picname}")
    print(f"miny={miny}, maxy={maxy}")

    plt.ylim([miny, maxy*1.1])

    mu = modFac(Ap,Bp)
    muErr = deltaMu(mu, Ap, Bp, Ap_err, Bp_err)

    plt.text(-np.pi, maxy, f"{G_mu}={mu*100:.2f}%"+r"$\pm$"+f"{muErr:.2f}%", fontsize=11)
    #plt.text(minbin*1.1, maxy, f"{G_mu}={mu*100:.2f}%"+r"$\pm$"+f"{muErr:.2f}")
    #plt.text(minbin*1.1, maxy*0.95, r"$N(\phi) = A_{P} + B_{P}\cdot cos^2(\phi-\phi_{0})$")
    #plt.text(minbin*1.1, maxy*0.90, f"Ap={Ap:.2f}, Bp={Bp:.2f}")

    plt.legend(loc='upper right')

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.grid()
    plt.savefig(f"{odir}/STOLEN-1DHist-{picname}.png")
    plt.close()

def simpleHistSpectrum(nuarray, nbins, minbin, maxbin, labels, picname, odir, scale=None):

    plt.figure()
  
    plt.figsize=(8,8)

    counts, bin_edges = np.histogram(nuarray, bins=nbins, range=(minbin,maxbin))
    maxbin_cnt = np.max(counts)
    minbin_cnt = np.min(counts)

    val_ibin = (maxbin-minbin)/nbins

    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    print(f"Range= {minbin} -> {maxbin}, with {nbins} bins gives {val_ibin} per bin")

    model = None

    peakbin = getMaxBin(counts[1:])
    peakbin+=1

    print(f"Found maximum bin at {peakbin} = {counts[peakbin]}")
    model = Model(gauss) 
    pars = model.make_params(A=maxbin_cnt, mu=peakbin*val_ibin, sigma=np.std(counts))
    print(f"Gauss Fit: Setting: A={maxbin_cnt}, mu={peakbin*val_ibin}, sigma={np.std(counts)}")   
 
    print(len(bin_centers[:-1]))
    print(len(counts))
    plt.hist(bin_centers, weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', facecolor='b')

    print(f"\n\ncounts={counts} \n\n")
    print(maxbin_cnt)
    print(minbin_cnt)
    print(f"max_amplitude={maxbin_cnt - minbin_cnt}")
    print("########## FITTING GASUSS FUNCTION #############")
    result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
    print(result.fit_report()) 
    if(result.params['mu']<=0):

        pars['mu'].min = peakbin*0.8*val_ibin
        pars['mu'].max = peakbin*1.2*val_ibin
 
        print("FIT FAILED: restricting fit parameters and re-fitting")
        result = model.fit(counts[:-1], pars, x=bin_centers[:-1] )

        print(result.fit_report()) 

    fitlab = ""
    miny,maxy = 0, 0
    A = result.params["A"].value
    mu = result.params["mu"].value
    sigma = result.params["sigma"].value 
    ax = plt.gca()
    miny,maxy = ax.get_ylim()
    #plt.yscale('log')
    plt.plot(bin_centers[:-1], result.best_fit, '--r')
    plt.text(minbin*1.1, maxy*0.9, f"A={round(A,2)}, {G_mu}={round(mu,2)}, {G_sigma}={round(sigma,2)}")

    plt.legend(loc='upper right')

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    if(scale is not None):
        plt.yscale(scale)
    plt.grid()
    plt.savefig(f"{odir}/1DHist-GausFit-{picname}.png")

    plt.close() 


def simpleHist(nuarray, nbins, minbin, maxbin, labels, picname, odir,fit=None):

    plt.figure(figsize=(10,10))

    datasize = None
    try:
        if(type(nuarray)=='numpy.ndarray'):
            datasize = nuarray.shape[0]
        else:
            datasize = len(nuarray)
        if(datgasize==0):
            print(f"data array for {picname} is empty!")
            return 0 
    except:
        print(f"Data type container for {picname} is ambiguous")
        pass

    print(f"Got {datasize} events selected for figure {picname}")

    #nuarray = nuarray[~np.isnan(nuarray)]

    counts, bin_edges = np.histogram(nuarray, bins=nbins, range=(minbin,maxbin))

    Q, U, modfac = None, None, None

    #plt.hist(nuarray, nbins, range=(minbin,maxbin), histtype='stepfilled', facecolor='b')
    plt.hist(bin_edges[:-1], weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', facecolor='b')
    if(fit):
        plt.figsize=(8,8)
        maxbin_cnt = np.max(counts)
        mean_cnt = np.max(counts)
        minbin_cnt = np.min(counts)
        #minval, maxval = max, counts[len(counts)-1]
        val_ibin = (maxbin-minbin)/nbins
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

        print(f"Range= {minbin} -> {maxbin}, with {nbins} bins gives {val_ibin} per bin")

        ########################################################
        model = None
        nfits = 0
        if(fit=="gaus"): 
            peakbin = getMaxBin(counts[1:])
            peakbin+=1

            print(f"Found maximum bin at {peakbin} = {counts[peakbin]}")
            model = Model(gauss) 
            pars = model.make_params(A=maxbin_cnt, mu=peakbin*val_ibin, sigma=np.std(counts))
            print(f"Gauss Fit: Setting: A={maxbin_cnt}, mu={peakbin*val_ibin}, sigma={np.std(counts)}")

        if(fit=="cosfunc"):            

            params, _ = curve_fit(cosfunc, bin_centers, counts, bounds=([0.0, 0.0,-np.pi],[np.inf,np.inf,np.pi]),maxfev=1000000)
            #params, _ = curve_fit(cosfunc, bin_centers, counts, bounds=([minbin_cnt*0.5, maxbin_cnt*0.5,-np.pi],[np.inf, np.inf,np.pi]),maxfev=1000000)
            #params, _ = curve_fit(cosfunc, bin_centers, counts, maxfev=1000000)
            Ap, Bp, phi = params

            ######################################################
            #model = Model(cosfunc)
            #pars = model.make_params(Ap=initAp_cnt,Bp=maxbin_cnt,phi0=0)

            #pars['Ap'].min = maxbin_cnt*0.8
            #pars['Ap'].max = minbin*1.2
            #pars['Bp'].min = minbin_cnt
            #pars['Bp'].max = np.inf
            #pars['phi0'].min = -2*np.pi/3
            #pars['phi0'].max = 2*np.pi/3

        print(f"\n\ncounts={counts} \n\n")
        print(maxbin_cnt)
        print(minbin_cnt)
        print(f"max_amplitude={maxbin_cnt - minbin_cnt}") 

        result = None
        if(fit=="cosfunc"):
            print("########## FITTING COS^2 FUNCTION #############")
        if(fit=="gaus"):
            result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
            print(result.fit_report()) 
            print("########## FITTING GASUSS FUNCTION #############")
        #result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
        
        nfits+=1

        if(fit=="gaus" and result.params['mu']<=0):

            #pars['A'].min = maxbin_cnt*0.8
            #pars['A'].max = maxbin_cnt*1.2

            pars['mu'].min = peakbin*0.8*val_ibin
            pars['mu'].max = peakbin*1.2*val_ibin
 
            #pars['sigma'].min = np.std(counts)*0.8
            #pars['sigma'].max = np.std(counts)*1.2

            print("FIT FAILED: restricting fit parameters and re-fitting")
            result = model.fit(counts[:-1], pars, x=bin_centers[:-1] )
            #result = model.fit(counts[peakbin-10:peakbin+10], pars, x=bin_centers[peakbin-10:peakbin+10])

            print(result.fit_report()) 
        
            nfits += 1
        ######### handing failed fit on modullation curve ################            
        #if(fit=="cosfunc" and (result.params['Ap']<=0 or result.params['Bp']<=0)):
        #    
        #    pars['Ap'].min = maxbin_cnt*0.5 
        #    #pars['Ap'].max = minbin_cnt*1.3

        #    pars['Bp'].min = 0.0
        #    #pars['Bp'].max = maxbin_cnt*1.3

        #    #pars['phi0'].min = -np.pi/2
        #    #pars['phi0'].max = np.pi/2 

        #    print("COSFUNC FIT FAILED: restricting fit parameters and re-fitting")
        #    result = model.fit(counts[:-1], pars, x=bin_centers[:-1] )
        #    nfits+=1

        fitlab = ""
        miny,maxy = 0, 0
        if(fit=="cosfunc"):
            #print("########## FITTING COS^2 FUNCTION #############")
            # these are for model = Model(cosfunc)
            #Ap = result.params['Ap'].value
            #Bp = result.params['Bp'].value
            #phi = result.params['phi0'].value
            #fitlab+=f"Ap={round(Ap,2)}, Bp={round(Bp,2)}, {G_phi}={round(phi,2)} "+r"($N_{fits}$="+f"{nfits})"
            #plt.plot(bin_centers[:-1], result.best_fit, '--r')
            #========================================
            plt.plot(bin_centers[:-1], cosfunc(bin_centers[:-1], Ap, Bp, phi), '--r')
            plt.hlines(Ap, -0.1, 3.15, colors='g', label=f"Ap={round(Ap,2)}")
            plt.hlines(Bp, -0.1, 3.15, colors='y', label=f"Bp={round(Bp,2)}")
            #plt.hlines(maxbin_cnt, -0.1, 3.14, colors='r',linestyles='--', label=f"MAXBIN-CTS")
            #plt.hlines(minbin_cnt, -0.1, 3.14, colors='r',linestyles='--', label=f"MINBIN-CTS")
            plt.vlines(phi, 0, np.max(counts)*1.05, colors='m',label=f"{G_phi}={round(phi,2)}({round(toDegrees(phi),2)} deg)")
            ax = plt.gca()
            miny,maxy = ax.get_ylim()
            mu = modFac(Ap,Bp)

            Q = (1/mu) * (Bp/2) * np.cos(2*phi)
            U = (1/mu) * (Bp/2) * np.sin(2*phi)  

            intensity = calcIntensity(Ap, Bp)

            print(f"Getting miny/maxy for histogram: {picname}")
            print(f"miny={miny}, maxy={maxy}")

            plt.text(-3.13 , maxy*0.96, r"$N(\phi) = A_{P} + B_{P}\cdot cos^2(\phi-\phi_{0})$"+r"($N_{fits}$="+f"{nfits})")
            plt.text(-3.13 , maxy*0.94, f"{G_mu}={mu:.2f}%")
            plt.text(-3.13 , maxy*0.92, f"I={intensity:.2f}")
            plt.text(-3.13 , maxy*0.90, f"P1={Q:.2f}")
            plt.text(-3.13 , maxy*0.88, f"P2={U:.2f}")

            print("________________________________________________________")
            print(f"Modulation factor = {mu}")
            print(f"\nStokes Parameters: \nQ(P1)={Q:.4f}, U(P2)={U:.4f}\n")
            print(f"Intensity = {intensity}")
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        elif(fit=="gaus"):
            #print("########## FITTING GAUS FUNCTION #############")
            A = result.params["A"].value
            mu = result.params["mu"].value
            sigma = result.params["sigma"].value 
            ax = plt.gca()
            miny,maxy = ax.get_ylim()
            #plt.yscale('log')
            plt.plot(bin_centers[:-1], result.best_fit, '--r')
            plt.text(minbin*1.1, maxy*0.9, f"A={round(A,2)}, {G_mu}={round(mu,2)}, {G_sigma}={round(sigma,2)}")

        plt.legend(loc='upper right')

    

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    #plt.yscale('log')
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
    plt.close()

def plot2DProjectionXY(matrix, labels, picname, odir):

    projectionX = matrix.sum(axis=0)
    projectionY = matrix.sum(axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(12,10))

    #ax[0].plot(projectionX)
    ax[0].bar(range(len(projectionX)), projectionX, width=1.0, align="center")
    ax[0].set_title(labels[0])
    ax[0].set_xlabel(labels[1])
    ax[0].set_ylabel(labels[2])

    #ax[1].plot(projectionY)
    ax[1].bar(range(len(projectionY)), projectionY, width=1.0, align="center")
    ax[1].set_title(labels[3])
    ax[1].set_xlabel(labels[4])
    ax[1].set_ylabel(labels[5])

    plt.tight_layout()
    plt.savefig(f"{odir}/XYproj-hist-{picname}.png")
    plt.close()

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


#def plot2dEvent(nuarray, info, picname, odir, plotMarker=None):
def plot2dEvent(nuarray, info, picname, odir, figtype=None):
 
    # ---- matrix 2d hist ----
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.86,0.1,0.05,0.8])
    #ms = ax.matshow(nuarray, cmap='plasma')
    ms = None
    if("total" in picname):
        #ms = ax.matshow(nuarray.T, cmap='hot')
        ms = ax.matshow(nuarray.T, cmap='gist_earth_r')
        ax.set_title("Pixel occupancy (Beam profile)")
        ax.set_xlabel("Pixel x")
        ax.set_ylabel("Pixel y")

    else:
        ms = ax.matshow(nuarray, cmap='hot')
    fig.colorbar(ms,cax=cax,orientation='vertical')
    start = 120
    ax.text(-90, start, info, fontsize=10,color='black' )    
    #if(plotMarker is not None):
    #    ax.scatter(plotMarker[0],plotMarker[1],c='blue', marker='*')
    #    ax.text(-90,200, f"redX={plotMarker[0]}\nredY={plotMarker[1]}", fontsize=10, color='black')

    if("total" in picname):
        ax.invert_yaxis()
    
    #peaks_x, peaks_y = countDiscont(nuarray)
    #plt.text(-90, 200, f"n_xpeaks={peaks_x}\nn_ypeaks={peaks_y}")

    plt.plot()
    if(figtype is None or type(figtype) is not str):
        fig.savefig(f"{odir}/reco-event-{picname}.png")
    else:
        fig.savefig(f"{odir}/reco-event-{picname}.{figtype}")
    
    plt.close()


def other2Dplot(nuarray, picname, odir):

    mean_x, mean_y, std_x, std_y = getClusterCenter(nuarray)
    #print("{},{}: {},{}".format(mean_x,mean_y,std_x,std_y))

    maxdev = 0
    if(std_x > std_y):
        maxdev = np.floor(std_x)
    else:
        maxdev = np.floor(std_y)

    xmin = int(np.floor(mean_x) - 4*maxdev) 
    xmax = int(np.floor(mean_x) + 4*maxdev)

    ymin = int(np.floor(mean_y) - 4*maxdev)
    ymax = int(np.floor(mean_y) + 4*maxdev)

    ROI = nuarray[xmin-2:xmax+2, ymin-2:ymax+2]
 
    #print("\n{},{}: {},{}\n".format(xmin,xmax,ymin,ymax))
    #ROI = nuarray[xmin:xmax, ymin:ymax]

    plt.figure(figsize=(8,8))

    #plt.imshow(ROI, cmap='viridis', origin='lower', extent=(ymin,ymax,xmin,xmax))
    plt.imshow(ROI, cmap='turbo', origin='lower', extent=(ymin,ymax,xmin,xmax))
    plt.gca().invert_yaxis() 
    plt.colorbar(orientation='vertical', label='Counts')

    plt.xlabel('x')
    plt.ylabel('y')
 
    plt.plot()
    plt.savefig(f"{odir}/reco-zoom-event-{picname}.png")
    #   plt.savefig(f"reco-zoom-event-{picname}.pdf")
    #else:
    #   plt.savefig(f"{odir}/reco-zoom-event-{picname}.png", dpi=200)
    plt.close()


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

tot_list = np.zeros(51, dtype=np.int64)
tot_edges = np.linspace(0,1000, 51+1)

alt_rotang = []    
tot_reduced = []   
TOTarray=[]

# for plotting 3d tracks
xlist, ylist, zlist = [], [], []

#matrix = np.zeros((256,256),dtype=np.uint16)
matrixTotal = np.zeros((256,256),dtype=np.uint16)
#matrixTOA = np.zeros((256,256),dtype=np.uint16)

nprotons = 0;

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
    
    #comToA = f.get_node(base_group_name+"ToACombined")
    #print(f"found VLarray TOACombined of size {type(comToA)}")
    #ToA = f.get_node(base_group_name+"ToA")
    #print(f"found VLarray TOT of size {type(ToA)}")
    ToT = f.get_node(base_group_name+"ToT")
    print(f"found VLarray TOT of size {type(ToT)}")
    centerX = f.get_node(base_group_name+"centerX")
    print(f"found centerX {type(centerX)}")
    centerY = f.get_node(base_group_name+"centerY")
    print(f"found centerY {type(centerY)}")
    hits = f.get_node(base_group_name+"hits")
    print(f"found hits {type(hits)}")
    excent = f.get_node(base_group_name+"eccentricity")
    print(f"found excentricity {type(excent)}")
    FIRT = f.get_node(base_group_name+"fractionInTransverseRms")
    print(f"found FIRT {type(FIRT)}")
    kurtosisL = f.get_node(base_group_name+"kurtosisLongitudinal")
    print(f"found kurtsisL {type(kurtosisL)}")
    kurtosisT = f.get_node(base_group_name+"kurtosisTransverse")
    print(f"found kurtosisT {type(kurtosisT)}")
    length = f.get_node(base_group_name+"length")
    print(f"found length {type(length)}")
    LDRT = f.get_node(base_group_name+"lengthDivRmsTrans")
    print(f"found LDRT {type(LDRT)}")
    RMS_L = f.get_node(base_group_name+"rmsLongitudinal")
    print(f"found RMS_L {type(RMS_L)}")
    RMS_T = f.get_node(base_group_name+"rmsTransverse")
    print(f"found RMS_T {type(RMS_T)}")
    rotAng = f.get_node(base_group_name+"rotationAngle")
    print(f"found rotAng {type(rotAng)}")
    skewL = f.get_node(base_group_name+"skewnessLongitudinal")
    print(f"found skewL {type(skewL)}")
    skewT = f.get_node(base_group_name+"skewnessTransverse")
    print(f"found skewT {type(skewT)}")
    sumTOT = f.get_node(base_group_name+"sumTot")
    print(f"found sumTOT {type(sumTOT)}")
    width = f.get_node(base_group_name+"width")
    print(f"found width {type(width)}")
    x = f.get_node(base_group_name+"x")
    print(f"found x {type(x)}")
    y = f.get_node(base_group_name+"y")
    print(f"found y {type(y)}")

    #firstangles, secondangles = None, None

    #print(f"centerX is {type(centerX)}, centerY is {type(centerY)}")    
    #print(f"centerX shape={centerX.shape}")
    #print(f"first ten centerX={centerX[0:10]}")
    #print(f"centerY shape={centerY.shape}")
    #print(f"first ten centerY={centerY[0:10]}")

    #print(f"ToT is {type(ToT)}")
    #print(f"x is {type(x)}")
    #print(f"y is {type(y)}")

    n_x = len(x)
    print(f"rate = {n_x/60} per second")

    #print("cehcking some variable length arrays:")
    #for i in range(5):
    #    print(f"x={x[i]}, y={y[i]}, TOT={ToT[i]}")

    # =========== plottin' shit =================

    print("Printing global variable data sets")
    #simpleSimRecoHist(ToT.reshape(-1), 51, np.nanmin(ToT.reshape(-1)), np.nanmax(ToT.reshape(-1)), ['TOT_Cycles','TOT','N'], plotname+"-TOT", False, outdir)
    #simpleSimRecoHist(centerX, 50, 6, 8, ['centerX','x','cnt'], plotname+"centerX", False, outdir)
    #simpleSimRecoHist(centerY, 50, 6, 8, ['centerY','Y','cnt'], plotname+"centerY", False, outdir)
    # - -------------------------------------------------------------------------
    simpleSimRecoHist(excent, 50, 0.9, 10 , ['Excentricity','','cnt'], plotname+"excent", True, outdir)
    #simpleSimRecoHist(FIRT, 50, np.min(FIRT), np.max(FIRT)+0.5, ['frac_In_rms_T','','cnt'], plotname+"fracInRmsTrans", True, outdir)
    #simpleSimRecoHist(kurtosisL, 50, np.nanmin(kurtosisL), np.nanmax(kurtosisL), ['kurtosisL','','cnt'], plotname+"kurtL", True, outdir)
    #simpleSimRecoHist(kurtosisT, 50, np.nanmin(kurtosisT), np.nanmax(kurtosisT), ['kurtosisT','','cnt'], plotname+"kurtT", True, outdir)
    simpleSimRecoHist(length, 50, np.min(length), np.max(length), ['length','','cnt'], plotname+"length", True, outdir)
    #simpleSimRecoHist(length, 50, np.min(length), 6, ['length','','cnt'], plotname+"length", False, outdir)

    simpleSimRecoHist(hits, 100, np.min(hits), np.max(hits), ['hits','','cnt'], plotname+"hits", True, outdir)
    #simpleSimRecoHist(LDRT, 50,0,50, ['lenDivRmsTrans','','cnt'], plotname+"LDRT", True, outdir)

    #simpleSimRecoHist(RMS_L, 50,np.min(RMS_L), 2.0, ['RMS_L','',''], plotname+"RMS_L", True, outdir)
    #simpleSimRecoHist(RMS_T, 50,np.min(RMS_T), 1.5, ['RMS_T','',''], plotname+"RMS_T", True, outdir)

    simpleSimRecoHist(rotAng, 50,np.min(rotAng),np.max(rotAng), ['rotAng','',''], plotname+"rotAng", True, outdir)
    #simpleSimRecoHist(skewL, 50,np.nanmin(skewL),np.nanmax(skewL), ['skewL','',''], plotname+"skewL", True, outdir)
    #simpleSimRecoHist(skewT, 50,np.nanmin(skewT),np.nanmax(skewT), ['skewT','',''], plotname+"skewT", True, outdir)
    simpleSimRecoHist(sumTOT, 100,np.min(sumTOT),np.max(sumTOT), ['sumTOT','',''], plotname+"sumTOT", True, outdir)
    #simpleSimRecoHist(width, 50,np.min(width), 2, ['width','',''], plotname+"width", True, outdir)
    # -----------------------------------------------------------------------------
    print("---------- CHINAZES! ------------")

    print(f"\nFile has {len(hits)} clusters\n")

    #exit(0)

    mean_glob_x = np.mean(centerX)   
    mean_glob_y = np.mean(centerY)   
    std_centerX = np.std(centerX)
    std_centerY = np.std(centerY)

    REAL_ALT_ROTANG = []
    hitsPerEvent = []
    naccepted = 0

    print(f"Mean position of the \"beam\" x={mean_glob_x},y={mean_glob_y}")
    print(f"sigmmas of the \"beamspot\" {round(std_centerX,2)}, {round(std_centerY,2)}")
    #plot2Dhist(excent.reshape(-1),FIRT.reshape(-1), ["\u03B5 vs FIRT","excentricity","FIRT"], "epsilon-vs-FIRT", outdir)    

    #exit(0)
    ################################################################
    ################################################################
    ################################################################

    TOAcomb = None
    ToA = None
    if(check_node(base_group_name+"ToACombined",f)):
        TOAComb = f.get_node(base_group_name+"ToACombined")
        ToA = f.get_node(base_group_name+"ToA")
        print("FOUND TOACombined!")

    oldXpolReco = check_node(base_group_name+"angle_fiststage", f)
    newXpolReco = check_node(base_group_name+"angle_firststage", f)

    #if(check_node(base_group_name+"angle_firststage", f)):
    if(oldXpolReco or newXpolReco):
        print("Found reconstruction data!")
        
        if(oldXpolReco):
            firstangles = f.get_node(base_group_name+"angle_fiststage")[:].T
        if(newXpolReco):
            firstangles = f.get_node(base_group_name+"angle_firststage")[:].T
        
        secondangles = f.get_node(base_group_name+"angle_secondstage")[:].T

        multiHist([firstangles,secondangles], 
                    101, 
                    [-np.pi,np.pi],
                    ['xray angles','Angles [radian]','#'],
                    ['stage1','stage2'],
                    False,
                    f"RecoAngles-{plotname}",
                    outdir) 

        simpleHist(firstangles, 
                    100, 
                    -np.pi,
                    np.pi,
                    ["Reco angles first stage", "Angle,[radian]", "N"],
                    "XPolFirstStage",
                    outdir,
                    fit="cosfunc")

        simpleHist(secondangles, 
                    100, 
                    -np.pi,
                    np.pi,
                    ["Reco angles second stage", "Angle,[radian]", "N"],
                    "XPolSecondStage",
                    outdir,
                    fit="cosfunc")

        exit(0)
        #fitAndPlotModulation(firstangles,
        #                     100,
        #                     -np.pi,
        #                     np.pi,
        #                     ["Reconstructed Angle Distribution", "Angle [radian]", r"$N_{Entries}$"],
        #                     "STOLEN-XpolFirstStage",
        #                     outdir)

        #fitAndPlotModulation(secondangles,
        #                     100,
        #                     -np.pi,
        #                     np.pi,
        #                     ["Reconstructed Angle Distribution", "Angle [radian]", r"$N_{Entries}$"],
        #                     "STOLEN-XpolSecondStage",
        #                     outdir)

        #exit(0)

    ntotal = ToT.shape[0]

    matrix_cut = np.zeros((256,256),dtype=int)

    print(f"\nTOTAL nr of clusters: {ntotal}\n")
    ievent, npics, mcevents = 0, 0, 0
    n_good = 0
    n_tracks = 0
    for event in ToT: 

        nhits = len(event)
        hitsPerEvent.append(nhits)
        #--------------------------------------------------------------
        ## --- related to proton counting ---
        #avg_hits_proton_track = 1403.5 # hits per proton track (counted by hand on a spreadsheet)
        #
        #if(nhits>np.ceil(avg_hits_proton_track/2.0)):
        #    nprotons += np.ceil(nhits/avg_hits_proton_track)

        #i_bincnt, _ = np.histogram(event, bins=tot_edges)
        #tot_list += i_bincnt       

        tot_reduced.append(sumTOT[ievent]/hits[ievent])
        TOTarray.append(sumTOT[ievent])
        #--------------------------------------------------------------
         
        rotAngDeg = round(rotAng[ievent]*180/3.1415,2)  
        characs = f"cX={round(centerX[ievent],2)}\ncY={round(centerY[ievent],2)}\n\u03B5={round(excent[ievent],2)}\nrotAng={rotAngDeg}\nnhits={nhits}\nlen={round(length[ievent],2)}\nwidth={round(width[ievent],2)}\nRMS:\n(T={round(RMS_T[ievent],2)},L={round(RMS_L[ievent],2)})\nFIRT={round(FIRT[ievent],2)}\nkurt:\nT={round(kurtosisT[ievent],2)}\n,L={round(kurtosisL[ievent],2)}\nLDRT={round(LDRT[ievent],2)}"        
        alt_rotang.append(rotAng[ievent])
       
        #--------------------------------------------------------------
        #          uncomment if analyzin' cyclotron data! 
        #
        #          old Timepix chip (most noisy channels)
        #
        #xskip = [96,206,183,56,101,128,103,106,194,204]
        #yskip = [161,84,123,64,103,125,112,127,202,245]

        #for xpos, ypos in zip(x[ievent],y[ievent]):
        #    if(xpos in xskip and ypos in yskip):
        #        continue
        #    else:
        #        np.add.at(matrixTotal, (xpos,ypos), 1)
        # 
        #--------------------------------------------------------------
        for xpos, ypos in zip(x[ievent],y[ievent]):
            np.add.at(matrixTotal, (xpos,ypos), 1)
 
        # IF EXCENTRICITY IS good :
        pol_angle = None
        if excent[ievent] > 1.3:
        #if excent[ievent] > 1.2 and nhits >= 80 and sumTOT[ievent] >= 2800 and sumTOT[ievent] < 5500 and length[ievent] <= 4.0:
        #if ((y[ievent]>99).all() and (y[ievent]<151).all() and (x[ievent]>99).all() and (x[ievent]<151).all()):
            yoba_pos = x[ievent],y[ievent]
            clusterangle = runAngleReco(yoba_pos,event)
            REAL_ALT_ROTANG.append(clusterangle) 
            pol_angle = clusterangle
            #for ix,iy in zip(x[ievent],y[ievent]):
            #    np.add.at(matrix_cut, (ix,iy), 1)
            naccepted+=1    
         
        if((npics < 50 and nhits > 25 and excent[ievent] > 2) or (nhits>2e4)): # this one get the actual tracks
        #if(sumTOT[ievent]>2e4): # EXTRA long tracks
        #if(npics < 50 and pol_angle is not None and pol_angle < 0):
        #if(npics < 100 and nhits > 20 and rotAng[ievent]>0.5 and rotAng[ievent]<2.5): # this one get the actual tracks
        #if(npics < 100 and sumTOT[ievent] < 200 and sumTOT[ievent] > 160):      

            matrix = np.zeros((256,256),dtype=np.uint16)
            n_good+=1
            #for i in range(nhits): 
            #    matrix[x[ievent][i],y[ievent][i]] = event[i]
            #    #matrixTOA[x[ievent][i],y[ievent][i]] = ToA[ievent][i]
            #    #matrixTOAcom[x[ievent][i],y[ievent][i]] = comToA[ievent][i]
            for k,l,m in zip(x[ievent],y[ievent],event):
                np.add.at(matrix, (k,l), m)               
 
            # this block is for plotting proton/ion tracks: ----------------------------
            #if(npics%10==0):
            #    plot2dEvent(matrix, characs, f"cluster-{ievent}-{plotname}", outdir, figtype="pdf") 
            #else:
            #    plot2dEvent(matrix, characs, f"cluster-{ievent}-{plotname}", outdir) 
            # --------------------------------------------------------------------------
            # this block is for plotting photoelectron tracks: -------------------------
            plot2dEvent(matrix, characs, f"cluster-{ievent}-{plotname}", outdir) 
            other2Dplot(matrix, f"cluster-{ievent}-{plotname}",outdir)

            #other2Dplot(matrixTOA,f"TOA-cluster-{ievent}-{plotname}",outdir)
            #other2Dplot(matrixTOAcom,f"TOAcom-cluster-{ievent}-{plotname}",outdir)
            
            #plotDbscan(nz_index, labels, ievent, nfound, outdir)
            npics+=1
            matrix = np.zeros((256,256),dtype=np.uint16)
            #matrixTOA = np.zeros((256,256),dtype=np.uint16)
            #matrixTOAcom = np.zeros((256,256),dtype=np.uint16)

            #if(ToA is not None):
            #   #print(f"TOAComb is not None and the event is {ievent}")
            #   xlist = x[ievent]
            #   ylist = y[ievent]
            #   zlist = (ToA[ievent]*25+1)/55.0
            #   plotInteractive3D(xlist,ylist,zlist,f"Event-{ievent}", f"-Evt-{ievent}-"+plotname, outdir)
 
        progress(ntotal, ievent)
        ievent+=1
        # ======== lookin' at absurd tracks in polarization data in 3D ========== 
        #if(TOAComb is not None and ievent==287911):
        #   print(f"TOAComb is not None and the event is {ievent}")
        #   xlist = x[ievent]
        #   ylist = y[ievent]
        #   zlist = TOAComb[ievent]
        #   plotInteractive3D(xlist,ylist,zlist,"Event-287911", plotname, outdir)
        #if(ievent==4000):
        #if("RomeRun" in recofile and ievent==30000):
        #    break

        #exit(0)

print(f"\nFOUND <{n_good}> events\n")

print(f"reduced TOT has <{len(tot_reduced)}> entries\n")

hits_arr = np.asarray(hitsPerEvent, dtype=np.uint16)

plot2dEvent(matrixTotal, "", "OCCUPANCY-total-run",outdir)
plot2DProjectionXY(matrixTotal, ["Occupancy X projection", "X [pixel]", r"$N_{Entries}$","Occupancy Y projection", "Y [pixel]", r"$N_{Entries}$"], plotname, outdir)
plot2dEvent(matrixTotal, "", "OCCUPANCY-total-run",outdir, figtype="pdf")

#cut_comment = "x>99 & x<151\ny>99 & y<151"
#plot2dEvent(matrix_cut, cut_comment, "OCCUPANCY-CUT-ON-position",outdir)

simpleHist(REAL_ALT_ROTANG, 100, -np.pi, np.pi, ["","Reconstructed angle,[rad]","Events,[N]"], "REAL_ALT_ROTANG", outdir, fit="cosfunc") 
print(f"\n Cut on EXCENTRICITY accepted only: {naccepted/ntotal*100:.2f}% of data\n")
simpleHist(alt_rotang, 100, 0.0, np.pi, ["","Reconstructed angle,[rad]","Events,[N]"], "ALT_ROTANG", outdir, fit="cosfunc") 
simpleHist(tot_reduced, 100, np.nanmin(tot_reduced), np.nanmax(tot_reduced), ["TOT Reduced by Number of Hits Per Event",r"$N {hits}$",r"$TOT/N_{Hits}$"], "TOT_REDUCED", outdir, fit="gaus")
simpleHist(TOTarray, 100, 0, 15000, [r"$\Sigma$ TOT Per Event",r"$N {Events}$",r"$\Sigma$ TOT"], "ALT_sumTOT", outdir, fit="gaus")
simpleHist(hits_arr, 100, np.nanmin(hits_arr), 250, ["Hits per Event",r"$_{hits}$",r"$N_{Events}$"], "HITS", outdir)

# ------------ cyclotron data here: --------------------------
#simpleHistSpectrum(TOTarray, 101, 0, 60000, ['TOT cyles per event (Before Irradiation)', 'TOT cycles','N'], "sumTOT", outdir, scale="linear")

print(f"TOTAL events: {ntotal}")
print(f"protons counted: {nprotons}")


