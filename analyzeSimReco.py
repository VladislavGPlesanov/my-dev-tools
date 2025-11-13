import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tables as tb
from sklearn.cluster import DBSCAN 
from scipy.optimize import curve_fit
from lmfit import Model
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math

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

## defining control chars for colored text ####

OUT_RST = '\033[0m'
OUT_RED = '\033[1;31m'
OUT_BLUE = '\033[1;34m'
OUT_GREEN = '\033[1;32m'
OUT_GRAY = '\033[1;37m'
OUT_YELLOW = '\033[1;33m'
OUT_CYAN = '\033[1;36m'
OUT_MAGEN = '\033[1;35m'

#######
# defining cut setting dictionary
#

CUTS_DICT={
    "DB-120Hz-0deg":{
        "sumTOT": [3750,6200],
        "length": 3.0,
        "width": 2.0,
        "nhits": 100, 
        "weightXY": -999, 
        "convX": [100,150], 
        "convY": [125,175], 
        "convXY":-999, 
        "Excent":5.0, 
        "RMSL":1.2
    },
    "DB-120Hz-30deg":{
        "sumTOT": [3750,6200],
        "length": 3.0,
        "width": 2.0,
        "nhits": 100, 
        "weightXY": [100,200], 
        "convX": [100,150], 
        "convY": [125,175], 
        "convXY":-999, 
        "Excent":5.0, 
        "RMSL":1.2
    },
    "DB-1260Hz-0deg":{
        "sumTOT": 6000,
        "length": 4.0,
        "width": 2.8,
        "nhits": [40,250], 
        "weightXY": -999, 
        "convX": [100,150], 
        "convY": [125,175], 
        "convXY":-999, 
        "Excent":50, 
        "RMSL":-999
    },
    "BGR":{
        "sumTOT": -999,
        "length": -999,
        "width": -999,
        "nhits": 25, 
        "weightXY": -999, 
        "convX": [3,252], 
        "convY": [3,252], 
        "convXY":-999, 
        "Excent":50, 
        "RMSL":-999
    }

}

#######

def cosfunc(x, Ap, Bp, phi0):

    return Ap + Bp*np.cos(x - phi0)**2

def gauss(x, A, mu, sigma):

    return A*np.exp(-((x-mu)**2)/(2*sigma**2))

def modFac(Ap,Bp):

    return Bp/(2*Ap+Bp)

def calcIntensity(A, B):

    return A + B/2.0

def deltaMu(mu, Ap, Bp, Ap_err, Bp_err):

    return mu * np.sqrt( (Bp_err / Bp)**2  + (2 * Ap_err / (Bp + 2*Ap) )**2 )

def polDeg(Ap, Bp, mu):

    return (1/mu)*(Bp/((2*Ap)+Bp))

def getPolDegree(Imax, Imin):

    return (Imax - Imin)/(Imax + Imin)

def check_node(gr_path, file):

    return gr_path in file

def toDegrees(angle):

    return angle*180.0/3.14159 

def toRadian(angle):

    return angle*3.14159/180

def removeNAN(data):

    return data[~np.isnan(data)]

def checkNAN(number):

    if(np.isnan(number)):
        return -1
    else:
        return number

def checkNANTheta(theta):

    if(np.isnan(theta)):
        return -999
    else:
        return theta

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

#def getHistRange(data):
#
#    data = np.array(data)
#    unqiue = np.sort(np.unique(data))
#    

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
        name=f"{labels[0]}"
 
    )) 
    #print("SUKA")   
 
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=f"{labels[1]}", range=[0,256]),
            yaxis=dict(title=f"{labels[2]}", range=[0,256]),
            #xaxis=dict(title=f"{labels[1]}"),
            #yaxis=dict(title=f"{labels[2]}"),
            zaxis=dict(title=f"{labels[3]}")
        ),
        margin=dict(l=0,r=0,b=0,t=0)
    )
    #print("PADLA")   
    fig.write_html(f"{odir}/Interactive-{picname}.html")

def calculateZpos(toa,ftoa,vdrift):

    z = ((toa*25+1) - ftoa*1.5625 )*vdrift/55.0
    # getting [um] here
    # multiplying by 1e-3 to get mm

    return z/1e-3

def getIthSokesParams(angle):

    i = 1
    q = np.cos(2*angle)
    u = np.sin(2*angle)
    #v = np.sqrt(1 - q**2 - u**2)

    return i,q,u

def checkClusterPosition(centerX, centerY, clusterX, clusterY, radius):

    r = np.sqrt((clusterX - centerX)**2 + (clusterY - centerY)**2)
    if(r>radius):
        return False
    else:
        return True

def checkClusterPositionEllipse(centerX, centerY, clusterX, clusterY, Major_radius, Minor_radius):

    ellipse_radius = ((clusterX - centerX)**2)/(Major_radius)**2 + ((clusterY - centerY)**2)/(Minor_radius)**2

    if(ellipse_radius > 1):
        return False
    else:
        return True

def runTwoStepAngleReco(coords, charges, radius_min=None, radius_max=None, weighting=None):
    """
    Combined two-stage reconstruction for 2D or 3D Timepix data.
    Computes both phi_1 and phi_2 in a single call.
    """

    if coords.shape[0] not in [2, 3]:
        raise ValueError("Coordinates must be 2D or 3D")

    # --- Center-of-charge shift ---
    center = np.average(coords, axis=1, weights=charges)
    X = coords - center[:, np.newaxis]

    # --- Step 1: Coarse principal axis determination ---
    cov = np.dot(X * charges, X.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # --- Angle (phi_1) in xy-plane ---
    projection_xy = np.array([principal_axis[0], principal_axis[1]])
    phi_1 = np.arctan2(projection_xy[1], projection_xy[0])

    # --- Compute projections and skewness ---
    proj_xy = np.dot(X[:2].T, principal_axis[:2])
    skewness_xy = np.sum(charges * proj_xy**3) / np.sum(charges)

    # --- Define start/end indices based on skewness ---
    left_indices = np.where(proj_xy < 0)[0]
    right_indices = np.where(proj_xy >= 0)[0]
    if skewness_xy <= 0:
        start_indices = left_indices
        end_indices = right_indices
    else:
        start_indices = right_indices
        end_indices = left_indices

    # --- Second-moment ratio (quality) ---
    m2_max = np.sum(charges * proj_xy**2) / np.sum(charges)
    min_axis = eigenvectors[:, np.argmin(eigenvalues)]
    m2_min = np.sum(charges * np.dot(X[:2].T, min_axis[:2])**2) / np.sum(charges)
    quality = m2_max / m2_min

    # --- Absorption point (within radius range if defined) ---
    radii2 = np.sum((X[:2])**2, axis=0)
    if radius_min is not None and radius_max is not None:
        r_inner = (radius_min * np.sqrt(m2_max))**2
        r_outer = (radius_max * np.sqrt(m2_max))**2
        circle_indices = np.where((radii2 > r_inner) & (radii2 < r_outer))[0]
        absorption_indices = np.intersect1d(start_indices, circle_indices)
    else:
        absorption_indices = start_indices

    if len(absorption_indices) > 0:
        absorption_point = np.average(coords[:, absorption_indices], axis=1, weights=charges[absorption_indices])
    else:
        absorption_point = np.full(coords.shape[0], np.nan)

    # --- Step 2: Fine refinement ---
    if weighting is not None:
        # Weight by distance from absorption point
        distances = np.sqrt(np.sum((coords - absorption_point[:, np.newaxis])**2, axis=0))
        w_charges = charges * np.exp(-distances / weighting)
        cov2 = np.dot((X * w_charges), X.T)
    else:
        # Restrict to start pixels only
        X2 = X[:, start_indices]
        c2 = charges[start_indices]
        cov2 = np.dot((X2 * c2), X2.T)

    eigenvalues2, eigenvectors2 = np.linalg.eig(cov2)
    principal_axis2 = eigenvectors2[:, np.argmax(eigenvalues2)]
    proj2_xy = np.array([principal_axis2[0], principal_axis2[1]])
    phi_2 = np.arctan2(proj2_xy[1], proj2_xy[0])

    # --- Ensure angular continuity between phi_1 and phi_2 ---
    dphi = phi_2 - phi_1
    if dphi > np.pi/2:
        phi_2 -= np.pi
    elif dphi < -np.pi/2:
        phi_2 += np.pi

    # --- Return everything ---
    #results = dict(
    #    phi_1=phi_1,
    #    phi_2=phi_2,
    #    center=center,
    #    start_indices=start_indices,
    #    end_indices=end_indices,
    #    absorption_point=absorption_point,
    #    skewness_xy=skewness_xy,
    #    quality=quality
    #)

    #return results
    return phi_1, phi_2, center, start_indices, end_indices, absorption_point, skewness_xy, quality

def runAngleReco(positions, charges):

    fTPX3 = True if len(positions)==3 else False

    xpos, ypos, zpos = None, None, None
    X = None
    center = np.average(positions, axis=1, weights=charges)
    if(fTPX3):
        xpos, ypos, zpos = positions
        X = np.vstack((xpos - center[0], ypos - center[1], zpos - center[2]))
    else:
        xpos, ypos = positions
        X = np.vstack((xpos - center[0], ypos - center[1]))
    
    # Covariance matrix 
    M = np.dot(X*charges, X.T)

    # getin' eigen val's n vectors for covariance matrix
    eigenVal, eigenVect = np.linalg.eig(M) 

    # get axis which maximizes second moment - eigenvector w biggest eigenvalue
    prime_axis = eigenVect[:, np.argmax(eigenVal)] 

    # projectin' new axis on x-y and calc its angle      
    projection_xy = np.array([prime_axis[0],prime_axis[1]])

    # getting phi_1 basically here
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

    # second step ends
    i_k = 1 
    q_k = np.cos(2*angle)
    u_k = np.sin(2*angle)

    return angle, i_k, q_k, u_k

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
    print(f"{OUT_MAGEN}[fitAndPlotModulation] --> {picname} --> {labels[0]} {OUT_RST}")
    clear_data = nuarray[~np.isnan(nuarray)]
    #counts, bin_edges = np.histogram(clear_data, bins = nbins, density=True)
    counts, bin_edges = np.histogram(clear_data, bins = nbins, range=(minbin,maxbin))
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    
    count_errors = np.sqrt(counts)

    mean_cts = np.mean(counts)
    y_dip = mean_cts - np.min(counts)
    y_hill = mean_cts + (mean_cts - np.min(counts))

    p0 = [y_dip, y_hill, 0.0]

    params, covariance = curve_fit(cosfunc, 
                                   bin_centers, 
                                   counts,
                                   p0=p0,
                                   bounds=([0,0,-np.pi/2],[np.inf,np.inf,np.pi/2]),
                                   maxfev=1000000)

    Ap, Bp, phi = params

    expected_cts = cosfunc(bin_centers, *params)
    
    chisquare = np.sum(((counts - expected_cts)**2 )/ count_errors**2)
    dof = len(counts) - len(params)
    chired = chisquare / dof

    Ap_err, Bp_err, phi_err = np.sqrt(np.diag(covariance))

    fBinColor = None

    if("BGR" in picname):
        fBinColor='gray'
    elif("CutOnPosition" in picname):
        fBinColor='darkorange'
    else:
        fBinColor='forestgreen'
        
    # plotting 
    plt.figure(figsize=(8,6))
    plt.hist(bin_edges[:-1], 
             weights=counts, 
             bins=nbins, 
             range=(minbin,maxbin), 
             align='left', 
             histtype='stepfilled', 
             facecolor=fBinColor)

    plt.plot(bin_centers[:-1], cosfunc(bin_centers[:-1], Ap, Bp, phi), '--r')
    plt.vlines(phi, 
                0, 
                np.max(counts)*1.05,
                colors='darkviolet',
                linestyles='dashed',
                label=f"{G_phi}={phi:.2f}"+r"$\pm$"+f"{phi_err:.2f}\n({toDegrees(phi):.2f} deg)")

    plt.hlines(Ap, -0.1, 3.15, colors='yellow', label=f"Ap={round(Ap,2)}")
    plt.hlines(Ap+Bp, -0.1, 3.15, colors='lightseagreen', label=f"Bp={round(Bp,2)}")
 
    ax = plt.gca()
    miny,maxy = ax.get_ylim()
    print(f"Getting miny/maxy for histogram: {picname}")
    print(f"miny={miny}, maxy={maxy}")

    plt.ylim([miny, maxy*1.2])

    mu = modFac(Ap,Bp)
    muErr = deltaMu(mu, Ap, Bp, Ap_err, Bp_err)
    intensity = calcIntensity(Ap, Bp)
    P = polDeg(Ap, Bp, mu)
 
    # Paolo, Polarimetry pdf, page 27
    Q = (1/mu) * (Bp/2) * np.cos(2*phi)
    U = (1/mu) * (Bp/2) * np.sin(2*phi)  

    

    # only walid for the caser on uncorrelated mu,phi,Bp
    # using as first estimate
    # need correlation term as well
    #
    dQ = np.sqrt(
        ((np.cos(2*phi)**2)*(Bp_err**2)/4*mu**2)+
        ((np.cos(2*phi)**2)*(muErr**2)*(Bp**2)/4*mu**4)+
        ((np.sin(2*phi)**2)*(phi_err**2)*(Bp)**2/mu**2)
                )

    dU = np.sqrt(
        ((np.sin(2*phi)**2)*(Bp_err**2)/4*mu**2)+
        (((np.sin(2*phi)**2)*(Bp**2)*(muErr**2))/4*mu**4)+
        (((np.cos(2*phi)**2)*(Bp**2)*(phi_err**2))/mu**2)
                )

    # Some publication 
    #Q = intensity * mu * np.cos(2*phi)
    #U = intensity * mu * np.sin(2*phi)  
    V = np.sqrt((P*intensity)**2 - Q**2 - U**2)

    #---------------------------------------------------
    # normalised modulation
    #normQ = Q/intensity    
    #normU = U/intensity
    #
    #Qr = 2/mu*normQ
    #Ur = 2/mu*normU
    #
    #Qr = 2/mu*Q
    #Ur = 2/mu*U
    #Vr = P*intensity - Qr**2 -Ur**2

    #---------------------------------------------------
    plt.text(-3.13, maxy*1.16, r"$\Sigma$"+f"(Entries)={len(nuarray)}",fontsize=11)
    plt.text(-3.13 , maxy*1.10, r"$N(\phi) = A_{P} + B_{P}\cdot cos^2(\phi-\phi_{0})$", fontsize=11)
    plt.text(-3.13 , maxy*1.04, f"{G_mu}={mu*100:.2f}%"+r"$\pm$"+f"{muErr:.2f}%", fontsize=11)
    plt.text(-3.13 , maxy*0.98, f"I={intensity:.2f}", fontsize=11)
    plt.text(-3.13 , maxy*0.92, f"Q={Q:.2f} +- {dQ:.2f} ({Q/intensity:.4f}+-{dQ/intensity:.4f})", fontsize=11)
    plt.text(-3.13 , maxy*0.86, f"U={U:.2f} +- {dU:.2f} ({U/intensity:.4f}+-{dU/intensity:.4f})", fontsize=11)
    plt.text(-3.13 , maxy*0.80, f"V={V:.2f})", fontsize=11)

    print("________________________________________________________")
    print(f"{OUT_RED} Modulation factor = {mu*100.0:.2f} {OUT_RST}")
    print(f"Polarization degree = {P:.2f}")
    print(f"\nStokes Parameters: \nQ(P1)={Q:.4f}, U(P2)={U:.4f}, V={V:.4f}\n")
    print(f"Intensity = {intensity}")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

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

    #print(f"\n\ncounts={counts} \n\n")
    #print(maxbin_cnt)
    #print(minbin_cnt)
    #print(f"max_amplitude={maxbin_cnt - minbin_cnt}")
    print("########## FITTING GASUSS FUNCTION #############")
    result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
    #print(result.fit_report()) 
    if(result.params['mu']<=0):

        pars['mu'].min = peakbin*0.8*val_ibin
        pars['mu'].max = peakbin*1.2*val_ibin
 
        print("FIT FAILED: restricting fit parameters and re-fitting")
        result = model.fit(counts[:-1], pars, x=bin_centers[:-1] )

        #print(result.fit_report()) 

    fitlab = ""
    miny,maxy = 0, 0
    A = result.params["A"].value
    mu = result.params["mu"].value
    sigma = result.params["sigma"].value 
    ax = plt.gca()
    miny,maxy = ax.get_ylim()
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

    print(f"{OUT_MAGEN}[simpleHist]-->{picname}-->{labels[0]}{OUT_RST}")

    plt.figure(figsize=(10,8))

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

    counts, bin_edges = np.histogram(nuarray, bins=nbins, range=(minbin,maxbin))

    Q, U = None, None
    Ap, Bp, phi = None, None, None
    Ap_err, Bp_err, phi_err = None, None, None

    plt.hist(bin_edges[:-1], weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', facecolor='b')
    if(fit):
        plt.figsize=(8,8)
        maxbin_cnt = np.max(counts)
        mean_cnt = np.max(counts)
        minbin_cnt = np.min(counts)
        val_ibin = (maxbin-minbin)/nbins
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

        #print(f"Range= {minbin:.4f} -> {maxbin:.4f}, with {nbins} bins gives {val_ibin:.4f} per bin")

        ########################################################
        model = None
        nfits = 0
        if(fit=="gaus"): 
            peakbin = getMaxBin(counts[1:])
            peakbin+=1

            #print(f"Found maximum bin at {peakbin} = {counts[peakbin]}")
            model = Model(gauss) 
            pars = model.make_params(A=maxbin_cnt, mu=peakbin*val_ibin, sigma=np.std(counts))
            #print(f"Gauss Fit: Setting: A={maxbin_cnt}, mu={peakbin*val_ibin}, sigma={np.std(counts)}")

        if(fit=="cosfunc"):            

            params, covar = curve_fit(cosfunc, bin_centers, counts, bounds=([0.0, 0.0,-np.pi],[np.inf,np.inf,np.pi]),maxfev=1000000)
            #params, _ = curve_fit(cosfunc, bin_centers, counts, bounds=([minbin_cnt*0.5, maxbin_cnt*0.5,-np.pi],[np.inf, np.inf,np.pi]),maxfev=1000000)
            #params, _ = curve_fit(cosfunc, bin_centers, counts, maxfev=1000000)
            Ap, Bp, phi = params
            Ap_err, Bp_err, phi_err = np.sqrt(np.diag(covar))

            ######################################################
            #model = Model(cosfunc)
            #pars = model.make_params(Ap=initAp_cnt,Bp=maxbin_cnt,phi0=0)

            #pars['Ap'].min = maxbin_cnt*0.8
            #pars['Ap'].max = minbin*1.2
            #pars['Bp'].min = minbin_cnt
            #pars['Bp'].max = np.inf
            #pars['phi0'].min = -2*np.pi/3
            #pars['phi0'].max = 2*np.pi/3

        #print(f"\n\ncounts={counts} \n\n")
        #print(maxbin_cnt)
        #print(minbin_cnt)
        #print(f"max_amplitude={maxbin_cnt - minbin_cnt}") 

        result = None
        if(fit=="cosfunc"):
            print("########## FITTING COS^2 FUNCTION #############")
        if(fit=="gaus"):
            result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
            #print(result.fit_report()) 
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

            #print(result.fit_report()) 
        
            nfits += 1
        ######### handing failed fit on modullation curve ################            
        fitlab = ""
        miny,maxy = 0, 0
        if(fit=="cosfunc"):
            #print("########## FITTING COS^2 FUNCTION #############")
            plt.plot(bin_centers[:-1], cosfunc(bin_centers[:-1], Ap, Bp, phi), '--r')
            plt.hlines(Ap, -0.1, 3.15, colors='g', label=f"Ap={round(Ap,2)}")
            plt.hlines(Ap+Bp, -0.1, 3.15, colors='y', label=f"Bp={round(Bp,2)}")
            plt.ylim([0, np.max(counts)*1.5])
            #plt.hlines(maxbin_cnt, -0.1, 3.14, colors='r',linestyles='--', label=f"MAXBIN-CTS")
            #plt.hlines(minbin_cnt, -0.1, 3.14, colors='r',linestyles='--', label=f"MINBIN-CTS")
            plt.vlines(phi, 0, np.max(counts)*1.05, colors='m',label=f"{G_phi}={phi:.2f} ({toDegrees(phi):.2f})")
            ax = plt.gca()
            miny,maxy = ax.get_ylim()
            mu = modFac(Ap,Bp)
            dmu = deltaMu(mu, Ap,Bp,Ap_err,Bp_err)

            Q = (1/mu) * (Bp/2) * np.cos(2*phi)
            U = (1/mu) * (Bp/2) * np.sin(2*phi)  

            intensity = calcIntensity(Ap, Bp)

            print(f"Getting miny/maxy for histogram: {picname}")
            print(f"miny={miny}, maxy={maxy}")

            plt.text(-3.13 , maxy*0.98, r"$N(\phi) = A_{P} + B_{P}\cdot cos^2(\phi-\phi_{0})$"+r"($N_{fits}$="+f"{nfits})")
            plt.text(-3.13 , maxy*0.96, f"{G_mu}={mu*100:.2f}%")
            plt.text(-3.13 , maxy*0.94, f"I={intensity:.2f}")
            plt.text(-3.13 , maxy*0.92, f"P1={Q:.2f}")
            plt.text(-3.13 , maxy*0.90, f"P2={U:.2f}")

            print("________________________________________________________")
            print(f"Modulation factor = {mu*100.0:.2f} %")
            print(f"\nStokes Parameters: \nQ(a0)={Q:.4f}, U(a1)={U:.4f}\n")
            print(f"Intensity = {intensity}")
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        elif(fit=="gaus"):
            #print("########## FITTING GAUS FUNCTION #############")
            A = result.params["A"].value
            mu = result.params["mu"].value
            sigma = result.params["sigma"].value 
            ax = plt.gca()
            miny,maxy = ax.get_ylim()
            plt.plot(bin_centers[:-1], result.best_fit, '--r')
            plt.text(minbin*1.1, maxy*0.9, f"A={round(A,2)}, {G_mu}={round(mu,2)}, {G_sigma}={round(sigma,2)}")

        plt.legend(loc='upper right')

    if(fit==None):
        sum_entries = np.sum(nuarray)
        mean_entries = np.mean(nuarray)
        stdev_entries = np.std(nuarray)
        ax = plt.gca()
        _, maxy = ax.get_ylim()
        _, xmax = ax.get_xlim()
        hist_comment = ""
        hist_comment += r"$\Sigma(Entries)$="+f"{sum_entries:.2f}\n"
        hist_comment += f"Mean={mean_entries:.2f}\n"
        hist_comment += r"$\sigma$="+f"{stdev_entries:.2f}"

        plt.text(xmax*0.8,maxy*0.95, hist_comment)

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    #logScalePlots = ["sumElec", "sumTOT", "tmp_hits"]
    #fLogscale = False
    #for name in logScalePlots:
    #    if(name in picname):
    #        fLogscale = True
    #if(fLogscale):
    #    plt.yscale('log')
    plt.grid()
    plt.savefig(f"{odir}/1DHist-{picname}.png")
    plt.close()


def plot2Dhist(x,y, labels, picname, odir):
    
    plt.figure(figsize=(8,6))

    plt.hist2d(x, y, bins=100, norm=LogNorm(), cmap="jet")
    plt.colorbar(label=r"$N_{Entries}$")

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    #plt.legend()
    plt.savefig(f"{odir}/2Dhist-{picname}.png")
    plt.close()

def plot2DProjectionXY(matrix, labels, picname, odir):

    projectionX = matrix.sum(axis=0)
    projectionY = matrix.sum(axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(10,10))

    ax[0].bar(range(len(projectionX)), projectionX, width=1.0, align="center")
    ax[0].set_title(labels[0])
    ax[0].set_xlabel(labels[1])
    ax[0].set_ylabel(labels[2])
    #ax[0].set_yscale('log')

    ax[1].bar(range(len(projectionY)), projectionY, width=1.0, align="center")
    ax[1].set_title(labels[3])
    ax[1].set_xlabel(labels[4])
    ax[1].set_ylabel(labels[5])
    #ax[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{odir}/XYproj-hist-{picname}.png")
    plt.close()

def plotDbscan(index, dblabels, ievent, nfound, odir):

    plt.figure(figsize=(8,6))
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
def plot2dEvent(nuarray, info, picname, odir, figtype=None, plotMarker=None):
 
    # ---- matrix 2d hist ----
    fig, ax = plt.subplots(figsize=(8,6))
    cax = fig.add_axes([0.86,0.1,0.05,0.8])
    #ms = ax.matshow(nuarray, cmap='plasma')
    ms = None
    if("total" in picname):
        #ms = ax.matshow(nuarray.T, cmap='viridis', norm=LogNorm(vmin=1,vmax=np.nanmax(nuarray)))
        ms = ax.matshow(nuarray.T, cmap='viridis', norm=LogNorm(vmin=1,vmax=np.nanmax(nuarray)))
        ax.set_title("Pixel occupancy (Beam profile)")
        fig.colorbar(ms,cax=cax,orientation='vertical',label='Occupancy')
    elif("events" in picname):
        #ms = ax.matshow(nuarray.T, cmap='jet', norm=LogNorm(vmin=1,vmax=np.nanmax(nuarray)))
        ms = ax.matshow(nuarray.T, cmap='viridis')
        ax.set_title("Tracks")
        fig.colorbar(ms,cax=cax,orientation='vertical',label='ToT')
    elif(info=="absorption"):
        ms = ax.matshow(nuarray.T, cmap='jet', norm=LogNorm(vmin=1,vmax=np.nanmax(nuarray)))
        ax.set_title("Absorption Points")
        fig.colorbar(ms,cax=cax,orientation='vertical',label=r'$N_{Entries}$')
    elif("title" in info):
        title = info.split(":")[1]
        ms = ax.matshow(nuarray.T, cmap='jet', norm=LogNorm(vmin=1,vmax=np.nanmax(nuarray)))
        ax.set_title(title)
        ax.grid(c='indigo', ls=":",lw=0.4)
        fig.colorbar(ms,cax=cax,orientation='vertical',label=r'$N_{Entries}$') 
    else:
        title_prefix = odir[4:]
        title_prefix = title_prefix[:-1]
        #ms = ax.matshow(nuarray.T, cmap='jet', norm=LogNorm(vmin=1,vmax=np.nanmax(nuarray)))
        ms = ax.matshow(nuarray.T, cmap='jet')
        ax.set_title(f"{title_prefix}\nPixel Map")
        fig.colorbar(ms,cax=cax,orientation='vertical')

    ax.set_xlabel("Pixel x")
    ax.set_ylabel("Pixel y")

    start = 123
    if("info" in info):
        comment = info.split(":")[1]
        ax.text(-90, start, comment, fontsize=9,color='black' )    
    if(plotMarker is not None):
        markx = plotMarker[0]
        marky = plotMarker[1]
        ax.scatter([markx],[marky],c='red', marker='d',s=80)

    ax.invert_yaxis()
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

    if(xmin < 0):
        xmin = 2
    if(ymin<0):
        ymin = 2
    if(xmax >256):
        xmax = 254
    if(ymax>256):
        ymax = 254

    ROI = nuarray[xmin-2:xmax+2, ymin-2:ymax+2]
 
    #print("\n{},{}: {},{}\n".format(xmin,xmax,ymin,ymax))
    #ROI = nuarray[xmin:xmax, ymin:ymax]

    plt.figure(figsize=(8,8))
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

#alt_rotang = []    
#tot_reduced = []   
#TOTarray=[]

# for plotting 3d tracks
xlist, ylist, zlist = [], [], []

matrixTotal = np.zeros((256,256),dtype=np.uint16)
matrixTotal_TOT = np.zeros((256,256),dtype=np.uint16)

matrix_instant_0 = np.zeros((256,256), dtype = int)
matrix_instant_1 = np.zeros((256,256), dtype = int)
matrix_cut = np.zeros((256,256),dtype=int)

n_instant_0, n_instant_1 = 0, 0
nprotons = 0

sum_U, sum_Q, sum_I = 0, 0, 0
altsum_U, altsum_Q, altsum_I, = 0, 0, 0

weightCenters = np.zeros((256,256),dtype=int)
weightCenters_cut = np.zeros((256,256),dtype=int)

# defining some temp lists to plot parameters after cuts

tmp_excent, tmp_hits, tmp_sumTOT =[], [], []
tmp_centerX, tmp_centerY = [], []
tmp_length,tmp_width = [], []

tmp_RMSL, tmp_RMST = [], []

tmp_densityRMS, tmp_densityWL = [], []
tmp_secondangles = []
tmp_secondangles_Y= []
tmp_BGRangles = []

tmp_theta = []
theta_secondstage = None

tmp_evtArea = []

conversionX, conversionY =  None, None
ConvX, ConvY = [],[]
#tmp_toaMean, tmp_toaRMS, tmp_toaSkew, tmp_toaLength =  [],[],[],[]

tmp_weightedY, tmp_weightedX = [], []

tmp_convX, tmp_convY = [],[]

cut_TOTMAP = np.zeros((256,256),dtype=int)

abs_stdx, abs_stdy, abs_meanx, abs_meany = None, None, None, None

electrons = None # variable for VLArray of electrons 

tmp_electrons, tmp_sumElec = [], []
tmp_elecPerHit = []

# flags here:

hasTheta = False
fTiming = False
fCharge = False
fConverted = False


with tb.open_file(recofile, 'r') as f:
   
    groups = f.walk_groups('/')
    grouplist = []
    for gr in groups:
        print(f'found {gr}')
        grouplist.append(gr)
    main_group = str(grouplist[len(grouplist)-1])
    print(f"last entry in walk_groups = \n{main_group}")

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

    basewords = None
    # =========== gettin' shit ==================
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

    n_x = len(x)
    print(f"rate = {n_x/60} per second")

    # =========== plottin' shit =================

#    print("Printing global variable data sets")
#    #simpleSimRecoHist(centerX, 50, 6, 8, ['centerX','x','cnt'], plotname+"centerX", False, outdir)
#    #simpleSimRecoHist(centerY, 50, 6, 8, ['centerY','Y','cnt'], plotname+"centerY", False, outdir)
#    # - -------------------------------------------------------------------------
#    simpleSimRecoHist(excent, 50, 0.9, 10 , ['Excentricity','','cnt'], plotname+"excent", True, outdir)
#    #simpleSimRecoHist(FIRT, 50, np.min(FIRT), np.max(FIRT)+0.5, ['frac_In_rms_T','','cnt'], plotname+"fracInRmsTrans", True, outdir)
#    #simpleSimRecoHist(kurtosisL, 50, np.nanmin(kurtosisL), np.nanmax(kurtosisL), ['kurtosisL','','cnt'], plotname+"kurtL", True, outdir)
#    #simpleSimRecoHist(kurtosisT, 50, np.nanmin(kurtosisT), np.nanmax(kurtosisT), ['kurtosisT','','cnt'], plotname+"kurtT", True, outdir)
#    simpleSimRecoHist(length, 50, np.min(length), np.max(length), ['length','','cnt'], plotname+"length", True, outdir)
#    #simpleSimRecoHist(length, 50, np.min(length), 6, ['length','','cnt'], plotname+"length", False, outdir)
#
#    simpleSimRecoHist(hits, 100, np.min(hits), np.max(hits), ['hits','','cnt'], plotname+"hits", True, outdir)
#    #simpleSimRecoHist(LDRT, 50,0,50, ['lenDivRmsTrans','','cnt'], plotname+"LDRT", True, outdir)
#
#    simpleSimRecoHist(RMS_L, 50,np.min(RMS_L), 2.0, ['RMS_L','',''], plotname+"RMS_L", True, outdir)
#    simpleSimRecoHist(RMS_T, 50,np.min(RMS_T), 1.5, ['RMS_T','',''], plotname+"RMS_T", True, outdir)
#
#    simpleSimRecoHist(rotAng, 50,np.min(rotAng),np.max(rotAng), ['rotAng','',''], plotname+"rotAng", True, outdir)
#    #simpleSimRecoHist(skewL, 50,np.nanmin(skewL),np.nanmax(skewL), ['skewL','',''], plotname+"skewL", True, outdir)
#    #simpleSimRecoHist(skewT, 50,np.nanmin(skewT),np.nanmax(skewT), ['skewT','',''], plotname+"skewT", True, outdir)
#    simpleSimRecoHist(sumTOT, 100,np.min(sumTOT),np.max(sumTOT), ['sumTOT','',''], plotname+"sumTOT", True, outdir)
#    simpleSimRecoHist(width, 100, np.min(width), np.max(width), ['width','',''], plotname+"width", True, outdir)
#    # -----------------------------------------------------------------------------
#    print("---------- CHINAZES! ------------")

    print(f"\nFile has {len(hits)} clusters\n")

    REAL_ALT_ROTANG = []
    SECOND_STEP_ALT_ROTANG = []

    absorption_points = np.zeros((256,256),dtype=int)
    absorption_points_pruned = np.zeros((256,256),dtype=int)

    #hitsPerEvent = []
    naccepted = 0

    ################################################################
    ################################################################
    ################################################################

    TOAcomb = None
    ToA = None
    fTOA = None
    toaLength, toaMean, toaRMS = None, None, None
    toaSkew, toaKurt = None, None

    if(check_node(base_group_name+"charge",f)):
        fCharge = True 
        electrons = f.get_node(base_group_name+"charge") 

    if(check_node(base_group_name+"ToACombined",f)):
        TOAComb = f.get_node(base_group_name+"ToACombined")
        ToA = f.get_node(base_group_name+"ToA")
        fTOA = f.get_node(base_group_name+"fToA")

        toaLength = f.get_node(base_group_name+"toaLength")
        toaMean = f.get_node(base_group_name+"toaMean")
        toaRMS = f.get_node(base_group_name+"toaRms")
        toaSkew = f.get_node(base_group_name+"toaSkewness")
        toaKurt = f.get_node(base_group_name+"toaKurtosis")

        fTiming = True

        #print(type(TOAComb))
        #print(type(ToA.shape))
        #print(type(fTOA.shape))

        #print("________TOACombined___________")
        #print(TOAComb[0:10])
        #print("________TOA___________")
        #print(ToA[0:10])
        #print("________fTOA___________")
        #print(fTOA[0:10])
        #print("________TOAlength___________")
        #print(toaLength[0:10])
        #print("________TOAmean___________")
        #print(toaMean[0:10])
        #print("________TOArms___________")
        #print(toaRMS[0:10])

        #'/home/vlad/readoutSW/tools/tmp-measurement-CP0-raw-2D/2Dhist-2D-Excent-Length.png' simpleSimRecoHist(TOAComb, 100 ,np.nanmin(TOAComb),np.nanmax(TOAComb), ['TOACombined','',''], plotname+"-TOACombined", True, outdir)
        #simpleSimRecoHist(np.array(ToA).reshape(-1), 100,np.nanmin(np.array(ToA).reshape(-1)),np.nanmax(np.array(ToA).reshape(-1)), ['ToA','',''], plotname+"-ToA", True, outdir)
        #simpleSimRecoHist(fTOA.reshape(-1), 100,np.nanmin(fTOA.reshape(-1)),np.nanmax(fTOA.reshape(-1)), ['fTOA','',''], plotname+"-fTOA", True, outdir)
        #simpleSimRecoHist(toaLength, 100,np.nanmin(toaLength), 200, ['toaLength','',''], plotname+"-toaLength", True, outdir)
        #simpleSimRecoHist(toaMean, 100,np.nanmin(toaMean), 50, ['toaMean','',''], plotname+"-toaMean", True, outdir)
        #simpleSimRecoHist(toaRMS, 100,np.nanmin(toaRMS), 50, ['toaRMS','',''], plotname+"-toaRMS", True, outdir)
        #simpleSimRecoHist(toaSkew, 100,np.nanmin(toaSkew),np.nanmax(toaSkew), ['toaSkew','',''], plotname+"-toaSkew", True, outdir)
        #simpleSimRecoHist(toaKurt, 100,np.nanmin(toaKurt),np.nanmax(toaKurt), ['toaKurtosis','',''], plotname+"-toaKurt", True, outdir)

        print("FOUND TOACombined!")

    oldXpolReco = check_node(base_group_name+"angle_fiststage", f)
    newXpolReco = check_node(base_group_name+"angle_firststage", f)

    hasTheta = check_node(base_group_name+"theta_secondstage",f)
    if(hasTheta):
        theta_secondstage = f.get_node(base_group_name+"theta_secondstage")
        simpleSimRecoHist(theta_secondstage, 100, np.nanmin(theta_secondstage), np.nanmax(theta_secondstage) , [r'Track:$\theta$','degrees, [radian]','CTS'], plotname+"THETA", True, outdir)

    if(oldXpolReco or newXpolReco):
        print("Found reconstruction data!")
        
        if(oldXpolReco):
            #firstangles = f.get_node(base_group_name+"angle_fiststage")[:].T
            firstangles = f.get_node(base_group_name+"angle_fiststage")
        if(newXpolReco):
            #firstangles = f.get_node(base_group_name+"angle_firststage")[:].T
            firstangles = f.get_node(base_group_name+"angle_firststage")
        
        #secondangles = f.get_node(base_group_name+"angle_secondstage")[:].T
        secondangles = f.get_node(base_group_name+"angle_secondstage")

        fitAndPlotModulation(secondangles,
                             100,
                             -np.pi,
                             np.pi,
                             ["Reconstructed Angle Distribution", "Angle [radian]", r"$N_{Entries}$"],
                             "STOLEN-XpolSecondStage",
                             outdir)

    if(check_node(base_group_name+"absorption_point_x",f)):   

        fConverted = True
        print("Found reconstructed absorption points")
        absorp_x = f.get_node(base_group_name+"absorption_point_x")
        absorp_y = f.get_node(base_group_name+"absorption_point_y")
        print(absorp_x[0:10])
        print(absorp_y[0:10])
        for ax,ay in zip(absorp_x,absorp_y):
            if(~np.isnan(ax) and ~np.isnan(ay)):
                ConvX.append(ax)
                ConvY.append(ay)
                ax = int(np.round(ax))
                ay = int(np.round(ay))
                np.add.at(absorption_points, (ax,ay), 1)
        plot2dEvent(absorption_points, "title:Reconstructed Absorption Points", "ABSORPTION", outdir)

        conversionX = absorp_x
        conversionY = absorp_y

        #absorp_x = absorp_x[np.where(~np.isnan(absorp_x))]
        #absorp_y = absorp_y[np.where(~np.isnan(absorp_y))]

        print(f"absorption x,y lengths: {absorp_x.shape[0]}\t{absorp_y.shape[0]}")
        print(f"Pruned lists          : {len(ConvX)}\t{len(ConvY)}")

        abs_stdx = np.std(ConvX)
        abs_stdy = np.std(ConvY)
        abs_meanx = np.mean(ConvX)
        abs_meany = np.mean(ConvY)

        print(f"{abs_stdx:.4f}")
        print(f"{abs_stdy:.4f}")
        print(f"{abs_meanx:.4f}")
        print(f"{abs_meany:.4f}")

        absorp_x, absorp_y = None, None
        print(f"Deviation in absorption x and y are:\n{OUT_BLUE} sigma_x={abs_stdx:.4f}, sigma_y={abs_stdy:.4f} {OUT_RST}")
        ConvX, ConvY = None, None

    #exit(0)
    ntotal = ToT.shape[0]

    print(f"\nTOTAL nr of clusters: {ntotal}\n")
    ievent, npics, mcevents = 0, 0, 0
    n_good = 0
    n_tracks = 0
    for event in ToT: 

        isumElec = 0
        if(fCharge):
            isumElec = np.sum(electrons[ievent])

        nhits = len(event)
        #hitsPerEvent.append(nhits)

        densityRMS = RMS_T[ievent]/RMS_L[ievent]
        densityWL = width[ievent]/length[ievent]
        areaWL = width[ievent]*length[ievent]
        
        #--------------------------------------------------------------
        ## --- related to proton counting ---
        #avg_hits_proton_track = 1403.5 # hits per proton track (counted by hand on a spreadsheet)
        #
        #if(nhits>np.ceil(avg_hits_proton_track/2.0)):
        #    nprotons += np.ceil(nhits/avg_hits_proton_track)

        #i_bincnt, _ = np.histogram(event, bins=tot_edges)
        #tot_list += i_bincnt       

        #tot_reduced.append(sumTOT[ievent]/hits[ievent])
        #TOTarray.append(sumTOT[ievent])
        #--------------------------------------------------------------
         
        rotAngDeg = round(rotAng[ievent]*180/3.1415,2)
        characs = "info:"
        characs += r"$\Sigma$(TOT)"+f"={sumTOT[ievent]}"
        characs += "\n"+r"$X_{c}$"+f"={centerX[ievent]:.2f}"
        characs += "\n"+r"$Y_{c}$"+f"={centerY[ievent]:.2f}"
        characs += "\n"+r"$\epsilon$"+f"={excent[ievent]:.2f}"
        characs += f"\nnhits={nhits}"
        characs += f"\nlen={length[ievent]:.2f}\nwidth={width[ievent]:.2f}"
        characs += "\n"+r"$RMS_{T}$"+f"={RMS_T[ievent]:.2f}"
        characs += "\n"+r"$RMS_{L}$"+f"={RMS_L[ievent]:.2f}"
        characs += f"\n"+r"$\rho$(RMS)="+f"{densityRMS:.2f}"

        if(toaMean is not None): 
            characs += f"\ntoaMEAN={toaMean[ievent]:.2f}"
            characs += f"\ntoaRMS={toaRMS[ievent]:.2f}"
            characs += f"\ntoaLength={toaLength[ievent]:.2f}"
            characs += f"\ntoaSkew={toaSkew[ievent]:.2f}"
            characs += f"\ntoaKurt={toaKurt[ievent]:.2f}"
        
        #alt_rotang.append(rotAng[ievent])
       
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
        for xpos, ypos, tot in zip(x[ievent],y[ievent], event):
            np.add.at(matrixTotal, (xpos,ypos), 1)
            np.add.at(matrixTotal_TOT, (xpos,ypos), tot)
            #if(ievent > 2000 and n_instant_0 < 10):
            #    np.add.at(matrix_instant_0, (xpos,ypos), tot)
            #    n_instant_0+=1
            #if(ievent > 3000 and n_instant_1 < 10):
            #    np.add.at(matrix_instant_1, (xpos,ypos), tot)
            #    n_instant_1+=1

        # checks/cuts:

        ## selecting only interesting events 
        goodLength = True if (length[ievent]<=20) else False
        goodSumTOT = True if (sumTOT[ievent]<=20000) else False
        goodHits = True if (hits[ievent]>=25 and hits[ievent]<1000) else False
        goodExcent = True if (excent[ievent] < 50) else False
        goodArea = True if (areaWL < 50) else False

        # Cuts for Background run file!  
        #goodLength = True if (length[ievent]<=6) else False
        #goodSumTOT = True if (sumTOT[ievent]<=8000) else False
        ##goodHits = True if (hits[ievent]>=50 and hits[ievent]<1000) else False
        #goodHits = True if (hits[ievent]<600) else False
        #goodExcent = True if (excent[ievent] < 15) else False
        #goodArea = True if (areaWL < 50) else False

        ## timing cuts
        #goodTOA_len = True if (toaLength[ievent]<=200) else False
        #goodTOA_rms = True if (toaRMS[ievent]<=100) else False
        #goodTOA_mean = True if (toaMean[ievent]<=100) else False
        ##goodTOA_skew = True if (toaSkew[ievent]<=) else False

        # selecting Based on 200k simulation of 10.541keV photons
        # Specific to: 153kHz data 
        #goodLength = True if (length[ievent]<=5) else False
        #goodWidth = True if (length[ievent]<=2.8) else False
        #goodSumTOT = True if (sumTOT[ievent]<=6000) else False
        #goodHits = True if (hits[ievent]>40 and hits[ievent]<250) else False
        #goodExcent = True if (excent[ievent] < 15) else False
        #goodRMSL = True if (RMS_L[ievent] <= 1.4) else False

        ## Specific to: 120Hz data 
        # 
        # COMMENT
        # 120Hz data seems to be complete shite 
        # cuz' cutting explicitly on thed supposed beamspot and apllying cuts 
        # on length/sumTOT/excenctricity and others 
        # seem to cut goddamn noise and 'tis the rejected part that shows soem modulation
        
        #goodLength = True if (length[ievent]<=3) else False
        #goodWidth = True if (length[ievent]<=2.) else False
        #goodSumTOT = True if (sumTOT[ievent]>= 3750 and sumTOT[ievent]<=6200) else False
        #goodHits = True if (hits[ievent]>100 and hits[ievent]<170) else False
        #goodExcent = True if (excent[ievent] <=5 ) else False
        #goodRMSL = True if (RMS_L[ievent] <= 1.2) else False
        #goodConvX = True if (conversionX[ievent] >= 100 and conversionX[ievent] <= 150) else False
        #goodConvY = True if (conversionY[ievent] >= 125 and conversionY[ievent] <= 175) else False 

        ## Specific to: 1,26kHz data 
        #goodLength = True if (length[ievent]<=4) else False
        #goodWidth = True if (length[ievent]<=2.8) else False
        #goodSumTOT = True if (sumTOT[ievent]<=6000) else False
        #goodHits = True if (hits[ievent]>40 and hits[ievent]<250) else False
        #goodExcent = True if (excent[ievent] < 50) else False
        #goodRMSL = True if (RMS_L[ievent] <= 1.2) else False

        ## Specific to: 15.1 kHz data 
        #goodLength = True if (length[ievent]<=5) else False
        #goodWidth = True if (length[ievent]<=2.8) else False
        #goodSumTOT = True if (sumTOT[ievent]<=7000) else False
        #goodHits = True if (hits[ievent]>85 and hits[ievent]<150) else False
        #goodExcent = True if (excent[ievent] < 20) else False
        #goodRMSL = True if (RMS_L[ievent] <= 1.25) else False

        # selecting basically everything
        #goodLength = True if (length[ievent] >= 0.1) else False
        #goodSumTOT = True if (sumTOT[ievent] >= 100) else False
        #goodHits = True if (hits[ievent] >= 25) else False
        #goodExcent = True if (excent[ievent] > 1 and excent[ievent] < 30) else False

        sum_TOTX, sum_TOTY = 0,0
        for itot, ix, iy in zip(event, x[ievent], y[ievent]):
            sum_TOTX += itot*ix
            sum_TOTY += itot*iy

        weightX = sum_TOTX/np.sum(event)
        weightY = sum_TOTY/np.sum(event)
       
        np.add.at(weightCenters, (int(np.round(weightX)), int(np.round(weightY))), 1)
 
        # overall cut to exclude outer part of chip
        goodXY = True if (weightX>=3 and weightX <=252 and weightY >= 3 and weightY <= 252) else False
        # cutting lower edge of matrix in BGR data
        #goodXY = True if (weightX>=3 and weightX <=252 and weightY >= 45 and weightY <= 252) else False
        # ROME DATA Cu 8keV: selecting the line of vertical scan in  the middle
        #goodXY = True if (weightX>=120 and 
        #                    weightX <=150 and 
        #                    weightY >= 40 and 
        #                    weightY <= 250) else False
        #goodConvX = True if (conversionX[ievent] >= 85 and conversionX[ievent] <= 170) else False
        #goodConvY = True if (conversionY[ievent] >= 100 and conversionY[ievent] <= 165) else False
        # ============== DIRECT BEAM ==============================
        # Direct beam cut on beamspot
        #goodXY = True if (weightX>=110 and weightX <=150 and weightY >= 130 and weightY <= 180) else False
        # Direct beam 1.26kHz 0deg beam spot (weighted enter of tracks)
        #goodXY = True if (weightX>=75 and weightX <= 175 and weightY >= 100 and weightY <= 200) else False
        #goodXY = True if ((weightX>=45 and weightX <= 85) or (weightX>=160 and weightX <= 210) and (weightY >= 70 and weightY <= 120) or (weightY >= 190 and weightY <= 230)) else False
        # Direct beam 15kHz 60deg beam spot (weighted enter of tracks)
        #goodXY = True if (weightX>=120 and weightX <= 210 and weightY >= 120 and weightY <= 185) else False
        # Direct beam 153kHz 0deg beam spot (weighted enter of tracks)
        #goodXY = True if (weightX>=80 and weightX <= 185 and weightY >= 100 and weightY <= 200) else False
        #goodConvX = True if (conversionX[ievent] >= 90 and conversionX[ievent] <= 160) else False
        #goodConvY = True if (conversionY[ievent] >= 120 and conversionY[ievent] <= 180) else False
        #goodConvXY = checkClusterPosition(abs_meanx, 
        #                                  abs_meany,
        #                                  conversionX[ievent],
        #                                  conversionY[ievent],
        #                                  abs_stdy
        #                                 ) 
        
        #X0 = 150
        #Y0 = 125
        #radius = 70
        #goodXY = checkClusterPosition(X0, Y0, weightX, weightY, radius) 

        #============== CHARGE PEAK ====================================
        # charge signal cut
        #goodXY = True if (weightX>=50 and weightX <=200 and weightY >= 60 and weightY <= 190) else False
        # chekin' supposed background 
        #goodXY = True if ((conversionX[ievent]>=30 and 
        #                conversionX[ievent] <=210 and 
        #                conversionY[ievent] >= 60 and 
        #                conversionY[ievent] <= 110) or
        #                (conversionX[ievent]>=30 and 
        #                conversionX[ievent] <=75 and 
        #                conversionY[ievent] >= 50 and 
        #                conversionY[ievent] <= 225
        #                )) else False
        #goodConvXY = checkClusterPosition(abs_meanx, 
        #                                  abs_meany,
        #                                  conversionX[ievent],
        #                                  conversionY[ievent],
        #                                  3*abs_stdx
        #                                 )  
        # using ellipse cut defined for major axis along x-axis
        goodConvXY = checkClusterPositionEllipse(abs_meanx, 
                                                 abs_meany-7,
                                                 conversionX[ievent],
                                                 conversionY[ievent],
                                                 3.2*abs_stdx,
                                                 2.5*abs_stdy
                                                 )  

        #============== MAGNETIC PEAK ====================================
        # peak position for for magnetic peak meas
        #goodXY = True if (weightX>=60 and weightX <=130 and weightY >= 75 and weightY <= 130) else False
        #goodXY = True if (weightX>=70 and weightX <=140 and weightY >= 75 and weightY <= 150) else False
        #goodConvX = True if (conversionX[ievent] >= 75 and conversionX[ievent] <= 150) else False
        #goodConvY = True if (conversionY[ievent] >= 80 and conversionY[ievent] <= 120) else False
        # Magnetic peak, first run 
        #goodConvXY = checkClusterPosition(105, 
        #                                  105,
        #                                  conversionX[ievent],
        #                                  conversionY[ievent],
        #                                  30
        #                                 )  
        # looking at the background part:
        #goodXY = True if (weightX>=150 and weightX <=225 and weightY >= 50 and weightY <= 150) else False

        # IF EXCENTRICITY IS good :
        pol_angle = None
        if (#goodLength and 
            #goodWidth and 
            #goodArea and
            #goodXY and 
            #goodConvX and 
            #goodConvY and 
            #not goodConvXY and 
            #goodRMSL and
            #goodSumTOT and 
            #goodHits and
            goodExcent #and
             ):# improved based on length cut

            yoba_pos =  [x[ievent],y[ievent]]
            clusterangle, i_k, q_k, u_k = runAngleReco(yoba_pos,event)
            #_, PHI2, _, _, _, _, _, _ = runTwoStepAngleReco(np.array(yoba_pos), event)
            #alt_ik, alt_qk, alt_uk = getIthSokesParams(PHI2)
  
            tmp_secondangles.append(secondangles[ievent]) 
            np.add.at(weightCenters_cut, (int(np.round(weightX)), int(np.round(weightY))), 1)
          
            #Weighting by excentricity 
            #wi = 1

            sum_I += i_k#*wi
            sum_Q += q_k#*wi
            sum_U += u_k#*wi
            
            ##################################
            #altsum_I += alt_ik
            #altsum_Q += alt_qk
            #altsum_U += alt_uk

            if(hasTheta):
                tmp_theta.append(checkNANTheta(theta_secondstage[ievent]))
            tmp_densityRMS.append(checkNAN(densityRMS))
            tmp_densityWL.append(checkNAN(densityWL))
            tmp_excent.append(checkNAN(excent[ievent]))
            tmp_hits.append(checkNAN(hits[ievent]))
            tmp_sumTOT.append(checkNAN(sumTOT[ievent]))
            tmp_length.append(checkNAN(length[ievent]))
            tmp_width.append(checkNAN(width[ievent]))
            tmp_RMST.append(checkNAN(RMS_T[ievent]))
            tmp_RMSL.append(checkNAN(RMS_L[ievent]))
            tmp_evtArea.append(areaWL)
            if(fConverted):
                tmp_convX.append(checkNAN(conversionX[ievent]))
                tmp_convY.append(checkNAN(conversionY[ievent]))
                np.add.at(absorption_points_pruned, (int(np.round(checkNAN(conversionX[ievent]))),int(np.round(checkNAN(conversionY[ievent])))),1)

            if(fCharge):
                tmp_sumElec.append(isumElec)
                tmp_elecPerHit.append(isumElec/nhits)

            if(goodXY):
            #if(goodConvXY):
                tmp_secondangles_Y.append(checkNAN(secondangles[ievent]))
                tmp_weightedY.append(checkNAN(weightY))
                tmp_weightedX.append(checkNAN(weightX))

            REAL_ALT_ROTANG.append(clusterangle)
            #SECOND_STEP_ALT_ROTANG.append(PHI2)
            #pol_angle = clusterangle
            for ix,iy,itot in zip(x[ievent],y[ievent], event):
                np.add.at(matrix_cut, (ix,iy), 1)
                np.add.at(cut_TOTMAP, (ix,iy), itot)
            naccepted+=1

        else:
            tmp_BGRangles.append(secondangles[ievent]) 
            yoba_pos =  [x[ievent],y[ievent]]
            clusterangle, i_k, q_k, u_k = runAngleReco(yoba_pos,event)
            sum_I -= i_k
            sum_Q -= q_k
            sum_U -= u_k
 
        #this_angle = True if (secondangles[ievent] >=-2.055 and secondangles[ievent] <= -1.92) else False
        #if((npics < 50 and nhits > 25 and excent[ievent] > 2) or (nhits>2e4)): # this one get the actual tracks

        if((npics < 10 and
           #goodLength  and
           #goodConvX and 
           #goodConvY and 
           #not goodConvXY and 
           #goodXY and
           #goodSumTOT and
           #density < 0 #and 
           #goodHits and
           goodExcent) or
           nhits > 5000
           ): # this one get the actual tracks

            matrix = np.zeros((256,256),dtype=np.uint16)
            n_good+=1
            for k,l,m in zip(x[ievent],y[ievent],event):
                np.add.at(matrix, (k,l), m)               
 
            # this block is for plotting photoelectron tracks: -------------------------
            if(fConverted):
                plot2dEvent(matrix, characs, f"cluster-{ievent}-{plotname}", outdir, [conversionX[ievent],conversionY[ievent]])
            else:
                plot2dEvent(matrix, characs, f"cluster-{ievent}-{plotname}", outdir, [weightX,weightY])

            #------------------------------------------------------------------
            # for viewing 3D tracks
            if(toaLength is not None):
                zpos = []
                vdrift_NeCO2 = 17.7867
                for toa, ftoa in zip(ToA[ievent], fTOA[ievent]):
                    zpos.append(calculateZpos(toa,ftoa,vdrift_NeCO2))

                plotInteractive3D(x[ievent],
                                y[ievent],
                                zpos,
                                [f"Reconstructed Track {ievent}", "x","y","z"], 
                                f"3D-Track-{ievent}-"+plotname, 
                                outdir)
            #------------------------------------------------------------------

            #try: 
            #    other2Dplot(matrix, f"zoom-cluster-{ievent}-{plotname}",outdir)
            #except:
            #    pass

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

#print(f"reduced TOT has <{len(tot_reduced)}> entries\n")

fitAndPlotModulation(np.array(tmp_secondangles),
                     100,
                     -np.pi,
                     np.pi,
                     ["Reconstructed Angle Distribution (Pruned by XY position)", "Angle [radian]", r"$N_{Entries}$"],
                     "STOLEN-XpolSecondStage-CutOnPosition",
                     outdir)

fitAndPlotModulation(np.array(tmp_BGRangles),
                     100,
                     -np.pi,
                     np.pi,
                     ["Reconstructed Angle Distribution (Pruned by XY position)", "Angle [radian]", r"$N_{Entries}$"],
                     "STOLEN-XpolSecondStage-CutOnPosition-BGRangles",
                     outdir)

#hits_arr = np.asarray(hitsPerEvent, dtype=np.uint16)

plot2dEvent(matrixTotal, "", "OCCUPANCY-total-run", outdir)
plot2dEvent(matrixTotal_TOT, "", "TOT-total-run", outdir)
plot2DProjectionXY(matrixTotal, ["Occupancy X projection", "X [pixel]", r"$N_{Entries}$","Occupancy Y projection", "Y [pixel]", r"$N_{Entries}$"], plotname, outdir)
plot2dEvent(matrixTotal, "", "OCCUPANCY-total-run", outdir, figtype="pdf")

plot2dEvent(weightCenters, "title:Weight Centers of All tracks", "weight-centers", outdir)
plot2dEvent(weightCenters_cut, "title:Weight Centers of All tracks (CUT)", "weight-centers-cut", outdir)
if(fConverted):
    plot2dEvent(absorption_points_pruned, "title:Absorption Points Pruned", "AbsPointsPruned",outdir)

#plot2dEvent(matrix_instant_0, "title:Events over 10 polling cycles", "OCCUPANCY-10-events", outdir)
#plot2dEvent(matrix_instant_1, "title:events over 10 polling cycles", "OCCUPANCY-another10-events", outdir)
#cut_comment = ""
try:
    plot2dEvent(matrix_cut, "title:Occupancy, Cut on Position", "OCCUPANCY-CUT-ON-position",outdir)
except:
    print("Failed to plot matrix_cut")
    pass
try:
    plot2dEvent(cut_TOTMAP, "title:", "TOT-map-cut", outdir)
except:
    print("Failed to plot cut_TOTMAP")
    pass

simpleHist(REAL_ALT_ROTANG, 100, -np.pi, np.pi, ["Filtered First Step Angle Reco","Reconstructed angle,[rad]","Events,[N]"], "REAL_ALT_ROTANG", outdir, fit="cosfunc") 
#simpleHist(SECOND_STEP_ALT_ROTANG, 100, -np.pi, np.pi, ["TWO-STEP Reconstructed Angles",r"$\phi_{0}$,[rad]",r"$N_{tracks}$,[N]"], "SECOND_STEP_ALT_ROTANG", outdir, fit="cosfunc") 
#simpleHist(tot_reduced, 100, np.nanmin(tot_reduced), np.nanmax(tot_reduced), ["TOT Reduced by Number of Hits Per Event",r"$N {hits}$",r"$TOT/N_{Hits}$"], "TOT_REDUCED", outdir, fit="gaus")

#totvals, totcts = np.unique(TOTarray, return_counts=True)
#totidx = np.argmax(totcts)
#totMode = totvals[totidx]
#totmaxbin = np.ceil(totMode*3)
#simpleHist(np.array(TOTarray), 100, 0, totmaxbin, [r"$\Sigma$ TOT Per Event",r"$N {Events}$",r"$\Sigma$ TOT"], "ALT_sumTOT", outdir, fit="gaus")
#
#hitvals, hitcts = np.unique(hits_arr, return_counts=True)
#hitidx = np.argmax(hitcts)
#hitMode = hitvals[hitidx]
#hitmaxbin = np.ceil(hitMode*3)
#simpleHist(hits_arr, 100, np.nanmin(hits_arr), hitmaxbin, ["Hits per Event",r"$_{hits}$",r"$N_{Events}$"], "HITS", outdir)

# ---------- plotting parameters after cuts ----------------------

simpleHist(np.array(tmp_excent), 100, np.nanmin(tmp_excent), np.nanmax(tmp_excent), ["Excentricity","Excentricity","Events,[N]"], "tmp_excent", outdir) 
simpleHist(np.array(tmp_hits), 100, np.nanmin(tmp_hits), np.nanmax(tmp_hits), ["Hits per Cluster",r"$N_{hits}$","Events,[N]"], "tmp_hits", outdir) 
simpleHist(np.array(tmp_sumTOT), 100, np.nanmin(tmp_sumTOT), np.nanmax(tmp_sumTOT), [r"$\Sigma$(TOT) per Cluster",r"$\Sigma$(TOT)","Events,[N]"], "tmp_sumTOT", outdir) 
simpleHist(np.array(tmp_densityRMS), 100, np.nanmin(tmp_densityRMS), np.nanmax(tmp_densityRMS), ["Cluster Density RMS Ratios",r"$RMS_{T}/RMS_{L}$","Events,[N]"], "tmp_densityRMS", outdir) 
simpleHist(np.array(tmp_densityWL), 100, np.nanmin(tmp_densityWL), np.nanmax(tmp_densityWL), ["Cluster Density Width/Length","width/length","Events,[N]"], "tmp_densityWL", outdir) 
simpleHist(np.array(tmp_evtArea), 100, np.nanmin(tmp_evtArea), np.nanmax(tmp_evtArea), ["Cluster Area",r"$width \cdot length$","Events,[N]"], "tmp_evtArea", outdir) 
if(hasTheta):
    simpleHist(np.array(tmp_theta), 100, -np.pi, np.pi, [r"Cluster $\theta$",r"$\theta$, [radian]","Events,[N]"], "tmp_theta", outdir) 
if(fCharge):
    simpleHist(np.array(tmp_sumElec), 100, np.nanmin(tmp_sumElec), np.nanmax(tmp_sumElec), ["Cluster Charge (electrons)",r"$\Sigma (e^{-})$,","Events,[N]"], "tmp_sumElec", outdir) 
    simpleHist(np.array(tmp_elecPerHit), 100, np.nanmin(tmp_elecPerHit), np.nanmax(tmp_elecPerHit), ["Electrons per Hit",r"$N_{e^{-}}/N_{Hits}$,","Events,[N]"], "tmp_elecPerHit", outdir) 

plot2Dhist(np.array(tmp_hits), np.array(tmp_densityRMS), [r"Cluster Hits Vs Cluster Density (RMS)", "Hits", r"$RMS_{T}/RMS_{L}$"], "2D-DesityRMS-vs-Hits", outdir)
plot2Dhist(np.array(tmp_hits), np.array(tmp_densityWL), [r"Cluster Hits Vs Cluster Density (WL)", "Hits", "width/length"], "2D-DesityWL-vs-Hits", outdir)
plot2Dhist(np.array(tmp_hits), np.array(tmp_excent), [r"Cluster $\epsilon$ vs Hits", "Hits", r"$\epsilon$"], "2D-Hits-vs-Epsilon", outdir)
plot2Dhist(np.array(tmp_hits), np.array(tmp_sumTOT), [r"$\Sigma$(TOT) vs Hits", "Hits", r"$\Sigma$(TOT)"], "2D-Hits-vs-sumTOT", outdir)
plot2Dhist(np.array(tmp_sumTOT), np.array(tmp_length), [r"Cluster Length $\Sigma$(TOT)s", r"$\Sigma$(TOT)", "Cluster Length"], "2D-sumTOT-vs-Length", outdir)
plot2Dhist(np.array(tmp_sumTOT), np.array(tmp_evtArea), [r"Cluster Area vs $\Sigma$(TOT)s", r"$\Sigma$(TOT)", r"Cluster ($width\cdot length$)"], "2D-ClusterArea-vs-sumTOT", outdir)
plot2Dhist(np.array(tmp_sumTOT), np.array(tmp_densityRMS), [r"Cluster Density vs $\Sigma$(TOT)", r"$\Sigma$(TOT)", r"$RMS_{T}/RMS_{L}$"], "2D-sumTOT-vs-density", outdir)
plot2Dhist(np.array(tmp_excent), np.array(tmp_length), ["Cluster Length vs Cluster Epsilon", r"$\epsilon$", "Cluster Length"], "2D-Epsilon-Length", outdir)
plot2Dhist(np.array(tmp_width), np.array(tmp_length), ["Cluster length vs Cluster width", "Cluster Width", "Cluster Length"], "2D-Width-Length", outdir)
plot2Dhist(np.array(tmp_RMSL), np.array(tmp_RMST), ["Transversal vs Longitudinal cluster RMS", r"$RMS_{L}$", r"$RMS_{T}$"], "2D-RMSL-RMST", outdir)
plot2Dhist(np.array(tmp_weightedY), np.array(tmp_secondangles_Y), ["Charge-Weighted cluster Y vs reco angle", "Y", "Reconstructed angle"], "2D-weightY-SecAng", outdir)
plot2Dhist(np.array(tmp_weightedX), np.array(tmp_secondangles_Y), ["Charge-Weighted cluster X vs reco angle", "X", "Reconstructed angle"], "2D-weightX-SecAng", outdir)

if(fConverted):
    plot2Dhist(np.array(tmp_convX), np.array(tmp_secondangles_Y), [r"Reconstructed $\phi$ vs reconstructed Absorption point X", "Absorption X", r"$\phi_{reco}$"], "2D-AbsPointX-SecAng", outdir)
    plot2Dhist(np.array(tmp_convY), np.array(tmp_secondangles_Y), [r"Reconstructed $\phi$ vs reconstructed Absorption point Y", "Absorption Y", r"$\phi_{reco}$"], "2D-AbsPointY-SecAng", outdir)
    plot2Dhist(np.array(tmp_convX), np.array(tmp_sumTOT), [r"Cluster $\Sigma$(TOT) X vs reconstructed Absorption X", "Absorption X", r"$\Sigma$(TOT)"], "2D-AbsPointX-vs-sumTOT", outdir)
    plot2Dhist(np.array(tmp_convY), np.array(tmp_sumTOT), [r"Cluster $\Sigma$(TOT) Y vs reconstructed Absorption Y", "Absorption Y", r"$\Sigma$(TOT)"], "2D-AbsPointY-vs-sumTOT", outdir)



#if(fTiming):
#    plot2Dhist(np.array(tmp_sumTOT), np.array(tmp_toaLength), [r"Cluster $\epsilon$ vs ToA length", r"$\epsilon$", "ToA Length [CLK]"], "2D-Hits-TOALength", outdir)
#    plot2Dhist(np.array(tmp_sumTOT), np.array(tmp_toaMean), [r"Cluster $\epsilon$ vs ToA Mean", r"$\epsilon$", "ToA Mean [CLK]"], "2D-Epsilon-TOAMean", outdir)
#    plot2Dhist(np.array(tmp_sumTOT), np.array(tmp_toaRMS), [r"Cluster $\epsilon$ vs ToA RMS", r"$\epsilon$", "ToA RMS [CLK]"], "2D-Epsilon-TOARMS", outdir)
#    plot2Dhist(np.array(tmp_sumTOT), np.array(tmp_toaSkew), [r"Cluster $\epsilon$ vs ToA Skeweness", r"$\epsilon$", "ToA Skew [CLK]"], "2D-Epsilon-TOASkew", outdir)

#plotInteractive3D(tmp_weightedX,
#                tmp_weightedY,
#                tmp_secondangles_Y,
#                ["Reconstructed Angle vs Weighted position of track", "x","y","PHI0"], 
#                f"3D-Angles-vs-weightedTrackPos-"+plotname, 
#                outdir)
# ------------ cyclotron data here: --------------------------
#simpleHistSpectrum(TOTarray, 101, 0, 60000, ['TOT cyles per event (Before Irradiation)', 'TOT cycles','N'], "sumTOT", outdir, scale="linear")

print(f"\n Cut on EXCENTRICITY accepted only: {naccepted/ntotal*100:.2f}% of data\n")
print("Calculated sum of Stokes parameters:")
print(f"I = {sum_I:.2f}  Q = {sum_Q:.2f} , U = {sum_U:.2f}")
normQ = sum_Q/sum_I
normU = sum_Q/sum_I
print(f"Normalized  :    Q = {normQ:.2f} , U = {normU:.2f}")
#pL = 2*0.2*np.sqrt(sum_Q**2 + sum_U**2)
#direction_psi = 0.5*math.atan(normU/normQ)
#print(f"Fraction of linear polarization: {pL:.2f} with direction {direction_psi*180/np.pi:.2f}")
#print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#print("Calculated ALTERNATIVE sum of Stokes parameters:")
#print(f"I = {altsum_I:.2f}  Q = {altsum_Q:.2f} , U = {altsum_U:.2f}")
#altnormQ = 2/0.2*altsum_Q/altsum_I
#altnormU = 2/0.2*altsum_Q/altsum_I
#print(f"Normalized  :       Q = {altnormQ:.2f} , U = {altnormU:.2f}")
#alt_pL = np.sqrt(altsum_Q**2 + altsum_U**2)/altsum_I
#alt_direction_psi = 0.5*math.atan(altnormU/altnormQ)
#print(f"Fraction of linear polarization: {alt_pL:.2f} with direction {alt_direction_psi*180/np.pi:.2f}")
#print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

print(f"TOTAL events: {ntotal}")
#print(f"protons counted: {nprotons}")


