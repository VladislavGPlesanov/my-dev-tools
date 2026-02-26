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
import glob
import datetime as dtime
from matplotlib.patches import Ellipse
import argparse as ap

import ROOT as rt 

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
OUT_RED_BGR = '\x1b[41m'
OUT_BLUE_BGR = '\x1b[44m'
OUT_GREEN_BGR = '\x1b[42m'
OUT_YEL_BGR = '\x1b[43m'
OUT_CYAN_BGR = '\x1b[46m'
OUT_WHITE_BGR = '\x1b[47m'
OUT_MAG_BGR = '\x1b[45m'


#######
# defining cut setting dictionary
#

# run times of the data sets fro direct beam
TIME_DICT={
    "DB-120Hz-0deg":3651.5069,
    "DB-1260Hz-0deg":1810.9238,
    "DB-1260Hz-30deg":1814.2037,
    "DB-1260Hz-60deg":2542.2456,
    "DB-1260Hz-90deg":2557.6083,
    "DB-15100Hz-0deg":910.3009,
    "DB-15100Hz-30deg":908.9542,
    "DB-15100Hz-60deg":909.7484,
    "DB-15100Hz-90deg":911.8904,
    "DB-153000Hz-0deg":460.7452,
    "DB-153000Hz-30deg":459.7881,
    "DB-153000Hz-60deg":459.2557,
    "DB-153000Hz-90deg":461.6843,
    "CP0": 2536.6465,
    "CP1": 18.6879,
    "CP2": 11.9262,
    "CP3": 1817.0506,
    "CP4": 1828.4342,
    "CP5": -1,
    "CP6": 1816.2043,
    "CP7": 18.6959,
    "MP1": 1809.7429,
    "MP2": 1813.8812,
    "MP3": 1213.6367,
    "MP4": 613.7001,
    "MP5": 1815.8047,
    "MP6": 1832.4772,
    "MP7": 3031.4198,
 
}

CUTS_DICT={
    "DB-120Hz-0deg":{
        "sumTOT": [3750,6200],
        "length": 3.0,
        "width": 2.0,
        "nhits": 100, 
        "weightXY": "none", 
        "convX": [100,150], 
        "convY": [125,175], 
        "convXY":"none", 
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
        "convXY":"none", 
        "Excent":5.0, 
        "RMSL":1.2
    },
    "DB-1260Hz-0deg":{
        "sumTOT": 6000,
        "length": 4.0,
        "width": 2.8,
        "nhits": [40,250], 
        "weightXY": "none", 
        "convX": [100,150], 
        "convY": [125,175], 
        "convXY":"none", 
        "Excent":50, 
        "RMSL":"none"
    },
    "BGR":{
        "sumTOT": "none",
        "length": "none",
        "width": "none",
        "nhits": 25, 
        "weightXY": "none", 
        "convX": [3,252], 
        "convY": [3,252], 
        "convXY":"none", 
        "Excent":50, 
        "RMSL":"none"
    },
    "SIM-TPX3":{
        "sumTOT": -20000,
        "length": -20,
        "width": "none",
        "nhits": 25, 
        "weightXY": "none", 
        "convX": [3,252], 
        "convY": [3,252], 
        "convXY":"none", 
        "Excent":-50, 
        "RMSL":"none"
    },
    "CP0":{
        "sumTOT": -20000,
        "length": -20,
        "width": "none",
        "nhits": [25,1000], 
        "weightXY": "none", 
        "convX": "none", 
        "convY": "none", 
        "convXY":"none", 
        "Excent":[1.2,20], 
        "RMSL":"none"
    }

}


# dictionary to store main results for output
main_results={
    "Mod_Orig":{},
    "Mod_BGR":{},
    "Mod_Cut":{}
}


def getCut(dict_entry, check_value):

    # e.g. if i pass CUTS_DICT["DB120Hz-0deg"]["sumTOT"]
    #                or cutset["sumTOT"]
    
    if(isinstance(dict_entry,list)):
        maxvalue = dict_entry[1]
        minvalue = dict_entry[0]
        return True if (check_value <= maxvalue and check_value >= minvalue) else False

    elif(isinstance(dict_entry,str)):
        if(dict_entry=="none"):
            return True
        else:
            return False
    else:
        if(dict_entry > 0):
            return True if (check_value >= abs(dict_entry)) else False
        else:
            return True if (check_value <= abs(dict_entry)) else False

    #return False



#######
def linefunc(x,A,B):

    return A*x + B

def cosfunc(x, Ap, Bp, phi0):

    return Ap + Bp*np.cos(x - phi0)**2

def gauss(x, A, mu, sigma):

    return A*np.exp(-((x-mu)**2)/(2*sigma**2))

def modFac(Ap,Bp):

    return Bp/(2*Ap+Bp)

def calcIntensity(A, B):

    return A + B/2.0

def calcIntensityError(sigA, sigB):

    return np.sqrt(sigA**2 + 0.5*sigB**2)

def deltaMu(mu, Ap, Bp, Ap_err, Bp_err):

    return mu * np.sqrt( (Bp_err / Bp)**2  + (2 * Ap_err / (Bp + 2*Ap) )**2 )

def polDeg(Ap, Bp, mu):

    return (1/mu)*(Bp/((2*Ap)+Bp))

def getPolDegree(Imax, Imin):

    return (Imax - Imin)/(Imax + Imin)

def getMDP(mu, nevents):

    return 4.29/(mu*np.sqrt(nevents))

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
def checkNANphi(phi):
    
    if(np.isnan(phi)):
        return 10
    else:
        return phi

def checkNANTheta(theta):

    if(np.isnan(theta)):
        return -999
    else:
        return theta


def safe_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as ex:
        print(f"{OUT_RED}[ERROR]: {ex}")
    else:
        return 0

def checkRateOrder(rate):

    number = rate
    rate = str(int(abs(rate)))
    
    if(len(rate)>0 and len(rate)<=3):
        return number, "Hz"
    elif(len(rate)>3 and len(rate)<=6):
        return round(number/1000.0,2), "kHz"
    elif(len(rate)>6 and len(rate)<=9):
        return round(number/1000000.0,2), "MHz"
    else:
        return number, "IMPOSIBRU!"    

def clearString(string, old_sep, new_sep, nolist):

    words = string.split(old_sep)
    filtered_words = [word for word in words if word not in nolist]
    return new_sep.join(filtered_words)

def makeFancySection(textlist, color=None):

    longest = max(len(text) for text in textlist)
    padded_list = [text.ljust(longest) for text in textlist]

    for text in padded_list:
        if(color is not None):
            print(f"{color}{text}{OUT_RST}")
        else:
            print(f"{text}")

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

def getSingleTrackSokes(angle):

    i = 1
    q = np.cos(2*angle)
    u = np.sin(2*angle)

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

def simpleScatter(xdata, ydata, labels, picname, odir):

    if(debug):
        print(len(xdata))
        print(len(ydata))

    if(len(xdata)!=len(ydata)):
        print(f"{OUT_RED}[simpleScatter] ERROR, data mismatch -> x={len(xdata)}, y={len(ydata)}{OUT_RST}")

    plt.figure(figsize=(14,8))
    plt.scatter(xdata,ydata, marker='.', c='green')
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.title(labels[0])
    plt.grid(which='both')
    #plt.legend(loc='upper right')
    if("GHOSTS" in picname):
        plt.yscale('log')
    plt.savefig(f"{odir}/1D-SCAT-{picname}.png")
    plt.close()

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

def getClusterMedian(matrix):
        
    x,y = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]),indexing='ij')
    
    norm = matrix.sum()

    medX = np.median(matrix*x)/norm
    medY = np.median(matrix*y)/norm

    return medX, medY

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


def fitAndPlotModulation(nuarray, nbins, minbin, maxbin, labels, picname, odir, debug):

    # faithfully stolen from Markus's plot_angles.py
    # fitting 
    if(debug):
        print(f"{OUT_MAGEN}[fitAndPlotModulation] --> {picname} --> {labels[0]} {OUT_RST}")
    clear_data = nuarray[~np.isnan(nuarray)]
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

    # ------ checking covarinces ----------
    if(debug):
        print(f"covariance matrix \nlen={len(covariance)}, \nshape={covariance.shape},\n{covariance}")

    # -------------------------------------

    Ap, Bp, phi = params

    expected_cts = cosfunc(bin_centers, *params)
    
    chisquare = np.sum(((counts - expected_cts)**2 )/ count_errors**2)
    dof = len(counts) - len(params)
    chired = chisquare / dof

    Ap_err, Bp_err, phi_err = np.sqrt(np.diag(covariance))

    fBinColor = None

    mukey, Apkey, Bpkey, phikey, chikey = "mu","Ap", "Bp", "phi", "chired"
    muerrkey, Aperrkey, Bperrkey, phierrkey = "muErr", "Aperr", "Bperr", "phierr"
    Ikey, Qkey, Ukey = "I","Q","U"
    Ierrkey, Qerrkey, Uerrkey = "Ierr", "Qerr", "Uerr"
    branch = None

    if("BGR" in picname):
        fBinColor='gray'
        branch = "Mod_BGR"
    elif("CutOnPosition" in picname):
        fBinColor='darkorange'
        branch = "Mod_Cut"
    else:
        fBinColor='forestgreen'
        branch = "Mod_Orig"
        
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

    plt.hlines(Ap, -0.1, 3.15, colors='yellow', label=f"Ap={Ap:.2f}")
    plt.hlines(Ap+Bp, -0.1, 3.15, colors='lightseagreen', label=f"Bp={Bp:.2f}")
 
    ax = plt.gca()
    miny,maxy = ax.get_ylim()
    if(debug):
        print(f"Getting miny/maxy for histogram: {picname}")
        print(f"miny={miny:.2f}, maxy={maxy:.2f}")

    plt.ylim([miny, maxy*1.2])

    mu = modFac(Ap,Bp)
    muErr = deltaMu(mu, Ap, Bp, Ap_err, Bp_err)
    intensity = calcIntensity(Ap, Bp)
    intensity_err = calcIntensityError(Ap_err, Bp_err)
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

    #V = np.sqrt((P*intensity)**2 - Q**2 - U**2)

    #---------------------------------------------------

    MDP = getMDP(mu, len(nuarray))

    #---------------------------------------------------
    plt.text(-3.13, maxy*1.16, r"$\Sigma$"+f"(Entries)={len(nuarray)} MDP(CL99%)={MDP*100:.2f}",fontsize=11)
    plt.text(-3.13 , maxy*1.10, r"$N(\phi) = A_{P} + B_{P}\cdot cos^2(\phi-\phi_{0})$", fontsize=11)
    plt.text(-3.13 , maxy*1.04, f"{G_mu}={mu*100:.2f}%"+r"$\pm$"+f"{muErr*100:.2f}%", fontsize=11)
    plt.text(-3.13 , maxy*0.98, f"I={intensity:.2f}"+r"$\pm$"+f"{intensity_err:.2f}", fontsize=11)
    plt.text(-3.13 , maxy*0.92, f"Q={Q:.2f} +- {dQ:.2f} ({Q/intensity:.4f}+-{dQ/intensity:.4f})", fontsize=11)
    plt.text(-3.13 , maxy*0.86, f"U={U:.2f} +- {dU:.2f} ({U/intensity:.4f}+-{dU/intensity:.4f})", fontsize=11)
    plt.text(-3.13 , maxy*0.80, r"$\chi_{red}^{2}$"+f"={chired:.2f}", fontsize=11)

    if(debug):
        print("________________________________________________________")
        print(f"{OUT_RED} Modulation factor = {mu*100.0:.2f} {OUT_RST}")
        print(f"Polarization degree = {P:.2f}")
        print(f"\nStokes Parameters: \nQ(P1)={Q:.4f}, U(P2)={U:.4f}")#, V={V:.4f}\n")
        print(f"Intensity = {intensity}")
        print(f"{OUT_RED}{G_chi}={chired:.2f}{OUT_RST}")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    main_results[branch][mukey] = mu
    main_results[branch][Apkey] = Ap
    main_results[branch][Bpkey] = Bp
    main_results[branch][phikey] = phi
    main_results[branch][chikey] = chired
    main_results[branch][muerrkey] = muErr
    main_results[branch][Aperrkey] = Ap_err
    main_results[branch][Bperrkey] = Bp_err
    main_results[branch][phierrkey] = phi_err
    main_results[branch][Ikey] = intensity
    main_results[branch][Qkey] = Q
    main_results[branch][Ukey] = U
    main_results[branch][Ierrkey] = intensity_err
    main_results[branch][Qerrkey] = dQ
    main_results[branch][Uerrkey] = dU

    plt.legend(loc='upper right')

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.grid()
    plt.savefig(f"{odir}/STOLEN-1DHist-{picname}.png")
    plt.close()

def simpleHistSpectrum(nuarray, nbins, minbin, maxbin, labels, picname, odir, debug, scale=None):

    plt.figure()
  
    plt.figsize=(8,8)

    counts, bin_edges = np.histogram(nuarray, bins=nbins, range=(minbin,maxbin))
    maxbin_cnt = np.max(counts)
    minbin_cnt = np.min(counts)

    val_ibin = (maxbin-minbin)/nbins

    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    if(debug):
        print(f"Range= {minbin:.2f} -> {maxbin:.2f}, with {nbins} bins gives {val_ibin} per bin")

    model = None

    peakbin = getMaxBin(counts[1:])
    peakbin+=1

    if(debug):
        print(f"Found maximum bin at {peakbin:.2f} = {counts[peakbin]}")
    model = Model(gauss) 
    pars = model.make_params(A=maxbin_cnt, mu=peakbin*val_ibin, sigma=np.std(counts))
    if(debug):
        print(f"Gauss Fit: Setting: A={maxbin_cnt:.2f}, mu={peakbin*val_ibin:.2f}, sigma={np.std(counts):.2f}")   
        print(len(bin_centers[:-1]))
        print(len(counts))

    plt.hist(bin_centers, weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', facecolor='b')

    #print(f"\n\ncounts={counts} \n\n")
    #print(maxbin_cnt)
    #print(minbin_cnt)
    #print(f"max_amplitude={maxbin_cnt - minbin_cnt}")
    
    if(debug):
        print("########## FITTING GASUSS FUNCTION #############")
    result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
    if(result.params['mu']<=0):

        pars['mu'].min = peakbin*0.8*val_ibin
        pars['mu'].max = peakbin*1.2*val_ibin
 
        print("FIT FAILED: restricting fit parameters and re-fitting")
        result = model.fit(counts[:-1], pars, x=bin_centers[:-1] )

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

def simpleMultiHist(datalist, nbins, minbin, maxbin, labels, picname, odir, yscale=None, fNorm=False):

    if(debug): 
        print(f"{OUT_MAGEN}[simpleMultiHist]-->{picname}-->{labels[0]} ({len(datalist)}) {OUT_RST}")
    plt.figure(figsize=(10,8))
    for data in datalist:
        if(debug):
            print(f"set: {len(data[1])}")
        counts, bin_edges = np.histogram(data[1], bins=nbins, range=(minbin,maxbin))
        if(not fNorm):
            plt.hist(bin_edges[:-1], weights=counts, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', alpha=0.2, label=f"{data[0]}")
        else:
            weights = counts/sum(counts)
            plt.hist(bin_edges[:-1], weights=weights, bins=nbins, range=(minbin,maxbin), align='left', histtype='stepfilled', alpha=0.2, label=f"{data[0]}")
        
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.legend(loc='upper right')
    plt.grid(True)
    if(yscale is not None and type(yscale)==str):
        plt.yscale(xscale)
    plt.savefig(f"{odir}/MultiHist-{picname}.png")

def simpleMultiHist3D(datalist, labels, xrange, picname, odir):

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    nbins = 100
    offsets = np.arange(0,len(datalist))
    offsets = offsets*10

    bin_width = bin_edges[1] - bin_edges[0]

    for data, yoffset in zip(datalist, offsets): 
        counts, bin_edges = np.histogram(clear_data, bins = nbins, range=xrange)
        xs = (bin_edges[:-1]+bin_edges[1:])/2
        ax.bar(xs, counts, zs=yoffset, alpha=0.5)     

    ax.set_title(f'{labels[0]}')
    ax.set_xlabel(f'{labels[1]}')
    ax.set_ylabel(f'{labels[2]}')
    ax.set_zlabel(f'{labels[3]}')

    plt.savefig(f"{odir}/MultiHist3D-{picname}.png")


def simpleHist(nuarray, nbins, minbin, maxbin, labels, picname, odir, debug, fit=None, yaxisscale=None):

    if(debug):
        print(f"{OUT_MAGEN}[simpleHist]-->{picname}-->{labels[0]} ({len(nuarray)}) {OUT_RST}")
    if(len(nuarray)==0):
        print(f"{OUT_RED}[ERROR::EMPTY_DATA]-->{picname}-->{labels[0]}{OUT_RST}")

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
        if(debug):
            print(f"Data type container for {picname} is ambiguous -> {type(nuarray)}")
        pass

    if(debug):
        print(f"Got {datasize} events selected for figure {picname}")

    residual_evt = 0
    cutoff = 0
    if("densityRMS" in picname):
        cutoff = 5000
    elif("tmp_excent" in picname):
        cutoff = 20
    elif("tmp_hits" in picname):
        cutoff = 2000
    elif("tmp_sumTOT" in picname):
        cutoff = 30000
    else:
        cutoff = 20e3

    if("GHOSTS" in picname):
       if("tmp_sumTOT" in picname):
            cutoff = 4000
       if("tmp_hits" in picname):
            cutoff = 100
   

    goodval = np.where(nuarray<=cutoff) 
    residual_evt = len(np.where(nuarray>cutoff))
    if(debug):
        print(f"[simpleHist]-->{picname}-->{labels[0]}: cutting away {residual_evt} events above {cutoff}")

    nuarray = nuarray[goodval]

    if(residual_evt>0):
        minbin = 0
        maxbin = cutoff

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

        result = None
        if(fit=="cosfunc" and debug):
            print("########## FITTING COS^2 FUNCTION #############")
            print(result.fit_report()) 
        if(fit=="gaus" and debug):
            result = model.fit(counts[:-1], pars, x=bin_centers[:-1])
            print(result.fit_report()) 
            print("########## FITTING GASUSS FUNCTION #############")
        
        nfits+=1

        if(fit=="gaus" and result.params['mu']<=0):

            pars['mu'].min = peakbin*0.8*val_ibin
            pars['mu'].max = peakbin*1.2*val_ibin
 
            if(debug):
                print("FIT FAILED: restricting fit parameters and re-fitting")
            result = model.fit(counts[:-1], pars, x=bin_centers[:-1] )
        
            nfits += 1
        ######### handing failed fit on modullation curve ################            
        fitlab = ""
        miny,maxy = 0, 0
        if(fit=="cosfunc"):
            if(debug):
                print("########## FITTING COS^2 FUNCTION #############")
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

            if(debug):
                print(f"Getting miny/maxy for histogram: {picname}")
                print(f"miny={miny}, maxy={maxy}")

            plt.text(-3.13 , maxy*0.98, r"$N(\phi) = A_{P} + B_{P}\cdot cos^2(\phi-\phi_{0})$"+r"($N_{fits}$="+f"{nfits})")
            plt.text(-3.13 , maxy*0.96, f"{G_mu}={mu*100:.2f}%")
            plt.text(-3.13 , maxy*0.94, f"I={intensity:.2f}")
            plt.text(-3.13 , maxy*0.92, f"P1={Q:.2f}")
            plt.text(-3.13 , maxy*0.90, f"P2={U:.2f}")

            if(debug):
                print("________________________________________________________")
                print(f"Modulation factor = {mu*100.0:.2f} %")
                print(f"\nStokes Parameters: \nQ(a0)={Q:.4f}, U(a1)={U:.4f}\n")
                print(f"Intensity = {intensity}")
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        elif(fit=="gaus"):
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
        if(residual_evt>0):
            hist_comment += "\n"+r"$N_{Overflow}=$"+f"{residual_evt}"

        plt.text(xmax*0.8,maxy*0.95, hist_comment)

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    if(yaxisscale is not None):
        plt.yscale(yaxisscale)
    plt.grid()
    plt.savefig(f"{odir}/1DHist-{picname}.png")
    plt.close()


def plot2Dhist(x,y, labels, picname, odir, debug):
 
    nentries_x = x.shape[0] 
    nentries_y = y.shape[0]

    if(debug):
        print(f"{OUT_GRAY}[plot2Dhist]--> {picname} --> {labels[0]} ({nentries_x},{nentries_y}) {OUT_RST}")
    if(nentries_x==0 or nentries_y==0):    
        print(f"{OUT_RED} [ERROR::EMPTY_DATA] --> {picname} MISSING DATA {OUT_RST}")
        return 0        

    mismatch = 0
    idx_stop = None
    if(nentries_x > nentries_y):
        idx_stop = nentries_y-1
        #idx_stop = nentries_y
        mismatch = nentries_y/nentries_x
        print(f"  MISMATCH in {picname} x={nentries_x}, y={nentries_y} -> {mismatch*100:.2f}%")
    elif(nentries_y > nentries_x):
        idx_stop = nentries_x-1
        #idx_stop = nentries_x
        mismatch = nentries_x/nentries_y
        print(f"  MISMATCH in {picname} x={nentries_x}, y={nentries_y} -> {mismatch*100:.2f}%")
    else:
        if(debug):
            print(f"  Data arrays equal") 

    if(mismatch>0.1):
        print(f"{OUT_RED}[plot2Dhist]: mismatch of {mismatch*100:.2f}% of data sets for {labels[0]} {OUT_RST}")

    plt.figure(figsize=(8,6))

    if("Theta" in picname or "theta" in labels[0]):
        idx = np.where(y>-999)
        plt.hist2d(x[idx], y[idx], bins=100, norm=LogNorm(), cmap="jet")
        #print("option [B]")
    elif("AbsPointY" in picname and "sumTOT" in picname):
        idx = np.where(x>-1)
        if(debug):
            print(f"Using {len(x[idx])} out of {len(x)}")
        plt.hist2d(x[idx], y[idx], bins=100, norm=LogNorm(), cmap="jet")
        #print("option [C]")
    elif("AbsPointY" in picname and "TOAMean" in picname):
        idy = np.where(y<40)
        x = x[idy]
        y = y[idy]
        idx = np.where(x>-1)
        plt.hist2d(x[idx], y[idx], bins=100, norm=LogNorm(), cmap="jet")
        #print("option [D]")
    elif("ToA RMS" in labels[0]):
        toa_cut = None
        if("GHOSTS" in picname):
            toa_cut = 40000
        else:
            toa_cut = 30
        idx = np.where(y<toa_cut)
        plt.hist2d(x[idx], y[idx], bins=100, norm=LogNorm(), cmap="jet")
        #print("option [E]")
    elif("ToA Length" in labels[0]):
        toa_cut = None
        if("GHOSTS" in picname):
            toa_cut = 40000
        else:
            toa_cut = 140
        idx = np.where(y<toa_cut)
        plt.hist2d(x[idx], y[idx], bins=100, norm=LogNorm(), cmap="jet") 
        #print("option [F]")
    elif("weightX-SecAng" in picname or "weightY-SecAng" in picname):
        idx = np.where(y<10)
        plt.hist2d(x[idx], y[idx], bins=100, norm=LogNorm(), cmap="jet") 
        #print("option [G]")
    elif("Epsilon-Length" in picname):
        idx = np.where(x<50)
        plt.hist2d(x[idx], y[idx], bins=100, norm=LogNorm(), cmap="jet") 
        #print("option [G]")
    elif("Hits-vs-Epsilon" in picname):
        idy = np.where(y<50)
        plt.hist2d(x[idy], y[idy], bins=100, norm=LogNorm(), cmap="jet") 
        #print("option [G]")
    else:
        #plt.hist2d(x[0:idx_stop], y[0:idx_stop], bins=100, norm=LogNorm(), cmap="jet")
        plt.hist2d(x[0:idx_stop], y[0:idx_stop], bins=100, norm=LogNorm(), cmap="jet")
        #print("option [else]")
    plt.colorbar(label=r"$N_{Entries}$")

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    #plt.legend()
    plt.savefig(f"{odir}/2Dhist-{picname}.png")
    plt.close()

def plotRoot2D(xdata,ydata,nbinsx,rangex,nbinsy,rangey,titles,picname,odir, debug):

    if(debug):
        print(f"{OUT_CYAN} [plotRoot2D]--> {picname} --> {titles[0]} ({len(xdata)},{len(ydata)}) {OUT_RST}")
    if(len(xdata)==0 or len(ydata)==0):
        print(f"{OUT_RED} [ERROR::EMPTY_DATA] --> {picname} MISSING DATA {OUT_RST}")
        return 0 

    hist2d = rt.TH2F(
        "hist","hist",
        nbinsx,rangex[0],rangex[1],
        nbinsy,rangey[0],rangey[1],
    )
    
    for ix,iy in zip(xdata,ydata):
        hist2d.Fill(ix,iy)

    can = rt.TCanvas("can","can",800,700)    
    hist2d.SetTitle(titles[0])
    hist2d.GetXaxis().SetTitle(titles[1])
    hist2d.GetYaxis().SetTitle(titles[2])
    hist2d.Draw("COLZ")
    can.SaveAs(f"{odir}/ROOT-2D-{picname}.png")

    del hist2d
    del can


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

def plot2dEvent(nuarray, info, picname, odir, debug, figtype=None, plotMarker=None):  
    
    if(debug):
        print(f"{OUT_YELLOW} [plot2dEvent]--> {picname} ({len(nuarray)}) {OUT_RST}")
 
    if(np.sum(nuarray)==0):
        print(f"{OUT_RED}[ERROR::EMPTY_DATA] {picname} matrix all zeros! {OUT_RST}")
        return 0

    # ---- matrix 2d hist ----
    fig, ax = plt.subplots(figsize=(8,6))
    cax = fig.add_axes([0.86,0.1,0.05,0.8])
    #ms = ax.matshow(nuarray, cmap='plasma')
    # ---------------------------------
    #
    ms = None

    deflabelx, deflabely = "x, [pixel]", "y, [pixel]"

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
        if("STOKES" in picname):
            if("STOKES-Q" in picname or "STOKES-U" in picname):
                ms = ax.matshow(nuarray.T, cmap='jet', vmin=-1,vmax=1)
            else:
                ms = ax.matshow(nuarray.T, cmap='jet')
            deflabelx = "x, [pixel/4]"
            deflabely = "y, [pixel/4]"
        else:
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

    ax.set_xlabel(deflabelx)
    ax.set_ylabel(deflabely)

    start = 123
    if("info" in info):
        comment = info.split(":")[1]
        ax.text(-90, start, comment, fontsize=9,color='black' )    
    if(plotMarker is not None):
        markx = plotMarker[0]
        marky = plotMarker[1]
        ax.scatter([markx],[marky],c='red', marker='d',s=80)

    if(("weight-centers" in picname or "AbsPoints" in picname) and ("cluster" not in picname)):
        #meanX = np.mean(nuarray, axis=0)/np.sum(nuarray)
        #meanY = np.mean(nuarray, axis=1)/np.sum(nuarray)
        meanX, meanY, stdX, stdY = getClusterCenter(nuarray)
        #medianX = np.median(nuarray, axis=0)
        #medianY = np.median(nuarray, axis=1)
        #medianX, medianY = getClusterMedian(nuarray)
 
        #ax.scatter([medianX],[medianY], c='magenta', s=64, marker="*")
        ax.scatter([meanX],[meanY], c='black', s=80, marker="+")
        #ax.vlines(meanX,0,256, colors='orange', linestyles=":")
        #ax.hlines(meanY,0,256, colors='orange', linestyles=":")

        ell_sigma = Ellipse( 
            xy=(meanX,meanY),
            width=stdX,
            height=stdY,
            edgecolor='black',
            facecolor='none',
            lw=1,
        )
        ax.add_patch(ell_sigma)

        ell_sigma_x2 = Ellipse( 
            xy=(meanX,meanY),
            width=stdX*2,
            height=stdY*2,
            edgecolor='black',
            facecolor='none',
            lw=1,
        )
        ax.add_patch(ell_sigma_x2)
        ax.text(0, -20, f"meanX={meanX:.2f},\nmeanY={meanY:.2f}", fontsize=10, color='darkblue')

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
    plt.close() 


def getEvolutionDirection(centerX, centerY, weightX, weightY, absX, absY):
    # basically dx/dy of charge center and absorption point

    deriv = ((absX-centerX) - (weightX-centerX))/((absY-centerY) - (weightY-centerY))
    if(deriv == np.inf or deriv == -np.inf):
        deriv = NaN

    return deriv

#def getMatrixProfile(matrix, labels, picname, odir):
def getMatrixProfile(matrix):

    # (NOTE) its easier to flip the calculation to slice y axis of matrix than x axis for 
    # angles >45 deg w.r.t x-axis
    # thus the names of containers for case of profileX are inverted for slicing in Y 
    # but they serve same purpose

    if(np.sum(matrix)==0):
        print(f"{OUT_RED}[ERROR::EMPTY_DATA] {picname} matrix all zeros! {OUT_RST}")
        return 0

    ny, nx = matrix.shape
    # accumulating stats from 5 columns 

    ncol = 5
    yidx = np.arange(ny)
    xidx = np.arange(nx)

    y_means, x_centers, devy = [], [], []

    for x0 in range(0,nx,ncol):
        x1 = min(x0+ncol,nx)
        #yprof = matrix[:,x0:x1].sum(axis=1)
        yprof = matrix[x0:x1,:].sum(axis=0)
        total = yprof.sum()
        if(total==0):
            continue
        
        ymean = np.sum(yidx*yprof)/total
        x_centers.append(0.5 * (x0+x1-1))
        y_means.append(ymean)

        vary = np.sum(yprof * (yidx-ymean)**2)/total
        devy.append(np.sqrt(vary/total))

    x_centers = np.array(x_centers)
    y_means = np.array(y_means)

    YERR = None
    par, cov = None, None
    slope_deg, chired = None, None

    plt.figure(figsize=(8,8))

    devy = np.array(devy)
    YERR = devy

    par, cov = curve_fit(linefunc, x_centers, y_means)  
    slope = np.arctan(par[0])
    slope_deg = slope*180/np.pi
    
    expected_pts = linefunc(x_centers, *par)
    dof = len(y_means) - len(par)
    chisquare = np.sum((y_means - expected_pts)**2/YERR**2)    
    chired = chisquare/dof

    return x_centers, y_means, YERR, chired

    #plt.errorbar(x_centers, y_means, yerr=YERR, fmt=".", color='darkblue', ecolor='black', capsize=4, label="ProfileY")
    #plt.plot(x_centers, linefunc(x_centers, *par), c="firebrick", label=f"Line: {par[0]:.4f}*x + {par[1]:.4f}")
    #plt.scatter([],[],color="white", label=r"$\angle\,$(x/y offset) "+f"= {slope_deg:.4f}"+r"$^{\circ}$"+f"[{slope:.4f} rad]")
    #plt.scatter([],[],color="white", label=r"$\chi^{2}_{\mathrm{red.}}$"+f" = {chired:.4f}")
    #plt.title(labels[0])
    #plt.xlabel(labels[1])
    #plt.ylabel(labels[2])
    #plt.xlim([0,256])
    #plt.ylim([0,256])
    #plt.grid(which='major', color='gray', linestyle='--', linewidth=0.5)
    #plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.25)
    #plt.minorticks_on()
    #plt.legend()
    #plt.savefig(f"{odir}/SCAT-MatrixProfileX-{picname}.png")
    #plt.close()

#####################################################################
#     MAIN STARTS HERE!
#####################################################################

parser = ap.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default="TEST", help="name for plots (suffix)")
parser.add_argument("-d", "--dir", type=str, default=None, help="Custom directory name")
parser.add_argument("-f", "--files", nargs="+", help="list of files ala ~/TPA/reco-weighted*\nCan be single file.")
parser.add_argument("--debug", action='store_true', help="Enable debug output")
parser.add_argument("--nocuts", action='store_true', help="Disable all cuts")
parser.add_argument("--timing", action='store_true', help="PLot Timinig-related histos")
parser.add_argument("--beamscan", action='store_true', help="Apply to Beam scan data")
parser.add_argument("--plotevents", action='store_true', help="Plot Individual events")
parser.add_argument("--combinedTOT", action='store_true', help="Plopt combined sumTOT plot for all data sets")
parser.add_argument("--stokes", action='store_true', help="Plot Stokes parameter realted data")
parser.add_argument("--plotroot", action='store_true', help="Plot ROOT histograms")
args = parser.parse_args()

plotname = args.name
recofiles = args.files
custom_directory = args.dir
debug = args.debug

debug_status = ""
if(debug):
    debug_status += f"{OUT_GREEN_BGR}{debug}{OUT_CYAN_BGR}"
else:
    debug_status += f"{OUT_RED_BGR}{debug}{OUT_CYAN_BGR}"
fancytext = ["Resolving Input:",
             f"plotname = {plotname}",
             f"custom dir = {custom_directory}",
             f"DEBUG = {debug_status}",
             "recofiles to be used:"]
makeFancySection(fancytext+recofiles, color=OUT_CYAN_BGR)

print("\n")
outdir = ""
if(custom_directory is not None):
    outdir += custom_directory
else:
    outdir += f"tmp-{plotname}/"

if not os.path.exists(outdir):
    os.makedirs(outdir)

# for plotting 3d tracks
xlist, ylist, zlist = [], [], []

matrixTotal = np.zeros((256,256),dtype=np.uint16)
matrixTotal_TOT = np.zeros((256,256),dtype=np.uint16)

matrix_cut = np.zeros((256,256),dtype=int)

n_instant_0, n_instant_1 = 0, 0
nprotons = 0

sum_U, sum_Q, sum_I = 0, 0, 0
altsum_U, altsum_Q, altsum_I, = 0, 0, 0

weightCenters = np.zeros((256,256),dtype=int)
weightCenters_GLOB = np.zeros((256,256),dtype=int)
weightCenters_cut = np.zeros((256,256),dtype=int)

absorption_points_pruned_GLOB = np.zeros((256,256),dtype=int)

# defining some temp lists to plot parameters after cuts

tmp_derivative = []
tmp_excent, tmp_hits, tmp_sumTOT, tmp_evtnr = [], [], [], []
tmp_centerX, tmp_centerY = [], []
tmp_length,tmp_width = [], []

tmp_RMSL, tmp_RMST = [], []

tmp_densityRMS, tmp_densityWL = [], []

tmp_secondangles_glob = []
tmp_secondangles_Y, tmp_secondangles_Y_conv = [], []
tmp_BGRangles = []

tmp_theta = []
theta_secondstage = None

#tmp_evtArea = []

# defining lists for rejected events

bad_hits, bad_sumTOT, bad_length, bad_excent = [], [], [], []

conversionX, conversionY =  None, None
ConvX, ConvY = [],[]
tmp_toaMean, tmp_toaRMS, tmp_toaSkew, tmp_toaLength =  [],[],[],[]

# ------------------------------------------
# some lists for global, cummulative plots 
#
tmp_secondangles = []
tmp_BGRangles_glob = []
GLOB_hits, GLOB_sumTOT,  = [], []

sumTOT_list = []

#############################################

abs_matrixList, weight_matrixList = [], []

#############################################

matrix_I = np.zeros((64,64), dtype=float)
matrix_Q = np.zeros((64,64), dtype=float)
matrix_U = np.zeros((64,64), dtype=float) 

glob_matrix_I = np.zeros((64,64),dtype=float)
glob_matrix_Q = np.zeros((64,64),dtype=float)
glob_matrix_U = np.zeros((64,64),dtype=float)

matrix_wI = np.zeros((64,64), dtype=float)
matrix_wQ = np.zeros((64,64), dtype=float)
matrix_wU = np.zeros((64,64), dtype=float) 

#############################################

tmp_weightedY, tmp_weightedX = [], []

tmp_convX, tmp_convY = [],[]

cut_TOTMAP = np.zeros((256,256),dtype=int)

abs_stdx, abs_stdy, abs_meanx, abs_meany = None, None, None, None

electrons = None # variable for VLArray of electrons 

tmp_electrons, tmp_sumElec = [], []
tmp_elecPerHit = []

glob_absorption_points = np.zeros((256,256),dtype=int)
 
####################################################
# flags for data here:
hasTheta = False
fNocuts = False
fTiming = False
fCharge = False
fConverted = False
fLongData = False

####################################################
# flags to enable certain plots
fBeamScan = args.beamscan
fNocuts = args.nocuts
fPlotSingleEvents = args.plotevents
fPlotROOT = args.plotroot
fPlotStokes = args.stokes
fPlotTiming = args.timing
fPlotCombinedTOT = args.combinedTOT
# ----
fPlotGlobalSumTOT = False
fPlotGlobalAngles = True
fPlotGlobalCenters = True
fPlotGlobalStokes = True
fPlotGlobalLineScan = False

if(len(recofiles)==1):
    fPlotGlobalSumTOT = False
    fPlotGlobalAngles = False
    fPlotGlobalCenters = False
    fPlotGlobalStokes = False
    fPlotGlobalLineScan = False

print("Flag Status:")

flagbox = [
    f"NoCuts = {fNocuts}",
    f"BeamScan = {fBeamScan}",
    f"PlotSingleEvents = {fPlotSingleEvents}",
    f"PlotROOT = {fPlotROOT}",
    f"PlotStokes = {fPlotStokes}",
    f"PlotTiming = {fPlotTiming}",
    f"PlotCombinedTOT = {fPlotCombinedTOT}",
    f"PlotGlobalSumTOT = {fPlotGlobalSumTOT}",
    f"PlotGlobalAngles = {fPlotGlobalAngles}",
    f"PlotGlobalCenters = {fPlotGlobalCenters}",
    f"PlotGlobalStokes = {fPlotGlobalStokes}",
    f"PlotGlobalLineScan = {fPlotGlobalLineScan}"
]

modified_flagbox = []
for line in flagbox:
    istatus = line.split("=")[1]
    if("False" in istatus):
        modified_flagbox.append(f"{OUT_RED_BGR}{line}{OUT_RST}")
    elif("True" in istatus):
        modified_flagbox.append(f"{OUT_GREEN_BGR}{line}{OUT_RST}")
    else: 
        modified_flagbox.append(line)

makeFancySection(modified_flagbox)

countangles = {"0":0, "30":0, "60":0, "90":0}

####################################################
time_now = dtime.datetime.now()
timestamp = time_now.strftime("%Y-%m-%d : %H:%M:%S")

freport = open(f"analReport-{plotname}.log",'a')
freport.write(f"\n========== {timestamp} ==========\n")
freport.write(f"ANALYZING: {recofiles}")
nfile = 0
for file in recofiles:

    if("BeamScan-bottom-Vanode" in file):
        fBeamScan = True

    print(f"{OUT_BLUE_BGR}ANALYZING: {file}{OUT_RST}\n")
    freport.write(f"\nCurrent file: {file}")
    freport.flush()

    # counting angles here
    if("-0deg" in file):
        countangles["0"]+=1
    if("-30deg" in file):
        countangles["30"]+=1
    if("-60deg" in file):
        countangles["60"]+=1
    if("-90deg" in file):
        countangles["90"]+=1

    with tb.open_file(file, 'r') as f:
 
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
        bgn_split = base_group_name.split('/')
        run_name = bgn_split[2]
        run_num = int(run_name[4:])

        if(debug):
            print(f'base group name is : <{base_group_name}>')
            print(bgn_split)
            print(f"<{run_name}>")
            print(f'run number is {run_num}')


        freport.write(f"\nRun number: {run_num}")

        ##############################################
        # cleaning up the base name resolving info
        groups = None
        grouplist = None
        main_group = None
        run_name, run_num, bgn_split = None, None, None 
        basewords = None

        # =========== gettin' shit ==================
        ToT = f.get_node(base_group_name+"ToT")
        if(debug):
            print(f"found VLarray TOT of size {type(ToT)}")
        centerX = f.get_node(base_group_name+"centerX")
        if(debug):
            print(f"found centerX {type(centerX)}")
        centerY = f.get_node(base_group_name+"centerY")
        if(debug):
            print(f"found centerY {type(centerY)}")
        hits = f.get_node(base_group_name+"hits")
        if(debug):
            print(f"found hits {type(hits)}")
        excent = f.get_node(base_group_name+"eccentricity")
        if(debug):
            print(f"found excentricity {type(excent)}")
        length = f.get_node(base_group_name+"length")
        if(debug):
            print(f"found length {type(length)}")
        RMS_L = f.get_node(base_group_name+"rmsLongitudinal")
        if(debug):
            print(f"found RMS_L {type(RMS_L)}")
        RMS_T = f.get_node(base_group_name+"rmsTransverse")
        if(debug):
            print(f"found RMS_T {type(RMS_T)}")
        rotAng = f.get_node(base_group_name+"rotationAngle")
        if(debug):
            print(f"found rotAng {type(rotAng)}")
        sumTOT = f.get_node(base_group_name+"sumTot")
        if(debug):
            print(f"found sumTOT {type(sumTOT)}")
        width = f.get_node(base_group_name+"width")
        if(debug):
            print(f"found width {type(width)}")
        x = f.get_node(base_group_name+"x")
        if(debug):
            print(f"found x {type(x)}")
        y = f.get_node(base_group_name+"y")
        if(debug):
            print(f"found y {type(y)}")
        EVENTNR = f.get_node(base_group_name+"eventNumber")
        if(debug):
            print(f"found eventNumber {type(EVENTNR)}")
        
        measurement_time = None
        for item in TIME_DICT.keys():
            if item in file:
                measurement_time = float(TIME_DICT[item])/float(len(recofiles))
                print(f"Found run time value [{measurement_time} seconds] for run [{item}]")
                break
        #if("1260Hz" in file):
        #    if("-0deg" in file):
        #        print("Found [1260Hz, 0deg]")
        #    elif("-30deg" in file):
        #        print("Found [1260Hz, 30deg]")
        #    elif("-60deg" in file):
        #        print("Found [1260Hz, 60deg]")
        #    else:
        #        print("Found [1260Hz, 90deg]")
        #else:
        #        print("This is not a [1260Hz] run")

        ############################################ 
        # detector efficiency for 10keV for NeCO2 80;20
        # based on simulation
        detector_epsilon_neco2 = 0.027476
        ############################################ 

        if(measurement_time is not None):
            n_clusters = len(hits)
            raw_rate = n_clusters/measurement_time
            if(debug): 
                print(f"{OUT_RED_BGR} Rate = {raw_rate:.4f} [Hz] {OUT_RST}")
            source_rate = raw_rate/detector_epsilon_neco2
            rateNum, rateMagn = checkRateOrder(source_rate)
            if(debug):
                print(f"{OUT_RED_BGR} Inferred source rate = {rateNum:.2f} [{rateMagn}] {OUT_RST}")
    
        # trying to find angle reconstruction constants used by Xraypolreco

        CONST = None
        if(check_node(base_group_name+"constants",f)):
            CONST = f.get_node(base_group_name+"constants")[:]
            if(debug):
                print(f"Found CONSTANTS: {CONST}")

        RMIN, RMAX, VDRIFT, WEIGHT = -1,-1,-1,-1
        if(CONST is not None):
            RMIN = CONST[0]
            RMAX = CONST[1]
            VDRIFT = CONST[2]
            WEIGHT = CONST[3]

        freport.write(f"\nAngle-Reco-const: {RMIN},{RMAX},{VDRIFT},{WEIGHT}")

        print("---------- CHINAZES! ------------")
    
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
            if(debug):
                print("FOUND TOACombined!")
        
        hasTheta = check_node(base_group_name+"theta_secondstage",f)
        if(hasTheta):
            theta_secondstage = f.get_node(base_group_name+"theta_secondstage")
            simpleSimRecoHist(theta_secondstage, 100, np.nanmin(theta_secondstage), np.nanmax(theta_secondstage) , [r'Track:$\theta$','degrees, [radian]','CTS'], plotname+f"THETA-{nfile}", True, outdir)
    
        if(check_node(base_group_name+"angle_secondstage",f)):
            if(debug):
                print("Found reconstruction data!")   
            secondangles = f.get_node(base_group_name+"angle_secondstage")
    
            fitAndPlotModulation(secondangles,
                                 100,
                                 -np.pi,
                                 np.pi,
                                 ["Reconstructed Angle Distribution", "Angle [radian]", r"$N_{Entries}$"],
                                 f"STOLEN-XpolSecondStage-{nfile}",
                                 outdir,
                                 debug)
    
        if(check_node(base_group_name+"absorption_point_x",f)):   
    
            fConverted = True
            if(debug):
                print("Found reconstructed absorption points")
            absorp_x = f.get_node(base_group_name+"absorption_point_x")
            absorp_y = f.get_node(base_group_name+"absorption_point_y")
            if(debug):
                print(absorp_x[0:10])
                print(absorp_y[0:10])
            for ax,ay in zip(absorp_x,absorp_y):
                if(~np.isnan(ax) and ~np.isnan(ay)):
                    ConvX.append(ax)
                    ConvY.append(ay)
                    ax = int(np.round(ax))
                    ay = int(np.round(ay))
                    np.add.at(absorption_points, (ax,ay), 1)
            plot2dEvent(absorption_points, 
                        "title:Reconstructed Absorption Points", 
                        f"ABSORPTION-{nfile}", 
                        outdir,
                        debug)
    
            conversionX = absorp_x
            conversionY = absorp_y
    
            if(debug):
                print(f"absorption x,y lengths: {absorp_x.shape[0]}\t{absorp_y.shape[0]}")
                print(f"Pruned lists          : {len(ConvX)}\t{len(ConvY)}")
            freport.write(f"\nAbsorption Point Data:\nLengths = {absorp_x.shape[0]}\t{absorp_y.shape[0]}\n")
            freport.write(f"\nPruned Length = {len(ConvX)}\t{len(ConvY)}")

            abs_stdx = np.std(ConvX)
            abs_stdy = np.std(ConvY)
            abs_meanx = np.mean(ConvX)
            abs_meany = np.mean(ConvY)
            abs_medianx = np.median(ConvX)
            abs_mediany = np.median(ConvY)
            
            if(debug):
                print(f"{abs_stdx:.4f}")
                print(f"{abs_stdy:.4f}")
                print(f"{abs_meanx:.4f}")
                print(f"{abs_meany:.4f}")
            freport.write(f"\nAbsorption peak: {abs_stdx:.4f}, {abs_stdy:.4f},{abs_meanx:.4f},{abs_meany:.4f}")
            freport.flush()

            absorp_x, absorp_y = None, None

            ConvX.clear()
            ConvY.clear()

            if(debug):
                print(f"Deviation in absorption x and y are:\n{OUT_BLUE} sigma_x={abs_stdx:.4f}, sigma_y={abs_stdy:.4f} {OUT_RST}")
    

        #exit(0)
        ntotal = ToT.shape[0]
        freport.write(f"\nN_clusters = {ntotal}")
    
        print(f"\nTOTAL nr of clusters: {ntotal}\n")
        ievent, npics, mcevents = 0, 0, 0
        n_good = 0
        n_tracks = 0

        # global cuts fro soem parameters
        gMaxHits, gMinHits = 25, 5000

        # checks/cuts:
    
        Rx0,Ry0 = None, None
        Rcut = abs_stdx if abs_stdx<=abs_stdy else abs_stdy
        minSumTOT, maxSumTOT = 100, 10000
        minHits, maxHits = 25, 1000
        minExcent, maxExcent = 0, 15 #1.5, 15
        maxLength = 5          
 
        maxToaLen = 50
        maxToaRms = 10
        maxToaMean = 100

        if("GHOSTS" in file):
            maxToaLen = 1e6
            maxToaRms = 1e3
            maxToaMean = 1e5

        #checking data based on reconstructed angle
        minPhi, maxPhi = 0.35, 0.69

        # weighted center boarder restriction
        wx_min, wx_max = 3, 252
        wy_min, wy_max = 3, 252

        if("DB" in file):                    
            Rcut = Rcut*2
            if("120Hz" in file):
                Rx0, Ry0 = 127.0, 145.0
                minSumTOT, maxSumTOT = 3900, 6700
            if("1260Hz" in file):
                minSumTOT, maxSumTOT = 3500, 6100
                if("-0deg" in file):
                    Rx0, Ry0 = 127.94, 151.76 
                elif("-30deg" in file):
                    Rx0, Ry0 = 82.18, 96.11 
                elif("-60deg" in file):
                    Rx0, Ry0 = 157.24, 153.27
                else:
                    Rx0, Ry0 = 82.82, 167.07  
            elif("15100Hz" in file):
                Rcut = Rcut*0.8 # common
                minSumTOT, maxSumTOT = 2500, 5600
                if("-0deg" in file):
                    Rx0, Ry0 = 128.62, 149.27 
                elif("-30deg" in file):
                    Rcut = Rcut*0.8
                    Rx0, Ry0 = 83.92, 92.29 
                elif("-60deg" in file):
                    Rx0, Ry0 = 154.79, 153.23 
                else:
                    Rx0, Ry0 = 81.76, 166.46
            elif("153000Hz" in file):
                if("-0deg" in file):
                    Rx0, Ry0 = 129.0, 148.0
                elif("-30deg" in file):
                    Rcut = Rcut*0.75
                    Rx0, Ry0 = 82.0, 92.0 
                elif("-60deg" in file):
                    Rcut = Rcut*0.6
                    Rx0, Ry0 = 159.0, 153.0 
                else: 
                    Rcut = Rcut*0.5
                    Rx0, Ry0 = 80.0, 175.0
            elif("1490000Hz" in file): # have only one for [0 deg]
                Rcut = Rcut*3
                Rx0, Ry0 = abs_meanx, abs_meany
                maxLength = 3.0
                maxSumTOT = 2200    
                minHits, maxHits = 10, 70
            else:
                Rx0, Ry0 = abs_meanx, abs_meany
        elif("MP" in file):
            if("MP2" in file or
               "MP1" in file or
                "MP3" in file or 
                "MP4" in file): # 0 deg
                Rx0, Ry0 = 105.0, 102.28 
                Rcut = abs_stdx*0.75
            elif("MP5" in file): # 90 deg
                Rx0, Ry0 = 107.0, 78.13 
                Rcut = abs_stdx*0.75
            elif("MP6" in file): # 60 deg
                Rx0, Ry0 = 122.84, 78.38 
                Rcut = abs_stdx*0.75
            elif("MP7" in file): # 30 deg
                Rx0, Ry0 = 125.0, 95.0
                Rcut = abs_stdx*0.75
            elif("MP-" in file and "400Vcm" in file):# drift field 400 Vcm
                Rcut = abs_stdx*0.9
                Rx0, Ry0 = 105.38, 112.95
            elif("MP-" in file and "450Vcm" in file):# 450 Vcm
                Rcut = abs_stdx*0.9
                Rx0, Ry0 = 105.38, 112.95
            elif("MP-" in file and "500Vcm" in file):# 500 Vcm
                Rcut = abs_stdx*0.9
                Rx0, Ry0 = 105.38, 112.95
            elif("pre-MP-" in file and "18-36-37" in file):
                Rcut = abs_stdx*3 if abs_stdx >= abs_stdy else abs_stdy*3
                Rx0, Ry0 = 107.0, 105.41
            else:
                Rx0, Ry0 = abs_meanx, abs_meany
        elif("CP" in file):
            Rcut = abs_stdx*2
            if("CP0" in file):
                minSumTOT, maxSumTOT = 2300, 5600 
                minHits, maxHits = 80, 150 
                Rx0, Ry0 = 123.50, 132.44  
            elif("CP2p5" in file):
                minSumTOT, maxSumTOT = 2300, 5800 
                minHits, maxHits = 75, 150 
                Rx0, Ry0 = 123.68, 132.51  
            elif("CP1" in file): #  0 deg
                minSumTOT, maxSumTOT = 450, 3200
                minHits, maxHits = 20, 90
                Rx0, Ry0 = 118.71, 137.28
            elif("CP3" in file): # 30 deg
                minSumTOT, maxSumTOT = 1950, 5150
                minHits, maxHits = 60,160
                Rx0, Ry0 = 125.44, 126.32 
            elif("CP4" in file): # 60 deg
                minSumTOT, maxSumTOT = 1950, 5300
                minHits, maxHits = 60, 160
                Rx0, Ry0 = 136.16, 135.11
            elif("CP5" in file): # 90 deg
                Rcut = Rcut*0.25
                minSumTOT, maxSumTOT = 2250, 5390
                minHits, maxHits = 70, 160
                Rx0, Ry0 = 109.21, 80.95 
            elif("CP6" in file): # 0deg, 11.950 keV ok rate 
                minSumTOT, maxSumTOT = 2600, 6200 
                minHits, maxHits = 85, 170 
                Rx0, Ry0 = 117.64, 128.96
            elif("CP7" in file): # 0deg, 11.950 keV rate++
                Rx0, Ry0 = 115.59, 126.72 
            else:
                Rx0, Ry0 = abs_meanx, abs_meany

        elif("gainVar" in file):
            Rx0, Ry0 = abs_meanx, abs_meany
            Rcut = abs_stdy*2
        elif("ROME" in file or "BeamScan-bottom-Vanode" in file):
            minSumTOT, maxSumTOT = 100, 32000
        else:
            Rx0, Ry0 = abs_meanx, abs_meany

        if("SIM-TPX3" in file):
            Rcut = Rcut*3

        # defining flags to enable/disable them in single call in the event loop

        goodXY, goodConvXY = True, True
        goodLength, goodSumTOT, goodHits, goodExcent, = True, True, True, True
        goodTOA_len, goodTOA_rms, goodTOA_mean = True, True, True

        # BEGINNING EVENT LOOP
        for event in ToT: 

            if("after-0deg-1p46MHz" in file and EVENTNR[ievent]>245000):
                print("BREAKING loop!")
                break

            isumElec = 0
            if(fCharge):
                isumElec = np.sum(electrons[ievent])
    
            nhits = len(event)
    
            densityRMS = nhits/(RMS_T[ievent]/RMS_L[ievent])
            #densityWL = nhits/(width[ievent]/length[ievent])
            areaWL = width[ievent]*length[ievent] 
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
            
            #--------------------------------------------------------------
            if(hits[ievent]<gMaxHits):
                for xpos, ypos, tot in zip(x[ievent],y[ievent], event):
                    np.add.at(matrixTotal, (xpos,ypos), 1)
                    np.add.at(matrixTotal_TOT, (xpos,ypos), tot)
    
            sum_TOTX, sum_TOTY = 0,0
            for itot, ix, iy in zip(event, x[ievent], y[ievent]):
                sum_TOTX += itot*ix
                sum_TOTY += itot*iy
    
            weightX = sum_TOTX/np.sum(event)
            weightY = sum_TOTY/np.sum(event)
           
            np.add.at(weightCenters, (int(np.round(weightX)), int(np.round(weightY))), 1)
            if(fPlotGlobalCenters):
                np.add.at(weightCenters_GLOB, (int(np.round(weightX)), int(np.round(weightY))), 1)
            
            if(not fNocuts):               
                if("ROME" in file or fBeamScan):
                    goodConvXY = True if ((conversionX[ievent] >= wx_min) and 
                                          (conversionX[ievent] <= wx_max) and 
                                          (conversionY[ievent] >= wy_min) and 
                                          (conversionY[ievent] <= wy_max)) else False
                else:
                    goodConvXY = checkClusterPosition(Rx0, 
                                                      Ry0,
                                                      conversionX[ievent],
                                                      conversionY[ievent],
                                                      Rcut)

                # selecting only interesting events 
                goodLength = True if (length[ievent]<=maxLength) else False
                goodSumTOT = True if (sumTOT[ievent]>= minSumTOT and sumTOT[ievent]<=maxSumTOT) else False
                goodHits = True if (hits[ievent]>=minHits and hits[ievent]<=maxHits) else False
                goodExcent = True if (excent[ievent]>= minExcent and excent[ievent] < maxExcent) else False
    
                ## timing cuts
                goodTOA_len = True if (toaLength[ievent]<=maxToaLen) else False
                goodTOA_rms = True if (toaRMS[ievent]<=maxToaRms) else False
                goodTOA_mean = True if (toaMean[ievent]<=maxToaMean) else False
           
                # overall cut to exclude outer part of chip and tracks
                # with bragg peak at the chip edges
                goodXY = True if (weightX >= wx_min and weightX <= wx_max and weightY >= wy_min and weightY <= wy_max) else False

            if (goodLength and 
                goodXY and 
                goodConvXY and 
                #not goodConvXY and 
                goodSumTOT and 
                goodHits and
                goodTOA_mean and 
                goodTOA_rms and 
                goodTOA_len and 
                goodExcent #and
                 ):# improved based on length cut
    
                tmp_secondangles.append(secondangles[ievent])
                if(fPlotGlobalAngles): 
                    tmp_secondangles_glob.append(secondangles[ievent]) 
                np.add.at(weightCenters_cut, (int(np.round(weightX)), int(np.round(weightY))), 1)
                
                if(hasTheta):
                    tmp_theta.append(checkNANTheta(theta_secondstage[ievent]))
    
                #tmp_densityRMS.append(checkNAN(densityRMS))
                tmp_excent.append(checkNAN(excent[ievent]))
                tmp_hits.append(checkNAN(hits[ievent]))
                tmp_sumTOT.append(checkNAN(sumTOT[ievent]))
                if(fLongData):
                    if(checkNAN(EVENTNR[ievent])%100==0):
                        tmp_evtnr.append(checkNAN(EVENTNR[ievent]))
                else:
                    tmp_evtnr.append(checkNAN(EVENTNR[ievent]))
                tmp_length.append(checkNAN(length[ievent]))
                tmp_width.append(checkNAN(width[ievent]))
                tmp_RMST.append(checkNAN(RMS_T[ievent]))
                tmp_RMSL.append(checkNAN(RMS_L[ievent]))
                #tmp_evtArea.append(areaWL)

                if(fPlotGlobalSumTOT):
                    GLOB_sumTOT.append(checkNAN(sumTOT[ievent]))
                    GLOB_hits.append(checkNAN(hits[ievent]))
     
                if(fConverted):
                    tmp_convX.append(checkNAN(conversionX[ievent]))
                    tmp_convY.append(checkNAN(conversionY[ievent]))
                    np.add.at(absorption_points_pruned, (int(np.round(checkNAN(conversionX[ievent]))),int(np.round(checkNAN(conversionY[ievent])))),1)
                    if(fPlotGlobalCenters):
                        np.add.at(absorption_points_pruned_GLOB, (int(np.round(checkNAN(conversionX[ievent]))),int(np.round(checkNAN(conversionY[ievent])))),1)
                    tmp_secondangles_Y_conv.append(checkNANphi(secondangles[ievent]))
            
                    # doing some maps of individual track stokes params (64x64 for now)

                    if(not np.isnan(conversionX[ievent]) and not np.isnan(conversionY[ievent])):
                        Itrack, Qtrack, Utrack = getSingleTrackSokes(secondangles[ievent])
                        downscale = 4.0
                        xscaled = conversionX[ievent]/downscale
                        yscaled = conversionY[ievent]/downscale
                        np.add.at(matrix_I, (int(xscaled),int(yscaled)), Itrack)
                        np.add.at(matrix_Q, (int(xscaled),int(yscaled)), Qtrack)
                        np.add.at(matrix_U, (int(xscaled),int(yscaled)), Utrack)

                        # plotting for weight centers
                        wx = int(weightX/downscale)
                        wy = int(weightY/downscale)
                        np.add.at(matrix_wI, (wx,wy), Itrack)
                        np.add.at(matrix_wQ, (wx,wy), Qtrack)
                        np.add.at(matrix_wU, (wx,wy), Utrack)


                if(fTiming and fPlotTiming):
                    tmp_toaLength.append(toaLength[ievent])
                    tmp_toaMean.append(toaMean[ievent])
                    tmp_toaRMS.append(toaRMS[ievent])
                    tmp_toaSkew.append(toaSkew[ievent])
    
                if(fCharge):
                    tmp_sumElec.append(isumElec)
                    tmp_elecPerHit.append(isumElec/nhits)
    
                if(goodXY or goodConvXY):
                    tmp_secondangles_Y.append(checkNANphi(secondangles[ievent]))
                    tmp_weightedY.append(checkNAN(weightY))
                    tmp_weightedX.append(checkNAN(weightX))
    
                for ix,iy,itot in zip(x[ievent],y[ievent], event):
                    np.add.at(matrix_cut, (ix,iy), 1)
                    np.add.at(cut_TOTMAP, (ix,iy), itot)
                naccepted+=1

            else:
                tmp_BGRangles.append(secondangles[ievent]) 
                if(fPlotGlobalAngles):
                    tmp_BGRangles_glob.append(secondangles[ievent]) 
                if(hits[ievent]<5000):
                    bad_hits.append(hits[ievent])
                    bad_excent.append(excent[ievent])
                    bad_length.append(length[ievent])
                    bad_sumTOT.append(sumTOT[ievent])
                 
 
            if(fPlotSingleEvents and 
               (npics < 5 and
               #goodLength  and
               #not goodConvXY and 
               goodConvXY and 
               goodXY and
               goodSumTOT and
               #goodHits and
               goodExcent) or
               nhits > 10000 
               #densityRMS > 1e6 
               #toaLength[ievent]>16000 or
               #(toaRMS[ievent]>1000 and toaRMS[ievent<4000]) or
               #(toaMean[ievent]>8000 and toaMean[ievent]<11000)
               ): # this one get the actual tracks

                # this block is for plotting photoelectron tracks: -------------------------   
                matrix = np.zeros((256,256),dtype=np.uint16)
                for k,l,m in zip(x[ievent],y[ievent],event):
                    np.add.at(matrix, (k,l), m)               
     
                if(fConverted):
                    plot2dEvent(matrix, characs, f"cluster-{nfile}-{ievent}-{plotname}", outdir, debug, [conversionX[ievent],conversionY[ievent]])
                else:
                    plot2dEvent(matrix, characs, f"cluster-{nfile}-{ievent}-{plotname}", outdir, debug, [weightX,weightY])

                characs = None
                matrix = np.zeros((256,256),dtype=np.uint16)

                #------------------------------------------------------------------
                # for viewing 3D tracks
                #if(toaLength is not None and npics < 50):
                #    zpos = []
                #    vdrift_NeCO2 = 17.7867
                #    for toa, ftoa in zip(ToA[ievent], fTOA[ievent]):
                #        zpos.append(calculateZpos(toa,ftoa,vdrift_NeCO2))
    
                #    plotInteractive3D(x[ievent],
                #                    y[ievent],
                #                    zpos,
                #                    [f"Reconstructed Track {ievent}", "x","y","z"], 
                #                    f"3D-Track-{nfile}-{ievent}-"+plotname, 
                #                    outdir)
                #------------------------------------------------------------------
                npics+=1
     
            progress(ntotal, ievent)
            ievent+=1
        
        ToT, centerX, centerY, hits = None, None, None, None
        excent, length, RMS_L, RMS_T = None, None, None, None
        rotAng, sumTOT, width = None, None, None
        x, y, EVENTNR = None, None, None

        f.close()
    if(debug):
        print(f"\nMade pics for <{npics}> events\n")
    freport.write(f"\nData after cuts: {naccepted}")
    freport.flush()

    if(debug):
        print("Resetting result dictionary...")
    main_results.fromkeys(main_results,0)

    # making file name suffixes 
    
    ommit = ['reco', 'weighted', 'eVelApp', '2D', 'P09']
    ifilename = file.split("/")[-1].split(".")[0]
    isuffix = clearString(ifilename, '-', '-', ommit)

    if(len(tmp_secondangles)>0):
        fitAndPlotModulation(np.array(tmp_secondangles),
                             100,
                             -np.pi,
                             np.pi,
                             ["Reconstructed Angle Distribution (Pruned by XY position)", "Angle [radian]", r"$N_{Entries}$"],
                             f"STOLEN-XpolSecondStage-CutOnPosition-{nfile}-{isuffix}",
                             outdir,
                             debug)
    tmp_secondangles.clear() 


    if(len(tmp_BGRangles)>0):
        fitAndPlotModulation(np.array(tmp_BGRangles),
                             100,
                             -np.pi,
                             np.pi,
                             ["Reconstructed Angle Distribution (Pruned by XY position)", "Angle [radian]", r"$N_{Entries}$"],
                             f"STOLEN-XpolSecondStage-CutOnPosition-BGRangles-{nfile}-{isuffix}",
                             outdir,
                             debug)
    tmp_BGRangles.clear()    

    plot2dEvent(matrixTotal, "", f"OCCUPANCY-total-run-{nfile}-{isuffix}", outdir, debug)
    plot2dEvent(matrixTotal_TOT, "", f"TOT-total-run-{nfile}-{isuffix}", outdir, debug)

    matrixTotal = np.zeros((256,256),dtype=np.uint16)
    matrixTotal_TOT = np.zeros((256,256),dtype=np.uint16)
    
    plot2dEvent(weightCenters, "title:Weight Centers of All tracks", f"weight-centers-{nfile}-{isuffix}", outdir, debug)

    if(fPlotGlobalLineScan):
        weight_plotdata = getMatrixProfile(weightCenters)
        weight_matrixList.append([weight_plotdata,f"{isuffix}"])
    if(np.sum(weightCenters_cut)>0):
        plot2dEvent(weightCenters_cut, "title:Weight Centers of All tracks (CUT)", f"weight-centers-cut-{nfile}-{isuffix}", outdir, debug)

    weightCenters = np.zeros((256,256),dtype=int)
    weightCenters_cut = np.zeros((256,256),dtype=int)

    if(fConverted and np.sum(absorption_points_pruned)>0):
        if(fPlotGlobalLineScan):
            plotdata = getMatrixProfile(absorption_points_pruned)
            abs_matrixList.append([plotdata, f"{isuffix}"])
        plot2dEvent(absorption_points_pruned, "title:Absorption Points Pruned", f"AbsPointsPruned-{nfile}-{isuffix}",outdir, debug)

    absorption_points = np.zeros((256,256),dtype=int)
    absorption_points_pruned = np.zeros((256,256),dtype=int)
    
    print("[MAIN]: Making Matrix plots")    

    # safe call wraps try:except block basically...
    #
    safe_call(plot2dEvent,
              matrix_cut, 
              "title:Occupancy, Cut on Position", 
              f"OCCUPANCY-CUT-ON-position-{nfile}-{isuffix}",
              outdir,
              debug)

    safe_call(plot2dEvent, 
            cut_TOTMAP, 
            "title:", 
            f"TOT-map-cut-{nfile}-{isuffix}", 
            outdir, 
            debug)
   
    matrix_cut = np.zeros((256,256),dtype=int)
    cut_TOTMAP = np.zeros((256,256),dtype=int)
    
    # ---------- plotting parameters after cuts ----------------------

    eventplot_title = r"$\Sigma$(TOT) vs Time"
    if(fLongData):
        eventplot_title+=" (Every 100th Event)"

    simpleScatter(tmp_evtnr,tmp_sumTOT,[r"$\Sigma$(TOT) vs Event Number","EvtNr",r"$\Sigma$(TOT)"],f"sumTOT-vs-EventNumber-{nfile}-{isuffix}",outdir)
    if("GHOSTS" in file):
        if(fTiming):
            simpleScatter(tmp_evtnr,tmp_toaMean,["Mean Cluster TOA vs Event Number","EvtNr",r"$\overline{TOA}$, [nCLK]"],f"TOAmean-vs-EventNumber-{nfile}-{isuffix}",outdir)
            simpleScatter(tmp_evtnr,tmp_toaLength,["Cluster Length (TOA) vs Event Number","EvtNr","Length, [nCLK]"],f"TOAlength-vs-EventNumber-{nfile}-{isuffix}",outdir)
            simpleScatter(tmp_evtnr,tmp_toaRMS,["Cluster RMS (TOA) vs Event Number","EvtNr",r"RMS(TOA), [nCLK]"],f"TOArms-vs-EventNumber-{nfile}-{isuffix}",outdir)
        
        simpleScatter(tmp_evtnr,tmp_sumTOT,[r"$\Sigma$(TOT) vs Event Number","EvtNr",r"$\Sigma$(TOT), [nCLK]"],f"sumTOT-vs-EventNumber-{nfile}-{isuffix}",outdir)        

    safe_call(simpleHist, 
            np.array(tmp_excent), 
            100, 
            np.nanmin(tmp_excent), 
            np.nanmax(tmp_excent), 
            ["Excentricity","Excentricity","Events,[N]"], 
            f"tmp_excent-{nfile}-{isuffix}", 
            outdir,
            debug, 
            yaxisscale='log') 
    
    safe_call(simpleHist,
            np.array(tmp_hits), 
            100, 
            np.nanmin(tmp_hits), 
            np.nanmax(tmp_hits), 
            ["Hits per Cluster",r"$N_{hits}$","Events,[N]"], 
            f"tmp_hits-{nfile}-{isuffix}", 
            outdir, 
            debug) 
    safe_call(simpleHist,
            np.array(tmp_sumTOT), 
            100, 
            np.nanmin(tmp_sumTOT), 
            np.nanmax(tmp_sumTOT), 
            [r"$\Sigma$(TOT) per Cluster",r"$\Sigma$(TOT)","Events,[N]"], 
            f"tmp_sumTOT-{nfile}-{isuffix}", 
            outdir, 
            debug) 
    
    #safe_call(simpleHist, 
    #         np.array(tmp_densityRMS),
    #         100,
    #         np.nanmin(tmp_densityRMS),
    #         np.nanmax(tmp_densityRMS), 
    #         ["Cluster Density RMS Ratios",r"$RMS_{T}/RMS_{L}$","Events,[N]"], 
    #         f"tmp_densityRMS-{nfile}-{isuffix}", 
    #         outdir, 
    #         debug, 
    #         yaxisscale='log') 

    if(hasTheta):
        
        safe_call(simpleHist,
             np.array(tmp_theta),
             100, 
             -np.pi, 
             np.pi, 
             [r"Cluster $\theta$",r"$\theta$, [radian]","Events,[N]"], 
             f"tmp_theta-{nfile}-{isuffix}", 
             outdir, 
             debug) 
   
    if(fCharge):
 
        safe_call(simpleHist,
            np.array(tmp_sumElec),
            100, 
            np.nanmin(tmp_sumElec), 
            np.nanmax(tmp_sumElec), 
            ["Cluster Charge (electrons)",r"$\Sigma (e^{-})$,","Events,[N]"], 
            f"tmp_sumElec-{nfile}-{isuffix}", 
            outdir, 
            debug) 

        safe_call(simpleHist, 
            np.array(tmp_elecPerHit), 
            100, 
            np.nanmin(tmp_elecPerHit), 
            np.nanmax(tmp_elecPerHit), 
            ["Electrons per Hit",r"$N_{e^{-}}/N_{Hits}$,","Events,[N]"], 
            f"tmp_elecPerHit-{nfile}-{isuffix}", 
            outdir, 
            debug) 

    if(fPlotROOT):
        print("[MAIN]: Making ROOT plots")    
        # plotting some ROOT histograms of parameters
        rt.gROOT.SetBatch(True)
        rt.gStyle.SetOptStat(0)

        rMaxSumTOT = 8000
        rMaxHits = 300
        if("ROME" in file or "BeamScan-bottom-Vanode" in file):
            rMaxSumTOT = 32000
            rMaxHits = 1000

        plotRoot2D(tmp_hits, tmp_sumTOT,
                   100,[0,rMaxHits],
                   100,[0,rMaxSumTOT],
                   [r"$\Sigma$(TOT) per track vs N hits per track (ACCEPTED)","Hits per track",r"$\Sigma$(TOT) per track"],
                   f"sumTOT-vs-Hits-ACC-{nfile}-{isuffix}",
                   outdir,
                   debug
        )
        
        plotRoot2D(bad_hits, bad_sumTOT,
                   100,[0,rMaxHits],
                   100,[0,rMaxSumTOT],
                   [r"$\Sigma$(TOT) per track vs N hits per track (REJECTED)","Hits per track",r"$\Sigma$(TOT) per track"],
                   f"sumTOT-vs-Hits-REJ-{nfile}-{isuffix}",
                   outdir,
                   debug
        )
        
        plotRoot2D(tmp_sumTOT, tmp_length,
                   100,[0,rMaxSumTOT],
                   100,[0,15],
                   [r"Track length vs $\Sigma$(TOT) per track (ACCEPTED)",r"$\Sigma$(TOT) per track", "Track length"],
                   f"length-vs-sumTOT-ACC-{nfile}-{isuffix}",
                   outdir,
                   debug
        )
        
        plotRoot2D(bad_sumTOT, bad_length,
                   100,[0,rMaxSumTOT],
                   100,[0,15],
                   [r"Track length vs $\Sigma$(TOT) per track (REJECTED)",r"$\Sigma$(TOT) per track", "Track length"],
                   f"length-vs-sumTOT-REJ-{nfile}-{isuffix}",
                   outdir,
                   debug
        )
        
        if(fConverted):
            plotRoot2D(tmp_convX,tmp_sumTOT,
                        256,[0,256],
                        200,[0,rMaxSumTOT],
                        [r"$\Sigma$(TOT) per track vs $X_{conversion}$",r"X_{conv.}",r"$\Sigma$(TOT) per track, [N CLK]"],
                        f"sumTOT-vs-AbsPointX-{nfile}-{isuffix}",
                        outdir,
                        debug
            )
            plotRoot2D(tmp_convY,tmp_sumTOT,
                        256,[0,256],
                        200,[0,rMaxSumTOT],
                        [r"$\Sigma$(TOT) per track vs $Y_{conversion}$",r"Y_{conv.}",r"$\Sigma$(TOT) per track, [N CLK]"],
                        f"sumTOT-vs-AbsPointY-{nfile}-{isuffix}",
                        outdir,
                        debug
            )


    print("[MAIN]: Making Python 2D hists")    
    #plot2Dhist(np.array(tmp_hits), np.array(tmp_densityRMS), [r"Cluster Hits Vs Cluster Density (RMS)", "Hits", r"$RMS_{T}/RMS_{L}$"], f"2D-DesityRMS-vs-Hits-{nfile}-{isuffix}", outdir, debug)
    #plot2Dhist(np.array(tmp_hits), np.array(tmp_densityWL), [r"Cluster Hits Vs Cluster Density (WL)", "Hits", "width/length"], f"2D-DesityWL-vs-Hits-{nfile}-{isuffix}", outdir)
    plot2Dhist(np.array(tmp_hits), np.array(tmp_excent), [r"Cluster $\epsilon$ vs Hits", "Hits", r"$\epsilon$"], f"2D-Hits-vs-Epsilon-{nfile}-{isuffix}", outdir, debug)
    plot2Dhist(np.array(tmp_hits), np.array(tmp_sumTOT), [r"$\Sigma$(TOT) vs Hits", "Hits", r"$\Sigma$(TOT)"], f"2D-Hits-vs-sumTOT-{nfile}-{isuffix}", outdir, debug)
    plot2Dhist(np.array(tmp_sumTOT), np.array(tmp_length), [r"Cluster Length $\Sigma$(TOT)s", r"$\Sigma$(TOT)", "Cluster Length"], f"2D-sumTOT-vs-Length-{nfile}-{isuffix}", outdir, debug)
    #plot2Dhist(np.array(tmp_sumTOT), np.array(tmp_evtArea), [r"Cluster Area vs $\Sigma$(TOT)s", r"$\Sigma$(TOT)", r"Cluster ($width\cdot length$)"], f"2D-ClusterArea-vs-sumTOT-{nfile}-{isuffix}", outdir, debug)
    #plot2Dhist(np.array(tmp_sumTOT), np.array(tmp_densityRMS), [r"Cluster Density vs $\Sigma$(TOT)", r"$\Sigma$(TOT)", r"$RMS_{T}/RMS_{L}$"], f"2D-sumTOT-vs-density-{nfile}-{isuffix}", outdir, debug)
    plot2Dhist(np.array(tmp_excent), np.array(tmp_length), ["Cluster Length vs Cluster Epsilon", r"$\epsilon$", "Cluster Length"], f"2D-Epsilon-Length-{nfile}-{isuffix}", outdir, debug)
    #plot2Dhist(np.array(tmp_width), np.array(tmp_length), ["Cluster length vs Cluster width", "Cluster Width", "Cluster Length"], f"2D-Width-Length-{nfile}-{isuffix}", outdir)
    #plot2Dhist(np.array(tmp_RMSL), np.array(tmp_RMST), ["Transversal vs Longitudinal cluster RMS", r"$RMS_{L}$", r"$RMS_{T}$"], f"2D-RMSL-RMST-{nfile}-{isuffix}", outdir)
    plot2Dhist(np.array(tmp_weightedY), np.array(tmp_secondangles_Y), ["Charge-Weighted cluster Y vs reco angle", "Y", "Reconstructed angle"], f"2D-weightY-SecAng-{nfile}-{isuffix}", outdir, debug)
    plot2Dhist(np.array(tmp_weightedX), np.array(tmp_secondangles_Y), ["Charge-Weighted cluster X vs reco angle", "X", "Reconstructed angle"], f"2D-weightX-SecAng-{nfile}-{isuffix}", outdir, debug)
        
    if(fConverted):
        print("[MAIN]: Plotting Converted data...")    
        plot2Dhist(np.array(tmp_convX), np.array(tmp_sumTOT), [r"Cluster $\Sigma$(TOT) X vs reconstructed Absorption X", "Absorption X", r"$\Sigma$(TOT)"], f"2D-AbsPointX-vs-sumTOT-{nfile}-{isuffix}", outdir, debug)
        plot2Dhist(np.array(tmp_convY), np.array(tmp_sumTOT), [r"Cluster $\Sigma$(TOT) Y vs reconstructed Absorption Y", "Absorption Y", r"$\Sigma$(TOT)"], f"2D-AbsPointY-vs-sumTOT-{nfile}-{isuffix}", outdir, debug)

        if(fPlotStokes): 
            print("[MAIN]: Plotting Stokes...")    
            # thhis one is mostly for MAgnetic peak data sets....
            plot2dEvent(matrix_I, "title:Stokes I for individual Tracks in 64x64 map", f"STOKES-I-{isuffix}", outdir, debug)
            plot2dEvent(matrix_Q/matrix_I, "title:Stokes Q for individual Tracks in 64x64 map", f"STOKES-Q-{isuffix}", outdir,debug)
            plot2dEvent(matrix_U/matrix_I, "title:Stokes U for individual Tracks in 64x64 map", f"STOKES-U-{isuffix}", outdir, debug)
           
            if(fPlotGlobalStokes):
                glob_matrix_I += matrix_I
                glob_matrix_Q += matrix_Q
                glob_matrix_U += matrix_U
            # plotting stokes for weight centers
            plot2dEvent(matrix_wI, "title:Stokes I for individual Weight Centers in 64x64 map", f"STOKES-I-weights-{isuffix}", outdir,debug)
            plot2dEvent(matrix_wQ/matrix_wI, "title:Stokes Q for individual Weight Centers in 64x64 map", f"STOKES-Q-weights-{isuffix}", outdir,debug)
            plot2dEvent(matrix_wU/matrix_wI, "title:Stokes U for individual Weight Centers in 64x64 map", f"STOKES-U-weights-{isuffix}", outdir,debug)

            matrix_I = np.zeros((64,64),dtype=float)
            matrix_Q = np.zeros((64,64),dtype=float)
            matrix_U = np.zeros((64,64),dtype=float)
 
            matrix_wI = np.zeros((64,64),dtype=float)
            matrix_wQ = np.zeros((64,64),dtype=float)
            matrix_wU = np.zeros((64,64),dtype=float)


    if(fTiming and fPlotTiming):
        print("[MAIN]: Plotting Timing Data...")    
        plot2Dhist(np.array(tmp_convX), np.array(tmp_sumTOT), [r"Cluster $\Sigma$(TOT) X vs reconstructed Absorption X", "Absorption X", r"$\Sigma$(TOT)"], f"2D-AbsPointX-vs-sumTOT-{nfile}-{isuffix}", outdir, debug)
        plot2Dhist(np.array(tmp_hits), np.array(tmp_toaLength), [r"Cluster ToA length vs Hits", "$N_{hits}$", "ToA Length [CLK]"], f"2D-Hits-TOALength-{nfile}-{isuffix}", outdir, debug)
        plot2Dhist(np.array(tmp_length), np.array(tmp_toaMean), [r"Cluster ToA Mean vs Cluster Length", r"$length$", "ToA Mean [CLK]"], f"2D-length-TOAMean-{nfile}-{isuffix}", outdir, debug)
        plot2Dhist(np.array(tmp_convX), np.array(tmp_toaMean), [r"Cluster ToA Mean vs Recosntructed Abs. X", r"$X_{Conversion}$", "ToA Mean [CLK]"], f"2D-AbsPointX-TOAMean-{nfile}-{isuffix}", outdir, debug)
        plot2Dhist(np.array(tmp_convY), np.array(tmp_toaMean), [r"Cluster ToA Mean vs Recosntructed Abs. Y", r"$Y_{Conversion}$", "ToA Mean [CLK]"], f"2D-AbsPointY-TOAMean-{nfile}-{isuffix}", outdir, debug)
        plot2Dhist(np.array(tmp_hits), np.array(tmp_toaRMS), [r"Cluster ToA RMS vs Hits", r"$N_{hits}$", "ToA RMS [CLK]"], f"2D-hits-TOARMS-{nfile}-{isuffix}", outdir, debug)

    if(debug):
        print("Writing Modulation fit results to report file...")
    freport.write("\nModulation factor Calculations:\n")
    outstring = ""
    for branch in main_results.keys():
        outstring+=f"{branch}:"
        subset = main_results[branch]
        nkeys = 0
        nsubset = len(subset)
        for key in subset.keys():
            outstring+=f"{key}={main_results[branch][key]}"
            nkeys+=1
            if(nkeys==nsubset):
                outstring+="\n"
            else:
                outstring+=","
        outstring+="\n"
   
    freport.write(outstring)
    outstring = None
 
    freport.write(f"\nFinished file {file}")
    freport.write("\n----------------------------------------\n")
    freport.flush()

    if(debug):
        print(f"\n {OUT_BLUE} DATA AFTER CUTS: {naccepted/ntotal*100:.2f}% survived {OUT_RST}\n")   
        print("RESETTING LISTS...")

    dname = file.split("/")[-1].split(".")[0]

    anono_list = ['reco','weighted','eVelApp'] 
    dlabel = clearString(dname,'-',',',anono_list)

    freport.write(f"Annotation:{dlabel}")
    freport.flush()
  
    if(fPlotCombinedTOT): 
        sumTOT_list.append([dlabel,np.array(tmp_sumTOT)])
        if(debug):
            print(f"len(sumTOT_list)=[{len(sumTOT_list)}]")

    tmp_evtnr.clear()
    tmp_sumTOT.clear()
    tmp_excent.clear()
    tmp_hits.clear()
    #tmp_densityRMS.clear()
    #tmp_densityWL.clear()
    #tmp_evtArea.clear()
    tmp_theta.clear()
    tmp_width.clear() 
    tmp_length.clear() 
    tmp_RMSL.clear()
    tmp_RMST.clear()
    tmp_weightedY.clear()
    tmp_weightedX.clear()
    tmp_secondangles_Y.clear()
    tmp_secondangles_Y_conv.clear()
    tmp_convX.clear()
    tmp_convY.clear()
    tmp_sumElec.clear()
    tmp_elecPerHit.clear()
    tmp_toaMean.clear()
    tmp_toaRMS.clear()
    tmp_toaLength.clear()
    tmp_toaSkew.clear()
    tmp_derivative.clear()
    tmp_centerX.clear()
    tmp_centerY.clear()
    
    abs_stdx, abs_stdy, abs_meanx, abs_meany = None, None, None, None
    
    electrons = None
    
    hasTheta = False
    fTiming = False
    fCharge = False
    fConverted = False

    nfile+=1

# plotting GLOBAL plots

if(fPlotGlobalSumTOT):
    simpleHist(np.array(GLOB_sumTOT), 100, np.nanmin(GLOB_sumTOT), np.nanmax(GLOB_sumTOT), [r"$\Sigma$(TOT) per Cluster",r"$\Sigma$(TOT)","Events,[N]"], f"tmp_sumTOT-GLOBAL", outdir, debug) 
    simpleHist(np.array(GLOB_hits), 100, np.nanmin(GLOB_hits), np.nanmax(GLOB_hits), ["NUmber of Hits per Cluster",r"$N_{hits}$,[#]",r"$N_{clusters}$,[#]"], f"tmp_hits-GLOBAL", outdir, debug) 

if(fPlotCombinedTOT):
    simpleMultiHist(sumTOT_list, # list of data lists
                    100, #nbins
                    0,    # minbin
                    10000, # maxbin
                    [r"$\Sigma$(TOT) for multiple MP runs", r"$\Sigma$(TOT)",r"$N_{clusters}/\Sigma(TOT)$"],# title,xlabel,ylabel
                    "sumTOT-GLOBAL", # picname
                    outdir, # output directory
                    debug,   # enable print()
                    fNorm=True) # enabling normalized plotting 
if(fPlotGlobalAngles):
    if(len(tmp_secondangles_glob)>0):
        fitAndPlotModulation(np.array(tmp_secondangles_glob),
                             100,
                             -np.pi,
                             np.pi,
                             ["Reconstructed Angle Distribution (Cut)", "Angle [radian]", r"$N_{Entries}$"],
                             f"STOLEN-XpolSecondStage-CutOnPosition-GLOB",
                             outdir,    
                             debug)
    tmp_secondangles_glob.clear() 
    if(len(tmp_BGRangles_glob)>0): 
        fitAndPlotModulation(np.array(tmp_BGRangles_glob),
                             100,
                             -np.pi,
                             np.pi,
                             ["Reconstructed Angle Distribution (Cut)", "Angle [radian]", r"$N_{Entries}$"],
                             f"STOLEN-XpolSecondStage-CutOnPosition-BGRangles-GLOB",
                             outdir,
                             debug)
    tmp_BGRangles_glob.clear() 


# checking here if i have only one type of angles or soemthing else
# don't print fof a set of different angles
# then the following global plots dont make sense

fAbortGlobal = sum(val != 0 for val in countangles.values()) > 1

if(debug):
    print(f"angles counted:\n{countangles.keys()}\n{countangles.values()}")

if(fAbortGlobal):
    fPlotGlobalCenters = False
    fPlotGlobalStokes = False
    if(debug):
        print(f"Detecting run with data for multiple angle settings: Ommiting GLobal cummulative matrix plots")

if(fPlotGlobalCenters):
    plot2dEvent(weightCenters_GLOB, "title:Weight Centers of All tracks (GLOBAL)", f"weight-centers-GLOBAL-{isuffix}", outdir, debug)
    plot2dEvent(absorption_points_pruned_GLOB, "title:Absorption Points Pruned (GLOBAL)", f"AbsPointsPruned-GLOB-{isuffix}",outdir, debug)
    weightCenters_GLOB = None

# plotting global I/Q/U parameter matrix 

if(fPlotGlobalStokes):
    plot2dEvent(glob_matrix_I, "title:Stokes I for individual Tracks in 64x64 map (All data sets)", f"GLOB-STOKES-I-{isuffix}", outdir, debug)
    plot2dEvent(glob_matrix_Q/glob_matrix_I, "title:Stokes Q for individual Tracks in 64x64 map (All data sets)", f"GLOB-STOKES-Q-{isuffix}", outdir, debug)
    plot2dEvent(glob_matrix_U/glob_matrix_I, "title:Stokes U for individual Tracks in 64x64 map (All data sets)", f"GLOB-STOKES-U-{isuffix}", outdir, debug)
 
# plotting some shite about HSCANs
if(fPlotGlobalLineScan):

    fig ,ax = plt.subplots(figsize=(10,8)) 
    
    for data in abs_matrixList:
        ax.errorbar(data[0][0],data[0][1],yerr=data[0][2],label=f"{data[1]}")
    ax.set_title("Reco. Abs. points")
    ax.set_xlabel("x, [pix]")
    ax.set_ylabel("y, [pix]")
    if(fBeamScan):
        ax.set_xlim([0,256])
        ax.set_ylim([0,65])
    else:
        ax.set_xlim([0,256])
        ax.set_ylim([0,256])
    ax.grid(which='major', color='gray', linestyle="--", linewidth='0.5')
    ax.grid(which='minor', color='gray', linestyle=":", linewidth='0.25')
    ax.minorticks_on()
    ax.legend()
    plt.savefig(f"{outdir}/MULTISCAT-MATRIX-Absorption-{plotname}.png")
    plt.close()
    
    ############################################################################

    fig ,ax = plt.subplots(figsize=(10,8)) 
    
    for data in weight_matrixList:
        ax.errorbar(data[0][0],data[0][1],yerr=data[0][2],label=f"{data[1]}")
    ax.set_title("Weigth Centers")
    ax.set_xlabel("x, [pix]")
    ax.set_ylabel("y, [pix]")
    if(fBeamScan):
        ax.set_xlim([0,256])
        ax.set_ylim([0,65])
    else:
        ax.set_xlim([0,256])
        ax.set_ylim([0,256])
    ax.grid(which='major', color='gray', linestyle="--", linewidth='0.5')
    ax.grid(which='minor', color='gray', linestyle=":", linewidth='0.25')
    ax.minorticks_on()
    ax.legend()
    plt.savefig(f"{outdir}/MULTISCAT-MATRIX-WeightCenters-{plotname}.png")
    plt.close()


freport.write(f"\n-SUCCESS-")
freport.close()
