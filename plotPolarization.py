import sys
import glob
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

    #return Ap + Bp*np.cos(x - phi0)**2
    return Bp + Ap*np.cos(x - phi0)**2

def modFac(Ap,Bp):

    return Bp/(2*Ap+Bp)

def deltaMu(mu, Ap, Bp, Ap_err, Bp_err):

    return mu * np.sqrt( (Bp_err / Bp)**2  + (2 * Ap_err / (Bp + 2*Ap) )**2 )

def polDeg(Ap, Bp, mu):

    return (1/mu)*(Bp/((2*Ap)+Bp))

def check_node(gr_path, file):

    return gr_path in file

def toDegrees(angle):

    return np.round(angle*180.0/3.14159,2)

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

def progress(ntotal, ith):

    try:
        perc = round(float(ith)/float(ntotal)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    finally:
        print(f"\r{perc}% done", end="",flush=True)

def getPlaneOffset(filename, keyword):
        
    words = filename.split("-")

    return words[words.index(keyword)-1]


def fitAndPlotModulation(nuarray, nbins, minbin, maxbin, labels, picname):

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
    plt.text(minbin*1.1, maxy*0.95, r"$N(\phi) = A_{P} + B_{P}\cdot cos^2(\phi-\phi_{0})$")
    plt.text(minbin*1.2, maxy*0.90, f"Ap={Ap:.2f}, Bp={Bp:.2f}")

    plt.legend(loc='upper right')

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.grid()
    plt.savefig(f"1DHist-{picname}.png")
    plt.close()

    return phi

#------------- main starts here ---------
recodir = sys.argv[1]
plotname = sys.argv[2]

recofiles = glob.glob(recodir+"*.h5")

angles1, angles2, planeOffset = [], [] , []

for file in recofiles:

    with tb.open_file(file) as f:

        print(f"Reading {f}")
        groups = f.walk_groups('/')
        grouplist = []
        for gr in groups:
            #print(f'found {gr}')
            grouplist.append(gr)
        main_group = str(grouplist[len(grouplist)-1])
        #print(f"last entry in walk_groups = \n{main_group}")
        grouplist = None 

        basewords = main_group.split('(')
        #print(basewords)
        #------ get angle from file name -----

        offset = getPlaneOffset(file, "deg") 
        planeOffset.append(float(offset))

        # ----------------------------------
        base_group_name = basewords[0][:-1]+'/'
        oldXpolReco = check_node(base_group_name+"angle_fiststage", f)
        newXpolReco = check_node(base_group_name+"angle_firststage", f)
        print("=> got base  group name")
        if(oldXpolReco):
            firstangles = f.get_node(base_group_name+"angle_fiststage")[:].T
        if(newXpolReco):
            firstangles = f.get_node(base_group_name+"angle_firststage")[:].T
        
        secondangles = f.get_node(base_group_name+"angle_secondstage")[:].T
        print("=> got angle data")

        first_angle = fitAndPlotModulation(firstangles,
                             100,
                             -np.pi,
                             np.pi,
                             ["Reconstructed Angle Distribution", "Angle [radian]", r"$N_{Entries}$"],
                             f"XpolFirstStage-OFFSET-{offset}",
                             )

        print("=> got FIRST_ANGLE")
        second_angle = fitAndPlotModulation(secondangles,
                             100,
                             -np.pi,
                             np.pi,
                             ["Reconstructed Angle Distribution", "Angle [radian]", r"$N_{Entries}$"],
                             f"XpolSecondStage-OFFSET-{offset}",
                             )
        print("=> got SECOND_ANGLE")
        
        angles1.append(toDegrees(first_angle))
        angles2.append(toDegrees(second_angle))
        print(f"DONE WITH {file}")

######## plottin' stuff ##########################

print("---PLOTTING---")
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(planeOffset, angles1)
ax.set_xlabel(r"Gridpix Plane Offset [$\circ$]")
ax.set_ylabel(r"Reconstructed $\phi$ [$\circ$]")
ax.set_title("Reconstructed Polarization angles vs. Angular Rotation of GridPix3")
fig.savefig(f"ANGLES-VS-ROTATION-{plotname}.png")
plt.close()
print("DONE")

