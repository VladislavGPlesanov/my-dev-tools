#import h5py
import sys
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import tables as tb
from scipy.optimize import curve_fit
from MyPlotter import myUtils, myColors, mySymbols, myPlotter
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm

from scipy.ndimage import gaussian_filter
from scipy import ndimage
from skimage.feature import canny
#from skimage.transform import hough_line, hough_line_peaks
#from skimage.transform import PiecewiseAffinetransform


MP = myPlotter()
MC = myColors()
MU = myUtils()
MS = mySymbols()

def checkNAN(number):

    if(np.isnan(number)):
        return -1
    else:
        return number

def toDegrees(angle):

    return angle*180.0/3.14159 

def cosfunc(x, Ap, Bp, phi0):

    return Ap + Bp*np.cos(x - phi0)**2

def modFac(Ap,Bp):

    return Bp/(2*Ap+Bp)

def deltaMu(Ap, Bp, Ap_err, Bp_err, covAB):
    # would use if A&B are uncorrelated
    #return mu * np.sqrt( (Bp_err / Bp)**2  + (2 * Ap_err / (Bp + 2*Ap) )**2 )
    # correlated case:

    #return np.sqrt((4*Bp**2*Ap_err+4*Ap**2*Bp_err-2*Ap*Bp*covAB)/(2*Ap+Bp)**4)

    # correlated case recalculated
    return 2*np.sqrt((Bp**2*Ap_err+Ap*(-2*Bp*covAB+Ap*Bp_err))/(2*Ap+Bp)**4)


def calcIntensity(A, B):

    return A + B/2.0

def calcIntensityError(sigA, sigB):

    return np.sqrt(sigA**2 + 0.5*sigB**2)

def calcStokesSigmaI(sA,sB,sAB):

    return np.sqrt(sA**2 + sB**2/4+sAB)

def calcStokesSigmaQ(sigmaA, sigmaB, covAB, phi):

    return np.sqrt(np.cos(2*phi)**2*(sigmaA**2 + (1/4)*sigmaB**2 +covAB))

def calcStokesSigmaU(sigmaA, sigmaB, covAB, phi):

    return np.sqrt(np.sin(2*phi)**2*(sigmaA**2 + (1/4)*sigmaB**2 +covAB))

#def fitAndPlotModulation(nuarray, nbins, minbin, maxbin, labels, picname, odir):
def fitAndPlotModulation(nuarray, nbins, minbin, maxbin, labels, picname):

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

    ## ------ checking covarinces ----------
    #if(debug):
    #    print(f"covariance matrix \nlen={len(covariance)}, \nshape={covariance.shape},\n{covariance}")
    #    condition = np.linalg.cond(covariance)
    #    print(f"fit condition={condition:.4f} (ideal:1, absurd: ill-conditioned)")

    ## -------------------------------------

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
             facecolor="royalblue")

    plt.plot(bin_centers[:-1], cosfunc(bin_centers[:-1], Ap, Bp, phi), '--r')
    plt.vlines(phi, 
                0, 
                np.max(counts)*1.05,
                colors='darkviolet',
                linestyles='dashed',
                label=f"{MS.phi}={phi:.2f}"+r"$\pm$"+f"{phi_err:.2f}\n({toDegrees(phi):.2f} deg)")

    pmstr = r"$\pm$"
    plt.hlines(Ap, -0.1, 3.15, colors='yellow', label=f"Ap={Ap:.2f}{pmstr}{Ap_err:.2f}")
    plt.hlines(Ap+Bp, -0.1, 3.15, colors='lightseagreen', label=f"Bp={Bp:.2f}{pmstr}{Bp_err:.2f}")
 
    ax = plt.gca()
    miny,maxy = ax.get_ylim()
    #if(debug):
    #    print(f"Getting miny/maxy for histogram: {picname}")
    #    print(f"miny={miny:.4f}, maxy={maxy:.4f}")

    plt.ylim([miny, maxy*1.2])

    mu = modFac(Ap,Bp)
    muErr = deltaMu(mu, Ap, Bp, Ap_err, Bp_err)
    intensity = calcIntensity(Ap, Bp)
    sigma_AB = covariance[0,1]
    intensity_err = calcStokesSigmaI(Ap_err,Bp_err,sigma_AB)
    #P = polDeg(Ap, Bp, mu)
    #P = getPolDegree(Bp+Ap, Ap)
 
    # Paolo, Polarimetry pdf, page 27
    Q = (1/mu) * (Bp/2) * np.cos(2*phi)
    U = (1/mu) * (Bp/2) * np.sin(2*phi)   

    # Now using uncertainties for Q,U for correlated case
    # Analytic solution in a Mathematica notebook
    dQ = calcStokesSigmaQ(Ap_err, Bp_err, sigma_AB, phi)
    dU = calcStokesSigmaU(Ap_err, Bp_err, sigma_AB, phi)

    #if(debug):
    #    print(f"Uncertainties dI = {intensity_err:.4f}, dQ = {dQ:.4f}, dU={dU:.4f}")

    #---------------------------------------------------

    #MDP = getMDP(mu, len(nuarray))

    #---------------------------------------------------
    #plt.text(-3.13, maxy*1.16, r"$\Sigma$"+f"(Entries)={len(nuarray)} MDP(CL99%)={MDP*100:.2f}",fontsize=11)
    plt.text(-3.13, maxy*1.16, r"$\Sigma$"+f"(Entries)={len(nuarray)}",fontsize=11)
    plt.text(-3.13 , maxy*1.10, r"$N(\phi) = A_{P} + B_{P}\cdot cos^2(\phi-\phi_{0})$", fontsize=11)
    #plt.text(-3.13 , maxy*1.04, f"{MS.mu}={mu*100:.2f}%"+r"$\pm$"+f"{muErr*100:.2f}%", fontsize=11)
    plt.text(-3.13 , maxy*1.04, f"{MS.mu}={mu*100:.2f}%"+r"$\pm$"+f"{muErr*100:.2f}% ("+r"$\frac{\delta\mu}{\mu}=$"+f"{muErr/mu*100:.2f}%)", fontsize=11)
    plt.text(-3.13 , maxy*0.98, f"I={intensity:.2f}"+r"$\pm$"+f"{intensity_err:.2f}", fontsize=11)
    plt.text(-3.13 , maxy*0.92, f"Q={Q:.2f} +- {dQ:.2f} ({Q/intensity:.4f}+-{dQ/intensity:.4f})", fontsize=11)
    plt.text(-3.13 , maxy*0.86, f"U={U:.2f} +- {dU:.2f} ({U/intensity:.4f}+-{dU/intensity:.4f})", fontsize=11)
    plt.text(-3.13 , maxy*0.80, r"$\chi_{red}^{2}$"+f"={chired:.4f}", fontsize=11)

    #if(debug):
    #    print("________________________________________________________")
    #    print(f"{OUT_RED} Modulation factor = {mu*100.0:.2f} {OUT_RST}")
    #    print(f"Polarization degree = {P:.2f}")
    #    print(f"\nStokes Parameters: \nQ(P1)={Q:.4f}, U(P2)={U:.4f}")#, V={V:.4f}\n")
    #    print(f"Intensity = {intensity}")
    #    print(f"{OUT_RED}{G_chi}={chired:.2f}{OUT_RST}")
    #    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    plt.legend(loc='upper right')

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.grid()
    plt.savefig(f"DISGUSTING-MODULATION-{picname}.png")
    plt.close()

    return mu, muErr, phi, phi_err

def plotMatrixWithRegion(globMatrix, leftbound, ycenter, reg_npixels, labels, picname):

    fig, ax = plt.subplots(figsize=(9,8))
    cax = fig.add_axes([0.86,0.1,0.05,0.8])
    ms = None
    ms = ax.matshow(globMatrix, cmap='jet', norm=LogNorm(vmin=1,vmax=np.nanmax(globMatrix)))
    fig.colorbar(ms, cax=cax, orientation='vertical')
    rect = Rectangle((leftbound, ycenter-reg_npixels/2),reg_npixels,reg_npixels,linewidth=1.5,edgecolor='red',facecolor='none')
    ax.add_patch(rect)
    ax.set_title(labels[0])
    ax.set_xlabel(labels[1])
    ax.set_xlabel(labels[2])
    ax.invert_yaxis()
    plt.savefig(f"{picname}.png")
    plt.close()





##############################################################

parser = ap.ArgumentParser()
parser.add_argument("-n","--name", type=str, default="HUYA")
parser.add_argument("-f","--files", nargs="+")
args = parser.parse_args()

filelist = args.files
picname = args.name

matrix_AP = np.zeros((256,256),dtype=int)

good_angles = []

swipe_area = 32 # pixels
swipe_steps = 16
swipe_centery = 95#160
xcenter_spacing = np.floor(256/swipe_steps)
swipe_ymin = swipe_centery-swipe_area/2
swipe_ymax = swipe_centery+swipe_area/2

sliding_reg = [ [] for i in range(int(swipe_steps))]
sliding_matrices = [np.zeros((256,256),dtype=int) for i in range(int(swipe_steps))]


sliding_bounds = np.linspace(swipe_area/2,256-swipe_area/2, swipe_steps)

square_left = np.array(sliding_bounds)-swipe_area/2
square_right = np.array(sliding_bounds)+swipe_area/2
#if(debug):
for c, l, r in zip(sliding_bounds, square_left, square_right):
    print(f"box {l:.2f}<---{c:.2f}--->{r:.2f}")

dummy_mat = np.ones((256,256),dtype=int)
ms = None
fig, ax = plt.subplots(figsize=(9,9))
ms = ax.matshow(dummy_mat.T, cmap='viridis')
for rc, rl, rr in zip(sliding_bounds, square_left, square_right):
    rect = Rectangle((rl,swipe_centery-swipe_area/2),swipe_area, swipe_area, linewidth=1.5,edgecolor='red', facecolor='none')
    ax.add_patch(rect)
ax.set_xlabel("x")
ax.set_xlabel("y")
ax.invert_yaxis()

plt.savefig(f"checking-sliding-regions-{picname}.png")
plt.close()

ifile=0
for file in filelist:
    ifile+=1
    with tb.open_file(file,'r') as f:
    
            groups = f.walk_groups('/')
            grouplist = []
            for gr in groups:
                #print(f'found {gr}')
                grouplist.append(gr)
            main_group = str(grouplist[len(grouplist)-1])
            #print(f"last entry in walk_groups = \n{main_group}")
        
            grouplist = None 
        
            basewords = main_group.split('(')
            print(basewords)
        
            base_group_name = basewords[0][:-1]+'/'
            #                              ^ removes space at the end of 'run_xxx/chip0 '
            #bgn_split = base_group_name.split('/')
            #run_name = bgn_split[2]
            #run_num = int(run_name[4:])
    
            ##############################################
            # cleaning up the base name resolving info
            groups = None
            grouplist = None
            main_group = None
            #run_name, run_num, bgn_split = None, None, None 
            basewords = None
    
            secondangles = f.get_node(base_group_name+"angle_secondstage") 
            absorp_x = f.get_node(base_group_name+"absorption_point_x")
            absorp_y = f.get_node(base_group_name+"absorption_point_y")
            totev = len(secondangles)
    
            for ievent, (secang, apx, apy) in enumerate(zip(secondangles,absorp_x,absorp_y)):
                
                iapx = int(np.floor(checkNAN(apx)))
                iapy = int(np.floor(checkNAN(apy)))
                isecang = checkNAN(secang)    

                np.add.at(matrix_AP, (iapx,iapy),1)
            
                for bleft, bright, slidreg, slidemat in zip(square_left, square_right, sliding_reg, sliding_matrices):
                   if(iapy>=swipe_ymin and iapy<=swipe_ymax):
                       if(iapx>=bleft and iapx<=bright):
                           slidreg.append(isecang)
                           np.add.at(slidemat,(iapx,iapy),1)
                       else:
                           continue
                   else:
                       break

                MU.progress_bar(ievent,totev)

    print(f"Done with {file} ({ifile}/{len(filelist)})")

######################################################
for modlist,lbound, rbound, imatrix in zip(sliding_reg, square_left, square_right, sliding_matrices):
    ititle="Moduation within "+r"y$\in$"+f"{swipe_ymin:.2f};{swipe_ymax:.2f}"+r"x=$\in$"+f"({lbound:.2f};{rbound:.2f})"
    region_name = f"x{int(np.floor(lbound))}_{int(np.floor(rbound))}_y{int(np.floor(swipe_ymin))}_{int(np.floor(swipe_ymax))}"
    print(f"Plotting: {region_name}")
    reg_mu, _, _,_ = fitAndPlotModulation(np.array(modlist),
                         100,
                         -np.pi,
                         np.pi,
                         [ititle,"angle, [rad]",r"$N_{entries}$"],
                         f"DISGUSTING-PART-MOD-{region_name}")
    #MP.plotMatrix(imatrix.T, f"AP-{region_name}", fLognorm=True)
    plotMatrixWithRegion(matrix_AP.T, lbound, swipe_centery, swipe_area, 
                            [ititle,"angle, [rad]",r"$N_{entries}$"],
                            f"AP-region-{region_name}")

 
######################################################
MP.plotMatrix(matrix_AP.T, f"GLOBAL-AP-{picname}", fLognorm=True)

img_smooth = gaussian_filter(matrix_AP, sigma=1.0)

MP.plotMatrix(img_smooth.T, f"GLOBAL-AP-SMOOTH-{picname}", fLognorm=True)

#sob_edges = ndimage.sobel(img_smooth)
#can_edges = canny(img_smooth/np.max(img_smooth),sigma=1.0)

#
#h,theta, d = hough_line(can_edges)
#
#lines = hough_line_peaks(h,theta,d)
#
#lines_v, lines_h = [],[]
#
#for angle, dist in zip(lines[1],lines[2]):
#    if(np.abs(angle) < np.pi/4):
#        lines_v.append((angle,dist))
#    else:
#        lines_h.append((angle,dist))
#
#print(f"FOUND VERT. LINES: {lines_v}")
#print(f"FOUND HORI. LINES: {lines_h}")
#
#plt.figure(figsize=9,9)
#for angle, dist in zip()
#
#
#
##tform = PiecewiseAffineTransform()
##tform.estimate()



        














