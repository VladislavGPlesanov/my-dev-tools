import numpy as np
import matplotlib.pyplot as plt
import sys
import tables as tb
from matplotlib.colors import LogNorm

def getBaseGroupName(file):

    print(f"Reading {f}")
    groups = file.walk_groups('/')
    grouplist = []
    for gr in groups:
        #print(f'found {gr}')
        grouplist.append(gr)
    main_group = str(grouplist[len(grouplist)-1])
    print(f"last entry in walk_groups = \n{main_group}")
    grouplist = None 
    
    basewords = main_group.split('(')
    print(basewords)
    
    base_group_name = basewords[0][:-1]+'/'

    return base_group_name

def getChargeCenter(xdata,ydata,charge):

    sum_TOTX, sum_TOTY = 0,0
    for itot, ix, iy in zip(charge, xdata, ydata):
        sum_TOTX += itot*ix
        sum_TOTY += itot*iy

    qx = sum_TOTX/np.sum(charge)
    qy = sum_TOTY/np.sum(charge)

    qxdev = np.std(xdata)
    qydev = np.std(ydata)
 
    return qx,qy, qxdev, qydev

def plotMatrix(matrix, labels, picname, auxpicname, plotMarker, plotVector):

    fig, ax = plt.subplots(figsize=(12,8))
    cax = fig.add_axes([0.86,0.1,0.05,0.8])
    ms = None
    use_cmap = 'jet'
    #ms = ax.matshow(matrix.T, cmap=use_cmap, norm=LogNorm(vmin=1,vmax=np.nanmax(matrix)))
    ms = ax.matshow(matrix.T, cmap=use_cmap)
    ax.scatter([qx],[qy],marker="*",c='m', s=50)
    ax.plot([plotVector[0],plotVector[2]],[plotVector[1],plotVector[3]], c="red", marker="d")
    #ax.plot([plotMarker[0],plotVector[0]],[plotMarker[1],plotVector[1]], c="red", marker="d")
    if(len(labels)>3):
        fig.colorbar(ms,cax=cax,orientation='vertical', label=labels[3])
    else:
        fig.colorbar(ms,cax=cax,orientation='vertical', label="TOT")
    ax.set_title(labels[2])
    ax.set_ylabel(labels[1])
    ax.set_xlabel(labels[0])
    ax.xaxis.set_label_position('top') 
    ax.invert_yaxis()
    plt.savefig(f"{picname}-{auxpicname}.png", dpi=400)
    ms = None
    plt.close()

def getVectorPoints(vector, center):

    centerX = center[0]
    centerY = center[1]

    #scaleF = 5       
 
    x1 = vector[0] + centerX
    #x2 = vector[0] - centerX
    y1 = vector[1] + centerY
    #y2 = vector[1] - centerY

    #return x1,x2,y1,y2
    return x1,y1

def getPrincipalAxisLine(center, eigen_vector, length=40):
    """
    Returns two endpoints of the main PCA axis line for plotting.
    `length` controls half-length of the line.
    """
    cx, cy = center
    vx, vy = eigen_vector / np.linalg.norm(eigen_vector)
    
    x1, y1 = cx - vx * length, cy - vy * length
    x2, y2 = cx + vx * length, cy + vy * length

    return x1, y1, x2, y2

def runAngleReco(positions, charges):

    center = np.average(positions, axis=1, weights=charges)
    print(f"CENTER AT:\n{center}")
    xpos, ypos = positions
    X = np.vstack((xpos - center[0], ypos - center[1]))
 
    print(f"X:\n{X}")
   
    # Covariance matrix 
    M = np.dot(X*charges, X.T)

    # getin' eigen val's n vectors for covariance matrix
    eigenVal, eigenVect = np.linalg.eig(M) 
    print(f"EIGENVAL:\n{eigenVal}\n")
    print(f"EIGENVECT:\n{eigenVect}\n")

    # get axis which maximizes second moment - eigenvector w biggest eigenvalue
    prime_axis = eigenVect[:, np.argmax(eigenVal)] 
    print(f"PRIME AXIS:\n{prime_axis}")

    # projectin' new axis on x-y and calc its angle      
    projection_xy = np.array([prime_axis[0],prime_axis[1]])
    print(f"PROJECTION_XY:\n{projection_xy}")

    # getting phi_1 basically here
    angle = np.arctan2(projection_xy[1],projection_xy[0])

    # projectin' data points onto new axis plane
    projection_xy_fit = np.dot(X[:2].T, prime_axis[:2])
    print(f"PROJECTION_XY_FIT:\nprojection_xy_fit")

    # calculatin' skewness
    skew_xy = np.sum(charges * projection_xy_fit**3)/np.sum(charges)
    print(f"Skeweness: {skew_xy}")

    if skew_xy > 0:
        if angle > 0: 
            angle = -np.pi + angle
        else:
            angle = np.pi + angle
    else:
        angle = angle

    ## second step ends
    #i_k = 1 
    #q_k = np.cos(2*angle)
    #u_k = np.sin(2*angle)

    #return angle, i_k, q_k, u_k
    return angle, center, eigenVect, projection_xy

#===================================================================================
#===================================================================================
#===================================================================================

infile = sys.argv[1]
picname = sys.argv[2]

with tb.open_file(infile,'r') as f:

    start_cut = 5000
    end_cut = 10000

    bg_name = getBaseGroupName(f) 
    cluster_x = f.get_node(bg_name+"x")[start_cut:end_cut]
    cluster_y = f.get_node(bg_name+"y")[start_cut:end_cut]
    cluster_tot = f.get_node(bg_name+"ToT")[start_cut:end_cut]
    cluster_epsilon = f.get_node(bg_name+"eccentricity")[start_cut:end_cut]   
    cluster_length = f.get_node(bg_name+"length")[start_cut:end_cut]   
    
    nevents = len(cluster_x)
    print(f"Selected {nevents} events")

    npics=0
    nclusters = 0
    for xpos, ypos, tot in zip(cluster_x, cluster_y, cluster_tot):
        ic_length = cluster_length[nclusters]
        ic_excent = cluster_epsilon[nclusters]
        nhits = len(xpos)
        matrix = np.zeros((256,256),dtype=int)
        np.add.at(matrix,(xpos,ypos),tot)
        if(nhits>90 and ic_excent > 2.0 and ic_length <= 4 and npics<=20):
            print(f"===============INTERSTING EVENT: [{nclusters}]===================")
            qx,qy,qxdev,qydev = getChargeCenter(xpos,ypos,tot)
            iangle, icenter, eVector, proj_xy = runAngleReco([xpos,ypos],tot)
            print(f"Weighted charge center at x={qx}, y={qy}")
            print(f"icenter is at {icenter}")
            #x1,x2,y1,y2 = getVectorPoints(eVector, icenter) 
            #x1,x2,y1,y2 = getVectorPoints(proj_xy, icenter) 
            #x1,y1 = getVectorPoints(proj_xy, icenter)
            
            x1,y1,x2,y2 = getPrincipalAxisLine(icenter, proj_xy) 

            #multiplier = 2
            #xrange_1 = np.round(qx-multiplier*qxdev) if qx-multiplier*qxdev >= 0 else 0 
            #xrange_2 = np.round(qx+multiplier*qxdev) if qx+multiplier*qxdev <= 256 else 256 
            #yrange_1 = np.round(qy-multiplier*qydev) if qx-multiplier*qydev >= 0 else 0 
            #yrange_2 = np.round(qy+multiplier*qydev) if qx+multiplier*qydev <= 256 else 256 

            plotMatrix(matrix, ["x","y",f"Event {nclusters}"], picname, f"BareEvent-C-{nclusters}",[qx,qy], [x1,x2,y1,y2])
            #plotMatrix(matrix, ["x","y",f"Event {nclusters}"], picname, f"BareEvent-C-{nclusters}",[qx,qy], [x1,y1])
            npics+=1
            nclusters+=1
        else:
            nclusters+=1
        matrix = None
    
    print(f"Run over {nclusters} clusters")
    

