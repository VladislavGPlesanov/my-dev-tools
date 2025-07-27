import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from MyPlotter import myUtils 
import glob
import os

location = sys.argv[1]
plotname = sys.argv[2]

dir_files = glob.glob(location+"*.OUT")
file_list = sorted(dir_files)
print(f"found <{len(file_list)}> degrad output files")
nFiles = len(file_list)
for i in range(10):
    print(f"file{i}->{file_list[i]}")


outdir = f"tmpHTML-{plotname}/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

MU = myUtils()

ngas1, ngas2 = 0, 0 

nfiles = 0
for file in file_list:

    fname = MU.removePath(str(file))
    evtNr = int(fname[16:21])

    if(os.stat(file).st_size==0):
        l_empty.append(cnt)
        cnt+=1
        nEmpty+=1
        continue


    f = open(file,'r')
    
    global_par, e_coordinates = None, None
    
    nlines = 0
    for line in f:
    
        if(nlines == 0):
    
            global_par = line
            nlines+=1
            continue
        if(nlines == 1):
    
            e_coordinates = line
    
    f.close()
    f = None
    
    header = ["nevent", 
              "ELECTRON CLUSTER SIZE (NCLUS)",
              "EXCITATION CLUSTER SIZE (NSTEXC)",
              "NUMBER OF EXC IN GAS1", 
              "NUMBER OF EXC IN GAS2", 
              "NUMBER OF EXC IN GAS3", 
              "NUMBER OF EXC IN GAS4", 
              "NUMBER OF EXC IN GAS5", 
              "NUMBER OF EXC IN GAS6", 
              "NUMBER OF PRIMARY COMPTON COLLISIONS (MCOMP)",
              "NUMBER OF PRIMARY PAIR PRODUCTION COLLISIONS (MPAIR)",
              "NULL COLLISION CLUSTER SIZE IN MOLECULAR GASES (NEXCNUL)"]
     
    nstep = 0
    for word, head in zip(global_par.split(),header):
        print(f"{head} \t -> \t{word}")
        if(nstep==3):
            ngas1 += int(word)
        if(nstep==4):
            ngas2 += int(word)
   
        nstep += 1     

    ex, ey, ez = None, None,None
    
    data_line = e_coordinates.strip()
    flat_data = list(map(float, e_coordinates.split()))
    points = np.array(flat_data).reshape(-1,7)
    
    ex = points[:,0]/10000.0 
    ey = points[:,1]/10000.0 
    ez = points[:,2]/10000.0 
    et = points[:,3]/10000.0
    
    ############ using plotly ##############################
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode='markers',
        marker=dict(size=3, color='red'),
        name='Track'
    ))
    
    fig.add_trace(go.Scatter3d(
        #x=[ex[0]], y=[ey[0]], z=[ez[0]],
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=5, color='green'),
        name='Start Point'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='x, mm',
            yaxis_title='y, mm',
            zaxis_title='z, mm'
            #zaxis=dict(autorange='reversed')  # Invert Z axis
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    fig.write_html(f"{outdir}photoelectronTrack-{evtNr}.html")

    if(nfiles==100):
        break

    nfiles+=1
###########################################################

print(f"ngas1={ngas1}")
print(f"ngas2={ngas2}")
print(f"RATIO: {ngas1/ngas2:.2f}")



