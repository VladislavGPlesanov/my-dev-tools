import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D

degrad = sys.argv[1]
picname = sys.argv[2]

f = open(degrad,'r')

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

for word, head in zip(global_par.split(),header):
    print(f"{head} \t -> \t{word}")

ex, ey, ez = None, None,None

data_line = e_coordinates.strip()
flat_data = list(map(float, e_coordinates.split()))
points = np.array(flat_data).reshape(-1,7)

ex = points[:,0]/10000.0 
ey = points[:,1]/10000.0 
ez = points[:,2]/10000.0 
et = points[:,3]/10000.0

############### just plotting it ###############################
#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(ex,ey,ez,s=10,c='red')
#ax.scatter(ex[0],ey[0],ez[0],s=70,c='green')
#ax.set_xlabel('x,mm')
#ax.set_ylabel('y,mm')
#ax.set_zlabel('z,mm')
#ax.invert_zaxis()
#plt.tight_layout()
#plt.savefig(f"TestTrack-{picname}-xyz.png")
#plt.close()

############ using plotly ##############################

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=ex, y=ey, z=ez,
    mode='markers',
    marker=dict(size=3, color='red'),
    name='Track'
))

fig.add_trace(go.Scatter3d(
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

fig.write_html(f"photoelectronTrack-{picname}.html")








