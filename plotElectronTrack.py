import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D

degrad = sys.argv[1]
finaltxt = sys.argv[2]
picname = sys.argv[3]

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

# reading file with final electron positions
#

xpos,ypos = [],[]
nFluor, nPairProd, nBremss = 0, 0, 0
matrix = np.zeros((256,256),dtype=np.uint16)

f = open(finaltxt,'r')
for line in f:
    if("x" in line):
        continue
    nums = line.split(',')
    xpos.append(float(nums[0]))
    ypos.append(float(nums[1]))
    np.add.at(matrix,(int(nums[3]),int(nums[4])),1)
    nFluor+=int(nums[5])
    nPairProd+=int(nums[6])
    nBremss+=int(nums[7])

print(f"Stats: fFluorescense:{nFluor}, fPairProd:{nPairProd}, fBremss:{nBremss}")
print(f"Retrieving some global info out of the finlename: {finaltxt}")

splitname = finaltxt.split(".")[0].split("-")

title_subtext = splitname[4]+"-"+splitname[5]+"-"+splitname[6]+"-"+splitname[7]

ex, ey, ez = None, None,None

data_line = e_coordinates.strip()
flat_data = list(map(float, e_coordinates.split()))
points = np.array(flat_data).reshape(-1,7)

ex = points[:,0]/10000.0 
ey = points[:,1]/10000.0 
ez = points[:,2]/10000.0 
et = points[:,3]/10000.0

############### just plotting it ###############################
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(ex,ey,ez,s=10,c='red')
ax.scatter(ex[0],ey[0],ez[0],s=70,c='green')
ax.set_xlabel('x,mm')
ax.set_ylabel('y,mm')
ax.set_zlabel('z,mm')
ax.invert_zaxis()
plt.tight_layout()
plt.savefig(f"photoelectronTrack-{picname}-xyz.png")
plt.close()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(ex,ey,et,s=10,c='red')
ax.scatter(ex[0],ey[0],et[0],s=70,c='green')
ax.set_xlabel('x,mm')
ax.set_ylabel('y,mm')
ax.set_zlabel('z,mm')
ax.invert_zaxis()
plt.tight_layout()
plt.savefig(f"photoelectronTrack-{picname}-xy_time.png")
plt.close()

############ using plotly ##############################

#fig = go.Figure()
#
#fig.add_trace(go.Scatter3d(
#    x=ex, y=ey, z=ez,
#    mode='markers',
#    marker=dict(size=3, color='red'),
#    name='Track'
#))
#
#fig.add_trace(go.Scatter3d(
#    #x=[ex[0]], y=[ey[0]], z=[ez[0]],
#    x=[0], y=[0], z=[0],
#    mode='markers',
#    marker=dict(size=5, color='green'),
#    name='Start Point'
#))
#
#fig.update_layout(
#    scene=dict(
#        xaxis_title='x, mm',
#        yaxis_title='y, mm',
#        zaxis_title='z, mm'
#        #zaxis=dict(autorange='reversed')  # Invert Z axis
#    ),
#    margin=dict(l=0, r=0, b=0, t=0)
#)

fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.7, 0.3],  # adjust width ratio
    specs=[[{'type': 'scene'}, {'type': 'heatmap'}]],
    subplot_titles=(f'3D Electron Track ({title_subtext})', '2D Matrix')
)

# 3D scatter
fig.add_trace(go.Scatter3d(
    x=ex, y=ey, z=ez,
    mode='markers',
    marker=dict(size=3, color='red'),
    #line=dict(color='red'),
    name='Track'
), row=1, col=1)

# Start point
fig.add_trace(go.Scatter3d(
    #x=[ex[0]], y=[ey[0]], z=[ez[0]],
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=6, color='green'),
    name='Start'
), row=1, col=1)

# 2D matrix heatmap
fig.add_trace(go.Heatmap(
    z=matrix,
    colorscale='Viridis',
    colorbar=dict(title="Counts"),
    showscale=True
), row=1, col=2)

# Layout tweaks
fig.update_layout(
    scene=dict(
        #xaxis=dict(title='x [mm]', range=[-0.25,0.25]),
        xaxis=dict(title='x [mm]'),
        yaxis=dict(title='y [mm]'),
        zaxis=dict(title='z [mm]')
    ),
    #xaxis2=dict(scaleanchor="y2", range=[-0.25,0.25]),
    #yaxis2=dict(scaleanchor="x2", range=[-0.25,0.25]),
    margin=dict(l=0, r=0, t=40, b=0),
    height=700,
    width=1550,
    legend=dict(
                orientation='h',
                x=0.25,
                y=-0.15,
                xanchor='center',
                yanchor='bottom',
                bgcolor='rgba(255,2355,255,0.7)',
                bordercolor='black',
                borderwidth=1
                )
   
)

fig.write_html(f"photoelectronTrack-{picname}.html")








