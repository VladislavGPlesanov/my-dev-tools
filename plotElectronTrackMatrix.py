import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import sys
#from mpl_toolkits.mplot3d import Axes3D

txtfile = sys.argv[1]
picname = sys.argv[2]

f = open(txtfile,'r')

header = []
#xpix, ypix, nhits = [],[],[]

matrix = np.zeros((256,256), dtype=np.uint16)

nlines = 0
for line in f:

    data = line.split()

    if("FEC" in line or "Board" in line or "Chip" in line):
        header.append(line)
        data = None
    else:
        #xpix.append(int(data[0]))
        #ypix.append(int(data[1]))
        #nhits.append(int(data[2]))
        np.add.at(matrix,(int(data[0]), int(data[1])), int(data[2]))       
    nlines+=1

f.close()

############### just plotting it ###############################
fig, ax = plt.subplots(figsize=(10,10))
cax = fig.add_axes([0.86, 0.1, 0.05, 0.8])
ms = ax.matshow(matrix.T, cmap='viridis')
fig.colorbar(ms,cax=cax, orientation='vertical', label='occupancy')
ax.set_xlabel("x")
ax.set_ylabel("y")
#ax.invert_yaxis()
plt.savefig(f"photoelectronMatrix-{picname}-xy.png")
plt.close()

############# using plotly ##############################
#
#fig = go.Figure()
#
#fig.add_trace(go.Scatter3d(
#    x=ex, y=ey, z=ez,
#    #x=ex, y=ey, z=et,
#    #mode='lines+markers',
#    mode='markers',
#    marker=dict(size=3, color='red'),
#    #line=dict(color='red',width=2),
#    name='Track'
#))
#
#fig.add_trace(go.Scatter3d(
#    x=[ex[0]], y=[ey[0]], z=[ez[0]],
#    #x=[ex[0]], y=[ey[0]], z=[et[0]],
#    mode='markers',
#    marker=dict(size=5, color='green'),
#    name='Start Point'
#))
#
#fig.update_layout(
#    scene=dict(
#        xaxis_title='x, mm',
#        yaxis_title='y, mm',
#        zaxis_title='z, mm',
#        zaxis=dict(autorange='reversed')  # Invert Z axis
#    ),
#    margin=dict(l=0, r=0, b=0, t=0)
#)
#
#fig.write_html(f"photoelectronTrack-{picname}.html")








