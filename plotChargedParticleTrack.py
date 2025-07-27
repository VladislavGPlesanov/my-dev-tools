import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

clusterfile = sys.argv[1]
electronfile = sys.argv[2]
picname = sys.argv[3]

f = open(clusterfile,'r')

x,y,z = [],[],[]
cluster_E, nElectrons = [],[]

for line in f:
    if("cluster" in line):
        continue
    words = line.split(',')
    
    x.append(float(words[0]))
    y.append(float(words[1]))
    z.append(float(words[2]))
    cluster_E.append(float(words[3]))
    nElectrons.append(float(words[5]))

    line = None
    words = None

f.close()
f = None

plt.figure(figsize=(8,8))
plt.scatter(x,y)
plt.grid(True)
plt.savefig("track.png")

counts, edges = np.histogram(cluster_E, bins=100)

binCenters = (edges[:-1]+edges[1:])/2

plt.figure(figsize=(8,8))
plt.hist(edges[:-1],weights=counts, bins=100, align='left',histtype='stepfilled')
plt.yscale('log')
plt.savefig("ClusterEnergy.png")

f = open(electronfile,'r')

ex,ey,ez,et = [],[],[],[]
energy = []

for line in f:

    words = line.split(',')

    ex.append(float(words[0]))
    ey.append(float(words[1]))
    ez.append(float(words[2]))
    et.append(float(words[3]))
    #energy.append(float(words[4]))

    line, words = None, None

f.close()
f = None

plt.figure()
plt.scatter(ex,ey)
plt.ylim([-0.15,0.15])
plt.grid(True)
plt.savefig("Primaryelectrontrack.png")

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(ex,ey,ez,s=10,c='red')
ax.scatter(ex[0],ey[0],ez[0],s=70,c='green')
ax.set_xlabel('x,mm')
ax.set_ylabel('y,mm')
ax.set_zlabel('z,mm')
#ax.invert_zaxis()
plt.tight_layout()
plt.savefig(f"Track-PrimaryElectrons-{picname}-xyz.png")
plt.close()

# ===========================================================

z_min, z_max = 0, 1.2
x_min, x_max = -0.7, 0.7
y_min, y_max = -0.7, 0.7

arr_ex = np.array(ex)
arr_ey = np.array(ey)
arr_ez = np.array(ez)

mask_inside = (arr_ex >= x_min) & (arr_ex <= x_max) & (arr_ey >= y_min) & (arr_ey <= y_max)

# walls
x1 = np.full((2, 2), -0.7)
x2 = np.full((2, 2), 0.7)
y1 = np.full((2, 2), -0.7)
y2 = np.full((2, 2), 0.7)
z = np.array([[z_min, z_min], [z_max, z_max]])
y_vals = np.array([[-0.7, 0.7], [-0.7, 0.7]])
x_vals = np.array([[-0.7, 0.7], [-0.7, 0.7]])

surfaces = [
    go.Surface(x=x1, y=y_vals, z=z, showscale=False, opacity=0.1, name='Wall -x'),
    go.Surface(x=x2, y=y_vals, z=z, showscale=False, opacity=0.1, name='Wall +x'),
    go.Surface(x=x_vals, y=y1, z=z, showscale=False, opacity=0.1, name='Wall -y'),
    go.Surface(x=x_vals, y=y2, z=z, showscale=False, opacity=0.1, name='Wall +y'),
]

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=ex, y=ey, z=ez,
    mode='markers',
    marker=dict(size=3, color='red'),
    name='Track'
))

fig.add_trace(go.Scatter3d(
    x=[ex[0]], y=[ey[0]], z=[ez[0]],
    mode='markers',
    marker=dict(size=2, color='green'),
    name='Start Point'
))

for surf in surfaces:
    fig.add_trace(surf)

fig.add_trace(go.Scatter3d(
    x=arr_ex[mask_inside],
    y=arr_ey[mask_inside],
    z=arr_ez[mask_inside],
    mode='markers',
    marker=dict(size=4, color='blue'),
    name='Inside sensitive area'
))

fig.update_layout(
    scene=dict(
        xaxis=dict(title='x, mm'), 
        #yaxis=dict(title='y, mm'),
        yaxis=dict(title='y, mm', range=[-4.0,4.0]),
        #zaxis=dict(title='z, mm') 
        zaxis=dict(title='z, mm', range=[z_min, z_max]) 
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

fig.write_html(f"ChargedParticleTrack-{picname}.html")

