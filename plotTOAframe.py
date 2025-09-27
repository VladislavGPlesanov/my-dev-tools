import sys
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def getFrameTime(SR, ST):
    return 256**SR * 46 * ST / 4e7


file = sys.argv[1]
picname = sys.argv[2]

frame = np.zeros((256,256),dtype=int)
perp_plane = np.zeros((256,256),dtype=int)

pure_file = file.split("/")[len(file.split("/"))-1]

framenum = pure_file.split("_")[3]

f = open(file, 'r')

pixpitch = 0.0055 #cm
# =================
maxTOA = 11810
sr = 0
st = 192
t_max = 295.25 # us
#t_frame = getFrameTime(sr, st)
#toa_adc_unit = t_frame/maxTOA
toa_adc_unit = maxTOA/t_max # adc/us
#eVelo = 0.545427 #cm/us ArCO2 80:20 @ 760, 25C, 504 Vcm # correct number 

# =================
# using incorrect number for now!
eVelo = 0.0545427 
# there's factor of 10 missing somewhere...

# =================

#print(f"t_frame={t_frame}")
print(f"toa_adc_unit={toa_adc_unit}")
print()

xlist, ylist, zlist = [], [], []
xcm, ycm = [], []

cnt=0
for line in f:

    if("FEC" in line or "Board" in line or "Chip" in line):
        continue

    #words = line.split('\t')
    words = line.split(' ')
    x = int(words[0])
    y = int(words[1])
    toa = int(words[2])
    z = eVelo*toa/toa_adc_unit # cm/us * adc * adc/us

    if(cnt%100==0):
        print(f"z={z}")
    
    xlist.append(x)
    ylist.append(y)
    zlist.append(z)

    np.add.at(frame, (x,y), toa)
    #np.add.at(perp_plane, (x,y), z)

    xcm.append(x*pixpitch)
    ycm.append(y*pixpitch)

    cnt+=1

f.close()

print("Checkin coords:")
for i in range(20):
    print(f"x={xlist[i]} y={ylist[i]} z={zlist[i]}")
    print(f"x[cm]={xlist[i]*0.0055}, y[cm]={ylist[i]*0.0055}")



# the plot

fig,ax = plt.subplots()
cax = fig.add_axes([0.86, 0.1, 0.05, 0.8])
ms = ax.matshow(frame)
fig.colorbar(ms, cax=cax, orientation='vertical')
ax.set_title(f"TOA frame {framenum}")
plt.plot()
#fig.savefig(f"TOA-frame-{framenum}-{picname}.png")
fig.savefig(f"TOA-frame-{picname}.png")

#fig,ax = plt.subplots()
#cax = fig.add_axes([0.86, 0.1, 0.05, 0.8])
#ms = ax.matshow(perp_plane)
#fig.colorbar(ms, cax=cax, orientation='vertical')
#ax.set_title(f"Perpendicular plane {framenum}")
#plt.plot()
#fig.savefig(f"TOA-frame-perpendicular-{framenum}-{picname}.png")

xarr = np.asarray(xcm)
yarr = np.asarray(ycm)
zarr = np.asarray(zlist)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')
#ax.scatter(xlist,ylist,zlist,s=10,c='red')
ax.scatter(xarr,yarr,zlist,s=10,c='red')

ax.set_xlabel('x,cm')
ax.set_ylabel('y,cm')
ax.set_zlabel('z,cm')
#
ax.set_zlim([0,3])
ax.set_xlim([0,1.42])
ax.set_ylim([0,1.42])
#ax.invert_zaxis()
plt.tight_layout()
#plt.savefig(f"TOA-frame-{framenum}-Track-{picname}-xyz.png")
plt.savefig(f"TOA-3D-frame-Track-{picname}-xyz.png")
plt.close()
fig = None

# using plotly
#

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=xarr, y=yarr, z=zarr,
    mode='markers',
    marker=dict(size=3, color='red'),
    name='Track'
))


fig.update_layout(
    scene=dict(
        xaxis=dict(title='x, cm', range=[0,1.42]),
        yaxis=dict(title='y, cm', range=[0,1.42]),
#        zaxis=dict(title='z, cm', range=[0,3])
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

#fig.write_html(f"TOA-frame-{framenum}-{picname}.html")
fig.write_html(f"TOA-frame-{picname}.html")

