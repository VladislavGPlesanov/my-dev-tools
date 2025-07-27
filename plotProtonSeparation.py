import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import numpy.ma as ma
import matplotlib.patches as patch
from matplotlib.patches import Ellipse

infile = sys.argv[1]
picname = sys.argv[2]

f = open(infile, 'r')

intLong, intShort, time = [],[],[]
LS_ratio = []

#p_region_minx = 

cnt=0
for line in f:

    if('timestamp' in line):
        continue

    words = line.split(',')

    nonempty = np.nonzero(np.array(words))

    idx_time = nonempty[0][0]
    idx_long = nonempty[0][1]
    idx_short = nonempty[0][2]

    t = float(words[idx_time])
    ilong = float(words[idx_long])
    ishort = float(words[idx_short])

    iratio = float(ilong - ishort)/float(ilong + ishort)

    if (iratio > 100):
        print("HUETA:")
        print(f't={t}, ilong={ilong}, ishort={ishort}')
        print("----------------------------------------")        

    #if(iratio>= 0 and iratio < 1 and ilong>0 and ilong < 60000 and ishort > 0):
    if(iratio>= 0 and iratio < 1 and ilong>0 and ilong < 60000):
        LS_ratio.append(iratio)
        intLong.append(ilong)
        time.append(t)

    if(cnt<10):
        print(line)
        print(f't={t}, ilong={ilong}, ishort={ishort}')

    cnt+=1


LSratio_cts, _ = np.histogram(np.array(LS_ratio), bins=100) 
intLong_cts, _ = np.histogram(np.array(intLong), bins=100) 

print(f"maximum in LS_ratio={max(LS_ratio)}")
print(f"maximum in intLong={max(intLong)}")
print(f"maximum in counts LS={max(LSratio_cts)}")
print(f"maximum in counts intLong={max(intLong_cts)}")
print(f"mean of LS_ratio = {np.mean(LS_ratio)}")

#wsignal, xedges, yedges = np.histogram2d(intLong, LS_ratio, bins=(100,100), range=[[0,60000],[0,1]])
wsignal, xedges, yedges = np.histogram2d(intLong, LS_ratio, bins=(100,100))

masked_cts = ma.masked_where(wsignal == 0, wsignal)

print(len(time))

time_total = time[len(time)-1]
#long_rate = np.array(intLong)/time_total
#discr_rate = np.array(LS_ratio)/time_total
wsignal_rate = np.array(masked_cts)/time_total
#  --------------------------------
print(masked_cts[0:5])
print(f'------------\ntotal_time={time_total}\n-------------')
print(wsignal_rate[0:5])


##pltTypes =['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
#
ptype = 'gist_earth_r'
#
#plt.figure()
#plt.hist2d(intLong, LS_ratio, bins=(100,100), cmap=ptype)
#plt.xlabel('long')
#plt.ylabel('(long-short)/long')
#plt.title(f'hueta')
#plt.colorbar(label='cnt')
#plt.xscale('log')
##plt.yscale('log')
##plt.ylim([5e-2,1])
#plt.savefig(f'protonSeparation-{picname}.png')
#
##############################################################################

fig = plt.figure(figsize=(8,8))
gs = GridSpec(2,3, width_ratios = [1,4,0.2], height_ratios=[4,1], hspace=0.05, wspace=0.05)

# adding square patch to emphasize proton peak

ax_main = fig.add_subplot(gs[0,1])
ax_xhist = fig.add_subplot(gs[1,1], sharex=ax_main)
ax_yhist = fig.add_subplot(gs[0,0], sharey=ax_main)
ax_cbar = fig.add_subplot(gs[0,2])

#h2d = ax_main.hist2d(intLong, LS_ratio, bins=(100,100), cmap=ptype, norm=LogNorm)
#mesh = ax_main.pcolormesh(xedges, yedges, masked_cts.T, cmap=ptype, norm=LogNorm())
mesh = ax_main.pcolormesh(xedges, yedges, wsignal_rate.T, cmap=ptype, norm=LogNorm())

protreg = Ellipse(
    xy=(np.mean(intLong), np.mean(LS_ratio)), 
    width=np.std(intLong)*3, 
    height=np.std(LS_ratio)*4, 
    edgecolor='red', 
    facecolor='none', 
    linewidth=2)
ax_main.add_patch(protreg)

print("ALALAA:")
print(np.mean(xedges))
print(np.mean(yedges))

ax_main.tick_params(labelbottom=False)
ax_main.tick_params(labelleft=False)
ax_main.set_title('Signal shape discriminant value vs. Collected QDC charge')

#cbar = plt.colorbar(h2d[3], cax=ax_cbar)
cbar = plt.colorbar(mesh, cax=ax_cbar)
cbar.set_label('Rate, [Hz]')

ax_xhist.hist(intLong, bins=100, color='blue', histtype='step')
ax_xhist.vlines(np.mean(intLong),0,max(intLong))
#ax_xhist.hist(long_rate, bins=100, color='blue', histtype='step')
ax_xhist.set_xlabel('Total charge (long), [QDC cts]')
ax_xhist.tick_params(axis='x', labelsize=8)
ax_xhist.tick_params(axis='y', labelleft=False)
#ax_xhist.set_xscale('log')

ax_yhist.hist(LS_ratio, bins=100, orientation='horizontal', color='blue', histtype='step')
ax_yhist.hlines(np.mean(LS_ratio),0,max(LS_ratio))
#ax_yhist.hist(discr_rate, bins=100, orientation='horizontal', color='blue', histtype='step')
ax_yhist.set_ylabel('Discriminant, (long-short)/long')
ax_yhist.tick_params(axis='x', labelbottom=False)
ax_yhist.tick_params(axis='y', labelsize=8)
#ax_yhist.set_yscale('log')

#ax_main.set_xlim([0,50000])
#ax_main.set_ylim([0,0.5])
plt.tight_layout()
plt.savefig(f'protonSeparation-Combined-{picname}.png')


stdx = np.std(intLong)
xshift = np.mean(intLong)
stdy = np.std(LS_ratio)
yshift= np.mean(LS_ratio)
n_prot = 0
for x,y in zip(intLong, LS_ratio):
    if(((x**2-xshift)/(stdx*3)**2)+ ((y**2-yshift)/(stdy*3)**2) < 1):
        n_prot+=1

print(f"TOTAL PROTONS SEEN: {n_prot}")
print(f"TOTAL RATE WITHIN REG-Of_INTEREST = {round(n_prot/time_total,2)} [Hz]")



#plt.figure()
#plt.hist(LSratio_cts, bins=100)
##plt.xlim([0,1])
#plt.savefig("PIZDEC-y.png")
#
#plt.figure()
#plt.hist(intLong_cts, bins=100)
##plt.xlim([0,1])
#plt.savefig("PIZDEC-x.png")
