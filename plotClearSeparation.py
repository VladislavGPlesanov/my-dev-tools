import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import numpy.ma as ma


def readFile(file, list_long, list_disc, list_time):

    for line in file:
    
        if('timestamp' in line):
            continue

        words = line.split(',')
    
        nonempty = np.nonzero(np.array(words))
    
        idx_time = nonempty[0][0]
        idx_short = nonempty[0][1]
        idx_long = nonempty[0][2]

        t = float(words[idx_time])
        short = float(words[idx_short])
        long = float(words[idx_long])

        discr = (long-short)/(long+short)

        if(discr>0):

           list_disc.append(discr)
           list_long.append(long)

    print(f'From file \"{file}\" recorded data lists\n => discriminant = {len(list_disc)} and long={len(list_long)}')

#######################################################################

base_path = '/home/vlad/IonTestBeams/ParticleSeparation/raw_data/'
signal_file = 'data_250514_133708.csv' # firist try file
#bgr_file = 'data_250514_154419.csv' # using activatioon bgr for now
bgr_file = 'data_250514_134244.csv' # dumping beam into Al plate

sfile = open(base_path+signal_file,'r')

sig_discriminant, sig_long, sig_time = [], [], []
bgr_discriminant, bgr_long, bgr_time = [], [], []

readFile(sfile, sig_discriminant, sig_long, )

bfile = open(base_path+bgr_file,'r')

readFile(bfile, bgr_discriminant, bgr_long)

signal_hist, xe_sig, ye_sig = np.histogram2d(sig_discriminant, sig_long, bins=(100,100))

bgr_hist, xe_bgr, ye_bgr = np.histogram2d(bgr_discriminant, bgr_long, bins=(100,100))

# ghetting rates

sig_tmax = max(sig_time)
bgr_tmax = max(bgr_time)

signal_rate = signal_hist/sig_tmax
brg_rate = bgr_hist/bgr_tmax

###############################################################
maptype = 'gist_earth_r'

plt.figure()
plt.pcolormesh(xe_sig, ye_sig, signal_rate.T, cmap=maptype)
plt.savefig("signal.png")

plt.figure()
plt.pcolormesh(xe_bgr, ye_bgr, bgr_rate.T, cmap=maptype)
plt.savefig("bgr.png")

SN_ratio = signal_rate/bgr_rate

plt.figure()
plt.pcolormesh(xe_sig,ye_sig,SN_ratio.T, cmap=maptype)
plt.savefig("diff.png")





