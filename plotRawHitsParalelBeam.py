import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def getFrameTime(SR,ST):

    return 256**SR * 46 * ST / 4e7

def fillList(path):

    avg_hits_per_proton = 1403.5 # average hits per single protons track in one frame

    nprotons = 0
    dlist = []

    flist = glob.glob(path+'/*.txt')
    flist=sorted(flist)

    nframes=0
    for file in flist:
        if(nframes%100):
            print(f"Opening File: {file}")
        #f = open(file,'r')
        nlines = 0
        ###############################################################
        #for line in f:
        #    if("Hits" in line):
        #        nhits = int(line.split(":")[1])
        #        if(nhits >= (avg_hits_per_proton/2.0)):
        #            nprotons += np.ceil(float(nhits)/avg_hits_per_proton)
        #        dlist.append(nhits)
        #    else:
        #        continue
        ###############################################################
        data = []
        with open(file, 'r') as f:
            data = f.readlines()[2]
        f.close()
        nhits = int(data.split(":")[1])
        if(nhits >= avg_hits_per_proton/2.0):
            #nprotons += np.ceil(float(nhits)/avg_hits_per_proton)
            nprotons += np.round(float(nhits)/avg_hits_per_proton,2)
        dlist.append(nhits)
        nframes+=1

    flist=None

    return dlist, nprotons, nframes

###############################################################

picname = sys.argv[1]

f_6666 = "/media/vlad/Paralel_beam_data/Run_006666_250723_10-42-55/"
f_6671 = "/media/vlad/Paralel_beam_data/Run_006671_250723_11-20-26/"
f_6672 = "/media/vlad/Paralel_beam_data/Run_006672_250723_11-25-55/"
f_6674 = "/media/vlad/Paralel_beam_data/Run_006674_250723_11-36-01/"
f_6676 = "/media/vlad/Paralel_beam_data/Run_006676_250723_11-45-27/"
f_6679 = "/media/vlad/Paralel_beam_data/Run_006679_250723_11-56-25/"

# run for SCX1 = 10mm with 0.2mm diaphragm
f_6696 = "/media/vlad/Paralel_beam_data/Run_006696_250723_13-06-44/"
f_6690 = "/media/vlad/Paralel_beam_data/Run_006690_250723_12-38-28/"

hits_6666, Nprot_006666, nframes_6666 = fillList(f_6666)
hits_6671, Nprot_006671, nframes_6671 = fillList(f_6671) 
hits_6672, Nprot_006672, nframes_6672 = fillList(f_6672)
hits_6674, Nprot_006674, nframes_6674 = fillList(f_6674)
hits_6676, Nprot_006676, nframes_6676 = fillList(f_6676)
hits_6679, Nprot_006679, nframes_6679 = fillList(f_6679)

_ , Nprot_006696, nframes_6696 = fillList(f_6696)
_ , Nprot_006690, nframes_6690 = fillList(f_6690)

#=== calculating rates based on raw hit txt data ===
frame_list = [nframes_6666, nframes_6671, nframes_6672, nframes_6674, nframes_6676, nframes_6679, nframes_6696, nframes_6690]
proton_list = [Nprot_006666, Nprot_006671, Nprot_006672, Nprot_006674, Nprot_006676, Nprot_006679, Nprot_006696, Nprot_006690]
SCX_pos = [0.5, 2.5, 5, 7.5, 9.5, 11.5, 10, 10]
ST_list = [10, 5, 1, 1, 1, 1, 250, 127]
SR_list = [1, 1, 1, 1, 1, 1, 0, 0]

t_dead = 2.5e-2

rate_list = []
total_proton_list = []
run_times = []

for st, sr, Np, nf in zip(ST_list, SR_list, proton_list, frame_list):

    t_frame = getFrameTime(sr,st)
    t_meas = (t_frame+t_dead)*nf
    #total_prot = (t_frame+t_dead)/t_frame*Np
    total_prot = t_dead/t_frame*Np
    rate = total_prot/t_meas
    run_times.append(t_meas)
    total_proton_list.append(np.floor(total_prot))
    rate_list.append(rate)

#=== ploting stuff for reconstruction based frame analysis ===

reco_Rate = [753.76, 1561.85, 5363.10, 6848.79, 8030.50, 7903.16]

# ============================================================

nbins = 100
minhits = 0
maxhits = 12000

counts_6666, edges = np.histogram(hits_6666, bins=nbins, range=(minhits,maxhits))
counts_6671, _ = np.histogram(hits_6671, bins=nbins, range=(minhits,maxhits))
counts_6672, _ = np.histogram(hits_6672, bins=nbins, range=(minhits,maxhits))
counts_6674, _ = np.histogram(hits_6674, bins=nbins, range=(minhits,maxhits))
counts_6676, _ = np.histogram(hits_6676, bins=nbins, range=(minhits,maxhits))
counts_6679, _ = np.histogram(hits_6679, bins=nbins, range=(minhits,maxhits))

bin_centers = (edges[:-1]+edges[1:])/2.0
bin_width = edges[1]-edges[0]

print(type(counts_6666))
print(len(counts_6666))
print(counts_6666[0:100])
print(max(counts_6666))

normcts_6666 = counts_6666/counts_6666.sum()
normcts_6671 = counts_6671/counts_6671.sum()
normcts_6672 = counts_6672/counts_6672.sum()
normcts_6674 = counts_6674/counts_6674.sum()
normcts_6676 = counts_6676/counts_6676.sum()
normcts_6679 = counts_6679/counts_6679.sum()

plt.figure(figsize=(8,6))
plt.hist(bin_centers[5:], weights=normcts_6666[5:], bins=nbins, range=(minhits,maxhits), alpha=0.5, histtype='stepfilled', fc='blue', label="run 6666")
plt.hist(bin_centers[5:], weights=normcts_6671[5:], bins=nbins, range=(minhits,maxhits), alpha=0.5, histtype='stepfilled', fc='limegreen', label="run 6671")
plt.hist(bin_centers[5:], weights=normcts_6672[5:], bins=nbins, range=(minhits,maxhits), alpha=0.5, histtype='stepfilled', fc='orangered', label="run 6672")
plt.title("Hits per Frame (Raw Data)")
plt.xlabel(r"$N_{\mathrm{Hits}}$, [$\mathrm{(frame)}^{-1}$]")
plt.ylabel(r"$N_{\mathrm{Frames}}$")
plt.grid(True)
plt.legend()
plt.savefig(f"Combined-RawHits-perFrame-Part1-{picname}.png")
plt.close()

plt.figure(figsize=(8,6))
plt.hist(bin_centers[5:], weights=normcts_6674[5:], bins=nbins, range=(minhits,maxhits), alpha=0.5, histtype='stepfilled', fc='blue', label="run 6674")
plt.hist(bin_centers[5:], weights=normcts_6676[5:], bins=nbins, range=(minhits,maxhits), alpha=0.5, histtype='stepfilled', fc='limegreen', label="run 6676")
plt.hist(bin_centers[5:], weights=normcts_6679[5:], bins=nbins, range=(minhits,maxhits), alpha=0.5, histtype='stepfilled', fc='orangered', label="run 6679")
plt.title("Hits per Frame (Raw Data)")
plt.xlabel(r"$N_{\mathrm{Hits}}$, [$\mathrm{(frame)}^{-1}$]")
plt.ylabel(r"$N_{\mathrm{Frames}}$")
plt.grid(True)
plt.legend()
plt.savefig(f"Combined-RawHits-perFrame-Part2-{picname}.png")
plt.close()

default_scx_pos = [0.5, 2.5, 5, 7.5, 9.5, 11.5]

plt.figure(figsize=(8,6))
plt.scatter(default_scx_pos, reco_Rate, marker='+', color='g', label="Based on reco. data")
plt.scatter(default_scx_pos, rate_list[0:6], marker='*', color='b', label="Based on Raw hits")
plt.scatter(SCX_pos[6], rate_list[6], marker='^', color='r', label=r" Raw hits for 200 $\mu$m")
plt.scatter(SCX_pos[7], rate_list[7], marker='v', color='g', label=r" Raw hits for 500 $\mu$m")
plt.title("Proton Rate vs SCX1 Position")
plt.xlabel("SCX1 position, [mm]")
plt.ylabel("Rate, [Hz]")
plt.yscale('log')
plt.xlim([0,14])
plt.ylim([1e2,1e5])
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
plt.legend(loc='upper left')
plt.savefig(f"RawData-calcRates-{picname}.png")
plt.savefig(f"RawData-calcRates-{picname}.pdf")
plt.close()

print(rate_list[0:6])
print("Integrated proton counts over all accessible data sets:")
print(sum(total_proton_list))
print("Total runtime:")
print("{} [min]".format(np.floor(sum(run_times)/60.0)))
