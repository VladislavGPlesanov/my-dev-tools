import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import json

picname = sys.argv[1]

data_tot = {

    6801:4018.78,
    6802:36703.07,
    6803:14011.05,
    6804:78617.98,
    6805:84698.45,
    6806:6512.7,
    6807:143087.29,
    6808:226137.49,
    6809:252373.74,
    6810:224486.78,
    6811:231440.87,
    6813:168925.33,
    6814:106672.63,
    6815:370692.57,
    6816:80032.61,
    6817:423494.01,
    6818:431511.06,
    6819:528999.09,
    6820:654455.94,
    6830:57659.63,
    6831:76425.65,
    6832:55708.55
}

data_toa = {

    6834: 480680.64,
    6835: 35178.25,
    6836: 797358.16,
    6837: 757023.09
}

SCXdata = {
    # TOT runs
    "Run_006788": [5.6],
    "Run_006792": [5.6],
    "Run_006801": [0.0,5.3],
    "Run_006802": [5.3],
    "Run_006803": [5.3,5.6],
    "Run_006804": [5.6],
    "Run_006805": [5.6],
    "Run_006806": [5.6,5.8],
    "Run_006807": [5.8],
    "Run_006808": [5.8],
    "Run_006809": [5.8],
    "Run_006810": [5.8],
    "Run_006811": [5.8],
    "Run_006812": [5.8],
    "Run_006813": [5.8],
    "Run_006814": [5.8],
    "Run_006815": [5.8],
    "Run_006816": [5.8],
    "Run_006817": [5.8],
    "Run_006818": [5.8],
    "Run_006819": [5.8],
    "Run_006820": [5.8],
    "Run_006831": [3.3],
    "Run_006832": [3.3],
    "Run_006833": [3.6],
    # good TOA runs
    "Run_006834": [3.6],     # Vgrid = 350
    "Run_006835": [3.6,3.7], # Vgrid = 350
    "Run_006836": [3.7],     # Vgrid = 350
    "Run_006837": [3.6],     # Vgrid = 450
    # Parallel beam runs
    "Run_006838": [3.6,3.8],
    "Run_006839": [3.8],
    "Run_006840": [3.8,4,2],
    "Run_006841": [4.2],
    "Run_006842": [4.2],
    "Run_006843": [4.2],
    "Run_006844": [4.2]
}

 
#runs, collectedIons = [6800], [0.0]
runs, collectedIons = ["TOT runs"], [0.0]
extrapIons, avgTOT, blobs, frameTOT = [], [], [], []

counting_results = "RunData-JSON/runIonCounts.json"
fsr_params = "RunData-JSON/fsr_parameters.json"


iondata = None
with open(counting_results,"r") as cr:
    try:
        iondata = json.load(cr)
        #print(HUYA)
        #iondata = HUYA
    except Exception as ex:
        print(f"Failed to load ion counting data because: {ex}")

cdata = iondata["counts"]

# for TOT-based runs
integ_avgTOT, integ_extrap, integ_blobs, integ_frameTOT = [0.0], [0.0], [0.0], [0.0]
n_avgTOT, n_extrap, n_blobs, n_frameTOT = 0.0, 0.0, 0.0, 0.0
runs_comb = ["TOA runs"]
# for TOA-based runs
integ_toa_dose, integ_toa_part = [], []
n_nitro_dose, n_part_dose = 0.0, 0.0
toa_runs = ["TOA runs"]
comb_dose = 0.0
integ_combined = [0.0]

for run in cdata.keys():
    rundata = cdata[run]
    #print(rundata)
    if(len(rundata.keys())>0):
        print(rundata)
        runtype = rundata["runtype"]
        if(runtype=="TOT"):
            iextrap = rundata["extrap"]   
            iavgtot = rundata["avgTOT"]   
            iblobs = rundata["blobCTS"]   
            iframetot = rundata["frameTOT"]   

            n_extrap+=iextrap
            n_avgTOT+=iavgtot
            n_blobs+=iblobs
            n_frameTOT+=iframetot

            comb_dose+=iavgtot
            integ_combined.append(comb_dose)

            extrapIons.append(iextrap)
            avgTOT.append(iavgtot)
            blobs.append(iblobs)
            frameTOT.append(iframetot)

            runnr = run.split("_")[1]

            runs.append(runnr)
            runs_comb.append(runnr)

            integ_avgTOT.append(n_avgTOT)
            integ_extrap.append(n_extrap)
            integ_blobs.append(n_blobs)
            integ_frameTOT.append(n_frameTOT)

        if(runtype=="TOA"):

            ndose = rundata["total_dose"]
            npart = rundata["part_dose"]

            n_nitro_dose+=ndose
            n_part_dose+=npart

            comb_dose+=ndose

            integ_toa_dose.append(n_nitro_dose)
            integ_toa_part.append(n_part_dose)
            integ_combined.append(comb_dose)

            runnr = run.split("_")[1]
            
            toa_runs.append(runnr)
            runs_comb.append(runnr)


lastAvgTOT = integ_avgTOT[-1]
integ_toa_offset = [n+lastAvgTOT for n in integ_toa_dose]
integ_toa_dose_adj = [integ_avgTOT[-1]] + integ_toa_offset

#total = 0
#for k, v in zip(data_tot.keys(), data_tot.values()):
#
#    total+=v
#    collectedIons.append(total)
#    runs.append(k)
#    k,v = None, None
#
#
#for k,v in zip(data_toa.keys(), data_toa.values()):
#
#    total+=v
#    collectedIons.append(total)
#    runs.append(k)

#total_avgTOT = np.sum(avgTOT)
#total_blobs = np.sum(blobCTS)
#total_frameTOT = np.sum(frameTOT)
#total_extrap = np.sum(extrapIons)

print()

fig, ax = plt.subplots(figsize=(11,8))

#ax.scatter([],[],' ',label=r"$14\mathrm{N}^{5++}$ @ 100 MeV")
#ax.plot(runs,collectedIons,color='firebrick',marker='.')
ax.plot(runs,
        integ_avgTOT,
        color='firebrick',
        marker='.',
        #label=r"$\overline{(\mathrm{TOT})}$"+f"={n_avgTOT:.2e}")
        label="Total ToT"+f" = {n_avgTOT:.2e}")
ax.plot(runs,
        integ_frameTOT,
        color='darkorange',
        marker='.',
        #label=r"$\sum\mathrm{(TOT)}_{i\mathrm{frame}}$"+f"={n_frameTOT:.2e}")
        label=f"ToT per frame = {n_frameTOT:.2e}")
ax.plot(runs,
        integ_blobs,
        color='royalblue',
        marker='.',
        #label=f"N(blobs)={n_blobs:.2e}")
        label=f"watershed N(objects) = {n_blobs:.2e}")
ax.plot(toa_runs,
        #np.array(integ_toa_dose)+integ_avgTOT[-1],
        integ_toa_dose_adj,
        marker='.',
        color="forestgreen",
        label=f"Time clusters N(TOA) = {n_nitro_dose:.2e}")

#ax.plot(runs_comb,integ_combined,color="forestgreen",label=f"Avg(TOT)+N(TOA) = {comb_dose:.2e}")
ax.scatter([],[],
            color="white",
            #label=r"$\overline{(\mathrm{TOT})}$"+f"+N(TOA) = {comb_dose:.2e}")
            label=f"Total ToT + Total ToA = {comb_dose:.2e}")

ax.set_title("")
ax.set_xlabel("Runs")
textSize = 15
ax.set_ylabel("Integrated Ion Count, [N]", fontsize=textSize)
ax.tick_params(axis='x', labelrotation=45, labelsize=textSize)
ax.tick_params(axis='y', labelsize=textSize)
ax.grid(which='major', color='grey', linestyle='-', linewidth='0.5')
ax.grid(which='minor', color='gray', linestyle='--', linewidth='0.25')
ax.set_yscale('log')
ax.legend(loc='lower right', title=r"$14\mathrm{N}^{5++}$ @ 100 MeV", fontsize=textSize-2)
plt.savefig(f"IntegratedIonExposure-{picname}.png",dpi=256)
plt.close()

# tryna' plot rates vs scx position.... 
runtimes, ionrate, runscx = [], [], []
#plotruns = [6801,6802,6804,6807]
plotruns = [6802,6804,6807]
fsrdata = None
with open(fsr_params,'r') as fp:
    try:
        fsrdata = json.load(fp)
    except Exception as ex:
        print(f"Failedd to load {fsr_params}: {ex}")

for runnr in fsrdata.keys():
    thisrun = int(runnr.split("_")[1])
    print(f"{runnr} --> {thisrun}")
    if(thisrun in plotruns):
        runtime = fsrdata[runnr]['t_run']
        runtimes.append(float(runtime))
        runions = cdata[runnr]["avgTOT"]
        ionrate.append(runions/runtime)
        runscx.append(SCXdata[runnr][0] )
        print(f"AHA! {runtime}, {runions}, {runscx}")

print(runscx)
print(ionrate)


avgSepScx = np.sum(np.diff(runscx))/(len(runscx)-1)
avgSepRate = np.sum(np.diff(ionrate))/(len(ionrate)-1)

xmin = np.min(runscx)-avgSepScx*0.5
xmax = np.max(runscx)+avgSepScx*0.5

ymin = np.min(ionrate)-avgSepRate*0.5
ymax = np.max(ionrate)+avgSepRate*0.5

fig,ax = plt.subplots(figsize=(9,7))
ax.scatter(runscx, ionrate, color='forestgreen',marker='*',s=64)
ax.set_title("")
ax.set_xlabel("SCX position, [mm]")
ax.set_ylabel("Ion rate, [Hz]")
ax.set_xlim([xmin,xmax])
ax.set_ylim([ymin,ymax])
ax.grid(True)
plt.savefig(f"IonRate-vs-SCX-{picname}.png")
    
    
    
    
    
    
    
    
    
    
    
    
    
    


