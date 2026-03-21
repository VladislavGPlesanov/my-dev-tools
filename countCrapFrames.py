import numpy as np
import glob 
import sys
import os
import matplotlib.pyplot as plt

from MyPlotter import myUtils

base_directory = "/media/vlad/NitroBeam/from-tpc23/"

substring_filter = input("Find me runs with: ")
picname = input("add picname: ")

hit_threshold = 30000

all_dirs = os.listdir(base_directory)

run_dirs = [
    d for d in all_dirs
    if os.path.isdir(os.path.join(base_directory, d))
    and substring_filter in d
    and d.startswith("Run_")
]

run_dirs.sort()

print(f"Found {len(run_dirs)} matching runs.")

# ==========================
# STORAGE
# ==========================

run_numbers = []
large_hit_counts = []

all_hit_values = []  # for optional histogram

# ==========================
# LOOP OVER RUNS
# ==========================

frame_hits = []

MU = myUtils()

for run_dir in run_dirs:

    full_run_path = os.path.join(base_directory, run_dir)

    # Extract run number (Run_XXXXXX_...)
    try:
        run_number = int(run_dir.split("_")[1])
    except (IndexError, ValueError):
        print(f"Skipping malformed directory name: {run_dir}")
        continue

    print(f"Processing Run {run_number}")

    file_list = os.listdir(full_run_path)

    n_large = 0

    # ----------------------
    # LOOP OVER FILES
    # ----------------------

    tmp_hits = []

    nfiles = len(file_list)
    ifile = 0

    for fname in file_list:

        ifile+=1
        file_path = os.path.join(full_run_path, fname)

        if not os.path.isfile(file_path):
            continue

        try:
            matrix = np.loadtxt(file_path, dtype=int)
        except Exception:
            continue

        nhits = np.count_nonzero(matrix)

        all_hit_values.append(nhits)
        tmp_hits.append(nhits)

        if nhits > hit_threshold:
            n_large += 1
        MU.progress(nfiles,ifile)


    #print(len(tmp_hits))
    run_numbers.append(run_number)
    large_hit_counts.append(n_large/nfiles)
    frame_hits.append([f"{run_number}",tmp_hits])
    tmp_hits = None

# ==========================
# SCATTER PLOT
# ==========================

plt.figure(figsize=(10,8))
plt.scatter(run_numbers, large_hit_counts)

plt.xlabel("Run Number")
plt.set_xticks(np.arange(len(run_numbers)))
plt.set_xticklabels(run_numbers)

plt.ylabel(f"Number of frames with hits > {hit_threshold}")
plt.title("High-Hit Frames per Run")

plt.xticks(rotation=45)
plt.minorticks_on()
plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)
plt.tight_layout()
plt.savefig(f"BS-frames-SCAT-per-run-{picname}.png")

# ==========================
# OPTIONAL HISTOGRAM
# ==========================

plt.figure(figsize=(10,9))

for data in frame_hits:
    #print(data)
    counts, edges = np.histogram(data[1], bins=100, range=(0,66000))
    #print(len(counts))
    #print(counts)
    plt.hist(edges[:-1], weights=counts, bins=100, align='left', alpha=0.25,label=f"{data[0]}")
    
plt.title("Hits per Frame for All Runs")
plt.xlabel(r"$N_{hits}$")
plt.ylabel(r"$N_{frames}$")
plt.yscale('log')
plt.legend(loc='upper center')
plt.savefig(f"Hists-overlayed-{picname}.png")
 

