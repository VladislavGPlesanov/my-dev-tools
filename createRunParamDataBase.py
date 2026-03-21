import os
import json
import glob
import re
from datetime import datetime

BASE_DIR = "/media/vlad/NitroBeam/from-tpc23/RUNPARAMS"
OUTPUT_JSON = "fsr_parameters.json"


def parse_fsr_file(filepath):
    """Read *_fsr.txt file and return parameters dictionary."""
    params = {}

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            key, val = parts
            try:
                params[key] = int(val)
            except ValueError:
                continue

    return params


#def parse_run_times(runfile):
#    """Extract start/end time and run duration."""
#    t_start = None
#    t_end = None
#
#    with open(runfile, "r") as f:
#        for line in f:
#
#            if "Run start time" in line:
#                timestamp = line.split()[-1]      # 250824_13-24-00
#                t_start = timestamp.split("_")[1]
#
#            if "Run end time" in line:
#                timestamp = line.split()[-1]
#                t_end = timestamp.split("_")[1]
#
#    if t_start and t_end:
#
#        fmt = "%H-%M-%S"
#
#        t0 = datetime.strptime(t_start, fmt)
#        t1 = datetime.strptime(t_end, fmt)
#
#        duration = (t1 - t0).total_seconds()
#
#        if duration < 0:
#            duration += 24 * 3600   # handle midnight wrap
#
#    else:
#        duration = None
#
#    return t_start, t_end, duration

def parse_run_times(runfile):

    t_start = None
    t_end = None

    time_pattern = re.compile(r"\d{6}_(\d{2}-\d{2}-\d{2})")

    with open(runfile, "r") as f:
        for line in f:

            if "Run start time" in line:

                if "TBD" in line:
                    t_start = -1
                else:
                    m = time_pattern.search(line)
                    if m:
                        t_start = m.group(1)

            if "Run end time" in line:

                if "TBD" in line:
                    t_end = -1
                else:
                    m = time_pattern.search(line)
                    if m:
                        t_end = m.group(1)

    # default duration
    duration = -1

    # only compute duration if both times are valid
    if isinstance(t_start, str) and isinstance(t_end, str):

        fmt = "%H-%M-%S"

        try:
            t0 = datetime.strptime(t_start, fmt)
            t1 = datetime.strptime(t_end, fmt)

            duration = (t1 - t0).total_seconds()

            if duration < 0:
                duration += 24 * 3600

        except Exception:
            duration = -1

    return t_start if t_start is not None else -1, \
           t_end if t_end is not None else -1, \
           duration

# --------------------------------------------------
# Load existing JSON if present
# --------------------------------------------------

if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r") as f:
        database = json.load(f)
else:
    database = {}


# --------------------------------------------------
# Scan run directories
# --------------------------------------------------

for dirname in os.listdir(BASE_DIR):

    if not dirname.startswith("RunParams_"):
        continue

    run_dir = os.path.join(BASE_DIR, dirname)

    if not os.path.isdir(run_dir):
        continue

    parts = dirname.split("_")
    if len(parts) < 2:
        continue

    run_id = f"Run_{parts[1]}"

    fsr_files = glob.glob(os.path.join(run_dir, "*_fsr.txt"))
    runfile = os.path.join(run_dir, "run.txt")

    if not fsr_files:
        continue

    params = parse_fsr_file(fsr_files[0])

    if os.path.exists(runfile):
        t_start, t_end, t_run = parse_run_times(runfile)
    else:
        t_start, t_end, t_run = None, None, None

    params["t_start"] = t_start
    params["t_end"] = t_end
    params["t_run"] = t_run


    # --------------------------------------------------
    # Update JSON database
    # --------------------------------------------------

    if run_id not in database:

        database[run_id] = params
        print(f"Added new run {run_id}")

    else:

        updated = False

        for key, val in params.items():

            if key not in database[run_id]:
                database[run_id][key] = val
                updated = True

            elif database[run_id][key] != val:
                print(f"{run_id}: updating {key} ({database[run_id][key]} -> {val})")
                database[run_id][key] = val
                updated = True

        if updated:
            print(f"{run_id} updated")


# --------------------------------------------------
# Save JSON
# --------------------------------------------------

with open(OUTPUT_JSON, "w") as f:
    json.dump(database, f, indent=4)

print(f"\nDatabase saved to {OUTPUT_JSON}")
