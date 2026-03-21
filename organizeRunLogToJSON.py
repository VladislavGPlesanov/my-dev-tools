import os
import json
import numpy as np

#DATA_DIR = "/path/to/bak/files"
#OUTPUT_JSON = "p09_runs.json"
DATA_DIR = "/home/vlad/Documents/P09-aux-files"              # directory with .fio files
OUTPUT_JSON = "P09Scan_database.json"


def load_existing_json():
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            return json.load(f)
    return {}


def save_json(data):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=4)


def parse_bak_file(filepath):

    params = {}
    col_index = {}

    reading_params = False
    reading_cols = False
    reading_data = False

    data_rows = []

    with open(filepath) as f:
        for line in f:

            line = line.strip()

            if line == "%p":
                reading_params = True
                continue

            if line == "%d":
                reading_params = False
                reading_cols = True
                continue

            if reading_params:

                if "=" in line:
                    key, val = line.split("=")
                    params[key.strip()] = float(val)

            elif reading_cols:

                if line.startswith("Col"):

                    parts = line.split()
                    idx = int(parts[1]) - 1
                    name = parts[2]

                    col_index[name] = idx

                else:
                    reading_cols = False
                    reading_data = True

            if reading_data:

                if not line or line.startswith("!"):
                    continue

                try:
                    row = [float(x) for x in line.split()]
                    data_rows.append(row)
                except:
                    continue

    if not data_rows:
        return None

    data = np.array(data_rows)

    # force 2D array (fix for single-row case)
    data = np.atleast_2d(data)

    result = {}

    result["del"] = params.get("del")
    result["pth"] = params.get("pth")
    result["gap"] = params.get("gap")

    if "abs_position" in col_index:
        result["abs_position"] = float(np.mean(data[:, col_index["abs_position"]]))

    if "abs_attenfactor" in col_index:
        result["abs_attenfactor"] = float(np.mean(data[:, col_index["abs_attenfactor"]]))

    if "timestamp" in col_index:
        timestamps = data[:, col_index["timestamp"]]
        result["t_run"] = float(timestamps[-1] - timestamps[0])

    return result


def main():

    database = load_existing_json()

    for file in os.listdir(DATA_DIR):

        if not file.endswith(".fio"):
            continue

        run_id = os.path.splitext(file)[0]

        filepath = os.path.join(DATA_DIR, file)

        parsed = parse_bak_file(filepath)

        if parsed is None:
            continue

        if run_id in database:

            changed = False

            for k, v in parsed.items():
                if database[run_id].get(k) != v:
                    database[run_id][k] = v
                    changed = True

            if changed:
                print(f"{run_id} updated")

        else:

            database[run_id] = parsed
            print(f"{run_id} added")

    save_json(database)


if __name__ == "__main__":
    main()
