import os
import pickle
import csv

from mediapipe.framework.formats.landmark_pb2 import Landmark

# === CONFIG ===
FOLDER = 'entireid/query_pose'  # path to pose .pkl files
LOG_FILE = 'malformed_files.log'
LABEL_CSV = 'pose_labels.csv'  # optional CSV file with: filename,label
APPLY_LABELS = True  # set to False if you're okay with dummy -1 labels

# === Load labels from CSV if available ===
label_map = {}
if APPLY_LABELS and os.path.exists(LABEL_CSV):
    with open(LABEL_CSV, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                label_map[row[0]] = int(row[1])
    print(f"[INFO] Loaded {len(label_map)} labels from {LABEL_CSV}")

# === Process and fix pose files ===
malformed = []

for filename in os.listdir(FOLDER):
    if not filename.endswith('.pkl'):
        continue

    file_path = os.path.join(FOLDER, filename)

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Valid format: (pose_list, label)
        if isinstance(data, tuple) and len(data) == 2:
            pose, label = data
            if isinstance(pose, list) and hasattr(pose[0], 'x'):
                continue  # File is good, skip fixing

        # Fix: Handle nested [[Landmark, Landmark...]]
        elif isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
            pose = data[0]
            if hasattr(pose[0], 'x'):
                label = label_map.get(filename, -1)
                with open(file_path, 'wb') as f:
                    pickle.dump((pose, label), f)
                print(f"[FIXED] {filename} - nested list")

            else:
                raise ValueError("Pose format invalid inside nested list.")

        # Fix: Handle flat list of Landmarks (but no label)
        elif isinstance(data, list) and hasattr(data[0], 'x'):
            pose = data
            label = label_map.get(filename, -1)
            with open(file_path, 'wb') as f:
                pickle.dump((pose, label), f)
            print(f"[FIXED] {filename} - flat list")

        else:
            raise ValueError("Unknown data format")

    except Exception as e:
        malformed.append((filename, str(e)))
        print(f"[ERROR] {filename}: {e}")

# === Log malformed files ===
if malformed:
    with open(LOG_FILE, 'w') as f:
        for filename, error in malformed:
            f.write(f"{filename}: {error}\n")
    print(f"\n[LOGGED] {len(malformed)} malformed files to {LOG_FILE}")
else:
    print("\n✅ All files valid or successfully fixed.")
