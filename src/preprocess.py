import os
import mne

DATA_PATH = "data/raw/physionet_eeg"

all_edf_files = []

for subject_folder in os.listdir(DATA_PATH):
    subject_path = os.path.join(DATA_PATH, subject_folder)
    if os.path.isdir(subject_path):
        for file in os.listdir(subject_path):
            if file.endswith(".edf"):
                full_path = os.path.join(subject_path, file)
                all_edf_files.append(full_path)

print("Found EDF files:\n")

for edf_file in all_edf_files:
    print(edf_file)
    
print(f"\nTotal number of EDF files found: {len(all_edf_files)}")