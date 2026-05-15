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

                all_edf_files.append({
                    "subject": subject_folder,
                    "file_path": full_path
                })


print("\n=== EDF FILES FOUND ===\n")

for item in all_edf_files:

    print(f"Subject: {item['subject']}")
    print(f"File: {item['file_path']}\n")


print(f"Total EDF Files: {len(all_edf_files)}")


first_file = all_edf_files[0]["file_path"]

print("\n=== LOADING FIRST EDF FILE ===\n")

raw = mne.io.read_raw_edf(first_file, preload=True)


print(raw)

print("\n=== CHANNEL NAMES ===")
print(raw.ch_names)

print("\n=== SAMPLING RATE ===")
print(raw.info['sfreq'])

print("\n=== DATA SHAPE ===")
print(raw.get_data().shape)