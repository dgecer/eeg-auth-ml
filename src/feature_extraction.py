import os
import mne
import numpy as np
import pandas as pd

DATA_PATH = "data/raw/physionet_eeg"

all_features = []

for subject_folder in os.listdir(DATA_PATH):

    subject_path = os.path.join(DATA_PATH, subject_folder)

    if os.path.isdir(subject_path):

        for file in os.listdir(subject_path):

            if file.endswith(".edf"):

                file_path = os.path.join(subject_path, file)

                print(f"\nProcessing: {file_path}")

                raw = mne.io.read_raw_edf(file_path, preload=True)

                data = raw.get_data()

                mean_value = np.mean(data)
                std_value = np.std(data)
                max_value = np.max(data)
                min_value = np.min(data)

                all_features.append({
                    "subject": subject_folder,
                    "mean": mean_value,
                    "std": std_value,
                    "max": max_value,
                    "min": min_value
                })


df = pd.DataFrame(all_features)

print("\n=== EXTRACTED FEATURES ===\n")
print(df.head())

df.to_csv("data/processed/features.csv", index=False)

print("\nFeatures saved to data/processed/features.csv")