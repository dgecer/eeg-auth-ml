import os
import mne
import numpy as np
import pandas as pd

from scipy.signal import welch

DATA_PATH = "data/raw/physionet_eeg"

all_features = []


def bandpower(data, sf, band):

    low, high = band

    freqs, psd = welch(data, sf)

    freq_resolution = freqs[1] - freqs[0]

    idx_band = np.logical_and(freqs >= low, freqs <= high)

    power = np.sum(psd[idx_band]) * freq_resolution

    return power


for subject_folder in os.listdir(DATA_PATH):

    subject_path = os.path.join(DATA_PATH, subject_folder)

    if os.path.isdir(subject_path):

        for file in os.listdir(subject_path):

            if file.endswith(".edf"):

                file_path = os.path.join(subject_path, file)

                print(f"\nProcessing: {file_path}")

                raw = mne.io.read_raw_edf(file_path, preload=True)

                data = raw.get_data()

                sf = raw.info['sfreq']

                flattened_data = data.flatten()


                delta = bandpower(flattened_data, sf, (0.5, 4))
                theta = bandpower(flattened_data, sf, (4, 8))
                alpha = bandpower(flattened_data, sf, (8, 13))
                beta = bandpower(flattened_data, sf, (13, 30))


                all_features.append({
                    "subject": subject_folder,
                    "delta": delta,
                    "theta": theta,
                    "alpha": alpha,
                    "beta": beta
                })


df = pd.DataFrame(all_features)

print("\n=== EXTRACTED EEG BAND FEATURES ===\n")
print(df.head())


df.to_csv("data/processed/features.csv", index=False)

print("\nFeatures saved to data/processed/features.csv")