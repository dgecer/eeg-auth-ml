import os
import mne
import numpy as np
import pandas as pd

from scipy.signal import welch


DATA_PATH = "data/raw/physionet_eeg"

all_features = []

WINDOW_SIZE_SECONDS = 5


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

                channel_names = raw.ch_names


                window_size_samples = int(WINDOW_SIZE_SECONDS * sf)

                total_samples = data.shape[1]


                for start in range(0, total_samples - window_size_samples, window_size_samples):

                    end = start + window_size_samples


                    feature_dict = {
                        "subject": subject_folder
                    }


                    for channel_index, channel_name in enumerate(channel_names):

                        channel_data = data[channel_index][start:end]


                        delta = bandpower(channel_data, sf, (0.5, 4))
                        theta = bandpower(channel_data, sf, (4, 8))
                        alpha = bandpower(channel_data, sf, (8, 13))
                        beta = bandpower(channel_data, sf, (13, 30))


                        feature_dict[f"{channel_name}_delta"] = delta
                        feature_dict[f"{channel_name}_theta"] = theta
                        feature_dict[f"{channel_name}_alpha"] = alpha
                        feature_dict[f"{channel_name}_beta"] = beta


                    all_features.append(feature_dict)


df = pd.DataFrame(all_features)


print("\n=== EXTRACTED EEG FEATURES ===\n")

print(df.head())

print(f"\nTotal Samples Created: {len(df)}")


df.to_csv("data/processed/features.csv", index=False)


print("\nFeatures saved to data/processed/features.csv")