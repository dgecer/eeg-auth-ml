import os
import json
import mne
import numpy as np
import pandas as pd
import joblib

from scipy.signal import welch
from collections import Counter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console()

mne.set_log_level("ERROR")


MODEL_PATH = "models/svm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
PCA_PATH = "models/pca.pkl"

PROFILE_PATH = "config/user_profiles.json"

WINDOW_SIZE_SECONDS = 5


def bandpower(data, sf, band):

    low, high = band

    freqs, psd = welch(data, sf)

    freq_resolution = freqs[1] - freqs[0]

    idx_band = np.logical_and(freqs >= low, freqs <= high)

    power = np.sum(psd[idx_band]) * freq_resolution

    return power


os.system("cls")


title = Text(
    "EEG BIOMETRIC AUTHENTICATION SYSTEM",
    style="bold bright_magenta"
)

console.print(
    Panel(
        title,
        subtitle="Neural Identity Recognition",
        border_style="magenta",
        expand=False
    )
)


subject_path = console.input(
    "\n[bold pink1]Enter subject folder path:[/bold pink1] "
)


console.print(
    "\n[bold orchid]Loading trained model...[/bold orchid]\n"
)


model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

pca = joblib.load(PCA_PATH)


with open(PROFILE_PATH, "r") as file:

    profiles = json.load(file)


all_feature_rows = []


for file in os.listdir(subject_path):

    if file.endswith(".edf"):

        file_path = os.path.join(subject_path, file)

        console.print(
            f"[hot_pink]Processing EEG recording:[/hot_pink] {file}"
        )

        raw = mne.io.read_raw_edf(file_path, preload=True)

        data = raw.get_data()

        sf = raw.info["sfreq"]

        channel_names = raw.ch_names


        window_size_samples = int(WINDOW_SIZE_SECONDS * sf)

        total_samples = data.shape[1]


        for start in range(
            0,
            total_samples - window_size_samples,
            window_size_samples
        ):

            end = start + window_size_samples


            feature_dict = {}


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


            all_feature_rows.append(feature_dict)


console.print(
    "\n[bold plum1]Extracting neural signature features...[/bold plum1]\n"
)


feature_df = pd.DataFrame(all_feature_rows)


features_scaled = scaler.transform(feature_df)

features_pca = pca.transform(features_scaled)


console.print(
    "[bold medium_purple]Running biometric identification...[/bold medium_purple]\n"
)


predictions = model.predict(features_pca)

probabilities = model.predict_proba(features_pca)


prediction_counts = Counter(predictions)

final_prediction = prediction_counts.most_common(1)[0][0]


average_probabilities = np.mean(probabilities, axis=0)

confidence_scores = dict(
    zip(model.classes_, average_probabilities)
)


profile = profiles.get(final_prediction)


console.print(
    Panel(
        f"[bold pink1]Authenticated User:[/bold pink1] {profile['name']}\n\n"
        f"[bold orchid]Predicted Subject:[/bold orchid] {final_prediction}",
        title="IDENTITY VERIFIED",
        border_style="bright_magenta"
    )
)


confidence_table = Table(
    title="Neural Confidence Scores",
    border_style="magenta"
)

confidence_table.add_column(
    "Subject",
    style="hot_pink"
)

confidence_table.add_column(
    "Confidence",
    style="plum1"
)


sorted_scores = sorted(
    confidence_scores.items(),
    key=lambda x: x[1],
    reverse=True
)


for subject, score in sorted_scores:

    confidence_table.add_row(
        subject,
        f"{score * 100:.2f}%"
    )


console.print(confidence_table)


while True:

    menu = Table(
        title="Neural Profile Access System",
        border_style="bright_magenta"
    )

    menu.add_column("Option", style="hot_pink")
    menu.add_column("Information", style="plum1")

    menu.add_row("1", "Address")
    menu.add_row("2", "Phone")
    menu.add_row("3", "Education")
    menu.add_row("4", "Social Media")
    menu.add_row("5", "Email")
    menu.add_row("6", "Physical Features")
    menu.add_row("7", "Exit")


    console.print(menu)


    choice = console.input(
        "\n[bold pink1]Enter your choice:[/bold pink1] "
    )


    if choice == "1":

        console.print(
            f"\n[hot_pink]Address:[/hot_pink] {profile['address']}"
        )

    elif choice == "2":

        console.print(
            f"\n[hot_pink]Phone:[/hot_pink] {profile['phone']}"
        )

    elif choice == "3":

        console.print(
            f"\n[hot_pink]Education:[/hot_pink] {profile['education']}"
        )

    elif choice == "4":

        console.print(
            f"\n[hot_pink]Social Media:[/hot_pink] {profile['social_media']}"
        )

    elif choice == "5":

        console.print(
            f"\n[hot_pink]Email:[/hot_pink] {profile['email']}"
        )

    elif choice == "6":

        console.print(
            f"\n[hot_pink]Physical Features:[/hot_pink] {profile['physical']}"
        )

    elif choice == "7":

        console.print(
            "\n[bold orchid]Exiting neural profile system...[/bold orchid]\n"
        )

        break

    else:

        console.print(
            "\n[bold red]Invalid choice.[/bold red]"
        )