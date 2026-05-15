from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import os


console = Console()


title = Text(
    "EEG BIOMETRIC AUTHENTICATION SYSTEM",
    style="bold bright_magenta"
)

console.print(
    Panel(
        title,
        subtitle="Neural Identity Recognition",
        expand=False,
        border_style="magenta"
    )
)


console.print(
    "\n[bold hot_pink]STEP 1[/bold hot_pink] -> PREPROCESSING EEG DATA\n"
)

os.system("python src/preprocess.py")


console.print(
    "\n[bold orchid]STEP 2[/bold orchid] -> EXTRACTING EEG FEATURES\n"
)

os.system("python src/feature_extraction.py")


console.print(
    "\n[bold plum1]STEP 3[/bold plum1] -> TRAINING MACHINE LEARNING MODEL\n"
)

os.system("python src/train_model.py")


console.print(
    "\n[bold medium_purple]STEP 4[/bold medium_purple] -> STARTING EEG IDENTIFICATION SYSTEM\n"
)

os.system("python src/predict.py")


console.print(
    Panel(
        "[bold pink1]SYSTEM SESSION FINISHED[/bold pink1]",
        border_style="bright_magenta"
    )
)