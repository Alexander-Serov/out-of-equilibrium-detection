import os
import socket
from pathlib import Path

FLOAT_FORMAT = ".6g"

scratch_folder = os.environ.get("MYSCRATCH", None)
if scratch_folder is not None:
    scratch_folder = Path(scratch_folder) / "out-of-equilibrium_detection"

data_folders = {
    "tars": scratch_folder,
    "maestro": scratch_folder,
    "onsager-dbc": Path(r"D:\calculated_data\out-of-equilibrium_detection"),
    "Desktopik": Path(r"D:\calculated_data\out-of-equilibrium_detection"),  # todo
    "": Path("data"),
}

hostname = socket.gethostname()
for key, folder in data_folders.items():
    if hostname.startswith(key) and folder is not None:
        data_folder = folder
        break

LOCK_TIMEOUT = 3  # s
PICKLE_PROTOCOL = 4

MLE_guess_file = "MLE_guesses.pyc"
stat_filename = "statistics.dat"
MLE_GUESSES_FOLDER = ".mle_guesses"
MLE_GUESSES_TO_KEEP = 10
JSON_INDENT = 2

ATOL = 1e-5
color_sequence = [
    [0.1008, 0.4407, 0.7238],
    [0.4353, 0.5804, 0],
    [0.9498, 0.4075, 0.1317],
    [0.4364, 0.2238, 0.5872],
    [0.5860, 0.4228, 0.2649],
]
