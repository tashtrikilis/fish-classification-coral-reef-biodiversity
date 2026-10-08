# Settings shared by all the job scripts. Edit IMAGES_DIR before the first run.

# Folder on Aqua holding frame_<N>.jpg. Must include frames 1000-2461 AND the
# out-of-range Forcepsfish donor frames 163-167 and 939-941.
IMAGES_DIR="$HOME/aqua_yolo/full_dataset"

# Python environment (same one your other YOLO jobs use)
VENV="$HOME/yolo-venv"

# Weights & Biases project every run logs to
WANDB_PROJECT="egh490-copypaste-experiment"

# If compute nodes can't reach the internet, uncomment this, then run
# `wandb sync runs/wandb/offline-run-*` (or `wandb sync wandb/offline-run-*`) from the login node.
# export WANDB_MODE=offline
