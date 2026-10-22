# Settings shared by all the job scripts. Edit IMAGES_DIR before the first run.

# Folder with all the frame_<N>.jpg files: frames 1000-2461 plus the donor frames outside
# that range (18, 22, 163-167, 939-941), i.e. images/ and donor_frames/images/ from the dataset zip.
IMAGES_DIR="$HOME/aqua_yolo/full_dataset"

# Python virtual environment with requirements.txt installed
VENV="$HOME/yolo-venv"

# Weights & Biases project every run logs to
WANDB_PROJECT="egh490-copypaste-experiment"

# If compute nodes can't reach the internet, uncomment this, then run
# `wandb sync runs/wandb/offline-run-*` (or `wandb sync wandb/offline-run-*`) from the login node.
# export WANDB_MODE=offline
