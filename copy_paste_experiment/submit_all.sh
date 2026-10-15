#!/bin/bash
# Submits the whole experiment with job dependencies:
#   Round 1: prepare  ->  4 training jobs (detect/segment x baseline/copypaste, seed 0)  ->  compare
#   (round 2 -- seeds 1-2 and the size-matched arm -- is submit_round2.sh)
# Run from this folder on the Aqua login node:  bash submit_all.sh
# Set SKIP_PREPARE=1 to reuse an existing data/ folder.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

if [ ! -d "$IMAGES_DIR" ]; then
    echo "IMAGES_DIR=$IMAGES_DIR does not exist -- edit config.sh"; exit 1
fi
# quick spot check of frames the experiment needs (prepare_experiment.py checks every one)
for F in 163 166 939 941 1000 1603 2381 2461; do
    if [ ! -f "$IMAGES_DIR/frame_$F.jpg" ]; then
        echo "missing $IMAGES_DIR/frame_$F.jpg -- needs frames 1000-2461 plus 163-167 and 939-941"; exit 1
    fi
done

# Download the pretrained weights here on the login node, in case compute nodes have no internet.
module load Miniconda3 >/dev/null 2>&1 || true
source "$VENV/bin/activate"
python - <<'PY'
from ultralytics import YOLO
for w in ("yolo26m.pt", "yolo26m-seg.pt"):
    YOLO(w)
print("pretrained weights ready")
PY

DEP=""
if [ "${SKIP_PREPARE:-0}" != "1" ]; then
    PREP=$(qsub job_prepare.pbs)
    echo "prepare: $PREP"
    DEP="-W depend=afterok:$PREP"
elif [ ! -f data/prepare_summary.json ]; then
    echo "SKIP_PREPARE=1 but data/prepare_summary.json is missing"; exit 1
fi

IDS=()
for TASK in detect segment; do
    for ARM in baseline copypaste; do
        ID=$(qsub $DEP -N "cp_${TASK:0:3}_${ARM}" -v TASK=$TASK,ARM=$ARM job_train.pbs)
        echo "train $TASK/$ARM: $ID"
        IDS+=("$ID")
    done
done

ALL=$(IFS=:; echo "${IDS[*]}")
CMP=$(qsub -W depend=afterany:$ALL job_compare.pbs)
echo "compare: $CMP"
echo "watch with:  qstat -u $USER"
