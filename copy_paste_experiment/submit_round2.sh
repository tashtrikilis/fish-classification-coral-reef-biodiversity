#!/bin/bash
# Round 2: repeat seeds + a size-matched copy-paste arm, for BOTH detection and segmentation.
#
#   prepare (rebuilds data/, adds the copypaste_scaled arm; round-1 images come out identical)
#     -> 14 training jobs:
#          baseline          seeds 1, 2   (round 1 already has seed 0)
#          copypaste         seeds 1, 2   (round 1 already has seed 0)
#          copypaste_scaled  seeds 0, 1, 2
#        each for detect and segment
#     -> compare (pools all seeds, round 1 included)
#
# Run from this folder on the Aqua login node:  bash submit_round2.sh
# Round 1's runs/ folders are kept and reused as seed 0.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

if [ ! -d "$IMAGES_DIR" ]; then
    echo "IMAGES_DIR=$IMAGES_DIR does not exist -- edit config.sh"; exit 1
fi
for R in detect_baseline detect_copypaste segment_baseline segment_copypaste; do
    if [ ! -f "runs/$R/focus_metrics.json" ]; then
        echo "warning: runs/$R/focus_metrics.json not found -- round 1's seed-0 run for $R is missing"
    fi
done

module load Miniconda3 >/dev/null 2>&1 || true
source "$VENV/bin/activate"
python - <<'PY'
from ultralytics import YOLO
for w in ("yolo26m.pt", "yolo26m-seg.pt"):
    YOLO(w)
print("pretrained weights ready")
PY

PREP=$(qsub job_prepare.pbs)
echo "prepare: $PREP"

IDS=()
submit() {  # task arm seed
    local ID
    ID=$(qsub -W depend=afterok:$PREP -N "cp_${1:0:3}_${2/copypaste_scaled/cpsc}_s$3" \
         -v TASK=$1,ARM=$2,SEED=$3 job_train.pbs)
    echo "train $1 / $2 / seed $3: $ID"
    IDS+=("$ID")
}
for TASK in detect segment; do
    for SEED in 1 2; do submit $TASK baseline $SEED; done
    for SEED in 1 2; do submit $TASK copypaste $SEED; done
    for SEED in 0 1 2; do submit $TASK copypaste_scaled $SEED; done
done

ALL=$(IFS=:; echo "${IDS[*]}")
CMP=$(qsub -W depend=afterany:$ALL job_compare.pbs)
echo "compare: $CMP"
echo "${#IDS[@]} training jobs queued. Watch with:  qstat -u ${USER:-$(whoami)}"
