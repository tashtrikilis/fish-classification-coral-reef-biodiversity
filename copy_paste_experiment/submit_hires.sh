#!/bin/bash
# Round 3: same experiment at HIGHER RESOLUTION (detection only).
#
# Frames are 1280x720. Training and evaluating at imgsz 2560 enlarges them 2x, so a 13 px
# test Forcepsfish reaches the model as ~26 px. Everything else is identical to round 2
# (same data/, split, donors, synthetic images, epochs, seeds).
#
#   9 training jobs: baseline / copypaste / copypaste_scaled x seeds 0, 1, 2   (detect, imgsz 2560)
#   -> compare, with round 2's imgsz-1280 baseline added as a reference column
#
# Output goes to runs_i2560/ so round 2's runs/ is never touched.
# Run from this folder on the Aqua login node:  bash submit_hires.sh
# Optional: IMGSZ=1920 bash submit_hires.sh  (outputs to runs_i1920/)
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

IMGSZ="${IMGSZ:-2560}"
OUT="runs_i${IMGSZ}"

for A in baseline copypaste copypaste_scaled; do
    if [ ! -f "data/det_${A}.yaml" ]; then
        echo "data/det_${A}.yaml not found -- run round 2's prepare first (qsub job_prepare.pbs)"; exit 1
    fi
done
if [ -d "$OUT" ] && ls "$OUT"/*/focus_metrics.json >/dev/null 2>&1; then
    echo "$OUT/ already has finished runs -- move it aside first if you want a fresh round"; exit 1
fi
mkdir -p "$OUT"

IDS=()
for ARM in baseline copypaste copypaste_scaled; do
    for SEED in 0 1 2; do
        ID=$(qsub -N "hr_${ARM/copypaste_scaled/cpsc}_s${SEED}" \
             -v TASK=detect,ARM=$ARM,SEED=$SEED,IMGSZ=$IMGSZ,PROJECT=$OUT job_train.pbs)
        echo "train detect / $ARM / seed $SEED / imgsz $IMGSZ: $ID"
        IDS+=("$ID")
    done
done

ALL=$(IFS=:; echo "${IDS[*]}")
CMP=$(qsub -W depend=afterany:$ALL -v RUNS_DIR=$OUT,REF_RUNS_DIR=runs job_compare.pbs)
echo "compare: $CMP"
echo "${#IDS[@]} training jobs queued. Results: $OUT/comparison/comparison_val.md (choose) and comparison.md (test)."
echo "Watch with:  qstat -u ${USER:-$(whoami)}"
