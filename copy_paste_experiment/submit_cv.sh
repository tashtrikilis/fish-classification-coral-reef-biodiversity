#!/bin/bash
# Step 1: Detection M (yolo26m), seed 0. 5-fold cross-validation, baseline vs copy-paste (imgsz 1280).
# Focus species: Forcepsfish, Orangespine Unicornfish, Convict Tang.
#
#   cv_prepare (CPU)  ->  10 training jobs (2 arms x 5 folds, H100)  ->  cv_evaluate (CPU)
#
# Needs cv_donor_candidates.csv and cv_donor_veto.txt (candidates excluded at the visual quality check; an empty file = keep all).
# Output: data_cv/, runs_cv/ and runs_cv/cv_results/. 
# Run from this folder on the Aqua login node:  bash submit_cv.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

for f in cv_prepare.py cv_evaluate.py cv_donor_candidates.csv experiment_donors.csv; do
    [ -f "$f" ] || { echo "$f is missing -- it is part of this folder"; exit 1; }
done
if [ ! -f cv_donor_veto.txt ]; then
    echo "cv_donor_veto.txt not found -- create it (empty = keep every candidate donor)"; exit 1
fi
[ -d "$IMAGES_DIR" ] || { echo "IMAGES_DIR=$IMAGES_DIR does not exist -- edit config.sh"; exit 1; }
if ls runs_cv/*/test_predictions.json >/dev/null 2>&1; then
    echo "runs_cv/ already has finished runs -- move it aside first for a fresh run"; exit 1
fi

PREP=$(qsub job_cv_prepare.pbs)
echo "prepare: $PREP"
IDS=()
for K in 0 1 2 3 4; do
    for ARM in baseline copypaste; do
        ID=$(qsub -W depend=afterok:$PREP -N "cv_${ARM:0:4}_f$K" \
             -v TASK=detect,ARM=$ARM,SEED=0,DATA_DIR=data_cv/fold$K,NAME=detect_${ARM}_f$K,PROJECT=runs_cv,SAVE_PREDS=1,TAGS=cv:fold$K,GROUP=cv \
             job_train.pbs)
        echo "train fold $K / $ARM: $ID"
        IDS+=("$ID")
    done
done
ALL=$(IFS=:; echo "${IDS[*]}")
EV=$(qsub -W depend=afterany:$ALL job_cv_evaluate.pbs)
echo "evaluate: $EV"
echo "${#IDS[@]} training jobs queued. Results: runs_cv/cv_results/cv_results.md"
echo "Watch with:  qstat -u ${USER:-$(whoami)}"
