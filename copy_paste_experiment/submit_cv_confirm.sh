#!/bin/bash
# Step 2: Detection M (yolo26m), second training seed (seed 1), same folds and same data_cv/.
#
#   10 training jobs (2 arms x 5 folds, seed 1)  ->  evaluate (seed 1; seeds 0+1; seed 0)
#
# Logs to W&B project egh490-cv-final by default. Reuses step 1's data_cv/ and runs_cv/.
# Run from this folder on the Aqua login node:  bash submit_cv_confirm.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
PROJ="${CONFIRM_WANDB_PROJECT:-egh490-cv-final}"

for K in 0 1 2 3 4; do
    for ARM in baseline copypaste; do
        [ -f "data_cv/fold$K/det_$ARM.yaml" ] || { echo "data_cv/fold$K/det_$ARM.yaml missing -- run submit_cv.sh first"; exit 1; }
        [ -f "runs_cv/detect_${ARM}_f$K/test_predictions.json" ] || { echo "step 1 run detect_${ARM}_f$K missing"; exit 1; }
        if [ -f "runs_cv/detect_${ARM}_f${K}_s1/test_predictions.json" ]; then
            echo "runs_cv/detect_${ARM}_f${K}_s1 already finished -- nothing to do for it? Move it aside to rerun."; exit 1
        fi
    done
done

IDS=()
for K in 0 1 2 3 4; do
    for ARM in baseline copypaste; do
        ID=$(qsub -N "cf_${ARM:0:4}_f$K" \
             -v TASK=detect,ARM=$ARM,SEED=1,DATA_DIR=data_cv/fold$K,NAME=detect_${ARM}_f${K}_s1,PROJECT=runs_cv,SAVE_PREDS=1,TAGS=cv:fold$K:seed1,GROUP=cv,WANDB_PROJECT=$PROJ \
             job_train.pbs)
        echo "train fold $K / $ARM / seed 1: $ID"
        IDS+=("$ID")
    done
done
ALL=$(IFS=:; echo "${IDS[*]}")
EV=$(qsub -W depend=afterany:$ALL -v EVAL_SETS=1+0:1+0,WANDB_PROJECT=$PROJ job_cv_evaluate.pbs)
echo "evaluate: $EV"
echo "${#IDS[@]} training jobs queued, logging to W&B project '$PROJ'."
echo "Results: runs_cv/cv_results_seeds0-1/cv_results.md"
echo "Watch with:  qstat -u ${USER:-$(whoami)}"
