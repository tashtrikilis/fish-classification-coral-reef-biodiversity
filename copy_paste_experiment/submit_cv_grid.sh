#!/bin/bash
# Step 3: the same cross-validated copy-paste experiment for the other five model configurations.
#   detection:     yolo26s, yolo26l            (yolo26m detection = steps 1-2)
#   segmentation:  yolo26s-seg, yolo26m-seg, yolo26l-seg
# Each set-up: baseline vs copy-paste x 5 folds x seeds 0 and 1 = 20 training jobs  ->  100 in total.
#
#   prepare (data_cv2/: same folds and synthetic images as data_cv/, plus segmentation labels; verified identical)
#     -> 100 training jobs (H100)
#     -> 8 evaluation jobs (box AP for all five set-ups, mask AP for the three segmentation set-ups),
#        each starting when its own 20 training jobs have finished
#
# Logs to W&B project egh490-cv-final (group = set-up). Results: runs_cv/cv_results_<setup>_<box|mask>_seeds0-1/
# Run from this folder on the Aqua login node:  bash submit_cv_grid.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
PROJ="${GRID_WANDB_PROJECT:-egh490-cv-final}"

for f in cv_prepare.py cv_evaluate.py cv_check_same_det.py train_experiment.py cv_donor_candidates.csv cv_donor_veto.txt; do
    [ -f "$f" ] || { echo "$f is missing - it is part of this folder"; exit 1; }
done
[ -d data_cv ] || { echo "data_cv/ (from submit_cv.sh) is missing - it is needed to check the new data matches"; exit 1; }
if ls runs_cv/detect_s_*/test_predictions.json runs_cv/segment_*/test_predictions.json >/dev/null 2>&1; then
    echo "runs_cv/ already has these runs - move them aside first for a fresh run"; exit 1
fi

module load Miniconda3 >/dev/null 2>&1 || true
source "$VENV/bin/activate"
python - <<'PY'
from ultralytics import YOLO
for w in ("yolo26s.pt", "yolo26l.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt"):
    YOLO(w)
print("pretrained weights ready")
PY

PREP=$(qsub job_cv_prepare_grid.pbs)
echo "prepare + check: $PREP"

# set-up name : task : weights
SETUPS="detect_s:detect:yolo26s.pt detect_l:detect:yolo26l.pt segment_s:segment:yolo26s-seg.pt segment_m:segment:yolo26m-seg.pt segment_l:segment:yolo26l-seg.pt"
N=0
for S in $SETUPS; do
    CFG=${S%%:*}; REST=${S#*:}; TASK=${REST%%:*}; MODEL=${REST#*:}
    IDS=()
    for SEED in 0 1; do
        SUF=""; [ "$SEED" = "1" ] && SUF="_s1"
        for K in 0 1 2 3 4; do
            for ARM in baseline copypaste; do
                ID=$(qsub -W depend=afterok:$PREP -N "g_${CFG/segment_/sg}_${ARM:0:2}_f${K}s${SEED}" \
                     -v TASK=$TASK,ARM=$ARM,SEED=$SEED,MODEL=$MODEL,DATA_DIR=data_cv2/fold$K,NAME=${CFG}_${ARM}_f${K}${SUF},PROJECT=runs_cv,SAVE_PREDS=1,TAGS=cv:grid:$CFG:fold$K:seed$SEED,GROUP=$CFG,WANDB_PROJECT=$PROJ \
                     job_train.pbs)
                IDS+=("$ID"); N=$((N+1))
            done
        done
    done
    ALL=$(IFS=:; echo "${IDS[*]}")
    E1=$(qsub -W depend=afterany:$ALL -N "ev_${CFG}_box" -v CFG=$CFG,IOU=box,WANDB_PROJECT=$PROJ job_cv_eval_grid.pbs)
    echo "$CFG: ${#IDS[@]} training jobs; box evaluation $E1"
    if [ "$TASK" = "segment" ]; then
        E2=$(qsub -W depend=afterany:$ALL -N "ev_${CFG}_mask" -v CFG=$CFG,IOU=mask,WANDB_PROJECT=$PROJ job_cv_eval_grid.pbs)
        echo "$CFG: mask evaluation $E2"
    fi
done
echo "$N training jobs queued, logging to W&B project '$PROJ'."
echo "Results: runs_cv/cv_results_<setup>_<box|mask>_seeds0-1/cv_results.md"
echo "Watch with:  qstat -u ${USER:-$(whoami)}"
