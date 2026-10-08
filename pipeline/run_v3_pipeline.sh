#!/usr/bin/env bash
# End-to-end pipeline: CVAT mask export -> v3 label conversion -> block-
# stratified dataset build (full 58-species + a top-10-priority build) ->
# top-10 class filtering -> copy-paste augmentation (2 experiments: equalize,
# minority-cap) for BOTH object detection and instance segmentation.
#
# Run this from your aqua_yolo/ folder on Aqua. EDIT THE TOP-10 KEEP LIST
# BELOW if your images-dir check shows a different frame range than assumed
# (min/max frame args) -- the ranking is computed by cvat_masks_to_yolo.py's
# printed report the first time you run it; re-run this script with an
# updated TOP10_KEEP if the ranking shifts once real image availability is
# confirmed.
#
# Usage: ./run_v3_pipeline.sh /path/to/full_dataset_images [min_frame] [max_frame]
# min_frame/max_frame default to 0/1911 (the full reviewed range) -- pass a
# narrower range to smoke-test the whole pipeline quickly before committing
# to the full run, e.g.: ./run_v3_pipeline.sh /path/to/images 0 200
set -euo pipefail

IMAGES_DIR="${1:?Usage: ./run_v3_pipeline.sh /path/to/full_dataset_images [min_frame] [max_frame]}"
MIN_FRAME="${2:-0}"
MAX_FRAME="${3:-1911}"

# From the real CVAT export analyzed so far -- re-check after step 1's printed
# report if your labelling has moved on since, and edit this list to match.
TOP10_KEEP="53,10,33,2,19,31,0,35,16,17"

echo "=== step 1: CVAT mask export -> flat YOLO label dirs + donor eligibility ==="
python3 cvat_masks_to_yolo.py \
    --xml annotations.xml \
    --seg-out dataset_seg_labels_v3 \
    --detect-out dataset_detect_labels_v3 \
    --donor-out dataset_donor_v3 \
    --meta-csv instance_meta_v3.csv \
    --edge-margin 2 \
    --min-frame "$MIN_FRAME" --max-frame "$MAX_FRAME"

echo
echo "=== step 2a: build dataset_seg_v3 / dataset_detect_v3 (whole-dataset variant, all 58 classes prioritized) ==="
python3 build_v3_dataset.py \
    --images-dir "$IMAGES_DIR" \
    --seg-labels dataset_seg_labels_v3 --detect-labels dataset_detect_labels_v3 --donor dataset_donor_v3 \
    --seg-dst dataset_seg_v3 --detect-dst dataset_detect_v3 \
    --seg-yaml data_seg_v3.yaml --detect-yaml data_detect_v3.yaml \
    --min-frame "$MIN_FRAME" --max-frame "$MAX_FRAME"

echo
echo "=== step 2b: build dataset_seg_v3_top10src / dataset_detect_v3_top10src (top-10 classes prioritized in the split) ==="
python3 build_v3_dataset.py \
    --images-dir "$IMAGES_DIR" \
    --seg-labels dataset_seg_labels_v3 --detect-labels dataset_detect_labels_v3 --donor dataset_donor_v3 \
    --seg-dst dataset_seg_v3_top10src --detect-dst dataset_detect_v3_top10src \
    --min-frame "$MIN_FRAME" --max-frame "$MAX_FRAME" \
    --priority-classes "$TOP10_KEEP"

echo
echo "=== step 3: filter the top10-prioritized build down to just those 10 classes ==="
python3 filter_topk_classes.py \
    --src dataset_seg_v3_top10src --dst dataset_seg_top10_v3 \
    --keep "$TOP10_KEEP" \
    --donor-src dataset_seg_v3_top10src/donor_eligibility --donor-dst dataset_seg_top10_v3/donor_eligibility \
    --out-yaml data_seg_top10_v3.yaml --force

python3 filter_topk_classes.py \
    --src dataset_detect_v3_top10src --dst dataset_detect_top10_v3 \
    --keep "$TOP10_KEEP" \
    --out-yaml data_detect_top10_v3.yaml --force

echo
echo "!!! CHECK the two tables just printed above -- confirm no class shows a ZERO train or val count"
echo "    before continuing to augmentation. If one does, re-run step 2b/3 with a different TOP10_KEEP"
echo "    (swap the problem class for the next-ranked one in instance_meta_v3.csv)."
echo

for VARIANT in full top10; do
  if [ "$VARIANT" = "full" ]; then
    SEG_SRC=dataset_seg_v3; DET_SRC=dataset_detect_v3
    SEG_YAML=data_seg_v3.yaml; DET_YAML=data_detect_v3.yaml
  else
    SEG_SRC=dataset_seg_top10_v3; DET_SRC=dataset_detect_top10_v3
    SEG_YAML=data_seg_top10_v3.yaml; DET_YAML=data_detect_top10_v3.yaml
  fi

  for MODE in equalize minority-cap; do
    SUFFIX=$(echo "$MODE" | tr -d '-')
    SEG_YAML_COPY="${SEG_YAML%.yaml}_${SUFFIX}.yaml"
    DET_YAML_COPY="${DET_YAML%.yaml}_${SUFFIX}.yaml"
    cp "$SEG_YAML" "$SEG_YAML_COPY"
    cp "$DET_YAML" "$DET_YAML_COPY"

    echo "=== step 4: copy-paste augment -- variant=$VARIANT mode=$MODE ==="
    python3 copy_paste_augment.py \
        --seg-src "$SEG_SRC" --detect-src "$DET_SRC" \
        --seg-yaml "$SEG_YAML_COPY" --detect-yaml "$DET_YAML_COPY" \
        --mode "$MODE" --minority-threshold 50 --minority-cap 30 \
        --blend matched --seed 42 --synth-suffix "$SUFFIX" \
        --report-csv "copy_paste_report_${VARIANT}_${SUFFIX}.csv"
    echo
  done
done

echo "=== done. Resulting data yamls, ready for qsub: ==="
ls data_*_v3_equalize.yaml data_*_v3_minoritycap.yaml 2>/dev/null | sort -u || true
