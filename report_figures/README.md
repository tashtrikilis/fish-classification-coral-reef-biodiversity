# Report figures

| Folder | Figure | Script |
|---|---|---|
| `annotation_fix/` | Original Coral Vision labels vs corrected labels (frames 1323, 2304–2305) | `find_mismatches.py` (finds every species mismatch → `species_mismatches.csv`), `make_annotation_fix.py`, `make_2304_context.py` |
| `cp_steps/` | Copy-paste pipeline: donor, mask, receiver, augmented frame | `make_cp_steps.py 1668 1950 50 3`, `make_b_mask_transparent.py` |
| `blend_compare/` | Direct paste vs edge-only Poisson blending | `make_blend_compare.py` |
| `box_vs_mask/` | Bounding boxes vs instance masks (frame 2189) | `make_box_vs_mask.py` |
| `scripts/` | Class distribution, forest plot of copy-paste effects | `class_distribution*.py`, `forest_parse.py` → `forest_plot.py` |
| `figures/` | Final figure files used in the report | |
| `video/` | Baseline vs copy-paste test detections | `make_test_video.py` |

Most scripts expect the folder layout of the original working directory: `full_dataset/` (frames) and
`copy_paste_experiment/` (labels and helpers) next to each other. Adjust the paths at the top of each script if
you run them from this repository.
