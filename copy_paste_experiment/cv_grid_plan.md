# Round 6 plan: copy-paste across model sizes and segmentation (written 11 Oct 2026, before any round-6 run)

## What is run

The same cross-validated experiment as rounds 4–5 (medium detection), repeated for five more model set-ups:

| Set-up | Task | Weights (COCO-pretrained) |
|---|---|---|
| `detect_s` | detection | yolo26s.pt |
| `detect_l` | detection | yolo26l.pt |
| `segment_s` | instance segmentation | yolo26s-seg.pt |
| `segment_m` | instance segmentation | yolo26m-seg.pt |
| `segment_l` | instance segmentation | yolo26l-seg.pt |

Each set-up is baseline vs copy-paste × 5 folds × seeds 0 and 1, which is 20 training jobs. The five set-ups make 100 jobs.

Everything else is identical to rounds 4–5:

- **Data:** the same folds, the same buffer and the same synthetic images. `data_cv2/` is rebuilt with segmentation labels added, and its detection part is checked to be byte-identical to `data_cv/` before any job starts.
- **Training:** imgsz 1280, at most 150 epochs, patience 50.
- **Model selection:** the best epoch is chosen on the validation blocks.
- **Evaluation:** test predictions are pooled across the 5 folds, every labelled fish is tested once, with a paired block bootstrap (2000 resamples of 49 blocks).

**Segmentation masks are scored like boxes.** Each segmentation job stores the mask IoU of every prediction with every labelled fish in the image. The evaluator then computes COCO-style mask AP over the pooled test folds.

## Primary tests (decided now)

One per set-up, so five in total: **the focus-species mean AP50 difference (copy-paste minus baseline), averaged over seeds 0 and 1.**

- For `detect_s` and `detect_l` it uses **box** AP50.
- For `segment_s`, `segment_m` and `segment_l` it uses **mask** AP50.
- Focus species: Forcepsfish, Orangespine Unicornfish and Convict Tang.

**Decision rule:**

- Compute the bootstrap p-value of each of the five tests.
- Apply the **Holm–Bonferroni** correction across the five, at a familywise α of 0.05.
- A set-up shows a significant copy-paste effect only if its test survives the correction.
- Report the 95% CI for every set-up, whether or not it survives.

Medium detection (rounds 4–5) is already reported and is not part of this family.

## Reported either way (secondary, descriptive)

- Per-species AP50 and recall differences, for each seed separately and for both seeds averaged.
- Box AP for the segmentation set-ups, so the boxes of detection and segmentation models can be compared.
- Any-fish (class-agnostic) AP50 and recall, and macro mAP50 over the classes with at least 20 test labels.
- **Model size:** baseline AP50 per species, any-fish AP50 and macro mAP50 for s, m and l (detection and segmentation).
- **Task:** detection vs segmentation box AP at each size.

## Known limitation, stated in advance

The bootstrap intervals cover which fish happen to be in the footage, not training randomness. Averaging two seeds reduces but doesn't remove that randomness.

## What won't be done

- No extra seeds, folds or settings to change any primary result.
- No change to the primary tests or the correction after the results are seen.
