# Data

## What's here

| Path | Contents |
|---|---|
| `cvat/annotations.xml` | CVAT 1.1 export of the instance masks (RLE `<mask>` elements), with an `occluded` attribute per instance. Covers all 2,817 frames. Exported 4 Oct 2026. |
| `labels_yolo_seg/labels/train/*.txt` | Ultralytics YOLO-segmentation export of the same project, exported 7 Oct 2026, which is the **latest** labels. One file per labelled frame (1,340), each line `class x1 y1 x2 y2 …` with normalised polygon coordinates. A frame with no file is labelled background. Files are named `nelha-trimmed_mp4-<frame>_jpg.rf.<hash>.txt`. |
| `labels_yolo_seg/data.yaml` | The 58 class names, with ids 0–57. |

The YOLO export is slightly newer than the CVAT XML. It includes Forcepsfish annotated in frames 939–941 that the XML doesn't.
The copy-paste experiment uses the YOLO export, and the v3 pipeline (`pipeline/`) uses the XML.

## Frames (not included)

The source footage is **2,817 JPEG frames (1280×720, about 455 MB)** from a fixed reef camera at Kona, Hawaiʻi
(Aqualink / MEGA Lab, 26 April 2021). It isn't stored in git because of its size and because the footage belongs to its
original owner.

To run anything, place the frames in a folder called `full_dataset/` at the **repository root**, named by frame number:

```
fish-classification-coral-reef-biodiversity/
├── full_dataset/
│   ├── frame_0.jpg
│   ├── frame_1.jpg
│   └── … frame_2816.jpg
├── data/
└── …
```

`full_dataset/` is listed in `.gitignore`, so it won't be committed by accident.

## Frame ranges used

| Range | Use |
|---|---|
| 1000–1771 | v2 / v3 train/val datasets (`pipeline/`) |
| 1000–2461 | copy-paste experiment: 2461 is the last labelled frame |
| 163–166, 939–941 | extra Forcepsfish copy-paste donors only. They're outside the evaluated range, so they never appear in val or test. |

The camera overlay timestamp gives about 30 frames per minute of real time.
