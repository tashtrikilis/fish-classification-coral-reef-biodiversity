# Demo videos and README images

Baseline vs targeted copy-paste augmentation on held-out test footage, one video per focus species:

| Video | Species | Length |
|---|---|---|
| [`demo_forcepsfish.mp4`](demo_forcepsfish.mp4) | Forcepsfish | 21 s |
| [`demo_orangespine_unicornfish.mp4`](demo_orangespine_unicornfish.mp4) | Orangespine Unicornfish | 18 s |
| [`demo_convict_tang.mp4`](demo_convict_tang.mp4) | Convict Tang | 10 s |

- **Panes:** baseline, augmented and ground truth (the manual labels), left to right. Model: YOLO26 Detection M,
  seed 0.
- **Test frames only:** each frame is predicted by the cross-validation fold model whose test set contains that frame.
  Neither model trained on it, and no copy-paste donor came from it or from the 15-frame buffer around it.
- **Boxes:** every prediction with confidence ≥ 0.25 (species and confidence) and every labelled fish (species only).
  The video's species is orange, everything else is white.
- Each frame is shown for 1.5 s. Consecutive frames are about 2 s apart in the original footage.

The clips are examples, not a random sample: they were picked from encounters where the baseline misses the fish and
the augmented model finds it. Every frame of each encounter is kept, including ones where the augmented model also
misses. For the overall effect see [`../results/`](../results/) and [`../supplementary_analysis/`](../supplementary_analysis/).

## Images

`copy_paste_pipeline.png`, `cv_folds.png` and `demo_forcepsfish_frame.jpg` (one frame of the Forcepsfish video) are the
images in the main README.

## Re-making the videos

`make_demo_videos.py` builds all three from the dataset zip and `analysis_bundle_public.tgz` (both on the Releases
page). It needs `Inter-400.ttf` and `Inter-700.ttf` (the [Inter](https://rsms.me/inter/) font, SIL Open Font
Licence) in `media/fonts/`, plus `pillow` and `ffmpeg`:

```bash
cd media
python make_demo_videos.py --images ../dataset/kona_reef_fish_dataset/images \
    --bundle ../supplementary_analysis/scripts/analysis_bundle --fonts fonts
```
