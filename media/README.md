# Demo videos

Baseline vs targeted copy-paste augmentation on held-out test footage, one video per focus species:

| Video | Species | Length |
|---|---|---|
| [`demo_forcepsfish.mp4`](demo_forcepsfish.mp4) | Forcepsfish | 21 s |
| [`demo_orangespine_unicornfish.mp4`](demo_orangespine_unicornfish.mp4) | Orangespine Unicornfish | 18 s |
| [`demo_convict_tang.mp4`](demo_convict_tang.mp4) | Convict Tang | 10 s |

- **Model:** YOLO26 Detection M, seed 0. Predicted boxes are shown, with confidence ≥ 0.25. The left pane is the
  baseline model and the right pane is the augmented model.
- **Test frames only:** each frame is predicted by the cross-validation fold model whose test set contains that frame.
  Neither model trained on it, and no copy-paste donor came from it or from the 15-frame buffer around it.
- **What is drawn:** every prediction with confidence ≥ 0.25. The video's species is shown in orange and all other
  species in white, with the species name and confidence on each box.
- Each frame is shown for 1.5 s. Consecutive frames are about 2 s apart in the original footage.

**These clips are illustrative, not a random sample.** They were chosen from encounters where the baseline misses the
fish and the augmented model finds it, to show what the improvement looks like. Frames where the augmented model also
misses or misclassifies the fish are kept within each encounter. The overall effect, including runs and frames where
augmentation did not help, is in [`../results/`](../results/) and [`../supplementary_analysis/`](../supplementary_analysis/).

## Re-making the videos

`make_demo_videos.py` builds all three from the dataset zip and `analysis_bundle_public.tgz` (both on the Releases
page). It needs `Inter-400.ttf` and `Inter-700.ttf` (the [Inter](https://rsms.me/inter/) font, SIL Open Font
Licence) in `media/fonts/`, plus `pillow` and `ffmpeg`:

```bash
cd media
python make_demo_videos.py --images ../dataset/kona_reef_fish_dataset/images \
    --bundle ../supplementary_analysis/scripts/analysis_bundle --fonts fonts
```
