# Copy-paste review

How the copy-paste experiment's settings were chosen, in the order it happened:

1. **`copy_paste_demo.ipynb`** covers species screening and blend comparison (naive, LAB-matched, Poisson). It includes a case study on why small donors fail: coarse masks, and colours washed out by distance underwater. That led to the `min_dim ≥ 20 px` donor rule.
2. **`blend_review_generate.py`**, **`blend_review_large_donors*.py`** and **`final_recommended_gallery.py`** generated folders of side-by-side paste examples for visual review of each blend method and donor size.
3. **`donor_acceptor_review.ipynb`** (built from `donor_acceptor_review_source.py`) shows galleries of every large eligible donor per species, and one candidate acceptor frame per minute of footage, for manual approval.
4. **`approved_donors.csv`** and **`approved_acceptor_frames.csv`** hold the outcome of that review, and they feed `copy_paste_experiment/`.

The scripts read `annotations.xml`, a link to `../data/cvat/annotations.xml`, and frames from `../full_dataset/`.
Generated example folders are git-ignored.
