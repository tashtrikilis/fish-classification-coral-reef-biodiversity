#!/usr/bin/env python3
"""Check that a rebuilt cross-validation folder (e.g. data_cv2, with segmentation roots added) has exactly the same
detection data as the original (data_cv): fold assignment, detection labels, image lists and synthetic images.
Exits with an error if anything differs, so the training jobs that depend on it never start on different data.

Usage:  python3 cv_check_same_det.py data_cv data_cv2
"""
import hashlib
import sys
from pathlib import Path


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    files = [p.relative_to(a) for p in a.glob("fold*/det_*/labels/*/*.txt")]
    files += [p.relative_to(a) for p in a.glob("fold*/synth_images/*.jpg")]
    files += [Path("folds.csv")] + [p.relative_to(a) for p in a.glob("fold*/synth_report.csv")]
    bad = [str(f) for f in files if not (b / f).exists() or md5(a / f) != md5(b / f)]
    la = sorted(str(p.relative_to(a)) for p in a.glob("fold*/det_*/images/*/*"))
    lb = sorted(str(p.relative_to(b)) for p in b.glob("fold*/det_*/images/*/*"))
    seg = sorted(b.glob("fold*/seg_*.yaml"))
    print(f"compared {len(files)} detection files: {len(bad)} differ; image lists identical: {la == lb}; "
          f"segmentation yamls in {b}: {len(seg)}")
    if bad or la != lb or len(seg) != 10:
        print("MISMATCH -- first differences:", bad[:10])
        sys.exit(1)
    print("OK: detection data identical, segmentation roots present")


if __name__ == "__main__":
    main()
