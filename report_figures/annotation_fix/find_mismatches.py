"""Compare original Roboflow labels with corrected CVAT labels (frames 1000-2461).
Match boxes per frame by IoU and list instances whose species changed."""
import os, re, glob, collections, csv
from PIL import Image
B = os.path.expanduser("~/mnt/EGH490-2_scripts")
RF = B + "/Kona- Hawaii.v1-biodiversity.yolo26"
CV = B + "/copy_paste_experiment/labels_export"
def names(path):
    t = open(path).read()
    m = re.search(r"names:\s*\[(.*?)\]", t, re.S)
    if m: return [x.strip().strip("'\"") for x in m.group(1).split(',') if x.strip()]
    out = {}
    for l in t.splitlines():
        mm = re.match(r"\s+(\d+):\s*(.+)", l)
        if mm: out[int(mm.group(1))] = mm.group(2).strip().strip("'\"")
    return [out[i] for i in sorted(out)]
def norm(n):
    n = re.sub(r"\s+", " ", n.replace("_", " ")).strip().lower()
    n = n.replace("butterfly fish", "butterflyfish").replace("lognnose", "longnose")
    n = n.replace("bullethead unicornfish", "bullethead parrotfish")
    return n
def box(vals):
    if len(vals) == 4:
        x, y, w, h = vals; return [x-w/2, y-h/2, x+w/2, y+h/2]
    xs, ys = vals[0::2], vals[1::2]; return [min(xs), min(ys), max(xs), max(ys)]
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    I = ix*iy; U = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - I
    return I/U if U > 0 else 0
def load(dirs, nm):
    d = collections.defaultdict(list); files = collections.defaultdict(list)
    for dd in dirs:
        for f in glob.glob(dd + "/*.txt"):
            m = re.search(r"mp4-(\d+)_jpg", f)
            if not m: continue
            fr = int(m.group(1))
            if not 1000 <= fr <= 2461: continue
            files[fr].append(f)
            if len(files[fr]) > 1: continue  # one copy per frame
            for l in open(f):
                p = l.split()
                if len(p) < 5: continue
                d[fr].append((norm(nm[int(p[0])]), nm[int(p[0])], box(list(map(float, p[1:])))))
    return d, files
rn = names(RF + "/data.yaml"); cn = names(CV + "/data.yaml")
rf, rff = load([RF + f"/{s}/labels" for s in ("train", "valid", "test")], rn)
cv, _ = load([CV + "/labels/train"], cn)
print("frames rf", len(rf), "cv", len(cv), "dup rf frames", sum(len(v) > 1 for v in rff.values()))
W, H = Image.open(B + "/full_dataset/frame_1000.jpg").size
rows = []
for fr in sorted(set(rf) | set(cv)):
    R, C = rf.get(fr, []), cv.get(fr, [])
    used = set()
    for i, (rnm, rraw, rb) in enumerate(R):
        best, bj = 0, None
        for j, (cnm, craw, cb) in enumerate(C):
            if j in used: continue
            v = iou(rb, cb)
            if v > best: best, bj = v, j
        if bj is not None and best >= 0.5:
            used.add(bj)
            cnm, craw, cb = C[bj]
            if rnm != cnm:
                w = (cb[2]-cb[0])*W; h = (cb[3]-cb[1])*H
                rows.append(dict(frame=fr, roboflow=rraw, corrected=craw, iou=round(best, 2),
                                 w_px=round(w), h_px=round(h), area=round(w*h),
                                 cx=round((cb[0]+cb[2])/2*W), cy=round((cb[1]+cb[3])/2*H)))
rows.sort(key=lambda r: -r["area"])
with open(os.path.dirname(__file__) + "/species_mismatches.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("mismatched instances:", len(rows), "image", W, H)
pc = collections.Counter((r["roboflow"], r["corrected"]) for r in rows)
for k, v in pc.most_common(): print(v, k)
print("--- largest, excluding longnose/forceps ---")
for r in rows:
    pair = {norm(r["roboflow"]), norm(r["corrected"])}
    if pair & {"longnose butterflyfish", "forcepsfish"}: continue
    print(r)
