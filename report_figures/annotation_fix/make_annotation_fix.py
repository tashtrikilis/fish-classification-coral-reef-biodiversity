"""Side-by-side panels: original Roboflow label (box) vs corrected label (mask + box) for one fish.
Usage: python3 make_annotation_fix.py  -> writes <frame>_a_original.png / <frame>_b_corrected.png"""
import os, re, glob
from PIL import Image, ImageDraw, ImageFont
B = os.path.expanduser("~/mnt/EGH490-2_scripts")
RF = B + "/Kona- Hawaii.v1-biodiversity.yolo26"; CV = B + "/copy_paste_experiment/labels_export"
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "find_mismatches.py")).read().split("rn = names")[0].split("def load")[0])
rn = names(RF + "/data.yaml"); cn = names(CV + "/data.yaml")
def lab(dirs, fr):
    for d in dirs:
        for f in glob.glob(f"{d}/*mp4-{fr}_jpg*.txt"):
            return [(int(l.split()[0]), list(map(float, l.split()[1:]))) for l in open(f) if len(l.split()) > 4]
import matplotlib
FONT = ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"), 34)
def tag(d, x, y, text, col):
    l, t, r, b = d.textbbox((0, 0), text, font=FONT); w, h = r - l, b - t
    y = y - h - 16 if y - h - 16 > 0 else y + 6
    d.rectangle([x, y, x + w + 16, y + h + 14], fill=col); d.text((x + 8 - l, y + 7 - t), text, fill="white", font=FONT)
def make(fr, cx, cy, crop=(300, 225), scale=3.2):
    im = Image.open(f"{B}/full_dataset/frame_{fr}.jpg").convert("RGB"); W, H = im.size
    R = lab([RF + f"/{s}/labels" for s in ("train", "valid", "test")], fr); C = lab([CV + "/labels/train"], fr)
    tgt = [cx / W, cy / H]
    def dist(b): return abs((b[0]+b[2])/2 - tgt[0]) + abs((b[1]+b[3])/2 - tgt[1])
    rc, rv = min(R, key=lambda r: dist(box(r[1]))); cc, cvp = min(C, key=lambda r: dist(box(r[1])))
    x0 = int(min(max(cx - crop[0]/2, 0), W - crop[0])); y0 = int(min(max(cy - crop[1]/2, 0), H - crop[1]))
    base = im.crop((x0, y0, x0 + crop[0], y0 + crop[1])).resize((int(crop[0]*scale), int(crop[1]*scale)), Image.LANCZOS)
    P = lambda x, y: ((x*W - x0)*scale, (y*H - y0)*scale)
    # (a) original Roboflow box
    a = base.copy(); d = ImageDraw.Draw(a); rb = box(rv)
    p0, p1 = P(rb[0], rb[1]), P(rb[2], rb[3]); RED = (214, 39, 40)
    d.rectangle([p0, p1], outline=RED, width=5); tag(d, p0[0], p0[1], rn[rc].replace("- ", " - "), RED)
    a.save(f"{fr}_a_original.png")
    # (b) corrected CVAT polygon
    b = base.copy().convert("RGBA"); ov = Image.new("RGBA", b.size, (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
    poly = [P(cvp[i], cvp[i+1]) for i in range(0, len(cvp), 2)]; GRN = (0, 158, 115)
    od.polygon(poly, fill=GRN + (40,)); b = Image.alpha_composite(b, ov).convert("RGB"); d = ImageDraw.Draw(b)
    d.line(poly + [poly[0]], fill=GRN, width=5); cb = box(cvp); q0, q1 = P(cb[0], cb[1]), P(cb[2], cb[3])
    d.rectangle([q0, q1], outline=GRN, width=3); tag(d, q0[0], q0[1], cn[cc], GRN)
    b.save(f"{fr}_b_corrected.png")
    print(fr, "original:", rn[rc], "| corrected:", cn[cc], "| crop origin", (x0, y0), crop)
make(1323, 324, 542)
make(1173, 117, 322)
make(2305, 820, 598)
