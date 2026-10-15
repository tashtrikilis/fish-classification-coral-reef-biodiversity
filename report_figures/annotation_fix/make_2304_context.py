"""Frame 2304 (one frame before 2305): the same fish, which Roboflow itself labelled Convict Tang.
Uses the same crop window as the 2305 panels so the three images line up."""
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_annotation_fix.py")).read()
exec(src.split("make(1323")[0])
fr, cx, cy, crop, scale = 2304, 749, 550, (300, 225), 3.2
x0, y0 = 670, 485                      # same window as frame 2305
im = Image.open(f"{B}/full_dataset/frame_{fr}.jpg").convert("RGB"); W, H = im.size
R = lab([RF + f"/{s}/labels" for s in ("train", "valid", "test")], fr)
def dist(b): return abs((b[0]+b[2])/2 - cx/W) + abs((b[1]+b[3])/2 - cy/H)
rc, rv = min(R, key=lambda r: dist(box(r[1])))
a = im.crop((x0, y0, x0 + crop[0], y0 + crop[1])).resize((int(crop[0]*scale), int(crop[1]*scale)), Image.LANCZOS)
P = lambda x, y: ((x*W - x0)*scale, (y*H - y0)*scale)
d = ImageDraw.Draw(a); rb = box(rv); p0, p1 = P(rb[0], rb[1]), P(rb[2], rb[3]); BLUE = (0, 114, 178)
d.rectangle([p0, p1], outline=BLUE, width=5); tag(d, p0[0], p0[1], rn[rc], BLUE)
a.save(f"{fr}_a_original.png"); print(fr, rn[rc], [round(v) for v in (p0 + p1)])
