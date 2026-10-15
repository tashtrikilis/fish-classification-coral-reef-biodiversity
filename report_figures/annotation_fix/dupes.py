"""Within the original Roboflow labels: boxes in the same frame, same position (IoU>=0.7), different species."""
import sys; sys.argv=['x']
exec(open("find_mismatches.py").read().split("rn = names")[0])
rn = names(RF + "/data.yaml"); rf, _ = load([RF + f"/{s}/labels" for s in ("train","valid","test")], rn)
cv, _ = load([CV + "/labels/train"], names(CV + "/data.yaml"))
n=0
for fr in sorted(rf):
    R = rf[fr]
    for i in range(len(R)):
        for j in range(i+1, len(R)):
            v = iou(R[i][2], R[j][2])
            if v >= 0.7 and R[i][0] != R[j][0]:
                n+=1
                b=R[i][2]; cvm=[c[1] for c in cv.get(fr,[]) if iou(c[2],b)>=0.5]
                print(fr, R[i][1], "|", R[j][1], "iou %.2f"%v, "w,h px", round((b[2]-b[0])*1280), round((b[3]-b[1])*720), "cvat:", cvm)
print("pairs", n)
print("frame 1323 roboflow:", [(r[1], [round(x,3) for x in r[2]]) for r in rf[1323]])
print("frame 1323 cvat:", [(r[1], [round(x,3) for x in r[2]]) for r in cv[1323]])
