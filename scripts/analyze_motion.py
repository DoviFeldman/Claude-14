import cv2, numpy as np
cap = cv2.VideoCapture("source.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
prev = None
rows = []
idx = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (379, 213))  # downscale for speed
    if prev is not None:
        d = cv2.absdiff(g, prev)
        # global mean diff = camera motion / global change
        rows.append((idx, float(d.mean())))
    prev = g
    idx += 1
cap.release()
arr = np.array(rows)
print("frames:", idx, "fps:", fps)
# Print per-0.5s window mean diff
import math
for s in range(0, int(idx/fps)+1):
    lo, hi = int(s*fps), int((s+1)*fps)
    seg = arr[(arr[:,0]>=lo)&(arr[:,0]<hi)]
    if len(seg):
        print(f"  t={s:2d}-{s+1:2d}s  meandiff={seg[:,1].mean():6.3f}  max={seg[:,1].max():6.3f}")
# Find first frame where a sustained low-motion period begins
vals = arr[:,1]
# smooth
k=5
sm = np.convolve(vals, np.ones(k)/k, mode='same')
thr = np.median(sm)*1.8
# find last index in first 150 frames exceeding thr (end of initial shake)
early = np.where(sm[:180] > thr)[0]
settle = (early.max()+5) if len(early) else 0
print("global median diff:", round(float(np.median(sm)),3), "thr:", round(float(thr),3))
print("estimated settle frame:", int(settle), "=> t=", round(settle/fps,2), "s")
np.save("framediff.npy", arr)
