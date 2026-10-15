import cv2, numpy as np, json
meta=json.load(open("meta.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
bg=np.load("bg_median.npy"); bg16=bg.astype(np.int16)

# PASS A: streaming Welford for per-pixel temporal mean/var (instability map)
cap=cv2.VideoCapture("clean.mp4")
mean=np.zeros((H,W),np.float64); M2=np.zeros((H,W),np.float64); k=0
while True:
    ok,fr=cap.read()
    if not ok: break
    g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY).astype(np.float64)
    k+=1; d=g-mean; mean+=d/k; M2+=d*(g-mean)
cap.release()
tstd=np.sqrt(M2/max(k-1,1)).astype(np.float32)
np.save("tstd.npy",tstd)
cv2.imwrite("tstd.png",np.clip(tstd*8,0,255).astype(np.uint8))
print("tstd median %.2f p90 %.2f p99 %.2f"%(np.median(tstd),np.percentile(tstd,90),np.percentile(tstd,99)))
# instability (jitter/flicker) mask: chronically changing pixels = edges, microwave display
jitter_mask=(tstd> max(8.0,np.percentile(tstd,92))).astype(np.uint8)
jitter_mask=cv2.dilate(jitter_mask,np.ones((3,3),np.uint8),iterations=1)
cv2.imwrite("jitter_mask.png",jitter_mask*255)
print("jitter_mask covers %.1f%% of frame"%(100*jitter_mask.mean()))
keep = (1-jitter_mask).astype(bool)

def detect_frame(gray, dark_thr=16):
    f=gray.astype(np.int16)
    fgp=(bg16-f)                      # dark fly signal
    fg=((fgp>dark_thr)&keep).astype(np.uint8)*255
    fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    fg=cv2.dilate(fg,np.ones((2,2),np.uint8),1)
    n,lab,stats,cent=cv2.connectedComponentsWithStats(fg,connectivity=8)
    dets=[]
    for i in range(1,n):
        a=stats[i,cv2.CC_STAT_AREA]; x,y,w,h=stats[i,0],stats[i,1],stats[i,2],stats[i,3]
        if not (2<=a<=70): continue
        if max(w,h)>16: continue
        extent=a/float(max(w*h,1))
        asp=max(w,h)/float(max(min(w,h),1))
        if extent<0.3: continue        # reject thin streaks
        if asp>3.5: continue           # reject elongated edge jitter
        # contrast: average darkening inside bbox
        dark=fgp[y:y+h,x:x+w].max()
        dets.append((float(cent[i][0]),float(cent[i][1]),int(a),int(dark)))
    return dets,fg

# PASS B: detect all frames + accumulate motion max-projection
cap=cv2.VideoCapture("clean.mp4"); t=0
counts=[]; all_dets=[]; maxproj=np.zeros((H,W),np.uint8)
while True:
    ok,fr=cap.read()
    if not ok: break
    g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    d,fg=detect_frame(g)
    counts.append(len(d)); all_dets.append(d)
    maxproj=np.maximum(maxproj,fg)
    t+=1
cap.release()
counts=np.array(counts)
json.dump(all_dets,open("dets_bg.json","w"))
np.save("counts_bg.npy",counts)
print("FILTERED det/frame: mean %.2f median %d max %d  frames>=1: %d/%d"%(counts.mean(),np.median(counts),counts.max(),(counts>=1).sum(),N))
# motion max projection on top of background
proj=cv2.cvtColor(bg,cv2.COLOR_GRAY2BGR)
proj[maxproj>0]=(0,0,255)
cv2.imwrite("motion_maxproj.png",proj)

# save a few overlays at high-count frames
order=np.argsort(-counts); pick=sorted(set([int(x) for x in order[:4]]+[N//2,100,300,450]))
cap=cv2.VideoCapture("clean.mp4"); t=0; want=set(pick)
while want:
    ok,fr=cap.read()
    if not ok: break
    if t in want:
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY); d,fg=detect_frame(g)
        vis=fr.copy()
        for (x,y,a,dk) in d: cv2.circle(vis,(int(x),int(y)),16,(0,0,255),2)
        cv2.putText(vis,f"f{t} dets={len(d)}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,0),3)
        cv2.imwrite(f"det2_{t:03d}.png",vis); want.discard(t)
    t+=1
cap.release()
print("pick",pick,"counts",[int(counts[p]) for p in pick])
