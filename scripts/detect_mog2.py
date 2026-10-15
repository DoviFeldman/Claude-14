import cv2, numpy as np, json
meta=json.load(open("meta.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
tstd=np.load("tstd.npy")
jitter=(tstd>max(8.0,np.percentile(tstd,92))).astype(np.uint8)
jitter=cv2.dilate(jitter,np.ones((3,3),np.uint8),1)
keep=(1-jitter).astype(bool)

bs=cv2.createBackgroundSubtractorMOG2(history=150,varThreshold=30,detectShadows=False)
def shape_ok(a,w,h):
    if not(2<=a<=70): return False
    if max(w,h)>16: return False
    if a/float(max(w*h,1))<0.3: return False
    if max(w,h)/float(max(min(w,h),1))>3.5: return False
    return True

dets_per=[]
cap=cv2.VideoCapture("clean.mp4"); t=0
while True:
    ok,fr=cap.read()
    if not ok: break
    g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    fg=bs.apply(g, learningRate=1.0/60)
    fg[~keep]=0
    fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,stats,cent=cv2.connectedComponentsWithStats((fg>0).astype(np.uint8),8)
    d=[]
    for i in range(1,n):
        a=stats[i,cv2.CC_STAT_AREA]; x,y,w,h=stats[i,:4]
        if shape_ok(a,w,h): d.append((float(cent[i][0]),float(cent[i][1]),int(a)))
    dets_per.append(d); t+=1
cap.release()
counts=np.array([len(d) for d in dets_per])
json.dump(dets_per,open("dets_mog2.json","w"))
print("MOG2 det/frame mean %.2f median %d max %d"%(counts.mean(),np.median(counts),counts.max()))
for s in range(0,N,30):
    seg=counts[s:s+30]; print(f" f{s:3d}-{s+29:3d} mean {seg.mean():5.1f} max {seg.max():3d}")
