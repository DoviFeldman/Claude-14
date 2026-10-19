import cv2, numpy as np, json, time
from scipy.optimize import linear_sum_assignment
meta=json.load(open("meta_hq.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
print(f"HD pipeline: {N} frames {W}x{H}")

# scale size caps for 1080p (old caps were for 1518x854; area x1.6, linear x1.265)
AMIN,AMAX,MAXSIDE=3,120,22
GATE=57; WARM=30

# ---- PASS A: streaming Welford tstd + median bg (sampled) ----
cap=cv2.VideoCapture("clean_hq.mp4")
mean=np.zeros((H,W),np.float64); M2=np.zeros((H,W),np.float64); k=0; samp=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    k+=1; d=g-mean; mean+=d/k; M2+=d*(g-mean)
    if k%4==1: samp.append(g.copy())
cap.release()
tstd=np.sqrt(M2/max(k-1,1)).astype(np.float32); np.save("tstd_hq.npy",tstd)
bg=np.median(np.stack(samp),axis=0).astype(np.uint8); del samp
cv2.imwrite("bg_hq.png",bg)
jit=cv2.dilate((tstd>max(8.0,np.percentile(tstd,92))).astype(np.uint8),np.ones((3,3),np.uint8),1)
keep=(1-jit).astype(bool)
print("tstd median %.2f p92 %.2f  jitter-mask covers %.1f%%"%(np.median(tstd),np.percentile(tstd,92),100*jit.mean()))

def shape_ok(a,w,h):
    return (AMIN<=a<=AMAX) and max(w,h)<=MAXSIDE and a/float(max(w*h,1))>=0.3 and max(w,h)/float(max(min(w,h),1))<=3.5

# ---- PASS B: MOG2 detection ----
bs=cv2.createBackgroundSubtractorMOG2(history=150,varThreshold=30,detectShadows=False)
dets_per=[]; cap=cv2.VideoCapture("clean_hq.mp4"); t0=time.perf_counter()
while True:
    ok,fr=cap.read()
    if not ok: break
    g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    fg=bs.apply(g,learningRate=1.0/60); fg[~keep]=0
    fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,st,ce=cv2.connectedComponentsWithStats((fg>0).astype(np.uint8),8)
    d=[(float(ce[i][0]),float(ce[i][1])) for i in range(1,n) if shape_ok(st[i,4],st[i,2],st[i,3])]
    dets_per.append(d)
cap.release()
det_time=time.perf_counter()-t0
counts=np.array([len(d) for d in dets_per])
json.dump(dets_per,open("dets_hq.json","w"))
print("MOG2 det/frame mean %.2f median %d max %d  (%.1f ms/frame)"%(counts.mean(),np.median(counts),counts.max(),det_time/N*1000))

# ---- PASS C: Kalman + Hungarian tracker ----
class KF:
    def __init__(s,x,y):
        s.x=np.array([x,y,0,0],float);s.P=np.eye(4)*10
        s.F=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],float)
        s.H=np.array([[1,0,0,0],[0,1,0,0]],float);s.Q=np.eye(4)*0.8;s.Q[2,2]=s.Q[3,3]=2;s.R=np.eye(2)*4
    def predict(s): s.x=s.F@s.x;s.P=s.F@s.P@s.F.T+s.Q;return s.x[:2]
    def update(s,z):
        y=z-s.H@s.x;S=s.H@s.P@s.H.T+s.R;Kk=s.P@s.H.T@np.linalg.inv(S);s.x=s.x+Kk@y;s.P=(np.eye(4)-Kk@s.H)@s.P
tr=[];fin=[];nid=[0]
def mk(d,fr):
    o=type("T",(),{})();o.kf=KF(d[0],d[1]);o.id=nid[0];nid[0]+=1;o.miss=0;o.pts={fr:(d[0],d[1])};return o
for fr in range(WARM,N):
    dd=dets_per[fr];pr=[t.kf.predict() for t in tr]
    if tr and dd:
        C=np.zeros((len(tr),len(dd)))
        for i,p in enumerate(pr):
            for j,d in enumerate(dd):C[i,j]=np.hypot(p[0]-d[0],p[1]-d[1])
        ri,ci=linear_sum_assignment(C);at=set();ad=set()
        for i,j in zip(ri,ci):
            if C[i,j]<=GATE:tr[i].kf.update(np.array(dd[j]));tr[i].miss=0;tr[i].pts[fr]=dd[j];at.add(i);ad.add(j)
        for j,d in enumerate(dd):
            if j not in ad:tr.append(mk(d,fr))
        for i,t in enumerate(tr):
            if i<len(pr) and i not in at:t.miss+=1
    else:
        for t in tr:t.miss+=1
        for d in dd:tr.append(mk(d,fr))
    al=[]
    for t in tr:(fin if t.miss>10 else al).append(t)
    tr=al
fin+=tr
fly=[]
for t in fin:
    frs=sorted(t.pts)
    if len(frs)<4: continue
    xs=np.array([t.pts[f][0] for f in frs]);ys=np.array([t.pts[f][1] for f in frs])
    span=frs[-1]-frs[0]+1;net=np.hypot(xs[-1]-xs[0],ys[-1]-ys[0]);path=np.hypot(np.diff(xs),np.diff(ys)).sum()
    if span<5 or (path<20 and net<15): continue
    fly.append(dict(id=int(t.id),frames=[int(f) for f in frs],x=[round(float(v),1) for v in xs],
        y=[round(float(v),1) for v in ys],net=round(float(net),1),path=round(float(path),1),span=int(span),hits=len(frs)))
fly.sort(key=lambda d:-d["hits"])
json.dump(dict(meta=meta,tracks=fly),open("fly_tracks_hq.json","w"))
print(f"\n=== HD RESULT: {len(fly)} fly tracks (low-quality baseline was 14) ===")
for t in fly[:30]:
    print(f"  id{t['id']:3d} hits{t['hits']:3d} span{t['span']:3d} net{t['net']:6.1f} path{t['path']:6.1f} f{t['frames'][0]}-{t['frames'][-1]}")
