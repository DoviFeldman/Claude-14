import cv2, numpy as np, json, time
from scipy.optimize import linear_sum_assignment
meta=json.load(open("meta.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
K=3; GAIN=5; DARK=95; BRIGHT=160; WARM=30
tstd=np.load("tstd.npy")
mask_keep=(tstd<=max(8.0,np.percentile(tstd,92))).astype(np.uint8)
mask_keep=cv2.erode(mask_keep,np.ones((3,3),np.uint8),1)

# ---- load gray frames once (for fair in-memory timing) ----
cap=cv2.VideoCapture("clean.mp4"); gray=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    gray.append(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY))
cap.release()
gray=np.stack(gray)  # N,H,W uint8
print("loaded",gray.shape)

def shape_ok(a,w,h):
    return (2<=a<=70) and max(w,h)<=14 and a/float(max(w*h,1))>=0.3 and max(w,h)/float(max(min(w,h),1))<=3.5

# ================= inRange detector (motion-amp + cv2.inRange) =================
def detect_inrange(t):
    cur=gray[t].astype(np.float32); prev=gray[max(0,t-K)].astype(np.float32)
    blend=0.5*cur+0.5*(255-prev)
    amp=np.clip(127.0+(blend-127.5)*GAIN,0,255).astype(np.uint8)   # motion AMPLIFICATION
    m=cv2.bitwise_or(cv2.inRange(amp,0,DARK),cv2.inRange(amp,BRIGHT,255))  # cv2.inRange color/value detector
    m=cv2.bitwise_and(m,m,mask=mask_keep)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,st,ce=cv2.connectedComponentsWithStats(m,8); d=[]
    for i in range(1,n):
        if shape_ok(st[i,4],st[i,2],st[i,3]): d.append((float(ce[i][0]),float(ce[i][1])))
    return d

# ================= MOG2 detector (baseline) =================
bs=cv2.createBackgroundSubtractorMOG2(history=150,varThreshold=30,detectShadows=False)
keep_mog=(1-cv2.dilate((tstd>max(8.0,np.percentile(tstd,92))).astype(np.uint8),np.ones((3,3),np.uint8),1)).astype(bool)
def detect_mog2(t):
    fg=bs.apply(gray[t],learningRate=1.0/60); fg[~keep_mog]=0
    fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,st,ce=cv2.connectedComponentsWithStats((fg>0).astype(np.uint8),8); d=[]
    for i in range(1,n):
        if shape_ok(st[i,4],st[i,2],st[i,3]): d.append((float(ce[i][0]),float(ce[i][1])))
    return d

# ---- BENCHMARK detector compute (exclude decode; frames in memory) ----
# warmup caches
for t in range(5): detect_inrange(t); detect_mog2(t)
t0=time.perf_counter(); dets_ir=[detect_inrange(t) for t in range(N)]; t1=time.perf_counter()
ir_time=t1-t0
bs=cv2.createBackgroundSubtractorMOG2(history=150,varThreshold=30,detectShadows=False)  # reset
t0=time.perf_counter(); dets_mg=[detect_mog2(t) for t in range(N)]; t1=time.perf_counter()
mg_time=t1-t0
print(f"\nDETECTOR SPEED (compute only, {N} frames @ {W}x{H}):")
print(f"  inRange(motion-amp): {ir_time*1000:.0f} ms total  = {ir_time/N*1000:.2f} ms/frame  = {N/ir_time:.0f} fps")
print(f"  MOG2 baseline:       {mg_time*1000:.0f} ms total  = {mg_time/N*1000:.2f} ms/frame  = {N/mg_time:.0f} fps")
print(f"  inRange is {mg_time/ir_time:.2f}x the speed of MOG2")

# ================= shared tracker =================
def track(dets):
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
        dd=dets[fr];pr=[t.kf.predict() for t in tr]
        if tr and dd:
            C=np.zeros((len(tr),len(dd)))
            for i,p in enumerate(pr):
                for j,d in enumerate(dd):C[i,j]=np.hypot(p[0]-d[0],p[1]-d[1])
            ri,ci=linear_sum_assignment(C);at=set();ad=set()
            for i,j in zip(ri,ci):
                if C[i,j]<=45:tr[i].kf.update(np.array(dd[j]));tr[i].miss=0;tr[i].pts[fr]=dd[j];at.add(i);ad.add(j)
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
    fin+=tr;fly=[]
    for t in fin:
        frs=sorted(t.pts)
        if len(frs)<4:continue
        xs=np.array([t.pts[f][0] for f in frs]);ys=np.array([t.pts[f][1] for f in frs])
        span=frs[-1]-frs[0]+1;net=np.hypot(xs[-1]-xs[0],ys[-1]-ys[0]);path=np.hypot(np.diff(xs),np.diff(ys)).sum()
        if span<5 or (path<16 and net<12):continue
        fly.append(dict(id=int(t.id),frames=[int(f) for f in frs],x=[round(float(v),1) for v in xs],
                        y=[round(float(v),1) for v in ys],net=round(float(net),1),path=round(float(path),1),span=int(span),hits=len(frs)))
    fly.sort(key=lambda d:-d["hits"])
    return fly

fly_ir=track(dets_ir); fly_mg=track(dets_mg)
json.dump(dict(meta=meta,tracks=fly_ir,params=dict(K=K,gain=GAIN,dark=DARK,bright=BRIGHT)),open("fly_tracks_inrange.json","w"))
print(f"\nTRACKS: inRange={len(fly_ir)}   MOG2={len(fly_mg)}")

# ---- accuracy vs MOG2 baseline: match tracks by space-time overlap ----
def centroids(t): 
    import numpy as np; return list(zip(t["frames"],t["x"],t["y"]))
def overlap(a,b):
    fa={f:(x,y) for f,x,y in centroids(a)}; fb={f:(x,y) for f,x,y in centroids(b)}
    common=set(fa)&set(fb)
    if not common: return 0,999
    ds=[np.hypot(fa[f][0]-fb[f][0],fa[f][1]-fb[f][1]) for f in common]
    return len(common),np.mean(ds)
matched=0; details=[]
used=set()
for a in fly_mg:
    best=None;bn=0;bd=999
    for j,b in enumerate(fly_ir):
        if j in used:continue
        nov,md=overlap(a,b)
        if nov>=3 and md<=30 and nov>bn: best=j;bn=nov;bd=md
    if best is not None: matched+=1;used.add(best);details.append((a["id"],fly_ir[best]["id"],bn,round(bd,1)))
print(f"MOG2 tracks matched by an inRange track (>=3 common frames, <=30px): {matched}/{len(fly_mg)}")
for mid,iid,nov,md in details: print(f"   MOG2#{mid} <-> inRange#{iid}  ({nov} common frames, {md}px mean dist)")
irpts=sum(t['hits'] for t in fly_ir); mgpts=sum(t['hits'] for t in fly_mg)
json.dump(dict(ir_tracks=len(fly_ir),mg_tracks=len(fly_mg),matched=matched,
    ir_points=irpts,mg_points=mgpts,ir_ms_per_frame=round(ir_time/N*1000,3),
    mg_ms_per_frame=round(mg_time/N*1000,3),speedup=round(mg_time/ir_time,2),
    ir_fps=round(N/ir_time),mg_fps=round(N/mg_time)),open("compare_inrange.json","w"))
print("saved compare_inrange.json")
