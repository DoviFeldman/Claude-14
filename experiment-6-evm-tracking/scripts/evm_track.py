import cv2, numpy as np, json, time
from scipy.signal import butter, filtfilt
from scipy.optimize import linear_sum_assignment
meta=json.load(open("meta_hq.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
DW,DH=640,360; sx=W/DW; sy=H/DH
# instability mask at work res
tstd=cv2.resize(np.load("tstd_hq.npy"),(DW,DH))
jit=cv2.dilate((tstd>max(6.0,np.percentile(tstd,92))).astype(np.uint8),np.ones((3,3),np.uint8),1)
keep=(1-jit).astype(bool)

t_load=time.perf_counter()
cap=cv2.VideoCapture("clean_hq.mp4"); small=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    small.append(cv2.cvtColor(cv2.resize(fr,(DW,DH)),cv2.COLOR_BGR2GRAY))
cap.release()
S=np.stack(small).astype(np.float32); load_t=time.perf_counter()-t_load

# ---- EVM temporal bandpass (TIMED) ----
t0=time.perf_counter()
b,a=butter(2,[1.5/(fps/2),13.0/(fps/2)],btype="band")
filt=filtfilt(b,a,S,axis=0).astype(np.float32)
noise=np.maximum(filt.std(axis=0),0.6)
Z=np.abs(filt)/noise
evm_t=time.perf_counter()-t0

# ---- detect on EVM z-field (TIMED) ----
ZT=4.5; AMIN,AMAX,MAXSIDE=1,45,11
def shape_ok(ar,w,h): return (AMIN<=ar<=AMAX) and max(w,h)<=MAXSIDE and max(w,h)/float(max(min(w,h),1))<=4.0
t0=time.perf_counter()
dets=[]
for t in range(N):
    m=((Z[t]>ZT)&keep).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,st,ce=cv2.connectedComponentsWithStats(m,8)
    d=[(float(ce[i][0])*sx,float(ce[i][1])*sy) for i in range(1,n) if shape_ok(st[i,4],st[i,2],st[i,3])]
    dets.append(d)
det_t=time.perf_counter()-t0
cnt=np.array([len(d) for d in dets])
print("EVM det/frame mean %.2f  frames>=1 %d/%d"%(cnt.mean(),(cnt>=1).sum(),N))

# ---- Kalman + Hungarian + stitch (TIMED) ----
t0=time.perf_counter()
GATE=60;COAST=25;WARM=30
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
            if C[i,j]<=GATE:tr[i].kf.update(np.array(dd[j]));tr[i].miss=0;tr[i].pts[fr]=dd[j];at.add(i);ad.add(j)
        for j,d in enumerate(dd):
            if j not in ad:tr.append(mk(d,fr))
        for i,t in enumerate(tr):
            if i<len(pr) and i not in at:t.miss+=1
    else:
        for t in tr:t.miss+=1
        for d in dd:tr.append(mk(d,fr))
    al=[]
    for t in tr:(fin if t.miss>COAST else al).append(t)
    tr=al
fin+=tr
raw=[]
for t in fin:
    frs=sorted(t.pts);xs=np.array([t.pts[f][0] for f in frs]);ys=np.array([t.pts[f][1] for f in frs])
    span=frs[-1]-frs[0]+1;net=np.hypot(xs[-1]-xs[0],ys[-1]-ys[0]);path=np.hypot(np.diff(xs),np.diff(ys)).sum()
    # STRICT movement filter: must really travel (reject static wall noise)
    if len(frs)<5 or span<6: continue
    if net<35 and path<60: continue
    raw.append({"frames":list(map(int,frs)),"x":list(map(float,xs)),"y":list(map(float,ys))})
nb=len(raw)
def endinfo(t):
    f=t["frames"];x=t["x"];y=t["y"];m=min(4,len(x))
    return f[-1],x[-1],y[-1],(x[-1]-x[-m])/max(1,m-1),(y[-1]-y[-m])/max(1,m-1)
tracks=[dict(t) for t in raw];merged=True
while merged:
    merged=False;tracks.sort(key=lambda t:t["frames"][0])
    for i in range(len(tracks)):
        if tracks[i] is None:continue
        ef,ex,ey,vx,vy=endinfo(tracks[i]);spd=np.hypot(vx,vy);best=None;bc=1e9
        for j in range(len(tracks)):
            if j==i or tracks[j] is None:continue
            b2=tracks[j];sf,sxx,syy=b2["frames"][0],b2["x"][0],b2["y"][0];gap=sf-ef
            if gap<1 or gap>18:continue
            d=np.hypot(ex+vx*gap-sxx,ey+vy*gap-syy);tol=min(max(40.0,spd*gap*0.6),130.0)
            gv=np.array([sxx-ex,syy-ey])
            if spd>1.5 and np.linalg.norm(gv)>1 and (gv@np.array([vx,vy]))/(np.linalg.norm(gv)*spd+1e-6)<0.5:continue
            if d<=tol and d<bc:bc=d;best=j
        if best is not None:
            A=tracks[i];B=tracks[best];g0=A["frames"][-1];g1=B["frames"][0];gap=g1-g0
            ax,ay=A["x"][-1],A["y"][-1];bx,by=B["x"][0],B["y"][0]
            for gi in range(1,gap):
                A["frames"].append(int(g0+gi));A["x"].append(float(ax+(bx-ax)*gi/gap));A["y"].append(float(ay+(by-ay)*gi/gap))
            A["frames"]+=B["frames"];A["x"]+=B["x"];A["y"]+=B["y"];tracks[best]=None;merged=True
    tracks=[t for t in tracks if t is not None]
track_t=time.perf_counter()-t0
fly=[]
for k,t in enumerate(tracks):
    frs=t["frames"];xs=np.array(t["x"]);ys=np.array(t["y"])
    span=frs[-1]-frs[0]+1;net=np.hypot(xs[-1]-xs[0],ys[-1]-ys[0]);path=np.hypot(np.diff(xs),np.diff(ys)).sum()
    fly.append(dict(id=k,frames=[int(f) for f in frs],x=[round(float(v),1) for v in xs],y=[round(float(v),1) for v in ys],
                    net=round(float(net),1),path=round(float(path),1),span=int(span),hits=len(frs)))
fly.sort(key=lambda d:-d["hits"])
json.dump(dict(meta=meta,tracks=fly),open("fly_tracks_evm.json","w"))
print(f"\nEVM tracking: pre-stitch {nb} -> stitched {len(fly)} tracks  mean path {round(np.mean([t['path'] for t in fly]),0)}  longest {max(t['hits'] for t in fly)}")
print(f"\n=== TIMING (520 frames @ work-res {DW}x{DH}) ===")
print(f"  EVM bandpass filter:  {evm_t*1000:7.0f} ms total = {evm_t/N*1000:.2f} ms/frame = {N/evm_t:.0f} fps")
print(f"  detection on z-field: {det_t*1000:7.0f} ms total = {det_t/N*1000:.2f} ms/frame = {N/det_t:.0f} fps")
print(f"  Kalman+Hungarian+stitch: {track_t*1000:5.0f} ms total = {track_t/N*1000:.2f} ms/frame")
tot=evm_t+det_t+track_t
print(f"  TOTAL (excl load/decode): {tot:.2f} s = {N/tot:.0f} fps  (video is 30fps)")
print(f"  (filtfilt is non-causal/offline; needs whole clip)")
