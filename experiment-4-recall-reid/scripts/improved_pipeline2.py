import cv2, numpy as np, json
from scipy.optimize import linear_sum_assignment
meta=json.load(open("meta_hq.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
tstd=np.load("tstd_hq.npy"); sigma=np.maximum(tstd,2.0)
bg=cv2.imread("bg_hq.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)
jit=cv2.dilate((tstd>max(9.0,np.percentile(tstd,94))).astype(np.uint8),np.ones((3,3),np.uint8),1)
keep=(1-jit).astype(bool)
def shape_ok(a,w,h): return (2<=a<=140) and max(w,h)<=24 and a/float(max(w*h,1))>=0.25 and max(w,h)/float(max(min(w,h),1))<=4.0
def blobs(mask):
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,st,ce=cv2.connectedComponentsWithStats(mask,8)
    return [(float(ce[i][0]),float(ce[i][1])) for i in range(1,n) if shape_ok(st[i,4],st[i,2],st[i,3])]
def dedup(pts,r=10):
    out=[]
    for p in pts:
        if all(np.hypot(p[0]-q[0],p[1]-q[1])>r for q in out): out.append(p)
    return out
cap=cv2.VideoCapture("clean_hq.mp4"); G=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    G.append(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY))
cap.release(); G=np.stack(G)
bs=cv2.createBackgroundSubtractorMOG2(history=150,varThreshold=25,detectShadows=False)
dets=[]
for t in range(N):
    fg=bs.apply(G[t],learningRate=1.0/60); fg[~keep]=0
    mog=blobs((fg>0).astype(np.uint8))
    f=G[t].astype(np.float32); z=np.abs(f-bg)/sigma
    mb=blobs((((z>5)&(np.abs(f-bg)>8)&keep).astype(np.uint8)*255))
    dets.append(dedup(mog+mb,10))
GATE=60; COAST=25; WARM=30
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
    span=frs[-1]-frs[0]+1; net=np.hypot(xs[-1]-xs[0],ys[-1]-ys[0]); path=np.hypot(np.diff(xs),np.diff(ys)).sum()
    if span<5 or (path<20 and net<15): continue
    raw.append({"frames":list(map(int,frs)),"x":list(map(float,xs)),"y":list(map(float,ys))})
n_before=len(raw)

# ---- CONSERVATIVE stitching with direction gate ----
MAXGAP=18
def endinfo(t):
    f=t["frames"];x=t["x"];y=t["y"]; m=min(4,len(x))
    vx=(x[-1]-x[-m])/max(1,m-1); vy=(y[-1]-y[-m])/max(1,m-1)
    return f[-1],x[-1],y[-1],vx,vy
tracks=[dict(t) for t in raw]
merged=True
while merged:
    merged=False
    tracks.sort(key=lambda t:t["frames"][0])
    for i in range(len(tracks)):
        if tracks[i] is None: continue
        ef,ex,ey,vx,vy=endinfo(tracks[i]); spd=np.hypot(vx,vy)
        best=None;bc=1e9
        for j in range(len(tracks)):
            if j==i or tracks[j] is None: continue
            b=tracks[j]; sf,sx,sy=b["frames"][0],b["x"][0],b["y"][0]
            gap=sf-ef
            if gap<1 or gap>MAXGAP: continue
            px=ex+vx*gap; py=ey+vy*gap
            d=np.hypot(px-sx,py-sy)
            tol=max(40.0, spd*gap*0.6); tol=min(tol,130.0)
            # direction gate: gap vector should align with velocity when moving
            gv=np.array([sx-ex,sy-ey]); 
            if spd>1.5 and np.linalg.norm(gv)>1:
                cosang=(gv@np.array([vx,vy]))/(np.linalg.norm(gv)*spd+1e-6)
                if cosang<0.5: continue   # >60deg off -> reject
            if d<=tol and d<bc: bc=d;best=j
        if best is not None:
            a=tracks[i];b=tracks[best]
            # interpolate across the gap so the track is continuous
            g0=a["frames"][-1]; g1=b["frames"][0]; gap=g1-g0
            ax,ay=a["x"][-1],a["y"][-1]; bx,by=b["x"][0],b["y"][0]
            for gi in range(1,gap):
                fr=g0+gi; al=gi/gap
                a["frames"].append(int(fr)); a["x"].append(float(ax+(bx-ax)*al)); a["y"].append(float(ay+(by-ay)*al))
            a["frames"]+=b["frames"]; a["x"]+=b["x"]; a["y"]+=b["y"]
            tracks[best]=None; merged=True
    tracks=[t for t in tracks if t is not None]
fly=[]
for k,t in enumerate(tracks):
    frs=t["frames"];xs=np.array(t["x"]);ys=np.array(t["y"])
    span=frs[-1]-frs[0]+1; net=np.hypot(xs[-1]-xs[0],ys[-1]-ys[0]); path=np.hypot(np.diff(xs),np.diff(ys)).sum()
    fly.append(dict(id=k,frames=[int(f) for f in frs],x=[round(float(v),1) for v in xs],y=[round(float(v),1) for v in ys],
                    net=round(float(net),1),path=round(float(path),1),span=int(span),hits=len(frs)))
fly.sort(key=lambda d:-d["hits"])
json.dump(dict(meta=meta,tracks=fly),open("fly_tracks_improved.json","w"))
base=json.load(open("fly_tracks_hq.json"))["tracks"]
mp=lambda ts: round(float(np.mean([t["path"] for t in ts])),0)
print(f"HD baseline: {len(base)} tracks, mean path {mp(base)}, longest {max(t['hits'] for t in base)}")
print(f"pre-stitch:  {n_before} tracks")
print(f"stitched:    {len(fly)} tracks, mean path {mp(fly)}, longest {max(t['hits'] for t in fly)}  (merged {n_before-len(fly)})")
for t in fly[:14]: print(f"  id{t['id']:3d} hits{t['hits']:3d} span{t['span']:3d} path{t['path']:7.1f} f{t['frames'][0]}-{t['frames'][-1]}")
