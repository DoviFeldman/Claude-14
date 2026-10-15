import cv2, numpy as np, json
from scipy.optimize import linear_sum_assignment
meta=json.load(open("meta.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
dets_per=json.load(open("dets_mog2.json"))
WARMUP=30
class KF:
    def __init__(s,x,y):
        s.x=np.array([x,y,0,0],float); s.P=np.eye(4)*10
        s.F=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],float)
        s.H=np.array([[1,0,0,0],[0,1,0,0]],float)
        s.Q=np.eye(4)*0.8; s.Q[2,2]=s.Q[3,3]=2.0; s.R=np.eye(2)*4
    def predict(s): s.x=s.F@s.x; s.P=s.F@s.P@s.F.T+s.Q; return s.x[:2]
    def update(s,z):
        y=z-s.H@s.x; S=s.H@s.P@s.H.T+s.R; K=s.P@s.H.T@np.linalg.inv(S)
        s.x=s.x+K@y; s.P=(np.eye(4)-K@s.H)@s.P
class Track:
    _id=0
    def __init__(s,d,fr):
        s.id=Track._id; Track._id+=1; s.kf=KF(d[0],d[1])
        s.hits=1; s.miss=0; s.pts={fr:(d[0],d[1])}
    def predict(s): return s.kf.predict()
    def update(s,d,fr): s.kf.update(np.array(d[:2],float)); s.hits+=1; s.miss=0; s.pts[fr]=(d[0],d[1])
GATE=45.0; tracks=[]; finished=[]
for fr in range(WARMUP,N):
    dets=dets_per[fr]
    preds=[t.predict() for t in tracks]
    if tracks and dets:
        C=np.hypot.outer if False else None
        C=np.zeros((len(tracks),len(dets)))
        for i,p in enumerate(preds):
            for j,d in enumerate(dets): C[i,j]=np.hypot(p[0]-d[0],p[1]-d[1])
        ri,ci=linear_sum_assignment(C); at=set(); ad=set()
        for i,j in zip(ri,ci):
            if C[i,j]<=GATE: tracks[i].update(dets[j],fr); at.add(i); ad.add(j)
        for j,d in enumerate(dets):
            if j not in ad: tracks.append(Track(d,fr))
        for i,t in enumerate(tracks):
            if i<len(preds) and i not in at: t.miss+=1
    else:
        for t in tracks: t.miss+=1
        for d in dets: tracks.append(Track(d,fr))
    alive=[]
    for t in tracks:
        (finished if t.miss>10 else alive).append(t)
    tracks=alive
finished+=tracks
fly=[]
for t in finished:
    frs=sorted(t.pts)
    if len(frs)<4: continue
    xs=np.array([t.pts[f][0] for f in frs]); ys=np.array([t.pts[f][1] for f in frs])
    span=frs[-1]-frs[0]+1; net=np.hypot(xs[-1]-xs[0],ys[-1]-ys[0])
    path=np.hypot(np.diff(xs),np.diff(ys)).sum()
    if span<5 or (path<16 and net<12): continue
    fly.append(dict(id=int(t.id),frames=[int(f) for f in frs],
        x=[round(float(v),1) for v in xs],y=[round(float(v),1) for v in ys],
        net=round(float(net),1),path=round(float(path),1),span=int(span),hits=len(frs)))
fly.sort(key=lambda d:-d["hits"])
json.dump(dict(meta=meta,tracks=fly),open("fly_tracks.json","w"))
print("fly tracks:",len(fly),"  total detections used:",sum(len(d) for d in dets_per[WARMUP:]))
for t in fly:
    print(f"  id{t['id']:3d} hits{t['hits']:3d} span{t['span']:3d} net{t['net']:6.1f} path{t['path']:6.1f} frames {t['frames'][0]}-{t['frames'][-1]}")
