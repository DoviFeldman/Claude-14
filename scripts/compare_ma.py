import cv2, numpy as np, json
from scipy.optimize import linear_sum_assignment
meta=json.load(open("meta.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
tstd=np.load("tstd.npy")
jitter=cv2.dilate((tstd>max(8.0,np.percentile(tstd,92))).astype(np.uint8),np.ones((3,3),np.uint8),1)
keep=(1-jitter).astype(bool)
def shape_ok(a,w,h):
    return (2<=a<=70) and max(w,h)<=16 and a/float(max(w*h,1))>=0.3 and max(w,h)/float(max(min(w,h),1))<=3.5
def detect_stream(path, invert=False):
    bs=cv2.createBackgroundSubtractorMOG2(history=150,varThreshold=30,detectShadows=False)
    dets=[]; cap=cv2.VideoCapture(path)
    while True:
        ok,fr=cap.read()
        if not ok: break
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        fg=bs.apply(g,learningRate=1.0/60); fg[~keep]=0
        fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
        n,lab,st,ce=cv2.connectedComponentsWithStats((fg>0).astype(np.uint8),8)
        d=[(float(ce[i][0]),float(ce[i][1])) for i in range(1,n) if shape_ok(st[i,4],st[i,2],st[i,3])]
        dets.append(d)
    cap.release(); return dets
def track(dets,warm=30):
    class KF:
        def __init__(s,x,y):
            s.x=np.array([x,y,0,0],float);s.P=np.eye(4)*10
            s.F=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],float)
            s.H=np.array([[1,0,0,0],[0,1,0,0]],float);s.Q=np.eye(4)*0.8;s.Q[2,2]=s.Q[3,3]=2;s.R=np.eye(2)*4
        def predict(s): s.x=s.F@s.x;s.P=s.F@s.P@s.F.T+s.Q;return s.x[:2]
        def update(s,z):
            y=z-s.H@s.x;S=s.H@s.P@s.H.T+s.R;K=s.P@s.H.T@np.linalg.inv(S);s.x=s.x+K@y;s.P=(np.eye(4)-K@s.H)@s.P
    tracks=[];fin=[];nid=[0]
    def mk(d,fr):
        t=type("T",(),{})();t.kf=KF(d[0],d[1]);t.id=nid[0];nid[0]+=1;t.miss=0;t.pts={fr:(d[0],d[1])};return t
    for fr in range(warm,len(dets)):
        dd=dets[fr];pr=[t.kf.predict() for t in tracks]
        if tracks and dd:
            C=np.zeros((len(tracks),len(dd)))
            for i,p in enumerate(pr):
                for j,d in enumerate(dd):C[i,j]=np.hypot(p[0]-d[0],p[1]-d[1])
            ri,ci=linear_sum_assignment(C);at=set();ad=set()
            for i,j in zip(ri,ci):
                if C[i,j]<=45:tracks[i].kf.update(np.array(dd[j]));tracks[i].miss=0;tracks[i].pts[fr]=dd[j];at.add(i);ad.add(j)
            for j,d in enumerate(dd):
                if j not in ad:tracks.append(mk(d,fr))
            for i,t in enumerate(tracks):
                if i<len(pr) and i not in at:t.miss+=1
        else:
            for t in tracks:t.miss+=1
            for d in dd:tracks.append(mk(d,fr))
        al=[]
        for t in tracks:(fin if t.miss>10 else al).append(t)
        tracks=al
    fin+=tracks;fly=[]
    for t in fin:
        frs=sorted(t.pts)
        if len(frs)<4:continue
        xs=np.array([t.pts[f][0] for f in frs]);ys=np.array([t.pts[f][1] for f in frs])
        span=frs[-1]-frs[0]+1;net=np.hypot(xs[-1]-xs[0],ys[-1]-ys[0]);path=np.hypot(np.diff(xs),np.diff(ys)).sum()
        if span<5 or (path<16 and net<12):continue
        fly.append((len(frs),span,round(net,1),round(path,1)))
    return fly
print("Detecting on RAW clean.mp4 ...")
raw=track(detect_stream("clean.mp4"))
print("Detecting on MOTION-AMPLIFIED ma_invert.mp4 ...")
ma=track(detect_stream("ma_invert.mp4"))
res=dict(raw_tracks=len(raw), raw_detpts=sum(f[0] for f in raw),
         ma_tracks=len(ma), ma_detpts=sum(f[0] for f in ma))
json.dump(res,open("compare_ma.json","w"))
print("\n=== RESULT ===")
print("RAW (no motion amp):   fly tracks =",len(raw),"  total track-points =",sum(f[0] for f in raw))
print("MOTION-AMPLIFIED:      fly tracks =",len(ma),"  total track-points =",sum(f[0] for f in ma))
print("raw track lengths:",sorted([f[0] for f in raw],reverse=True))
print("ma  track lengths:",sorted([f[0] for f in ma],reverse=True))
