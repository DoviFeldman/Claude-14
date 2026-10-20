#!/usr/bin/env python3
"""
Live, real-time, CAUSAL Eulerian-magnification fly tracker.
- Causal IIR temporal band-pass (no future frames) -> works on a live stream.
- Online per-pixel noise estimate -> auto-suppresses static wall noise, no precomputed mask.
- Blob detect + Kalman + Hungarian, all streaming.
Runs at full resolution. Basic-hardware friendly: ~a dozen numpy ops/frame, a few MB of state.

Usage:
  python live_evm_tracker.py --source 0            # webcam
  python live_evm_tracker.py --source phone.mp4    # a video file (e.g. your uncompressed clip)
  python live_evm_tracker.py --source clip.mp4 --out out.mp4 --no-display   # headless render
Keys: q = quit.
"""
import cv2, numpy as np, time, argparse
from scipy.optimize import linear_sum_assignment

def alpha_for(fc, fs):           # first-order lowpass coefficient for cutoff fc at sample-rate fs
    return 1.0 - np.exp(-2.0*np.pi*fc/fs)

class CausalEVM:
    """Per-pixel causal band-pass + online z-score. O(1) state per pixel.
    Optimized: preallocated scratch buffers + in-place numpy (no per-frame allocation).
    Noise floor = SLOW ungated EMA of band-power. A fly visits any pixel only ~3 frames,
    which barely moves a multi-second average, so the floor tracks noise (not flies) without
    needing a gate (gating froze chronically-spiky edge pixels into permanent false positives)."""
    def __init__(self, shape, fs, f_lo=1.5, f_hi=13.0, noise_tc=6.0, pfloor=1.0):
        self.a_hi=alpha_for(f_hi,fs); self.a_lo=alpha_for(f_lo,fs)
        self.b_pow=alpha_for(1.0/noise_tc, fs)   # SLOW noise-power EMA (~noise_tc sec)
        self.pfloor=pfloor                        # min noise power (avoids blow-up at ultra-quiet pixels)
        self.lp_hi=None; self.lp_lo=None; self.powv=None; self.n=0
    def _alloc(self, g):
        self.lp_hi=g.copy(); self.lp_lo=g.copy()
        self.powv=np.full_like(g, 4.0)            # start at a modest floor, converge from there
        self.band=np.empty_like(g); self.tmp=np.empty_like(g); self.z=np.empty_like(g)
    def update(self, gray_f):
        if self.lp_hi is None: self._alloc(gray_f)
        tmp,band=self.tmp,self.band
        np.subtract(gray_f,self.lp_hi,out=tmp); tmp*=self.a_hi; self.lp_hi+=tmp   # lp_hi += a_hi*(x-lp_hi)
        np.subtract(gray_f,self.lp_lo,out=tmp); tmp*=self.a_lo; self.lp_lo+=tmp   # lp_lo += a_lo*(x-lp_lo)
        np.subtract(self.lp_hi,self.lp_lo,out=band)                               # band = lp_hi-lp_lo
        # z from the PREVIOUS frame's noise floor (fly doesn't inflate its own z)
        np.maximum(self.powv,self.pfloor,out=tmp); np.sqrt(tmp,out=tmp)           # denom = sqrt(max(powv,floor))
        np.abs(band,out=self.z); self.z/=tmp                                      # z = |band|/denom
        np.multiply(band,band,out=tmp); tmp-=self.powv; tmp*=self.b_pow; self.powv+=tmp  # slow ungated EMA
        self.n+=1
        return band, self.z

class KF:
    def __init__(s,x,y):
        s.x=np.array([x,y,0,0],float);s.P=np.eye(4)*10
        s.F=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],float)
        s.H=np.array([[1,0,0,0],[0,1,0,0]],float);s.Q=np.eye(4)*0.8;s.Q[2,2]=s.Q[3,3]=2;s.R=np.eye(2)*4
    def predict(s): s.x=s.F@s.x;s.P=s.F@s.P@s.F.T+s.Q;return s.x[:2]
    def update(s,z):
        y=z-s.H@s.x;S=s.H@s.P@s.H.T+s.R;Kk=s.P@s.H.T@np.linalg.inv(S);s.x=s.x+Kk@y;s.P=(np.eye(4)-Kk@s.H)@s.P

class Tracker:
    """Online Kalman + Hungarian. Confirms after min_hits, coasts max_miss frames."""
    def __init__(s,gate=60,max_miss=25,min_hits=4):
        s.gate=gate;s.max_miss=max_miss;s.min_hits=min_hits;s.tracks=[];s.nid=0
    def step(s,dets):
        for t in s.tracks: t["pred"]=t["kf"].predict()
        if s.tracks and dets:
            C=np.zeros((len(s.tracks),len(dets)))
            for i,t in enumerate(s.tracks):
                for j,d in enumerate(dets): C[i,j]=np.hypot(t["pred"][0]-d[0],t["pred"][1]-d[1])
            ri,ci=linear_sum_assignment(C); at=set();ad=set()
            for i,j in zip(ri,ci):
                if C[i,j]<=s.gate:
                    t=s.tracks[i];t["kf"].update(np.array(dets[j]));t["miss"]=0;t["hits"]+=1
                    t["trail"].append((float(t["kf"].x[0]),float(t["kf"].x[1])));at.add(i);ad.add(j)
            for j,d in enumerate(dets):
                if j not in ad: s._birth(d)
            for i,t in enumerate(s.tracks):
                if i not in at: t["miss"]+=1
        else:
            for t in s.tracks: t["miss"]+=1
            for d in dets: s._birth(d)
        s.tracks=[t for t in s.tracks if t["miss"]<=s.max_miss]
        return [t for t in s.tracks if t["hits"]>=s.min_hits]
    def _birth(s,d):
        s.tracks.append(dict(kf=KF(d[0],d[1]),id=s.nid,miss=0,hits=1,trail=[(d[0],d[1])]));s.nid+=1

def detect(z, keep_z=4.5, amin=2, amax=140, maxside=24):
    m=(z>keep_z).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,st,ce=cv2.connectedComponentsWithStats(m,8); out=[]
    for i in range(1,n):
        a=st[i,4];w=st[i,2];h=st[i,3]
        if amin<=a<=amax and max(w,h)<=maxside and max(w,h)/float(max(min(w,h),1))<=4.0:
            out.append((float(ce[i][0]),float(ce[i][1])))
    return out

def moved(trail, min_disp=22.0):
    """A real fly travels; static noise doesn't. True if the recent trail displaced enough."""
    if len(trail)<4: return False
    xs=[p[0] for p in trail]; ys=[p[1] for p in trail]
    net=((xs[-1]-xs[0])**2+(ys[-1]-ys[0])**2)**0.5
    path=sum(((xs[k]-xs[k-1])**2+(ys[k]-ys[k-1])**2)**0.5 for k in range(1,len(xs)))
    return net>=min_disp or path>=min_disp*2

def run(source, out=None, display=True, scale=1.0, warmup=60, keep_z=6.0):
    cap=cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    fs=cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok,fr=cap.read()
    if not ok: print("no frames"); return
    if scale!=1.0: fr=cv2.resize(fr,None,fx=scale,fy=scale)
    H,W=fr.shape[:2]
    evm=CausalEVM((H,W),fs,noise_tc=8.0,pfloor=2.0); trk=Tracker()
    writer=None
    if out: writer=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'),fs,(W,H))
    colors={}; rng=np.random.default_rng(7)
    fcount=0; t_proc=0.0
    while True:
        if fcount>0:
            ok,fr=cap.read()
            if not ok: break
            if scale!=1.0: fr=cv2.resize(fr,None,fx=scale,fy=scale)
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY).astype(np.float32)
        t0=time.perf_counter()
        band,z=evm.update(g)
        dets=detect(z,keep_z) if fcount>=warmup else []
        live=trk.step(dets)
        t_proc+=time.perf_counter()-t0
        fcount+=1
        # ---- draw (only tracks that have actually moved = flies, not static noise) ----
        live=[t for t in live if moved(t["trail"])]
        vis=fr.copy()
        for t in live:
            if t["id"] not in colors: colors[t["id"]]=tuple(int(c) for c in rng.integers(80,255,3))
            c=colors[t["id"]]; tr=t["trail"][-30:]
            for k in range(1,len(tr)): cv2.line(vis,(int(tr[k-1][0]),int(tr[k-1][1])),(int(tr[k][0]),int(tr[k][1])),c,2)
            x,y=t["kf"].x[0],t["kf"].x[1]; cv2.circle(vis,(int(x),int(y)),14,c,2)
            cv2.putText(vis,f"#{t['id']}",(int(x)+15,int(y)-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
        inst_fps=fcount/max(t_proc,1e-6)
        cv2.putText(vis,f"LIVE causal EVM  {W}x{H}  flies:{len(live)}  proc:{inst_fps:.0f}fps",(15,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        if writer is not None: writer.write(vis)
        if display:
            cv2.imshow("live EVM tracker",vis)
            if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release()
    if writer is not None: writer.release()
    if display: cv2.destroyAllWindows()
    print(f"processed {fcount} frames @ {W}x{H}: {fcount/max(t_proc,1e-6):.1f} fps compute ({t_proc/fcount*1000:.2f} ms/frame)")
    return fcount/max(t_proc,1e-6), (W,H)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="0")
    ap.add_argument("--out",default=None)
    ap.add_argument("--no-display",dest="display",action="store_false")
    ap.add_argument("--scale",type=float,default=1.0)
    ap.add_argument("--keep-z",type=float,default=4.5)
    a=ap.parse_args()
    run(a.source,a.out,a.display,a.scale,keep_z=a.keep_z)
