import cv2, numpy as np, json
from scipy.signal import butter, filtfilt
meta=json.load(open("meta_hq.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
DW,DH=640,360
cap=cv2.VideoCapture("clean_hq.mp4"); frames=[]; small=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    frames.append(fr); small.append(cv2.cvtColor(cv2.resize(fr,(DW,DH)),cv2.COLOR_BGR2GRAY))
cap.release()
S=np.stack(small).astype(np.float32)
b,a=butter(2,[1.5/(fps/2),13.0/(fps/2)],btype="band")
filt=filtfilt(b,a,S,axis=0).astype(np.float32)
noise=np.maximum(filt.std(axis=0),0.6); Z=np.abs(filt)/noise

# ---- (1) EVM tracking overlay ----
D=json.load(open("fly_tracks_evm.json"))["tracks"]
rng=np.random.default_rng(11); colors={t["id"]:tuple(int(c) for c in rng.integers(80,255,3)) for t in D}
TAIL=30
vw=cv2.VideoWriter("tracked_evm.mp4",cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
byf={}
for t in D:
    for f in t["frames"]: byf[f]=byf.get(f,0)+1
for f in range(N):
    vis=frames[f].copy()
    for t in D:
        col=colors[t["id"]]
        pts=[(ff,x,y) for ff,x,y in zip(t["frames"],t["x"],t["y"]) if f-TAIL<=ff<=f]
        for k in range(1,len(pts)):
            cv2.line(vis,(int(pts[k-1][1]),int(pts[k-1][2])),(int(pts[k][1]),int(pts[k][2])),col,3)
        for x,y in [(x,y) for ff,x,y in zip(t["frames"],t["x"],t["y"]) if ff==f]:
            cv2.circle(vis,(int(x),int(y)),16,col,3); cv2.putText(vis,f"#{t['id']}",(int(x)+18,int(y)-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
    cv2.putText(vis,f"EVM + Kalman/Hungarian  frame {f}  active {byf.get(f,0)}  tracks {len(D)}",(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
    vw.write(vis)
vw.release()

# ---- (2) GLOW ON BLACK: flies glow white on black (your idea, via EVM z-field) ----
vw2=cv2.VideoWriter("evm_glow_black.mp4",cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,720))
for t in range(N):
    g=np.clip((Z[t]-3.0)/6.0*255,0,255).astype(np.uint8)   # fly signal -> white
    g=cv2.resize(g,(1280,720))
    glow=cv2.GaussianBlur(g,(0,0),6)                         # bloom/glow
    out=np.clip(g.astype(np.int32)+glow.astype(np.int32),0,255).astype(np.uint8)
    out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
    # tint the glow slightly cyan-white for a nicer look
    out[:,:,0]=np.clip(out[:,:,0].astype(np.int32)+30,0,255)
    cv2.putText(out,f"EVM glow-on-black  frame {t}",(15,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,255,255),2)
    vw2.write(out)
vw2.release()
print("wrote tracked_evm.mp4 + evm_glow_black.mp4")
