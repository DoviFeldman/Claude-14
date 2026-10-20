import cv2, numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
meta=json.load(open("meta_hq.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
D=json.load(open("fly_tracks_hq.json"))["tracks"]
bg=cv2.imread("bg_hq.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)
cap=cv2.VideoCapture("clean_hq.mp4"); G=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    G.append(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY))
cap.release(); G=np.stack(G)

# ---- camera model (phone wide ~ 70 deg HFOV) ----
HFOV=np.deg2rad(68.0)
fpx=(W/2)/np.tan(HFOV/2)        # focal length in pixels
cx,cy=W/2,H/2
FLY_MM=9.0                      # assumed apparent fly size (body+blur), mm
# ---- kitchen box estimate (metres), from the photo ----
KIT=dict(width=3.0, height=2.4, depth=3.0)   # documented guess

def apparent_size(f,x,y,R=14):
    x,y=int(round(x)),int(round(y))
    if x-R<0 or y-R<0 or x+R>=W or y+R>=H: return None
    d=np.clip(bg[y-R:y+R,x-R:x+R]-G[f,y-R:y+R,x-R:x+R].astype(np.float32),0,255)
    m=(d>0.4*max(d.max(),8)).astype(np.uint8)
    n,lab,st,ce=cv2.connectedComponentsWithStats(m,8)
    if n<2: return None
    i=1+np.argmax(st[1:,4]); a=st[i,4]
    return float(np.sqrt(a))     # ~ linear apparent size in px

def smooth(v,k=7):
    v=np.array(v,float); 
    if len(v)<k: return v
    pad=np.pad(v,(k//2,k//2),mode='edge'); return np.convolve(pad,np.ones(k)/k,mode='valid')[:len(v)]

tracks3d=[]
for t in D:
    frs=t["frames"]; xs=t["x"]; ys=t["y"]
    sizes=[]
    for f,x,y in zip(frs,xs,ys):
        s=apparent_size(f,x,y); sizes.append(s)
    # fill missing sizes with median
    valid=[s for s in sizes if s]; 
    if len(valid)<4: continue
    med=np.median(valid); sizes=[s if s else med for s in sizes]
    sizes=smooth(sizes,7); sizes=np.clip(sizes,2.0,40.0)
    depth = FLY_MM*fpx/sizes/1000.0          # metres (bigger apparent -> closer)
    depth = np.clip(depth,0.35,KIT["depth"]) # clamp to kitchen volume
    depth = smooth(depth,9)
    X=(np.array(xs)-cx)/fpx*depth            # right (+)
    Y=(np.array(ys)-cy)/fpx*depth            # down (+)
    Z=depth                                   # into-scene
    tracks3d.append(dict(id=t["id"],X=list(map(float,X)),Y=list(map(float,Y)),Z=list(map(float,Z)),
                         frames=frs, path3d=float(np.sum(np.sqrt(np.diff(X)**2+np.diff(Y)**2+np.diff(Z)**2)))))
json.dump(dict(camera=dict(fpx=float(fpx),HFOV_deg=68.0,fly_mm=FLY_MM),kitchen=KIT,tracks=tracks3d),open("tracks3d.json","w"))
print("reconstructed",len(tracks3d),"3D tracks")
for t in tracks3d[:8]:
    print(f"  id{t['id']}: depth {min(t['Z']):.2f}-{max(t['Z']):.2f}m  3D path {t['path3d']:.2f}m")

# ---- 3D render (world: X right, Z depth, Y up) ----
fig=plt.figure(figsize=(13,9)); ax=fig.add_subplot(111,projection="3d")
rng=np.random.default_rng(7)
for t in tracks3d:
    c=rng.random(3)*0.7+0.2
    ax.plot(t["X"], t["Z"], [-y for y in t["Y"]], color=c, lw=2)
# draw kitchen box
w,h,dp=KIT["width"],KIT["height"],KIT["depth"]
for s,e in [((-w/2,0,-h/2),(w/2,0,-h/2)),((-w/2,dp,-h/2),(w/2,dp,-h/2))]:
    pass
ax.set_xlabel("X left-right (m)"); ax.set_ylabel("Z depth into kitchen (m)"); ax.set_zlabel("height (m)")
ax.set_title("Estimated 3D fly flight paths (monocular, size->depth)")
ax.set_xlim(-w/2,w/2); ax.set_ylim(0,dp); ax.set_zlim(-h/2,h/2)
ax.view_init(elev=18,azim=-70); plt.tight_layout(); plt.savefig("kitchen3d_view1.png",dpi=90)
ax.view_init(elev=12,azim=-120); plt.savefig("kitchen3d_view2.png",dpi=90)
import os; os.makedirs("k3frames",exist_ok=True)
fr=[]
for i,az in enumerate(range(-100,80,4)):
    ax.view_init(elev=16,azim=az); fn=f"k3frames/k{i:03d}.png"; plt.savefig(fn,dpi=80); fr.append(fn)
plt.close(); print("rendered",len(fr),"rotation frames")
