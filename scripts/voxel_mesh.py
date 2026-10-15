import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np, json, cv2
from scipy.spatial import Delaunay
D=json.load(open("fly_tracks.json")); meta=D["meta"]; tr=D["tracks"]
N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]

# ---------- (A) Spatio-temporal MESH (Delaunay over all track points, 2D) ----------
allpts=[]; 
for t in tr:
    for x,y in zip(t["x"],t["y"]): allpts.append((x,y))
allpts=np.array(allpts)
bg=cv2.imread("bg_median.png")
canvas=(bg*0.5).astype(np.uint8)
if len(allpts)>=3:
    tri=Delaunay(allpts)
    for s in tri.simplices:
        # skip huge triangles (connect only local structure)
        pr=allpts[s]
        per=sum(np.hypot(*(pr[i]-pr[(i+1)%3])) for i in range(3))
        if per>600: continue
        for i in range(3):
            p1=pr[i].astype(int); p2=pr[(i+1)%3].astype(int)
            cv2.line(canvas,tuple(p1),tuple(p2),(0,200,255),1,cv2.LINE_AA)
for t in tr:
    for x,y in zip(t["x"],t["y"]):
        cv2.circle(canvas,(int(x),int(y)),3,(0,0,255),-1)
cv2.putText(canvas,"Spatio-temporal mesh: Delaunay triangulation of all fly detections",
            (20,35),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
cv2.imwrite("mesh_spatiotemporal.png",canvas)
print("wrote mesh_spatiotemporal.png  points:",len(allpts))

# ---------- (B) VOXEL grid (x,y,t) ----------
GX,GY,GT=48,27,60
vox=np.zeros((GX,GY,GT),np.int32)
pts3=[]
for t in tr:
    for f,x,y in zip(t["frames"],t["x"],t["y"]):
        gx=min(GX-1,int(x/W*GX)); gy=min(GY-1,int(y/H*GY)); gt=min(GT-1,int(f/N*GT))
        vox[gx,gy,gt]+=1; pts3.append((x,y,f,t["id"]))
pts3=np.array(pts3,float)
occ=int((vox>0).sum()); print("occupied voxels:",occ,"/",vox.size)

# 3D scatter view (x, time, y) with track-colored paths
fig=plt.figure(figsize=(13,9))
ax=fig.add_subplot(111,projection="3d")
rng=np.random.default_rng(7)
cmap={}
for t in tr:
    c=rng.random(3)*0.7+0.2
    xs=np.array(t["x"]); ys=np.array(t["y"]); fs=np.array(t["frames"])/fps
    ax.plot(xs, fs, H-ys, color=c, lw=2)
    ax.scatter(xs, fs, H-ys, color=c, s=14)
ax.set_xlabel("X (px)"); ax.set_ylabel("time (s)"); ax.set_zlabel("Y (flipped, px)")
ax.set_xlim(0,W); ax.set_zlim(0,H); ax.set_ylim(0,N/fps)
ax.set_title("Voxel/space-time view of fly tracks (X, time, Y)")
ax.view_init(elev=22,azim=-60)
plt.tight_layout(); plt.savefig("voxel_3d_view1.png",dpi=90)
ax.view_init(elev=18,azim=-120); plt.savefig("voxel_3d_view2.png",dpi=90)
print("wrote voxel_3d_view1/2.png")

# rotating MP4
import os; os.makedirs("vframes",exist_ok=True)
frames=[]
for i,az in enumerate(range(-90,90,3)):
    ax.view_init(elev=20,azim=az); 
    fn=f"vframes/v{i:03d}.png"; plt.savefig(fn,dpi=80); frames.append(fn)
plt.close()
print("rendered",len(frames),"rotation frames")
