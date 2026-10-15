import cv2, numpy as np, json
D=json.load(open("fly_tracks.json")); meta=D["meta"]; tr=D["tracks"]
N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
# color per track
rng=np.random.default_rng(7)
colors={t["id"]:tuple(int(c) for c in rng.integers(60,255,3)) for t in tr}
# index positions by frame
byframe={}
for t in tr:
    for f,x,y in zip(t["frames"],t["x"],t["y"]):
        byframe.setdefault(f,[]).append((t["id"],x,y))
TAIL=25
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
vw=cv2.VideoWriter("tracked_coords.mp4",fourcc,fps,(W,H))
cap=cv2.VideoCapture("clean.mp4"); f=0
while True:
    ok,fr=cap.read()
    if not ok: break
    vis=fr.copy()
    # draw each track's recent tail up to current frame
    for t in tr:
        col=colors[t["id"]]
        pts=[(ff,x,y) for ff,x,y in zip(t["frames"],t["x"],t["y"]) if f-TAIL<=ff<=f]
        for k in range(1,len(pts)):
            a=int(255*(k/len(pts)))
            cv2.line(vis,(int(pts[k-1][1]),int(pts[k-1][2])),(int(pts[k][1]),int(pts[k][2])),col,2)
        # current marker
        cur=[(x,y) for ff,x,y in zip(t["frames"],t["x"],t["y"]) if ff==f]
        for x,y in cur:
            cv2.circle(vis,(int(x),int(y)),13,col,2)
            cv2.putText(vis,f"#{t['id']}",(int(x)+15,int(y)-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
    active=len(byframe.get(f,[]))
    cv2.putText(vis,f"frame {f}  active flies: {active}  total tracks: {len(tr)}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
    vw.write(vis); f+=1
cap.release(); vw.release()
print("wrote tracked_coords.mp4")

# temporal path map: all paths colored by time (voxel t->color 2D projection)
bg=cv2.imread("bg_median.png"); bg=cv2.cvtColor(bg,cv2.COLOR_GRAY2BGR) if bg.ndim==2 else bg
canvas=(bg*0.5).astype(np.uint8)
for t in tr:
    fsn=np.array(t["frames"]); xs=t["x"]; ys=t["y"]
    for k in range(1,len(xs)):
        tc=fsn[k]/N
        col=cv2.applyColorMap(np.uint8([[tc*255]]),cv2.COLORMAP_JET)[0,0].tolist()
        cv2.line(canvas,(int(xs[k-1]),int(ys[k-1])),(int(xs[k]),int(ys[k])),col,2)
    cv2.circle(canvas,(int(xs[0]),int(ys[0])),4,(255,255,255),-1)
cv2.putText(canvas,"Fly flight paths (color=time, blue=early -> red=late)",(20,35),
            cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
cv2.imwrite("paths_temporal.png",canvas)
print("wrote paths_temporal.png")
