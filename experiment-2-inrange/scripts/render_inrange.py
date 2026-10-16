import cv2,numpy as np,json
D=json.load(open("fly_tracks_inrange.json")); tr=D["tracks"]; meta=D["meta"]
N=meta["N"];W=meta["W"];H=meta["H"];fps=meta["fps"]
rng=np.random.default_rng(3)
colors={t["id"]:tuple(int(c) for c in rng.integers(60,255,3)) for t in tr}
TAIL=25
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
vw=cv2.VideoWriter("tracked_inrange.mp4",fourcc,fps,(W,H))
byframe={}
for t in tr:
    for f,x,y in zip(t["frames"],t["x"],t["y"]): byframe.setdefault(f,[]).append(1)
cap=cv2.VideoCapture("clean.mp4"); f=0
while True:
    ok,fr=cap.read()
    if not ok: break
    vis=fr.copy()
    for t in tr:
        col=colors[t["id"]]
        pts=[(ff,x,y) for ff,x,y in zip(t["frames"],t["x"],t["y"]) if f-TAIL<=ff<=f]
        for k in range(1,len(pts)):
            cv2.line(vis,(int(pts[k-1][1]),int(pts[k-1][2])),(int(pts[k][1]),int(pts[k][2])),col,2)
        cur=[(x,y) for ff,x,y in zip(t["frames"],t["x"],t["y"]) if ff==f]
        for x,y in cur:
            cv2.circle(vis,(int(x),int(y)),13,col,2)
            cv2.putText(vis,f"#{t['id']}",(int(x)+14,int(y)-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)
    cv2.putText(vis,f"inRange+motion-amp  frame {f}  active {len(byframe.get(f,[]))}  tracks {len(tr)}",
                (20,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
    vw.write(vis); f+=1
cap.release(); vw.release()
# temporal path map
bg=cv2.imread("bg_median.png"); canvas=(bg*0.5).astype(np.uint8)
for t in tr:
    fs=np.array(t["frames"]); xs=t["x"]; ys=t["y"]
    for k in range(1,len(xs)):
        col=cv2.applyColorMap(np.uint8([[fs[k]/N*255]]),cv2.COLORMAP_JET)[0,0].tolist()
        cv2.line(canvas,(int(xs[k-1]),int(ys[k-1])),(int(xs[k]),int(ys[k])),col,2)
    cv2.circle(canvas,(int(xs[0]),int(ys[0])),4,(255,255,255),-1)
cv2.putText(canvas,f"inRange+motion-amp: {len(tr)} tracks (color=time)",(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
cv2.imwrite("paths_inrange.png",canvas)
print("wrote tracked_inrange.mp4 + paths_inrange.png  tracks:",len(tr))
# also report per-track for judgement
for t in tr: print(f"  #{t['id']:3d} hits{t['hits']:3d} span{t['span']:3d} net{t['net']:6.1f} path{t['path']:6.1f} f{t['frames'][0]}-{t['frames'][-1]}")
