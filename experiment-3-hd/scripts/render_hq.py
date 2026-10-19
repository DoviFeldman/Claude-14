import cv2,numpy as np,json
D=json.load(open("fly_tracks_hq.json")); tr=D["tracks"]; meta=D["meta"]
N=meta["N"];W=meta["W"];H=meta["H"];fps=meta["fps"]
rng=np.random.default_rng(7)
colors={t["id"]:tuple(int(c) for c in rng.integers(60,255,3)) for t in tr}
# path map
bg=cv2.imread("bg_hq.png"); canvas=(bg*0.5).astype(np.uint8)
for t in tr:
    fs=np.array(t["frames"]); xs=t["x"]; ys=t["y"]
    for k in range(1,len(xs)):
        col=cv2.applyColorMap(np.uint8([[fs[k]/N*255]]),cv2.COLORMAP_JET)[0,0].tolist()
        cv2.line(canvas,(int(xs[k-1]),int(ys[k-1])),(int(xs[k]),int(ys[k])),col,2)
    cv2.circle(canvas,(int(xs[0]),int(ys[0])),5,(255,255,255),-1)
cv2.putText(canvas,f"HD: {len(tr)} fly tracks (color=time, blue->red)",(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),2)
cv2.imwrite("paths_hq.png",canvas)
# overlay video
TAIL=30
vw=cv2.VideoWriter("tracked_hq.mp4",cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
byframe={}
for t in tr:
    for f in t["frames"]: byframe.setdefault(f,0); byframe[f]+=1
cap=cv2.VideoCapture("clean_hq.mp4"); f=0
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
            cv2.circle(vis,(int(x),int(y)),15,col,2)
            cv2.putText(vis,f"#{t['id']}",(int(x)+17,int(y)-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
    cv2.putText(vis,f"HD MOG2+Kalman+Hungarian  frame {f}  active {byframe.get(f,0)}  tracks {len(tr)}",
                (20,45),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
    vw.write(vis); f+=1
cap.release(); vw.release()
conc=[byframe.get(f,0) for f in range(N)]
print("wrote paths_hq.png + tracked_hq.mp4  max concurrent flies:",max(conc),"frames>=2:",sum(c>=2 for c in conc))
