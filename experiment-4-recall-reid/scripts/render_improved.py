import cv2,numpy as np,json
D=json.load(open("fly_tracks_improved.json")); tr=D["tracks"]; meta=D["meta"]
N=meta["N"];W=meta["W"];H=meta["H"];fps=meta["fps"]
rng=np.random.default_rng(1)
colors={t["id"]:tuple(int(c) for c in rng.integers(70,255,3)) for t in tr}
bg=cv2.imread("bg_hq.png"); canvas=(bg*0.5).astype(np.uint8)
for t in tr:
    col=colors[t["id"]]; xs=t["x"]; ys=t["y"]
    for k in range(1,len(xs)):
        cv2.line(canvas,(int(xs[k-1]),int(ys[k-1])),(int(xs[k]),int(ys[k])),col,2)
    cv2.circle(canvas,(int(xs[0]),int(ys[0])),6,(255,255,255),-1)
    cv2.putText(canvas,f"#{t['id']}",(int(xs[0])+6,int(ys[0])-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
cv2.putText(canvas,f"IMPROVED: union+stitch = {len(tr)} tracks (each color = one fly)",(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)
cv2.imwrite("paths_improved.png",canvas)
# overlay video
TAIL=40
vw=cv2.VideoWriter("tracked_improved.mp4",cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
byframe={}
for t in tr:
    for f in t["frames"]: byframe[f]=byframe.get(f,0)+1
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
    cv2.putText(vis,f"IMPROVED (union+stitch)  frame {f}  active {byframe.get(f,0)}  tracks {len(tr)}",
                (20,45),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
    vw.write(vis); f+=1
cap.release(); vw.release()
print("wrote paths_improved.png + tracked_improved.mp4")
