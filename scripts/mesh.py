import cv2, numpy as np, json
from scipy.spatial import Delaunay
D=json.load(open("fly_tracks.json")); meta=D["meta"]; tr=D["tracks"]
N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
# positions + velocity per frame per track
byframe={}
for t in tr:
    fs=t["frames"]; xs=t["x"]; ys=t["y"]
    for k,f in enumerate(fs):
        vx=vy=0.0
        if k>0: vx=xs[k]-xs[k-1]; vy=ys[k]-ys[k-1]
        byframe.setdefault(f,[]).append((t["id"],xs[k],ys[k],vx,vy))
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
vw=cv2.VideoWriter("tracked_mesh.mp4",fourcc,fps,(W,H))
cap=cv2.VideoCapture("clean.mp4"); f=0
while True:
    ok,fr=cap.read()
    if not ok: break
    vis=(fr*0.6).astype(np.uint8)
    act=byframe.get(f,[])
    P=np.array([[a[1],a[2]] for a in act],float)
    # Delaunay mesh when >=3 concurrent flies
    if len(P)>=3:
        try:
            tri=Delaunay(P)
            for s in tri.simplices:
                for i in range(3):
                    p1=P[s[i]].astype(int); p2=P[s[(i+1)%3]].astype(int)
                    cv2.line(vis,tuple(p1),tuple(p2),(0,220,220),1,cv2.LINE_AA)
        except Exception: pass
    elif len(P)==2:
        cv2.line(vis,tuple(P[0].astype(int)),tuple(P[1].astype(int)),(0,220,220),1,cv2.LINE_AA)
    # nodes + velocity vectors
    for (tid,x,y,vx,vy) in act:
        cv2.circle(vis,(int(x),int(y)),6,(0,0,255),-1)
        cv2.arrowedLine(vis,(int(x),int(y)),(int(x+vx*4),int(y+vy*4)),(0,255,0),2,tipLength=0.4)
        cv2.putText(vis,f"#{tid}",(int(x)+8,int(y)-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(vis,f"frame {f}  mesh nodes: {len(P)}  (Delaunay of concurrent flies + velocity)",
                (20,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
    vw.write(vis); f+=1
cap.release(); vw.release()
print("wrote tracked_mesh.mp4")
# how often are >=3 flies concurrent
conc=[len(byframe.get(f,[])) for f in range(N)]
import numpy as np
print("max concurrent flies:",max(conc),"frames with >=2:",sum(c>=2 for c in conc),">=3:",sum(c>=3 for c in conc))
