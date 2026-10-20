import cv2, numpy as np, json, time
from scipy.signal import butter, filtfilt
meta=json.load(open("meta_hq.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
D=json.load(open("fly_tracks_hq.json"))["tracks"]
cap=cv2.VideoCapture("clean_hq.mp4"); G=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    G.append(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY))
cap.release(); G=np.stack(G)
bg=cv2.imread("bg_hq.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)
R=18  # ROI half-size

def roi_stack(track):
    """track-stabilized ROIs: body ~centered, so wing flicker stands out."""
    frs=track["frames"]; xs=track["x"]; ys=track["y"]; stack=[]; dark=[]
    for f,x,y in zip(frs,xs,ys):
        x,y=int(round(x)),int(round(y))
        if x-R<0 or y-R<0 or x+R>=W or y+R>=H: continue
        roi=G[f,y-R:y+R,x-R:x+R].astype(np.float32)
        bgr=bg[y-R:y+R,x-R:x+R]
        stack.append(roi); dark.append(np.clip(bgr-roi,0,255))  # darkness = fly presence
    return np.stack(stack) if stack else None, np.stack(dark) if dark else None

results=[]
t0=time.perf_counter()
# use the longest tracks
for track in D[:6]:
    S,Dk=roi_stack(track)
    if S is None or len(S)<12: continue
    # Method A: high-freq temporal band on stabilized ROI = wing-beat energy
    b,a=butter(2,[8.0/(fps/2),14.0/(fps/2)],btype="band")
    hi=filtfilt(b,a,S,axis=0)
    wing_energy=hi.std(axis=0)                     # where does fast flicker concentrate
    # Method B: body = mean darkness (compact dark core)
    body_map=Dk.mean(axis=0)
    # locate peaks
    we=cv2.GaussianBlur(wing_energy,(0,0),1.2); bm=cv2.GaussianBlur(body_map,(0,0),1.2)
    wy,wx=np.unravel_index(np.argmax(we),we.shape)
    by,bx=np.unravel_index(np.argmax(bm),bm.shape)
    sep=float(np.hypot(wx-bx,wy-by))
    results.append(dict(id=track["id"],n=len(S),sep=round(sep,1),
                        wing_peak=float(we.max()),body_peak=float(bm.max())))
    # visualize one: side by side body map, wing map, overlay
    def norm(m): return cv2.applyColorMap(np.clip(m/ (m.max()+1e-6)*255,0,255).astype(np.uint8),cv2.COLORMAP_INFERNO)
    big=lambda m: cv2.resize(m,(180,180),interpolation=cv2.INTER_NEAREST)
    bmi=big(norm(bm)); wei=big(norm(we))
    cv2.putText(bmi,"BODY (darkness)",(3,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
    cv2.putText(wei,"WING (hi-freq)",(3,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
    # overlay body(red)+wing(cyan) peaks on mean ROI
    mean_roi=cv2.cvtColor(big(cv2.resize(S.mean(axis=0).astype(np.uint8),(R*2,R*2))),cv2.COLOR_GRAY2BGR)
    sc=180/(R*2)
    cv2.circle(mean_roi,(int(bx*sc),int(by*sc)),6,(0,0,255),2)
    cv2.circle(mean_roi,(int(wx*sc),int(wy*sc)),6,(255,255,0),2)
    cv2.putText(mean_roi,f"id{track['id']} sep{sep:.1f}px",(3,172),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
    row=np.hstack([mean_roi,bmi,wei])
    cv2.imwrite(f"wing_id{track['id']}.png",row)
proc=time.perf_counter()-t0
print("wing analysis on",len(results),"tracks  (%.2f s, %.1f ms/track)"%(proc,proc/max(len(results),1)*1000))
for r in results: print("  ",r)
json.dump(results,open("wing_results.json","w"))
