import cv2, numpy as np, json
SRC="source.mp4"; START=78
cap=cv2.VideoCapture(SRC)
fps=cap.get(cv2.CAP_PROP_FPS)
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames=[]; i=0
while True:
    ok,fr=cap.read()
    if not ok: break
    if i>=START: frames.append(fr)
    i+=1
cap.release()
N=len(frames)
# optional light translation-only stabilization via phase correlation to ref
ref=cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY).astype(np.float32)
def shift_of(g):
    (dx,dy),resp=cv2.phaseCorrelate(ref, g.astype(np.float32))
    return dx,dy,resp
raw_res=[]; 
small=lambda im:cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(379,213))
for k in range(1,N):
    raw_res.append(float(cv2.absdiff(small(frames[k]),small(frames[k-1])).mean()))
print("RAW trimmed residual mean diff: %.3f  median: %.3f  max: %.3f"%(np.mean(raw_res),np.median(raw_res),np.max(raw_res)))
# phase-correlation translation stabilization
stab=[]; shifts=[]
for k in range(N):
    g=cv2.cvtColor(frames[k],cv2.COLOR_BGR2GRAY)
    dx,dy,resp=shift_of(g)
    shifts.append((dx,dy))
    M=np.float32([[1,0,-dx],[0,1,-dy]])
    stab.append(cv2.warpAffine(frames[k],M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT))
pres=[]
for k in range(1,N):
    pres.append(float(cv2.absdiff(small(stab[k]),small(stab[k-1])).mean()))
print("PHASE-CORR translation stabilized residual mean: %.3f  median: %.3f"%(np.mean(pres),np.median(pres)))
sh=np.array(shifts); print("max |shift| dx,dy:",round(np.abs(sh[:,0]).max(),2),round(np.abs(sh[:,1]).max(),2),"px")

use_stab = np.mean(pres) < np.mean(raw_res)*0.95
print("=> using", "PHASE-STAB" if use_stab else "RAW (already stable)")
out = stab if use_stab else frames
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
vw=cv2.VideoWriter("clean.mp4",fourcc,fps,(W,H))
for im in out: vw.write(im)
vw.release()
json.dump(dict(N=N,W=W,H=H,fps=fps,start=START,stabilized=bool(use_stab)),open("meta.json","w"))
print("wrote clean.mp4  N=",N)
