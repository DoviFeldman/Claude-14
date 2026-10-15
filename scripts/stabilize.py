import cv2, numpy as np, json
SRC="source.mp4"; START=78
cap=cv2.VideoCapture(SRC)
fps=cap.get(cv2.CAP_PROP_FPS)
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# collect frames from START
frames=[]
i=0
while True:
    ok,fr=cap.read()
    if not ok: break
    if i>=START: frames.append(fr)
    i+=1
cap.release()
N=len(frames); print("clean frames:",N,"res",W,H,"fps",fps)

# --- Global alignment to reference (frame 0) using ORB + partial affine ---
ref_gray=cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
orb=cv2.ORB_create(2000)
kp_ref,des_ref=orb.detectAndCompute(ref_gray,None)
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
transforms=[np.eye(2,3,dtype=np.float32)]
for idx in range(1,N):
    g=cv2.cvtColor(frames[idx],cv2.COLOR_BGR2GRAY)
    kp,des=orb.detectAndCompute(g,None)
    M=np.eye(2,3,dtype=np.float32)
    if des is not None and des_ref is not None:
        matches=bf.match(des_ref,des)
        if len(matches)>=12:
            matches=sorted(matches,key=lambda m:m.distance)[:200]
            src=np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst=np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            Mest,inl=cv2.estimateAffinePartial2D(dst,src,method=cv2.RANSAC,ransacReprojThreshold=3)
            if Mest is not None: M=Mest.astype(np.float32)
    transforms.append(M)

# residual motion after stabilization (diagnostic)
stab=[cv2.warpAffine(frames[k],transforms[k],(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT) for k in range(N)]
res=[]
small=lambda im:cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(379,213))
for k in range(1,N):
    res.append(float(cv2.absdiff(small(stab[k]),small(stab[k-1])).mean()))
print("post-stab residual mean diff: %.3f (was ~0.18 raw baseline)"%np.mean(res))

# write stabilized clean color video
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
vw=cv2.VideoWriter("clean_stab.mp4",fourcc,fps,(W,H))
for im in stab: vw.write(im)
vw.release()
# also save stabilized gray stack (downsized not needed; keep full but as uint8 memmap)
np.save("transforms.npy",np.array(transforms))
meta=dict(N=N,W=W,H=H,fps=fps,start=START)
json.dump(meta,open("meta.json","w"))
# save a mid screenshot
cv2.imwrite("clean_mid.png",stab[N//2])
print("wrote clean_stab.mp4, meta.json")
