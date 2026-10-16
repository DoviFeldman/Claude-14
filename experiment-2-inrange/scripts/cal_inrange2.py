import cv2,numpy as np
cap=cv2.VideoCapture("clean.mp4"); frames=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY))
cap.release()
N=len(frames); K=3
tstd=np.load("tstd.npy")
mask_keep=(tstd<=max(8.0,np.percentile(tstd,92))).astype(np.uint8)  # keep stable pixels
mask_keep=cv2.erode(mask_keep,np.ones((3,3),np.uint8),1)
def ma_amp(t,gain):
    cur=frames[t].astype(np.float32); prev=frames[t-K].astype(np.float32)
    blend=0.5*cur+0.5*(255-prev); amp=127.0+(blend-127.5)*gain
    return np.clip(amp,0,255).astype(np.uint8)
def detect(amp,dark,bright,amin=2,amax=70,maxside=14):
    m=cv2.bitwise_or(cv2.inRange(amp,0,dark),cv2.inRange(amp,bright,255))
    m=cv2.bitwise_and(m,m,mask=mask_keep)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,st,ce=cv2.connectedComponentsWithStats(m,8); d=[]
    for i in range(1,n):
        a=st[i,4]; w=st[i,2]; h=st[i,3]
        if amin<=a<=amax and max(w,h)<=maxside and a/float(max(w*h,1))>=0.3 and max(w,h)/float(max(min(w,h),1))<=3.5:
            d.append((ce[i][0],ce[i][1],a))
    return d
test_fly=[312,318,324,470,476,425,430,80,82,276]; test_quiet=[160,170,185,195,230,245,340,355]
print("WITH instability mask:")
for gain in [4,5,6]:
    for dark,bright in [(95,160),(90,165),(85,170)]:
        fc=np.mean([len(detect(ma_amp(t,gain),dark,bright)) for t in test_fly])
        qc=np.mean([len(detect(ma_amp(t,gain),dark,bright)) for t in test_quiet])
        print(f"  gain{gain} <{dark}/>{bright}: fly {fc:.1f}  quiet {qc:.1f}  ratio {fc/max(qc,0.01):.1f}")
