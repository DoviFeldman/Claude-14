import cv2, numpy as np, json
meta=json.load(open("meta_hq.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]
tstd=np.load("tstd_hq.npy")
bg=cv2.imread("bg_hq.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)
jit=cv2.dilate((tstd>max(9.0,np.percentile(tstd,94))).astype(np.uint8),np.ones((3,3),np.uint8),1)
keep=(1-jit).astype(bool)
sigma=np.maximum(tstd,2.0)  # floor noise
AMIN,AMAX,MAXSIDE=2,140,24
def shape_ok(a,w,h): return (AMIN<=a<=AMAX) and max(w,h)<=MAXSIDE and a/float(max(w*h,1))>=0.25 and max(w,h)/float(max(min(w,h),1))<=4.0
def blobs(mask):
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,st,ce=cv2.connectedComponentsWithStats(mask,8)
    return [(ce[i][0],ce[i][1]) for i in range(1,n) if shape_ok(st[i,4],st[i,2],st[i,3])]
cap=cv2.VideoCapture("clean_hq.mp4"); G=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    G.append(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY))
cap.release(); G=np.stack(G)
# sample frames for calibration
test=list(range(40,N,20))
for C in [5,6,7,8]:
    for fl in [8,10]:
        cnt=[]
        for t in test:
            f=G[t].astype(np.float32); z=np.abs(f-bg)/sigma
            m=((z>C)&((np.abs(f-bg))>fl)&keep).astype(np.uint8)*255
            cnt.append(len(blobs(m)))
        print(f"C={C} floor={fl}:  mean det/frame {np.mean(cnt):.2f}  max {max(cnt)}")
