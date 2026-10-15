import cv2, numpy as np, json
meta=json.load(open("meta.json")); fps=meta["fps"]; W=meta["W"]; H=meta["H"]
cap=cv2.VideoCapture("clean.mp4")
frames=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    frames.append(fr)
cap.release()
N=len(frames); print("frames",N)
K=3  # time shift in frames

fourcc=cv2.VideoWriter_fourcc(*'mp4v')
vw1=cv2.VideoWriter("ma_invert.mp4",fourcc,fps,(W,H))      # literal requested method
vw2=cv2.VideoWriter("ma_enhanced.mp4",fourcc,fps,(W,H))    # deviation-boosted, color-mapped

for t in range(N):
    cur=frames[t].astype(np.float32)
    prev=frames[max(0,t-K)].astype(np.float32)
    inv_prev=255.0-prev
    # 1) literal: average of current and inverted-shifted -> static ~127 gray
    blend=0.5*cur+0.5*inv_prev
    vw1.write(np.clip(blend,0,255).astype(np.uint8))
    # 2) enhanced: deviation from mid-gray, amplified, on a dark canvas with heat colormap
    dev=blend-127.5                      # 0 where static
    mag=np.abs(dev).mean(axis=2)         # motion magnitude (gray)
    mag=np.clip(mag*6.0,0,255).astype(np.uint8)   # amplify 6x
    heat=cv2.applyColorMap(mag,cv2.COLORMAP_TURBO)
    # dim original as context
    ctx=(frames[t]*0.25).astype(np.uint8)
    out=cv2.addWeighted(ctx,1.0,heat,0.9,0)
    vw2.write(out)
vw1.release(); vw2.release()
# sample screenshots mid-video
mid=N//2
cv2.imwrite("ma_invert_mid.png", cv2.imread.__self__ if False else None) if False else None
# re-render one mid frame for screenshot
cur=frames[mid].astype(np.float32); prev=frames[mid-K].astype(np.float32)
blend=0.5*cur+0.5*(255-prev)
cv2.imwrite("ma_invert_mid.png",np.clip(blend,0,255).astype(np.uint8))
dev=blend-127.5; mag=np.clip(np.abs(dev).mean(axis=2)*6,0,255).astype(np.uint8)
cv2.imwrite("ma_enhanced_mid.png",cv2.addWeighted((frames[mid]*0.25).astype(np.uint8),1.0,cv2.applyColorMap(mag,cv2.COLORMAP_TURBO),0.9,0))
print("wrote ma_invert.mp4, ma_enhanced.mp4 + screenshots; K=",K)
