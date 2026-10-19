import cv2, numpy as np, json
from scipy.signal import butter, filtfilt
meta=json.load(open("meta_hq.json")); N=meta["N"]; W=meta["W"]; H=meta["H"]; fps=meta["fps"]
DW,DH=640,360
cap=cv2.VideoCapture("clean_hq.mp4"); frames=[]; small=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    frames.append(fr)
    small.append(cv2.cvtColor(cv2.resize(fr,(DW,DH)),cv2.COLOR_BGR2GRAY))
cap.release()
S=np.stack(small).astype(np.float32)
lo,hi=1.5,13.0
b,a=butter(2,[lo/(fps/2),hi/(fps/2)],btype="band")
filt=filtfilt(b,a,S,axis=0).astype(np.float32)
# per-pixel in-band noise -> z-score so flies (strong transient) pop, noise floor stays dark
noise=np.maximum(filt.std(axis=0),0.6)
Z=np.abs(filt)/noise            # (N,DH,DW)
print("Z p50 %.2f p99 %.2f p99.9 %.2f max %.2f"%(np.percentile(Z,50),np.percentile(Z,99),np.percentile(Z,99.9),Z.max()))
# reveal: show z above ~3 (flies), scaled
def toheat(zf):
    m=np.clip((zf-3.0)/7.0*255,0,255).astype(np.uint8)  # z 3..10 -> 0..255
    return m
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
vw=cv2.VideoWriter("evm_reveal.mp4",fourcc,fps,(1280,720))
maxz=np.zeros((DH,DW),np.float32)
for t in range(N):
    m=toheat(Z[t]); maxz=np.maximum(maxz,Z[t])
    heat=cv2.applyColorMap(cv2.resize(m,(1280,720)),cv2.COLORMAP_TURBO)
    ctx=cv2.resize((frames[t]*0.4).astype(np.uint8),(1280,720))
    out=cv2.addWeighted(ctx,1.0,heat,0.9,0)
    cv2.putText(out,f"Eulerian Video Magnification (band {lo}-{hi}Hz, z-score)  frame {t}",(15,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    vw.write(out)
vw.release()
# clean max projection of z (fly paths)
mp=np.clip((maxz-3.0)/9.0*255,0,255).astype(np.uint8)
base=cv2.resize((cv2.imread("bg_hq.png")*0.55).astype(np.uint8),(DW,DH))
proj=cv2.addWeighted(base,1.0,cv2.applyColorMap(mp,cv2.COLORMAP_TURBO),0.85,0)
cv2.imwrite("evm_maxproj.png",cv2.resize(proj,(1280,720)))
# single sample frame at a known fly moment for inspection
for tf in [315]:
    m=toheat(Z[tf]); heat=cv2.applyColorMap(cv2.resize(m,(1280,720)),cv2.COLORMAP_TURBO)
    ctx=cv2.resize((frames[tf]*0.4).astype(np.uint8),(1280,720))
    cv2.imwrite(f"evm_frame{tf}.png",cv2.addWeighted(ctx,1.0,heat,0.9,0))
print("wrote evm_reveal.mp4, evm_maxproj.png, evm_frame315.png")
