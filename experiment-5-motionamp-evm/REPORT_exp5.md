# Experiment 5 — Motion-amp tracking WITHOUT MOG2, + Eulerian Video Magnification

Two things in one: (A) drop MOG2 entirely and track from motion amplification, to chase the flies you still see on the white cabinets; (B) build real Eulerian Video Magnification to *reveal* those subtle flies.

---

## Part A — Motion-amplification tracking, no MOG2

**Pipeline:** motion-amp blend (inverted/time-shifted, gain 5) → `cv2.inRange` (dark+bright deviations) → instability mask + size caps → **Kalman + Hungarian** → track-stitching. No background model at all.

**Result on HD:**
| | MOG2 (Exp 3) | Improved (Exp 4) | **Motion-amp, no MOG2** |
|---|---|---|---|
| Tracks | 26 | 24 | **89** |
| Detections/frame | 1.24 | 1.21 | **2.62** |
| Coverage | upper frame | upper+lower | **whole frame, incl. white cabinets** |

**This explains the mystery.** Motion-amp lights up the white cabinets with many more fly paths — the ones you were seeing that MOG2 missed.

**Why MOG2 missed them:** MOG2 *adapts* (learning-rate ≈ 1/60, so it forgets in ~2 s). A fly that **hovers or lingers** near the cabinets sits in roughly one spot long enough that MOG2 **learns it into the background** and stops reporting it. Motion amplification has **no memory** — it only compares frame *t* to frame *t−3*, so any wing-buzz or drift re-reveals the fly every frame. That's exactly why a slow/hovering fly on a clean white cabinet is invisible to MOG2 but obvious to motion-amp.

**The trade-off:** motion-amp has no background model, so those 89 tracks include **more false positives** (handles, faucet, reflections) along with the real extra flies. It's **high recall, lower precision** — the mirror image of MOG2 (**high precision, lower recall**). The ideal production system fuses both: MOG2 for confident moving flies + motion-amp (or the z-score median-bg from Exp 4) for the hoverers, with tracking to clean up.

---

## Part B — Eulerian Video Magnification (EVM)

**What it is:** a 2012 MIT technique that amplifies tiny, near-invisible changes in video. Instead of *following* an object (Lagrangian), it fixes each pixel and watches how its value changes over **time** (Eulerian). It runs a **temporal band-pass filter** per pixel to keep only changes in a chosen frequency band, multiplies them by a big gain, and adds them back — so subtle motion/color becomes dramatically visible (classic demos: seeing a pulse in a face, a wall vibrating).

**My implementation (`evm2.py`):**
1. Downscale to 640×360 (memory-safe for a full temporal filter).
2. Per-pixel **Butterworth band-pass, 1.5–13 Hz** along the time axis (`scipy.signal.filtfilt`). This band rejects DC + slow camera settle (<1.5 Hz) and keeps fast fly transits, below the 15 Hz Nyquist.
3. **Z-score normalize** by each pixel's own in-band noise, so the sensor-noise floor stays dark and only genuine events (flies) pop. (First try without this just amplified noise everywhere — a key lesson.)
4. Render the amplified signal as a Turbo heatmap over the dimmed kitchen.

**Result:** flies show up as bright glowing blobs/streaks moving through the scene — including the subtle ones on the white cabinets. See `evm_frame315.png` (a fly glowing on the upper-left cabinet + two streaks by the microwave) and the `10_eulerian_magnification.mp4` video.

**Note:** EVM here is a **visualization/detection aid**, not a tracker — but you *could* run the same blob-detect + Kalman/Hungarian on the EVM z-score field as yet another recall-boosting detector.

---

## Takeaways
- MOG2 misses hovering flies **because it adapts**; motion-amp catches them **because it doesn't**. Recall vs precision.
- Frequency is a real signal: EVM's temporal band-pass isolates "fly-speed" motion from slow drift — something none of our spatial methods do.
- Best next step for accuracy remains a **learned detector** (Exp 5 idea), now optionally trained on EVM-revealed crops.

## Files
- `video/09_tracking_motionamp_noMOG2.mp4`, `video/10_eulerian_magnification.mp4`
- `figures/paths_motionamp.png`, `figures/evm_maxproj.png`, `figures/evm_frame315.png`
- `scripts/motionamp_track.py`, `scripts/evm2.py`
- `data/fly_tracks_motionamp.json`
