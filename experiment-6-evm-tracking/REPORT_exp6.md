# Experiment 6 — Tracking ON Eulerian magnification (the best tracker yet) + glow-on-black

## The big result
Running **blob-detect → Kalman → Hungarian → stitch** on the **EVM z-score field** gives **12 clean tracks** — each a smooth, complete fly flight-path, with **no wall-noise tracks and no double-counting**. This is the cleanest tracker of all 6 experiments.

| Method | Tracks | Character |
|---|---|---|
| MOG2 (Exp 3/4) | 26 | precise but misses hoverers |
| motion-amp, no MOG2 (Exp 5) | 89 | high recall, noisy, **double-counts** |
| **EVM + Kalman/Hungarian (this)** | **12** | **clean paths, no noise, no doubles** |

**Why EVM tracks so cleanly:**
- **No double-counting.** The inverted-overlay motion-amp compares frame *t* vs *t−3*, so a moving fly appears twice (now + ghost of then). EVM filters each pixel over *time*, so the fly appears **once, where it is.**
- **Static noise is gone by construction.** EVM's temporal band-pass only passes "fly-speed" frequencies (1.5–13 Hz). A non-moving speck on the wall has no energy in that band → it never fires. That's the "only flies, not wall noise" you wanted, for free.
- A **strict movement filter** (track must travel ≥35 px net or ≥60 px path) drops the last few barely-moving detections.

## EVM vs motion amplification (your question)
- **Motion amplification** (what we did first): a *spatial* trick — overlay the frame with an inverted copy from a few frames back. Cheap, instant, but it shows motion as a **difference between two moments** → ghosting / double images, and it amplifies *all* change including noise.
- **EVM**: a *temporal-frequency* method — sit on each pixel, band-pass its brightness over time, amplify only the chosen frequencies. It isolates motion by **how fast it repeats**, so it separates flies (fast) from drift (slow) and static (zero) — and shows each fly once. Cleaner, but needs a window of frames.

## Real-time? (your question)
Timing at 640×360 working resolution, 520 frames:
| Stage | ms/frame | fps |
|---|---|---|
| EVM band-pass filter | 10.4 | 96 |
| detection on z-field | 0.7 | 1506 |
| Kalman+Hungarian+stitch | 0.2 | — |
| **Total compute** | **11.3** | **89 fps** |

So the **throughput is ~89 fps — ~3× faster than the 30 fps video.** BUT the filter I used (`filtfilt`) is **zero-phase / non-causal**: it looks both forward and backward in time, so it needs the whole clip (or a buffered window) — it's "faster-than-real-time **offline**," not live. For a true **live** stream you'd swap in a **causal one-pass IIR band-pass** (the original EVM paper's first-order recursive filter), which updates each frame from the last — that runs in real time with a fraction-of-a-second latency. Totally doable as a follow-up.

## Bonus — your glow-on-black idea
You described: keep moving things glowing white on a black background. I built exactly that from the EVM z-field (`12_evm_glow_on_black.mp4`): flies glow white on black, with a bloom filter. It's the same spirit as your "diff against the first frame + glow" trick, but EVM's frequency filtering handles the residual camera motion that would have smeared a raw first-frame difference.

## Notes on your other observations
- **"motion-amp sees double"** — correct, and EVM fixes it (above).
- **"Kalman/Hungarian caught barely-moving things"** — tightened the movement filter here; those are gone.
- **first-frame-reference trick** — great for a locked-off tripod; our handheld-on-wall clip has enough residual drift that a static reference smears, which is why the median-background (Exp 4) and EVM (frequency) approaches work better here.

## Files
- `video/11_tracking_on_EVM.mp4` — the 12-track overlay
- `video/12_evm_glow_on_black.mp4` — glow-on-black
- `figures/paths_evm.png`, `figures/glow_sample.png`
- `scripts/evm_track.py` (detect+track+timing), `scripts/evm_render_glow.py`
- `data/fly_tracks_evm.json`
