# Live real-time causal EVM fly tracker

A **streaming, causal** (no future frames) Eulerian-magnification fly tracker that runs in
**real time at full 1080p on basic hardware** (no GPU). One self-contained file: `live_evm_tracker.py`.

## Run it
```bash
pip install numpy opencv-python scipy
python live_evm_tracker.py --source 0                 # your webcam, live
python live_evm_tracker.py --source YOURPHONE.mp4     # your uncompressed phone clip
python live_evm_tracker.py --source clip.mp4 --out tracked.mp4 --no-display   # headless render
python live_evm_tracker.py --source clip.mp4 --scale 0.5   # weaker hardware / more headroom
```
Press `q` to quit. Flags: `--keep-z` (detection sensitivity, default 6), `--scale` (downscale factor).

## Measured performance
520 frames @ **1920×1080** on a 4-core cloud VM (no GPU): **~31 fps** end-to-end (filter + detect + track + draw).
That is faster than the 30 fps video = real time. Downscale with `--scale` for Raspberry-Pi-class devices.

## How it works (and how it differs from the offline EVM)
| | Offline EVM (Exp 6) | **Live causal (this)** |
|---|---|---|
| Temporal filter | `filtfilt` Butterworth (looks forward+back → needs whole clip) | **causal IIR**: two running low-pass EMAs, `band = lp_fast − lp_slow` (uses only past) |
| Noise floor | per-pixel std over the whole clip | **online slow EMA** of band-power, updated every frame |
| Static-noise rejection | precomputed instability mask | **automatic**: the online floor rises on chronically-noisy pixels → their z stays low |
| Latency | whole clip | ~1 frame |
| State/memory | full stack in RAM | **5 small images** (a few MB), O(1) per pixel |

Pipeline per frame: causal band-pass → online z-score → threshold+size-filter blobs → Kalman + Hungarian → a "has it moved?" filter so only real flies (not static noise) are shown.

## Tuning notes / lessons
- **Causal ≠ filtfilt.** The one-pass filter is gentler, so z-values are lower — recalibrate `keep_z` (6 here vs 4.5 offline).
- **Noise floor is the whole game.** A fast EMA lets a fly poison its own threshold (z caps out); a gate freezes noisy edges into permanent false positives. The fix that works: a **slow (~8 s) ungated EMA** — a fly visits any pixel only ~3 frames, too briefly to move a multi-second average.
- **No offline stitching live** → track IDs churn (you'll see high numbers). The Kalman coast bridges short gaps; full re-ID needs the offline pass.
