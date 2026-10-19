# Experiment 3 — Same best method on the HIGH-QUALITY source

**Goal:** re-run the Experiment-1 best pipeline (trim → instability mask → **MOG2 adaptive background** → **Kalman filter + Hungarian association**) on the full-quality clip and see how much the quality helps.

**Source:** `PXL_20260916_232522481.TS.mp4` — **1920×1080, HEVC, ~20 Mbps** (vs the old 1518×854, 1.1 Mbps). Same footage.

## Result

| Metric | Low-quality | **HD** |
|---|---|---|
| Fly tracks | 14 | **26** |
| Total detections kept | 107 | **317** |
| Longest track | 17 frames | **41 frames** |
| Longest flight path | 227 px | **821 px** |
| Mean path length | 83 px | **210 px** |
| Max flies airborne at once | 2 | **5** |

**Why HD helps so much:** at 1.1 Mbps, compression smeared the tiny flies, so detections blinked in and out and tracks fragmented into stubs. At 20 Mbps the fly signal stays solid frame-to-frame, so MOG2 catches them consistently and the tracker stitches long, smooth flight paths.

Parameters scaled for 1080p: blob size 3–120 px² / ≤22 px side; tracker gate 57 px, 10-frame coast.

## Files
- `figures/paths_hq.png` — all HD flight paths (color = time)
- `video/tracked_hq_h264.mp4` — HD tracking overlay (downscaled to 1280 for size; raw 50 MB source not included)
- `scripts/hq_pipeline.py`, `scripts/render_hq.py`
- `data/fly_tracks_hq.json` — the 26 tracks

## Known limitations (next experiment)
- **Recall:** flies that hover (absorbed by MOG2's adaptation) or sit on low-contrast/dark backgrounds are missed — visible to the eye but untracked.
- **Continuity:** a fly that briefly disappears gets a new ID → one fly becomes several tracks. Needs gap-bridging / re-ID.
