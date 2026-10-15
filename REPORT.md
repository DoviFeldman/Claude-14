# Fly Tracking via Motion Amplification — Project Report

**Source:** `PXL_20260916_232522481.TS.mp4` — 1518×854, 30 fps, 20.0 s, 599 frames.
Phone held flat against a wall, looking into a kitchen; a few flies in the air; microwave clock flickers.

---

## 1. Headline results

- **14 clean fly flight-tracks** recovered across the clip.
- Flies are **small dark specks (~2–40 px²)** that move against the bright white cabinets — confirmed real by frame-to-frame motion and by the fact that every track sits in the **upper third** of the frame (where flies actually congregate), not scattered randomly.
- **Max 2 flies airborne simultaneously** (never 3+) — they mostly appear one at a time.
- **Motion-amplification finding:** detecting on the raw video with an adaptive background subtractor (**14 tracks**) *beat* detecting on the motion-amplified video (**12 tracks, shorter**). Motion-amp is excellent for the human eye but worse for automated detection — your hunch was right.

---

## 2. Pipeline (every stage, in order)

| # | Stage | What it does | Why |
|---|-------|--------------|-----|
| 0 | **Probe** | `ffprobe` for res/fps/frames; extract sample frames | Understand the material |
| 1 | **Shake detection** | Per-frame global mean abs-diff on downscaled gray | Find where the opening camera move ends |
| 2 | **Trim** | Drop first 2.6 s (78 frames) → 521 frames / 17.4 s | Remove the shake that ruins motion-amp |
| 3 | **Stabilization test** | ORB+affine and phase-correlation, measured residual | Footage was already stable (phone on wall) — stabilizing made it *worse*, so skipped |
| 4 | **Motion amplification** | `blend = 0.5·frame(t) + 0.5·(255 − frame(t−3))` | The inverted/time-shifted overlay you asked for: static → flat gray, motion → visible |
| 5 | **Temporal instability map** | Streaming Welford per-pixel std over all frames | Pixels that *always* flicker = edges/handles/microwave clock |
| 6 | **Jitter mask** | Mask the top ~8% most-unstable pixels (13.6% of frame after dilation) | Kills edge-jitter false positives |
| 7 | **Background model** | **MOG2 adaptive** subtractor (history=150, varThr=30, shadows off) | Absorbs the slow hand-settling drift the first ~5 s; a *fixed* median background could not |
| 8 | **Blob detection** | Connected components + shape filter (area 2–70, max side ≤16, extent ≥0.3, aspect ≤3.5) | Keep compact round specks; reject thin streaks |
| 9 | **Tracking** | Kalman (constant-velocity) + Hungarian association, 45 px gate, 10-frame coast | Link detections into tracks; this is what rejects fixed-location jitter |
| 10 | **Track filtering** | Keep tracks ≥4 hits, span ≥5, that actually move (path ≥16 or net ≥12 px) | A fly *moves*; jitter sits still |
| 11 | **Visualizations** | Coordinate trajectories, mesh, voxel grid | The three requested views |
| 12 | **A/B experiment** | Same detector on raw vs motion-amplified video | Measure whether motion-amp helps detection |

---

## 3. The three tracking visualizations

1. **Coordinate arrays** (`03_tracking_coordinates.mp4`, `paths_temporal.png`)
   Each track is an array of (frame, x, y). Rendered as fading trajectory tails + live markers + IDs on the video, plus a single time-colored path map (blue=early → red=late).

2. **Mesh** (`04_tracking_mesh.mp4`, `mesh_spatiotemporal.png`)
   - *Per-frame:* Delaunay triangulation of concurrently-active flies + Kalman velocity arrows. (Rarely ≥3 flies at once, so this is mostly nodes+vectors.)
   - *Spatio-temporal:* Delaunay triangulation of **all** detections across time — the "activity web" of where flies flew.

3. **Voxel grid** (`05_voxel_spacetime_3d.mp4`, `voxel_3d_view1/2.png`)
   (x, y, t) space. A 48×27×60 occupancy grid (54 occupied cells) + a 3D matplotlib view plotting every track as a ribbon in X–time–Y, rotating. This is the clearest way to *see* that flies occur at different times.

---

## 4. Libraries used

- **OpenCV 5** (`opencv-contrib-python-headless`) — I/O, MOG2, connected components, Kalman math, drawing, warps.
- **NumPy** — arrays, Welford std, geometry.
- **SciPy** — `optimize.linear_sum_assignment` (Hungarian association), `spatial.Delaunay` (mesh).
- **Matplotlib** (Agg) — 3D voxel/space-time rendering.
- **scikit-image** — installed, available for future morphology/registration.
- **FFmpeg** — frame extraction, final H.264 (`yuv420p`, `+faststart`) transcode for universal playback.

---

## 5. Tricks & tips (the stuff worth keeping)

- **Inverted-overlay motion-amp math:** averaging a frame with the *inverted* version of a frame a few steps back makes static pixels cancel to mid-gray (X and 255−X average to 127.5). Deviation from gray = motion. Simple, no Eulerian pyramid needed.
- **Don't stabilize already-stable footage.** ORB features on repetitive tile/cabinet texture produce noisy transforms that *add* jitter. Always measure residual before and after.
- **Temporal std is a free "what-always-moves" map.** Compute it streaming (Welford) so you never hold the whole stack in RAM. Mask the chronically-unstable pixels — this removes edge jitter *and* the flickering microwave clock automatically.
- **Adaptive > fixed background when the camera slowly settles.** A fixed median background flagged 230 false blobs/frame early (hand creep); MOG2 adapts the drift away and dropped it to ~0.5/frame. Give MOG2 a ~30-frame warmup you then discard.
- **Shape-filter before you track.** Flies are compact (high extent, near-1 aspect). Edge jitter is thin/elongated. One cheap filter removes most noise.
- **Tracking is the real noise filter.** The decisive fly-vs-noise cue is *coherent motion over time.* Require a track to translate a minimum distance; stationary flicker never qualifies.
- **Memory:** never `np.stack` 500+ full-res float frames (that's the OOM I hit). Stream, sample for the background, use incremental stats.
- **Playback:** OpenCV writes `mp4v` which many players/Telegram choke on. Always re-encode to `libx264 + yuv420p + faststart`.

---

## 6. Motion-amp A/B result

| Detector input | Fly tracks | Total track-points |
|---|---|---|
| **Raw frames + MOG2** | **14** | **107** |
| Motion-amplified (inverted overlay) + MOG2 | 12 | 82 |

Motion-amp halves local contrast and adds a time-shifted ghost, which fragments the already-tiny fly blobs — so automated detection is *better on raw frames*. Keep motion-amp as a **visualization / human-review** layer, not a detection pre-filter.

---

## 7. Limitations & next steps (for the bigger project)

- **Subtle/slow flies** that hover get absorbed by MOG2's adaptation — a dual model (fast adaptive + slow reference) would catch both.
- **Re-identification across gaps:** when a fly leaves and re-enters frame it becomes a new ID. Add appearance/motion-model re-ID to stitch tracks.
- **Settling drift** cost us the first ~5 s of usable detection. A proper sub-pixel stabilizer (ECC on a masked background ROI) could reclaim it.
- **Scale up detection quality:** try (a) 3-frame differencing `min(|t−t−d|,|t−t+d|)` for jitter-immune motion, (b) a tiny CNN/blob-CNN or `blob_log` from skimage for sub-pixel centers, (c) true Eulerian Video Magnification (Laplacian-pyramid, temporal bandpass) to amplify *only* fly-frequency motion.
- **3D / depth:** single camera = no depth. Two phones against two walls → stereo → real 3D voxel occupancy.
- **Metrics:** add ground-truth labels on a few hundred frames to measure precision/recall and tune thresholds objectively.

---

## 8. File manifest

**Videos (`deliver/`):**
- `01_motion_amplification.mp4` — inverted/time-shifted overlay (requested method)
- `02_motion_amp_heatmap.mp4` — 6× amplified motion, Turbo colormap (shows noise problem)
- `03_tracking_coordinates.mp4` — trajectory overlay
- `04_tracking_mesh.mp4` — Delaunay mesh + velocity vectors
- `05_voxel_spacetime_3d.mp4` — rotating 3D (x, y, t) view

**Images:** `paths_temporal.png`, `mesh_spatiotemporal.png`, `voxel_3d_view1.png`, `voxel_3d_view2.png`, `motion_maxproj.png`, `tstd.png`, `jitter_mask.png`, `bg_median.png`

**Scripts:** `analyze_motion.py`, `trim_clean.py`, `motion_amp.py`, `detect2.py`, `detect_mog2.py`, `track2.py`, `render_coords.py`, `mesh.py`, `voxel_mesh.py`, `compare_ma.py`

**Data:** `fly_tracks.json` (final tracks), `dets_mog2.json`, `tstd.npy`, `bg_median.npy`, `compare_ma.json`
