# Experiment 4 — Better recall (catch missed flies) + continuity (re-ID / gap-bridging)

Built on the HD pipeline (Exp 3). Two fixes for the two problems you spotted.

## Problem 1 — missed flies (recall)
**Why MOG2 misses them:** MOG2 *adapts* — a fly that hovers or drifts slowly gets learned into the background and disappears; and low-contrast flies over the dark stove/microwave never cross the threshold. My instability mask + size filter could also discard real flies.

**Fix — a UNION of three complementary detectors, merged per frame:**
1. **MOG2** — catches *moving* flies (what we had).
2. **z-score median-background** — the temporal-median image contains *no* flies, so even a *still/hovering* fly stands out. Thresholded per-pixel as `|frame − bg| > 5·σ(pixel)` (σ = that pixel's temporal noise), so clean cabinets flag tiny flies while noisy texture needs a bigger jump. This is the key addition for hoverers.
3. (3-frame differencing available too, jitter-immune, but added little here.)

Plus a slightly looser instability mask (p94 vs p92) so flies over textured areas aren't thrown away.

**Effect:** detections now appear in the **lower frame (stove/counter)** where the MOG2-only baseline saw nothing — several new candidate tracks. (Caveat: the cluttered lower area is also where false positives are likeliest, so a few of those may not be flies.)

## Problem 2 — one fly becoming many IDs (continuity)
**Why:** when a fly briefly vanishes (occlusion / missed frame / dark patch) its track dies and a new ID is born.

**Fix — two parts:**
- **Longer coast:** the Kalman track now survives 25 missed frames (was 10) before dying, so short gaps are bridged *inside* the tracker.
- **Track stitching (re-ID):** after tracking, greedily link a track's END to another track's START when the gap is ≤18 frames, the first track's velocity *predicts* the second's start position (within an adaptive tolerance), AND the directions agree (<60°). Bridged gaps are linearly interpolated so the path is continuous.

**Important lesson:** my first stitch was too aggressive (tolerance + gap too large) and drew long straight lines joining *different* flies across the frame. Tightening the gap, adding a **direction gate**, and capping the tolerance fixed it — precision over recall for the merge step.

## Results vs HD baseline

| | HD baseline (Exp 3) | **Improved (Exp 4)** |
|---|---|---|
| Detector | MOG2 only | MOG2 ∪ z-score median-bg |
| Coast | 10 frames | 25 frames |
| Stitching | none | yes (conservative, direction-gated) |
| Tracks | 26 | 24 |
| Mean flight-path length | 210 px | **371 px** |
| Longest track | 41 frames | **104 frames (~3.5 s)** |
| Confident stitch merges | — | 4 |
| Coverage | upper frame | upper **+ lower** frame |

Track count went *down* (26→24) even though we detect more — because stitching correctly **merges fragments of the same fly**. The real wins are **longer coherent tracks** and **new coverage of previously-missed areas**, not the raw count.

## What it would take to catch EVERY fly you can see
The honest answer: classical detectors (MOG2, differencing, median-bg) all rely on contrast/motion, so a fly that's faint, still, or over busy background will always be borderline. To get human-level recall you'd train a **tiny learned detector** (a small CNN / blob classifier, or a YOLO-style model) on a few hundred hand-labeled fly crops from this footage. That learns what a fly *looks like* regardless of motion, and would catch the ones we still miss. That's the natural Experiment 5.

## Files
- `figures/paths_improved.png` — the 24 stitched tracks (one color per fly)
- `video/08_tracking_improved.mp4` — overlay
- `scripts/improved_pipeline2.py`, `scripts/render_improved.py`
- `data/fly_tracks_improved.json`
