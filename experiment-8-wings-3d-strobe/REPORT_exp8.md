# Experiment 8 — Four exploratory ideas (wings, 3D, strobe, clutter)

Squeezing the most out of the one clip, and ruling things out. Honest results — including what failed.

---
## #1 — Tracking only the WINGS (the main one)

**First question: are wings even resolvable here?** No. At 30 fps / 1080p a flying fly is an **8–15 px motion-blur streak** (see `wing_zoom.png`) — individual wings are sub-pixel. So "track the wings" becomes "find the wing *region/activity* within the fly," and I tried three ways:

**Method A — track-stabilized high-frequency EVM.** Follow the fly so its body is centred, then band-pass (8–14 Hz) the stabilized patch; wings beat fast so their flicker should localize.
→ **FAILED.** Stabilizing on the fly makes the *static background* slide through the patch, and background edges (cabinet lines, grout) dominate the high-freq band — the "wing" map is just vertical background stripes (`wing_id74.png`, right panel). Body/wing peak separation was noisy (6–21 px; 21 px is bigger than the fly = artifact).

**Method B — darkness map.** `background − frame` inside the fly ROI.
→ **WORKS, but it finds the whole fly,** not the wings. It cleanly localizes the fly as an elongated streak (`wing_id74.png`, middle). Great for body/fly localization, useless for wing isolation.

**Method C — per-frame core-vs-halo split.** Dense dark core = body; softer surrounding = blur/wing region; PCA major axis = motion-blur (flight) direction.
→ **PARTIAL.** On a clean background (ids 39, 80) it gives a sensible red body-core + cyan halo (`wing_methodC.png`); near a cabinet edge (id65) it latches onto the background and fails. The "halo" is mostly *motion blur*, not resolved wings, and the PCA axis measures flight direction, not wing-beat.

**Verdict:** wing-level tracking is **not achievable from this clip.** The body/streak is localizable; wings are not. What it would take: a **high-speed camera (240–1000 fps)** to freeze the wing-beat (then Method A works because the wing-beat becomes the dominant fast signal and the body is nearly frozen), **higher resolution**, and the **fly closer** (filling 50+ px). 
**Timing:** Method A ≈ 31 ms/track, Method C per-frame is ~free — not a bottleneck, so the limit is purely the camera, not compute.

---
## #2 — Guessing the 3D flight path (monocular)

**Kitchen size estimate (from the photo):** a U-shaped kitchen roughly **3.0 m wide × 2.4 m tall × 3.0 m deep** (counter ~0.9 m high, upper cabinets ~1.4–2.1 m, camera held against the near wall).

**Method:** pinhole camera (phone ~68° horizontal FOV → focal ≈ 1430 px). Depth from apparent fly size: `depth = fly_mm × focal_px / apparent_px` (bigger blob = closer), assuming a ~9 mm apparent fly, smoothed and clamped to the kitchen volume. Then back-project each 2D track point to 3D world metres.

**Result (`kitchen3d_view1.png`, `kitchen3d_rotate.mp4`):** 25 reconstructed 3D paths. Flies land at **~1.8–3.0 m depth, in the upper-back volume** near the cabinets — which **matches reality** (that's where they fly). 3D path lengths 0.7–2.4 m.

**Honest caveats:** the image-plane X/Y is solid; **depth is the weak axis** — apparent size is noisy (±1–2 px → tens-of-% depth error), inflated by motion blur, and assumes a fixed real fly size. So treat depth as a *plausible guess*, not a measurement. Two cameras (stereo) would make it real — your two-phones idea.

---
## #3 — Strobe idea (temporal part simulated)

The depth-dependent lighting of a real strobe **cannot** be simulated from a static-lit clip. The *temporal* half — detect only on sparse "lit" frames, predict through "dark" ones — I did simulate:

| Strobe | detections fed | tracks | mean track length | compute |
|---|---|---|---|---|
| 1/1 (full) | 599 | 32 | 11.5 | 1× |
| 1/2 | 358 | 21 | 7.6 | ½× |
| 1/3 | 265 | 13 | 6.1 | ⅓× |
| 1/4 | 229 | 6 | 4.5 | ¼× |
| 1/6 | 191 | **0** | — | ⅙× |

**Finding:** for *these* fast, erratic flies, strobing **hurts** — they change direction faster than the Kalman can predict across the dark gaps, so continuity collapses by ~1/4 and fails at 1/6. It saves compute but loses tracks. The genuinely useful part of your idea (strobe lighting to separate big-motion objects from pixel noise and isolate near objects) needs real strobe hardware + a dark room to test; worth a real-world try, not a simulation.

---
## #4 — Moving people / clutter (analysis, not faked)

You said skip if it's not real — compositing fake hands would mislead more than teach, so here's the reasoning instead:
- **Big objects are already handled:** the size filter rejects blobs > 24 px, so a hand/arm (hundreds of px) never becomes a "fly." Good news.
- **But their edges spawn many small blobs:** a moving hand's boundary + cast shadows produce lots of small moving fragments that *do* pass the size filter → a burst of false detections along the edge. The movement + tracking filters remove most (they don't form coherent point-tracks), but some short false tracks would appear.
- **Occlusion is the real cost:** when a person/arm passes in front of a fly, the fly's track breaks (needs the re-ID/stitching from Exp 4).
- **Mitigation worth testing for real:** add an *upper* size gate + reject blobs adjacent to large moving regions (dilate big-blob mask, suppress detections inside it).

---
## Files
- `figures/wing_zoom.png`, `figures/wing_id74.png` (Method A fail), `figures/wing_methodC.png` (Method C)
- `figures/kitchen3d_view1.png`, `video/14_kitchen_3d_rotate.mp4`
- `scripts/wings.py`, `scripts/kitchen3d.py`
- `data/wing_results.json`, `data/tracks3d.json`
