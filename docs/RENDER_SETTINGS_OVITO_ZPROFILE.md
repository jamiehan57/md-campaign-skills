# OVITO Structure Render + Dopant z-Profile Movie — Campaign Settings (shareable)

> SEM_MLCC / 03_SURF_HETERO convention (finalized 2026-07-23). Two movies are rendered from the
> SAME strided trajectory frames so they play side-by-side frame-for-frame:
> `structure_ovito_<run>_<T>K.mp4` (OVITO/Tachyon) + `dopant_zmovie_<run>_<T>K.mp4` (matplotlib).
> Scripts: `03_SURF_HETERO/scripts/prod_structure_movie.py`, `prod_dopant_movie.py`.

## 1. Color code & display radii (single source of truth)

Defined once in `scripts/_species_colors.py`; matplotlib figures AND OVITO/Blender renders import
from it, so plots and renders cannot drift apart.

| Species | Hex | RGB (0–1) | Display radius (Å) | Note |
|---|---|---|---|---|
| Ba | `#1EEF2C` | (0.118, 0.937, 0.173) | 1.35 | green |
| Ti | `#78CAFF` | (0.471, 0.792, 1.000) | 0.90 | sky blue |
| O  | `#FE0300` | (0.996, 0.012, 0.000) | 0.55 | red — kept small so dopants stay visible |
| Dy | `#3106FC` | (0.192, 0.024, 0.988) | 1.25 | blue (dopant) |
| Mg | `#FB7B15` | (0.984, 0.482, 0.082) | 0.85 | orange (dopant) |

- These are **user-specified exact hex codes (2026-07-16)**, NOT VESTA defaults (VESTA Dy is teal,
  Ti near-white). The older palette is kept as `SPECIES_COLOR_LEGACY` for a one-line revert.
- Radii are **visual, not physical** — covalent-scale radii render a perovskite as one solid mass.
- Blender/io_mesh_atomic keys off FULL element names ("Barium", not "Ba") — symbol lookups
  silently miss (`ELEMENT_FULLNAME` map in the same module).

## 2. OVITO structure movie — exact settings

**Environment** (broken PySide6/Qt in the base env — a dedicated env is required):

```bash
OVITO_THREAD_COUNT=2 QT_QPA_PLATFORM=offscreen \
/home/jamie/anaconda3/envs/ovito-render/bin/python scripts/prod_structure_movie.py \
    --stride 20 --limit 151 --repeat 1x10x2 --no-cell --zvert <run_dir>
```

| Setting | Value | Why |
|---|---|---|
| Renderer | `TachyonRenderer(shadows=False, direct_light_intensity=1.1, ambient_occlusion=True, ambient_occlusion_samples=12)` | soft, readable depth without harsh shadows |
| Background | white `(1,1,1)`, `alpha=False` | slide-ready |
| Image size | 1600×760 default (`--size WxH`) | |
| Viewport | `Front`, `zoom_all()` on frame 0, then camera **frozen** (`camera_pos/dir/fov` reused every frame) | per-frame zoom_all would "breathe" as atoms wander |
| Positions | `atoms.wrap()` before styling | new prods write unwrapped positions |
| Tiling | `--repeat 1x10x2` (b×10, c×2) | pairing convention: 2 periods along z to match the tiled z-profile |
| Orientation | `--zvert`: rotate cell b→screen-x, so **z (c-axis) is VERTICAL, increasing UP** | matches the z-profile movie axis; without it, default is z horizontal (rotate 90° about y) |
| Cell box | `--no-cell` for the pairing convention (else `line_width 0.25`, color (0.25,0.25,0.25)) | tiled render, box edges just clutter |
| Per-type style | `type.color = rgb(name)`, `type.radius = SPECIES_RADIUS[name]` from `_species_colors.py` | §1 palette |
| OVITO ≥ 3.15 gotcha | build a **fresh `Pipeline(StaticSource(...))` per frame** — mutating/reassigning `source.data` after the first render is silently ignored (cached evaluation; verified by identical frame md5s) | otherwise all frames render frame 0 |

**ffmpeg encode** (shared-server safe — process limit means threading fails):

```bash
ffmpeg -y -loglevel error -threads 1 -framerate 25 -i f%05d.png \
       -vcodec libx264 -threads 1 -crf 20 -pix_fmt yuv420p \
       -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" out.mp4
```

- `-threads 1` needed **both before `-i` (PNG decoder) and after (encoder)**.
- libx264 needs EVEN pixel dims — the pad filter guarantees it (for matplotlib movies, pick
  figsize×dpi that lands even: 4.62 in × 110 dpi = 508 px ✓, 4.6 → 505 px ✗).

## 3. Dopant z-profile movie — exact settings

```bash
env OMP_NUM_THREADS=1 python scripts/prod_dopant_movie.py \
    --stride 20 --limit 151 --zsigma 1.0 <run_dir>
```

| Setting | Value |
|---|---|
| Layout (2026-07-23 slide convention) | ONE box spanning **TWO z-periods** (ylim 0 → 2·Lz, one-period histogram tiled with no seam), z on the **VERTICAL axis increasing UP** — lines up 1:1 with the `--repeat 1x10x2 --zvert` structure render |
| Curves | Dy alone, or Dy+Mg overlaid (colors from §1); dotted t=0 profile of each species stays as reference |
| Per-frame data | **block average** of the `--stride` traj frames it covers (no frames thrown away; 48 Dy × 20 frames ≫ shot noise of one frame) |
| z-binning | 0.5 Å bins, number density (atoms/Å) |
| Smoothing | `--zsigma 1.0` (Å): area-preserving Gaussian along z melts per-lattice-plane spikes into an envelope (slide look); `--window N` optional extra running mean across frames (default off) |
| Legend | NONE in the movie — saved once as `dopant_zmovie_legend_<run>_<T>K.png` (dotted "X, t=0" + solid "X" per species) |
| Text | title on top, density label bottom only |
| Figure | figsize 4.62 × 11.2 in (even-pixel rule, §2) |

## 4. Matching the two movies (timescale / frame speed)

**Campaign movie convention: every production movie covers 3 ns as 151 frames = 20 ps/frame,
6 s at 25 fps.** Pass the SAME `--stride`/`--limit`/`--fps` to BOTH scripts:

| Traj frame spacing | Runs | Flags for BOTH scripts |
|---|---|---|
| 1 ps/frame (dt 1 fs × 1000-step interval) | 07/08 compensated prods | `--stride 20 --limit 151` |
| 2 ps/frame (dt 2 fs × 1000-step interval) | 09, uncomp dt-2fs jobs, older 0X_hetero | `--stride 10 --limit 151` |

- Frame k of each movie shows/averages traj frames `[k·stride, (k+1)·stride)` with the same time
  label — that's what makes side-by-side playback line up.
- Check frame spacing first (`run_meta.json`: dt_fs, time_ps vs traj length) — the wrong stride
  silently halves/doubles movie coverage (a stride-10 habit on a 1 ps/frame run gives 1.5 ns, and
  the movie just "moves faster"; nothing errors).
- `--partial` renders a still-running traj (outputs suffixed `_partial`).
- Continuation runs are STITCHED: pass the part2 dir (its `run_meta.json` has
  `continuation_from`/`t0_ps`); part1 is prepended, the duplicated seam frame dropped, outputs
  named `<source>_stitched*`.
- **Outputs land in `<struct>/<subdir>/rendering/`** (convention since 2026-07-23): both mp4s,
  the separate legend PNG, and the extxyz trajectory export (`traj_<run>.xyz`, complete runs
  only — OVITO opens it directly). `plot/` holds only the static analysis figures.

## 5. Server etiquette (shared login node)

- `OMP_NUM_THREADS=1` for the matplotlib movie; `OVITO_THREAD_COUNT=2` for OVITO;
  ffmpeg 1 thread (§2). Process limit on the box is low — thread-happy defaults die with
  `fork: Resource temporarily unavailable` / `ff_frame_thread_encoder_init failed`.
- Long renders: `nohup ... > log 2>&1 &`, one at a time, check with `grep`, not `cat`.
