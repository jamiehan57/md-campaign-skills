---
name: prod-movies
description: Render the paired production movies for MD runs — OVITO/Tachyon structure mp4 + dopant z-profile mp4 — with matched timescale, frame speed, and campaign color code so they play side-by-side frame-for-frame. Triggers - /prod-movies, structure movie, 구조 무비, ovito 렌더 무비, z profile movie, z프로파일 무비, dopant distribution movie, paired movies, 무비 렌더, production movie render.
---

# prod-movies — paired OVITO structure + dopant z-profile movies

Renders TWO mp4s from the SAME strided trajectory frames so they line up
frame-for-frame in side-by-side playback:

- `structure_ovito_<run>_<T>K.mp4` — OVITO/Tachyon render, z VERTICAL up, cell tiled 2× along z
- `dopant_zmovie_<run>_<T>K.mp4` — Dy(+Mg) z-density animation, one box spanning TWO z-periods

Scripts live in `/home/jamie/SEM_MLCC/02_RUN/03_SURF_HETERO/scripts/`
(`prod_structure_movie.py`, `prod_dopant_movie.py`). Full settings reference:
`claude-work/SEM_MLCC~02_RUN/03_SURF_HETERO/RENDER_SETTINGS_OVITO_ZPROFILE.md`.

## Commands (the pairing convention)

```bash
cd /home/jamie/SEM_MLCC/02_RUN/03_SURF_HETERO

# 1) structure movie (NEEDS the ovito-render env; base env's ovito is broken)
OVITO_THREAD_COUNT=2 QT_QPA_PLATFORM=offscreen \
/home/jamie/anaconda3/envs/ovito-render/bin/python scripts/prod_structure_movie.py \
    --stride 20 --limit 151 --repeat 1x10x2 --no-cell --zvert <run_dir>

# 2) z-profile movie (sem-neq env fine; single-thread BLAS)
env OMP_NUM_THREADS=1 python scripts/prod_dopant_movie.py \
    --stride 20 --limit 151 --zsigma 1.0 <run_dir>
```

## Timescale matching — THE rule

**Movie convention: 3 ns coverage = 151 frames = 20 ps/frame = 6 s at 25 fps.**
Pass the SAME `--stride`/`--limit`/`--fps` to BOTH scripts.

| Traj frame spacing | Which runs | Flags |
|---|---|---|
| 1 ps/frame (dt 1 fs) | 07/08 compensated prods | `--stride 20 --limit 151` |
| 2 ps/frame (dt 2 fs) | 09, uncomp dt-2fs, older 0X_hetero | `--stride 10 --limit 151` |

CHECK the frame spacing first (`run_meta.json` dt_fs / time_ps vs `len(traj)`): the wrong
stride silently renders the wrong time span (movie just "moves faster" — no error).
Frame k of each movie covers traj frames `[k*stride, (k+1)*stride)` with the same time label;
the z-profile block-averages those frames, the structure movie renders the block start.

## Style facts (do not improvise)

- Colors/radii: `scripts/_species_colors.py` ONLY (Ba `#1EEF2C` · Ti `#78CAFF` · O `#FE0300` ·
  Dy `#3106FC` · Mg `#FB7B15`; radii Ba 1.35 / Ti 0.90 / O 0.55 / Dy 1.25 / Mg 0.85 Å).
  Change colors THERE, never inline — plots and renders share the module.
- Structure: Tachyon (no shadows, direct_light 1.1, AO 12 samples), white bg, 1600×760,
  Front viewport, camera frozen at frame 0, `atoms.wrap()`, fresh Pipeline per frame
  (OVITO ≥3.15 caches StaticSource — reassigning data silently re-renders frame 0).
- z-profile: 0.5 Å bins, ylim 0→2·Lz tiled seamlessly, z vertical UP, dotted t=0 reference,
  title top / density label bottom, NO in-movie legend — legend saved separately as
  `dopant_zmovie_legend_<run>_<T>K.png`. Slide look = `--zsigma 1.0`.
- ffmpeg: `-threads 1` BOTH before `-i` and after (shared-box process cap); libx264 needs
  even pixel dims (pad filter in structure script; figsize 4.62×11.2 in the profile script).
- Continuations: pass the part2 dir; `run_meta.json` `continuation_from`/`t0_ps` triggers
  stitching (seam frame dropped), outputs named `<source>_stitched*`.
- `--partial` for still-running trajs (`_partial` suffix). Outputs → `<struct>/<subdir>/plot/`.

## Workflow

1. Identify run dir(s) and their traj frame spacing → pick stride per the table.
2. Run both scripts with matched flags (long renders: `nohup` + log, ONE at a time).
3. Verify: both mp4s have the same frame count (151) and duration; spot-check one PNG
   (`--keep-png`) or the mp4 with the user before rendering many runs.
