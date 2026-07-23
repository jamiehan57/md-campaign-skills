# md-campaign-skills

Claude Code skills + reference docs for the SEM_MLCC-style MD campaign analysis and rendering
pipeline (BaTiO3 heterostructure MLIP-MD): paired production movies (OVITO structure render +
dopant z-profile animation, frame-aligned) and the static analysis figure suite (CN, MSD
xyz/par-perp, z-profile, stability), all sharing one color code.

## Contents

| Path | What |
|---|---|
| `skills/prod-movies/SKILL.md` | Skill: paired OVITO structure mp4 + dopant z-profile mp4, matched stride/limit/fps so the two movies play side-by-side frame-for-frame |
| `skills/prod-analysis/SKILL.md` | Skill: CN / MSD-xyz / z-profile / E_tot-T stability figures, 3 ns cut convention, separate-legend rules |
| `docs/RENDER_SETTINGS_OVITO_ZPROFILE.md` | Full shareable settings reference: exact hex color code, display radii, Tachyon renderer settings, camera/orientation, tiling, ffmpeg flags, timescale-matching table |
| `scripts/` | The actual pipeline scripts the skills drive (single source of truth for colors: `_species_colors.py`) |

## Install the skills (Claude Code)

Copy each skill folder into your skills directory:

```bash
cp -r skills/prod-movies skills/prod-analysis ~/.claude/skills/
```

Then `/prod-movies` or `/prod-analysis` in Claude Code, or just ask for "structure movie",
"CN plot", etc.

## Color code (single source of truth: `scripts/_species_colors.py`)

Ba `#1EEF2C` green · Ti `#78CAFF` sky · O `#FE0300` red · Dy `#3106FC` blue · Mg `#FB7B15` orange
(display radii Ba 1.35 / Ti 0.90 / O 0.55 / Dy 1.25 / Mg 0.85 Å — visual, not physical).
Matplotlib figures and OVITO/Blender renders both import this module, so plots and renders
cannot drift apart. Change colors there only.

## The one rule that matters

**Every production movie covers 3 ns as 151 frames = 20 ps/frame = 6 s at 25 fps.**
Pass the same `--stride`/`--limit`/`--fps` to both movie scripts:
`--stride 20 --limit 151` on 1 ps/frame trajectories, `--stride 10 --limit 151` on 2 ps/frame.
Check your trajectory's frame spacing first — the wrong stride silently renders the wrong
time span.

## Environment notes

- OVITO rendering needs a dedicated conda env (`ovito` + PySide6 conflicts with system Qt in
  mixed envs); run with `QT_QPA_PLATFORM=offscreen OVITO_THREAD_COUNT=2`.
- On shared login nodes keep everything single-threaded: `OMP_NUM_THREADS=1`, ffmpeg
  `-threads 1` (before AND after `-i`).
- Scripts expect ASE `.traj` trajectories with `run_meta.json` metadata; continuation runs are
  stitched automatically via `continuation_from`/`t0_ps`.
