---
name: prod-analysis
description: Campaign-convention static analysis figures for MD production runs — CN (cation-O coordination), MSD xyz/par-perp, z-profile, and E_tot/T stability plots, with the 3 ns cut and separate-legend conventions. Triggers - /prod-analysis, CN plot, coordination number plot, stability plot, 안정성 플랏, cn 그래프, zprofile 플랏, 분석 그림, analysis figures, cation analysis, E_tot plot.
---

# prod-analysis — CN / MSD-xyz / z-profile / stability figures

Static analysis suite for production runs, in the shared campaign style
(`_analysis_style.py`: Arial, SC=1.4 font scaling, Agg, BLAS pinned to 1 thread —
ALWAYS imported before numpy). Colors from `scripts/_species_colors.py`
(Ba `#1EEF2C` · Ti `#78CAFF` · O `#FE0300` · Dy `#3106FC` · Mg `#FB7B15`);
dopants emphasized (lw 2.6, opaque, on top), hosts faded (lw 1.3, alpha 0.5).

Scripts (server ccel_147_snu):
- `~/SEM_MLCC/02_RUN/03_SURF_HETERO/scripts/prod_cation_analysis.py`
- `~/claude/stability_plots.py` (edit its RUNS dict to add runs)
- mid-run MSD helpers: `~/claude/msd_fromtraj.py`, `~/claude/msd_z_dopant.py`

## Commands

```bash
cd /home/jamie/SEM_MLCC/02_RUN/03_SURF_HETERO
# full per-run suite -> <struct>/<subdir>/plot/
env OMP_NUM_THREADS=1 python scripts/prod_cation_analysis.py --tmax 3000 [--partial] <run_dir ...>
# stability figures (md.log based)
env OMP_NUM_THREADS=1 python ~/claude/stability_plots.py
```

## Figures & conventions

| Figure | File | Convention |
|---|---|---|
| Total MSD | `msd_<run>_<T>K_{raw,smooth}.png` | hop_screen.plot_msd style; needs msd.dat (only exists AFTER completion) |
| MSD by axis | `msd_xyz_...` | 3 panels x/y/z, shared y |
| MSD in/out-of-plane | `msd_parperp_...` | 2 panels: in-plane (x+y) vs z |
| CN | `cn_cationO_<run>_<T>K.png` | species-averaged cation-O CN vs t; per-species cutoff from first minimum of r²-normalized X-O histogram; **10×5.6 in exact canvas** (matches stability figure); **NO legend** — saved separately as `cn_cationO_legend_<system>_<T>K.png`; dashed per-species mean-CN guides keep right-side value labels |
| z-profile | `zprofile_<run>_<T>K.png` | per-element density along z, mean of first vs last 10 frames |
| Stability | `{etot,temp}_<system>_<T>K_{raw,smooth}.png` | from md.log (0.1 ps): E_tot black, T grey `#858585` with dashed target line; if combined in one stacked twin-axis figure, band-stack the ylims (E top half, T bottom half) or the curves overlap unreadably |

## Rules that silently break things

- **Analysis cut: `--tmax 3000` (suffix `_to3ns`)** — the campaign compares everything at 3 ns
  even when runs are 5 ns. Smooth = 10 ps running mean, edge-corrected (NOT bare mode="same").
- msd.dat is written by the MD driver every 0.2 ps but ONLY exists after completion — mid-run,
  use `msd_fromtraj.py` (all species, from traj) / `msd_z_dopant.py` (Dy/Mg z-only).
  Mind traj frame spacing: 1 ps/frame runs → TMAX_FRAMES 3001; 2 ps/frame → 1501 AND time axis ×2.
- `--partial` for incomplete runs (`_partial` suffix, discovery skips them otherwise).
- Continuations stitched via `run_meta.json` `continuation_from`/`t0_ps` → `<source>_stitched`.
- Login node: everything single-thread (`env OMP_NUM_THREADS=1`), no GPU, no SLURM needed.
- LAMMPS-dump analysis (03_uncomp_lammps etc.): ASE reads dump types as Z=1..N — remap
  `{1:56, 2:66, 3:12, 4:8, 5:22}` for Ba/Dy/Mg/O/Ti order or every species mask is empty (MSD=0).

## Workflow

1. Confirm run status (`run_meta.json`) → decide `--partial` and msd.dat availability.
2. Run the suite with `--tmax 3000`; add the run to `stability_plots.py` RUNS if new.
3. Show the user 1–2 rendered PNGs to confirm style, not all of them.
