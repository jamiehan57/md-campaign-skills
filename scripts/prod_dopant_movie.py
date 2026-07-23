#!/usr/bin/env python3
"""
Animated dopant z-distribution for 03_SURF_HETERO production runs, saved next to
the static figures in <struct>/<subdir>/plot/ as

    dopant_zmovie_<run>_<T>K.mp4     (or .gif with --gif)

One animation frame per --stride trajectory frames (default 10). traj.traj is
written every TRAJ_INTERVAL = 1000 steps at dt = 2 fs, i.e. 1 frame / 2 ps, so a
5 ns run is 2501 traj frames -> 251 animation frames at 20 ps each (~10 s at the
default 25 fps). prod_structure_movie.py renders the same strided frames, so the
two movies line up frame-for-frame.

Curves: Dy alone if the run has no Mg, Dy + Mg overlaid otherwise (same
SPECIES_COLOR as every other campaign figure). The t = 0 profile of each species
stays on the axes as a dotted reference, so the spreading out of the starting
reservoir band (Dy z ~ 43-57 A) is visible against where it began.

Layout (2026-07-23, slide convention): ONE box spanning TWO periods along z
(ylim 0 -> 2*Lz, the one-period profile tiled with no seam), z on the VERTICAL
axis increasing UPWARD -- the paired structure movie is rendered
`--repeat 1x10x2 --zvert` (cell tiled 2x along z, OVITO camera z-up), so the
tiled profile lines up 1:1 with the two repeated cells when the movies sit side
by side. Title on top, density label on the bottom only. The plot carries NO
legend; it is written once per run as dopant_zmovie_legend_<run>_<T>K.png.

A single frame's histogram of 18 Dy atoms is mostly shot noise, so each animation
frame is the BLOCK AVERAGE of its --stride trajectory frames (20 ps of data per
plotted profile, no frames thrown away). --window adds a further running mean
across animation frames on top of that; default 1 = off. --zsigma S smooths the
SHAPE of each profile with an area-preserving Gaussian of width S Angstrom along
z, melting the per-lattice-plane spikes into an envelope; default 0 = raw
histogram.

Run (login node, single-thread BLAS):
  env OMP_NUM_THREADS=1 python scripts/prod_dopant_movie.py [--partial] [--gif]
                          [--fps 25] [--stride 10] [--window 1] [run_dir ...]
No args: same discovery as prod_cation_analysis.py.
"""
import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _analysis_style as S            # noqa: F401  pins BLAS threads, Arial style
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import hop_screen as H                 # SPECIES_COLOR / DOPANTS
from ase.io.trajectory import Trajectory

ROOT = HERE.parent
DZ = 0.5                               # z-bin width (A), matches zprofile figure
FPS = 25
STRIDE = 10                            # traj frames per animation frame (block avg)
WINDOW = 1                             # extra running mean across animation frames
ZSIGMA = 0.0                           # Gaussian smoothing of the profile along z (A)


def _traj_z(path):
    """Wrapped z (A) per frame per atom for one trajectory file."""
    traj = Trajectory(str(path))
    nfr = len(traj)
    sym = np.array(traj[0].get_chemical_symbols())
    z = np.empty((nfr, len(sym)))
    Lz = np.empty(nfr)
    for i, at in enumerate(traj):
        cell = at.get_cell().array
        fz = (at.get_positions() @ np.linalg.inv(cell))[:, 2] % 1.0
        z[i] = fz * cell[2, 2]
        Lz[i] = cell[2, 2]
    return z, Lz, sym


def load_wrapped_z(rd):
    """Wrapped z (A) per frame per atom -- a density profile wants the atoms
    inside the cell, NOT the unwrapped paths prod_cation_analysis.py builds.
    Continuation runs (16_continue_nvt.py: run_meta.json carries
    continuation_from/t0_ps) are STITCHED: the source trajectory is prepended
    and the continuation's frame 0 dropped (it duplicates the source's last
    frame), so one movie covers part1+part2 with correct time labels."""
    meta = {}
    mp = rd / "run_meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = {}
    z, Lz, sym = _traj_z(rd / "traj.traj")
    src = meta.get("continuation_from")
    if src:
        z_src, Lz_src, _ = _traj_z(Path(src) / "traj.traj")
        z = np.concatenate([z_src, z[1:]])
        Lz = np.concatenate([Lz_src, Lz[1:]])
    nfr = len(z)
    total_ps = float(meta.get("time_ps", nfr - 1))
    if src:
        total_ps += float(meta.get("t0_ps", 0.0))
    dt = total_ps / max(1, nfr - 1)
    if meta.get("status") != "complete" and "dt_fs" in meta:
        dt = 1000 * float(meta["dt_fs"]) * 1e-3      # TRAJ_INTERVAL * dt
    T = int(round(float(meta.get("temperature_K", 1463))))
    return z, Lz, sym, dt, T


def hist_series(z, mask, bins, stride, window, zsigma=ZSIGMA):
    """(n_anim, nbin) number density (atoms/A). Animation frame k is the mean
    histogram of traj frames [k*stride, (k+1)*stride) -- every frame's data is
    used, so 18 Dy atoms x 10 frames = 180 samples per profile instead of 18."""
    h = np.stack([np.histogram(zf[mask], bins=bins)[0] for zf in z]) / DZ
    blocks = [h[i:i + stride].mean(axis=0) for i in range(0, len(h), stride)]
    h = np.stack(blocks)
    if window > 1:
        k = np.ones(window)
        num = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, h)
        den = np.convolve(np.ones(len(h)), k, mode="same")[:, None]
        h = num / den
    if zsigma > 0:
        # smooth the SHAPE along z: normalized Gaussian kernel preserves the
        # area (atom count); edge renormalization keeps it exact at the ends
        half = int(np.ceil(3 * zsigma / DZ))
        x = np.arange(-half, half + 1) * DZ
        k = np.exp(-0.5 * (x / zsigma) ** 2)
        k /= k.sum()
        num = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 1, h)
        den = np.convolve(np.ones(h.shape[1]), k, mode="same")[None, :]
        h = num / den
    return h


def save_legend(species, out):
    """The movie panels carry no legend -- write it once as its own PNG."""
    from matplotlib.lines import Line2D
    handles = []
    for sp in species:
        c = H.SPECIES_COLOR.get(sp, "k")
        handles.append(Line2D([], [], color=c, lw=1.4, ls=":", alpha=0.75,
                              label=f"{sp}, $t$=0"))
        handles.append(Line2D([], [], color=c, lw=2.6, label=sp))
    fig = plt.figure(figsize=(2.0, 1.4))
    fig.legend(handles=handles, loc="center", frameon=True)
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {out}")


def make_movie(rd, system, gif=False, fps=FPS, stride=STRIDE, window=WINDOW,
               zsigma=ZSIGMA, outdir=None, limit=None):
    z, Lz, sym, dt, T = load_wrapped_z(rd)
    if limit:
        # cap at `limit` animation frames (same meaning as
        # prod_structure_movie.py --limit, so paired movies stay frame-aligned)
        z = z[:limit * stride]
    species = [sp for sp in H.DOPANTS if sp in sym]
    if not species:
        print(f"  SKIP (no Dy/Mg): {rd}")
        return
    bins = np.arange(0.0, Lz[0] + DZ, DZ)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    prof = {sp: hist_series(z, sym == sp, bins, stride, window, zsigma)
            for sp in species}
    # animation frame k covers traj frames [k*stride, ...) -> label its start,
    # the same instant prod_structure_movie.py renders for frame k
    dt_anim = dt * stride
    ymax = 1.15 * max(p.max() for p in prof.values())

    save_legend(species, outdir / f"dopant_zmovie_legend_{system}_{T}K.png")

    # ONE box spanning TWO periods stacked along z (0 -> 2*Lz, profile tiled,
    # no seam at the middle) = the 2x z-repeat of the paired structure render
    # (--repeat 1x10x2 --zvert). z is the VERTICAL axis and increases UPWARD,
    # matching OVITO's z-up camera in --zvert mode. No BTO-grain shading: the
    # perovskite fills the whole cell along z (Ti runs 0 -> Lz with no gap), so
    # a Ti marker would just stripe every lattice plane and mark nothing. The
    # dotted t=0 curve is the reservoir reference instead.
    # figsize*110 dpi must land on EVEN pixel counts (libx264 yuv420p rejects
    # odd frame sizes; 4.6 in truncates to 505 px) -- 4.62x11.2 -> 508x1232
    Lz0 = Lz[0]
    ctr2 = np.concatenate([ctr, ctr + Lz0])
    fig, ax = plt.subplots(figsize=(4.62, 11.2), constrained_layout=True)
    lines, fills = {}, {}
    for sp in species:
        c = H.SPECIES_COLOR.get(sp, "k")
        p0 = np.tile(prof[sp][0], 2)
        ax.plot(p0, ctr2, color=c, lw=1.4, ls=":", alpha=0.75)
        lines[sp], = ax.plot(p0, ctr2, color=c, lw=2.6, zorder=3)
        fills[sp] = ax.fill_betweenx(ctr2, 0.0, p0, color=c, alpha=0.25, lw=0)
    ax.set_ylim(0, 2 * Lz0)
    ax.set_xlim(0, ymax)
    ax.set_ylabel(r"z ($\AA$)")
    ax.set_xlabel(r"number density (atoms/$\AA$)", fontsize=S.fs(16))
    ax.set_title("Dopant distribution along z")
    clock = ax.text(0.96, 0.99, "", transform=ax.transAxes,
                    ha="right", va="top", fontweight="bold",
                    fontsize=S.fs(19))

    nfr = len(next(iter(prof.values())))

    def update(i):
        for sp in species:
            p = np.tile(prof[sp][i], 2)
            lines[sp].set_xdata(p)
            fills[sp].remove()
            fills[sp] = ax.fill_betweenx(
                ctr2, 0.0, p,
                color=H.SPECIES_COLOR.get(sp, "k"), alpha=0.25, lw=0)
        clock.set_text(f"{i * dt_anim:7.0f} ps")
        if i and i % 50 == 0:
            print(f"     frame {i}/{nfr}", flush=True)
        return ()

    ext = "gif" if gif else "mp4"
    out = outdir / f"dopant_zmovie_{system}_{T}K.{ext}"
    # -threads 2 is REQUIRED, not tuning: this box has ulimit -u 500 with ~1.4k
    # threads already live, so libx264's default 15 threads fail to spawn and
    # the encoder dies with "ff_frame_thread_encoder_init failed" for any frame
    # taller than ~480 px. Quality knob is -crf; matplotlib's bitrate= would
    # emit a bare "-b", which this ffmpeg build rejects as ambiguous.
    writer = (PillowWriter(fps=fps) if gif else
              FFMpegWriter(fps=fps, codec="libx264",
                           extra_args=["-threads", "2", "-crf", "20",
                                       "-pix_fmt", "yuv420p"]))
    anim = FuncAnimation(fig, update, frames=nfr, blit=False)
    anim.save(str(out), writer=writer, dpi=110)
    plt.close(fig)
    print(f"   -> {out}  ({nfr} frames from {len(z)} traj frames, stride "
          f"{stride} = {dt_anim:g} ps/frame, {nfr / fps:.0f} s at {fps} fps, "
          f"species={'+'.join(species)})")


# ------------------------------------------------------------------- driver
def discover():
    runs = []
    for sd in sorted(ROOT.glob("0[0-9]_hetero_*")):
        if not sd.is_dir():
            continue
        for tr in sorted(sd.glob("*/prod*/traj.traj")):
            rd = tr.parent.resolve()
            if rd not in runs:
                runs.append(rd)
    return runs


def animate(rd, partial=False, **kw):
    rd = rd.resolve()
    outdir = rd.parent / "plot"
    outdir.mkdir(exist_ok=True)
    system = rd.name
    meta_p = rd / "run_meta.json"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            src = meta.get("continuation_from")
            if src:
                # stitched part1+part2 movie, named after the source run
                system = Path(src).name + "_stitched"
            status = meta.get("status")
            if status != "complete":
                if not partial:
                    print(f"  SKIP (status={status}): {rd}")
                    return
                system += "_partial"
                print(f"  [partial] status={status}: animating traj so far")
        except Exception:
            pass
    print(f"== {rd.parent.parent.name} / {system}")
    make_movie(rd, system, outdir=outdir, **kw)


if __name__ == "__main__":
    argv = sys.argv[1:]
    partial = "--partial" in argv
    gif = "--gif" in argv
    fps, stride, window, zsigma, limit = FPS, STRIDE, WINDOW, ZSIGMA, None
    rest = []
    it = iter([a for a in argv if a not in ("--partial", "--gif")])
    for a in it:
        if a == "--fps":
            fps = int(next(it))
        elif a == "--stride":
            stride = int(next(it))
        elif a == "--window":
            window = int(next(it))
        elif a == "--zsigma":
            zsigma = float(next(it))
        elif a == "--limit":
            limit = int(next(it))
        else:
            rest.append(Path(a))
    runs = rest or discover()
    if not runs:
        print("no production runs found")
    for r in runs:
        animate(r, partial=partial, gif=gif, fps=fps, stride=stride,
                window=window, zsigma=zsigma, limit=limit)
