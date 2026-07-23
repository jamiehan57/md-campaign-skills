#!/usr/bin/env python3
"""
Per-production-run cation analysis for 03_SURF_HETERO, saved into each
structure directory's plot/ subdir (e.g. 02_hetero_Dy2O3_pristine_crystal/plot/).
Four figures per run, all in the shared campaign style (_analysis_style /
hop_screen); <run> is the production dir name (prod_1ns, prod_5ns, ...) so
different-length runs of the same structure don't overwrite each other:

  1. msd_<run>_<T>K_{raw,smooth}.png      total per-species MSD (hop_screen.plot_msd)
  2. msd_xyz_<run>_<T>K_{raw,smooth}.png  MSD split by x/y/z displacement,
                                          3 panels in a row, shared y axis
  2b. msd_parperp_<run>_<T>K_{raw,smooth}.png
                                          MSD in-plane (x+y, parallel to the
                                          interface) vs out-of-plane (z),
                                          2 panels, shared y axis
  3. cn_cationO_<run>_<T>K.png            species-averaged cation-O coordination
                                          number vs time; per-species cutoff from
                                          the first minimum of the X-O distance
                                          histogram (g(r)-like, r^2-normalised)
  4. zprofile_<run>_<T>K.png              per-element density profile along z,
                                          mean of the first vs last N_AVG frames

Run (login node, single-thread BLAS):
  env OMP_NUM_THREADS=1 python scripts/prod_cation_analysis.py [--partial] [run_dir ...]
No args: discovers every 0?_hetero_*/*/prod*/traj.traj (skips runs whose
run_meta.json is not status=complete).
--partial: analyse still-running runs from the traj written so far; outputs
are suffixed _partial so the eventual full-run plots don't collide. msd.dat
only exists after completion, so the total-MSD figure is skipped mid-run
(the xyz / par-perp MSD figures cover it from traj.traj).
"""
import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _analysis_style as S            # noqa: F401  pins BLAS threads, Arial style
import numpy as np
import matplotlib.pyplot as plt
import hop_screen as H                 # SPECIES_COLOR / DOPANTS / mic / plot_msd
from ase.io.trajectory import Trajectory

ROOT = HERE.parent
CATIONS = ("Ba", "Ti", "Dy", "Mg")
N_AVG = 10                             # frames averaged for start/end z-profiles
SMOOTH_PS = 10.0                       # running-mean window for "smooth" curves
CN_SAMPLE_FR = 16                      # frames sampled for the cutoff histogram


def runmean(y, w):
    """Edge-corrected running mean (no mode='same' dip at the ends)."""
    if len(y) < w:
        return y
    k = np.ones(w)
    return np.convolve(y, k, mode="same") / np.convolve(np.ones(len(y)), k,
                                                        mode="same")


def style(sp):
    dop = sp in H.DOPANTS
    return dict(color=H.SPECIES_COLOR.get(sp, "k"), lw=2.6 if dop else 1.3,
                alpha=1.0 if dop else 0.5, zorder=3 if dop else 1)


def _read_traj(path):
    traj = Trajectory(str(path))
    nfr = len(traj)
    sym = np.array(traj[0].get_chemical_symbols())
    pos = np.empty((nfr, len(sym), 3))
    cells = np.empty((nfr, 3, 3))
    for i, at in enumerate(traj):
        pos[i] = at.get_positions()
        cells[i] = at.get_cell().array
    return pos, cells, sym


def load_run(rd, tmax_ps=None):
    meta = {}
    mp = rd / "run_meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = {}
    pos, cells, sym = _read_traj(rd / "traj.traj")
    # continuation runs (16_continue_nvt.py): prepend the source trajectory,
    # dropping the continuation's frame 0 (duplicate of the source's last)
    src = meta.get("continuation_from")
    if src:
        pos0, cells0, _ = _read_traj(Path(src) / "traj.traj")
        pos = np.concatenate([pos0, pos[1:]])
        cells = np.concatenate([cells0, cells[1:]])
    nfr = len(pos)
    # traj positions are wrapped into the cell (in-plane L ~ 10-11.5 A), so a
    # boundary crossing looks like a ~L jump; rebuild continuous paths from
    # per-frame MIC steps before any displacement-based analysis
    for i in range(1, nfr):
        pos[i] = pos[i - 1] + H.mic(pos[i] - pos[i - 1], cells[i])
    total_ps = float(meta.get("time_ps", nfr - 1))
    if src:
        total_ps += float(meta.get("t0_ps", 0.0))
    dt = total_ps / max(1, nfr - 1)
    if meta.get("status") != "complete" and "dt_fs" in meta:
        # partial (still-running) run: meta time_ps is the PLANNED length, so
        # use the actual frame spacing (nvt_production TRAJ_INTERVAL = 1000
        # steps between traj writes)
        dt = 1000 * float(meta["dt_fs"]) * 1e-3
    if tmax_ps is not None:
        n = int(round(tmax_ps / dt)) + 1
        if n < nfr:
            pos, cells = pos[:n], cells[:n]
    T = int(round(float(meta.get("temperature_K", 1463))))
    return pos, cells, sym, dt, T, meta


def pair_dists(pos_f, cell, xi, oi):
    """|X-O| MIC distances for one frame -> (n_X, n_O)."""
    dv = pos_f[oi][None, :, :] - pos_f[xi][:, None, :]
    dv = H.mic(dv.reshape(-1, 3), cell).reshape(len(xi), len(oi), 3)
    return np.linalg.norm(dv, axis=-1)


# ---------------------------------------------------------------- 2. MSD xyz
def plot_msd_xyz(pos, sym, dt, system, T, outdir):
    t = np.arange(len(pos)) * dt
    disp2 = (pos - pos[0]) ** 2                       # unwrapped trajectory
    species = [s for s in ("Ba", "Ti", "Dy", "Mg", "O") if s in sym]
    msd = {sp: disp2[:, sym == sp, :].mean(axis=1) for sp in species}  # (nfr, 3)
    win = max(3, int(round(SMOOTH_PS / dt)))
    for mode in ("raw", "smooth"):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True,
                                 constrained_layout=True)
        for k, (ax, lab) in enumerate(zip(axes, ("x", "y", "z"))):
            for sp in species:
                y = msd[sp][:, k]
                if mode == "smooth":
                    y = runmean(y, win)
                ax.plot(t, y, label=sp, **style(sp))
            ax.set_title(lab)
            ax.set_xlabel("time (ps)")
        axes[0].set_ylabel(r"MSD ($\AA^2$)")
        axes[-1].legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        borderaxespad=0.)
        fig.suptitle("Mean Squared Displacement by component", fontweight="bold")
        fig.savefig(outdir / f"msd_xyz_{system}_{T}K_{mode}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------- 2b. MSD parallel/perpendicular
def plot_msd_par_perp(pos, sym, dt, system, T, outdir):
    """Two panels: in-plane MSD (x+y, parallel to the interface) and
    out-of-plane MSD (z, normal to the interface)."""
    t = np.arange(len(pos)) * dt
    disp2 = (pos - pos[0]) ** 2                       # unwrapped trajectory
    species = [s for s in ("Ba", "Ti", "Dy", "Mg", "O") if s in sym]
    comp = {sp: disp2[:, sym == sp, :].mean(axis=1) for sp in species}
    msd = {sp: {"par": comp[sp][:, 0] + comp[sp][:, 1],
                "perp": comp[sp][:, 2]} for sp in species}
    win = max(3, int(round(SMOOTH_PS / dt)))
    titles = (("par", r"parallel to interface (x+y)"),
              ("perp", r"normal to interface (z)"))
    for mode in ("raw", "smooth"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True,
                                 constrained_layout=True)
        for ax, (key, lab) in zip(axes, titles):
            for sp in species:
                y = msd[sp][key]
                if mode == "smooth":
                    y = runmean(y, win)
                ax.plot(t, y, label=sp, **style(sp))
            ax.set_title(lab)
            ax.set_xlabel("time (ps)")
        axes[0].set_ylabel(r"MSD ($\AA^2$)")
        axes[-1].legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        borderaxespad=0.)
        fig.suptitle("Mean Squared Displacement — in-plane vs out-of-plane",
                     fontweight="bold")
        fig.savefig(outdir / f"msd_parperp_{system}_{T}K_{mode}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------- 3. cation-O CN(t)
def first_shell_cutoff(pos, cells, sym, sp):
    """Cutoff = first minimum after the first peak of the r^2-normalised X-O
    distance histogram, pooled over CN_SAMPLE_FR frames."""
    xi = np.where(sym == sp)[0]
    oi = np.where(sym == "O")[0]
    frames = np.linspace(0, len(pos) - 1, CN_SAMPLE_FR).astype(int)
    d = np.concatenate([pair_dists(pos[f], cells[f], xi, oi).ravel()
                        for f in frames])
    d = d[d < 6.0]
    hist, edges = np.histogram(d, bins=220, range=(0.5, 6.0))
    r = 0.5 * (edges[1:] + edges[:-1])
    g = runmean(hist / r ** 2, 5)
    pk = int(np.argmax(np.where(r < 3.4, g, 0.0)))
    jmax = int(np.searchsorted(r, min(r[pk] + 2.0, 4.3)))
    j = pk + int(np.argmin(g[pk:jmax]))
    return float(r[j])


def plot_cn(pos, cells, sym, dt, system, T, outdir):
    t = np.arange(len(pos)) * dt
    oi = np.where(sym == "O")[0]
    cations = [sp for sp in CATIONS if sp in sym]
    cut = {sp: first_shell_cutoff(pos, cells, sym, sp) for sp in cations}
    cn = {sp: np.empty(len(pos)) for sp in cations}
    xidx = {sp: np.where(sym == sp)[0] for sp in cations}
    for f in range(len(pos)):
        for sp in cations:
            d = pair_dists(pos[f], cells[f], xidx[sp], oi)
            cn[sp][f] = (d < cut[sp]).sum(axis=1).mean()
    win = max(3, int(round(SMOOTH_PS / dt)))
    # size matches the stability figure (10 x 5.6, exact canvas, no tight
    # crop); fixed margins reserve room for the outside mean-CN labels
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.11, right=0.88, bottom=0.14, top=0.90)
    for sp in cations:
        st = style(sp)
        ax.plot(t, cn[sp], color=st["color"], lw=1.0, alpha=0.25, zorder=1)
        ax.plot(t, runmean(cn[sp], win), label=f"{sp}–O ($r_c$={cut[sp]:.2f} $\\AA$)",
                **{**st, "alpha": 1.0, "zorder": st["zorder"] + 1})
    # time-averaged CN: dashed guide line per species; the value is written
    # OUTSIDE the axes, just right of the spine at the height of its own line
    # (y in data coords, x in axes coords), de-overlapped when means are close
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    lab_h = 0.062 * yr                     # approx label height in data units
    means = {sp: float(cn[sp].mean()) for sp in cations}
    for sp in cations:
        ax.axhline(means[sp], color=H.SPECIES_COLOR.get(sp, "k"),
                   ls="--", lw=1.1, alpha=0.6, zorder=4)
    order = sorted(cations, key=lambda s: means[s])
    ys = [means[s] for s in order]
    sep = 1.3 * lab_h                      # min center-to-center label spacing
    for _ in range(20):                    # push apart labels closer than sep
        for i in range(len(ys) - 1):
            gap = ys[i + 1] - ys[i]
            if gap < sep:
                ys[i] -= 0.5 * (sep - gap)
                ys[i + 1] += 0.5 * (sep - gap)
    for sp, y in zip(order, ys):
        ax.text(1.015, y, f"{means[sp]:.2f}",
                transform=ax.get_yaxis_transform(), clip_on=False,
                color=H.SPECIES_COLOR.get(sp, "k"), fontweight="bold",
                ha="left", va="center")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("coordination number")
    ax.set_title("Cation–O Coordination Number")
    fig.savefig(outdir / f"cn_cationO_{system}_{T}K.png", dpi=300)
    plt.close(fig)
    # legend saved as its own PNG (slide convention, like the movie legend)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=style(sp)["color"], lw=style(sp)["lw"],
                      label=f"{sp}–O ($r_c$={cut[sp]:.2f} $\\AA$)")
               for sp in cations]
    figl = plt.figure(figsize=(2.8, 1.6))
    figl.legend(handles=handles, loc="center", frameon=True)
    figl.savefig(outdir / f"cn_cationO_legend_{system}_{T}K.png", dpi=300,
                 bbox_inches="tight")
    plt.close(figl)
    return {sp: {"cutoff_A": cut[sp], "cn_mean_last100ps":
                 float(cn[sp][t >= t[-1] - 100.0].mean())} for sp in cations}


# ------------------------------------------------------------ 4. z-profile
def plot_zprofile(pos, cells, sym, dt, system, T, outdir):
    species = [s for s in ("Ba", "Ti", "Dy", "Mg", "O") if s in sym]
    Lz = cells[0][2, 2]
    bins = np.arange(0.0, Lz + 0.5, 0.5)
    ctr = 0.5 * (bins[1:] + bins[:-1])

    def profile(frames):
        # wrapped fractional z, averaged over the frame block -> atoms per A
        prof = {}
        for sp in species:
            m = sym == sp
            h = np.zeros(len(ctr))
            for f in frames:
                fz = (pos[f][m] @ np.linalg.inv(cells[f]))[:, 2] % 1.0
                h += np.histogram(fz * cells[f][2, 2], bins=bins)[0]
            y = h / len(frames) / 0.5
            yp = np.r_[y[-3:], y, y[:3]]          # periodic smoothing along z
            prof[sp] = runmean(yp, 3)[3:-3]
        return prof

    p0 = profile(range(N_AVG))
    p1 = profile(range(len(pos) - N_AVG, len(pos)))
    t_ps = (len(pos) - 1) * dt
    fig, axes = plt.subplots(len(species), 1, figsize=(12, 2.1 * len(species)),
                             sharex=True, constrained_layout=True)
    for ax, sp in zip(np.atleast_1d(axes), species):
        c = H.SPECIES_COLOR.get(sp, "k")
        # start: light shade + dotted line; end: darker shade + solid line
        ax.fill_between(ctr, p0[sp], color=c, alpha=0.10, lw=0)
        ax.plot(ctr, p0[sp], color=c, lw=1.6, ls=":",
                label=f"first {N_AVG} frames")
        ax.fill_between(ctr, p1[sp], color=c, alpha=0.38, lw=0)
        ax.plot(ctr, p1[sp], color=c, lw=1.6, label=f"last {N_AVG} frames")
        ax.text(0.015, 0.90, sp, transform=ax.transAxes, color=c,
                fontweight="bold", ha="left", va="top")
        ax.set_ylim(bottom=0)
    np.atleast_1d(axes)[0].legend(ncol=1, loc="upper left",
                                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    np.atleast_1d(axes)[-1].set_xlabel(r"z ($\AA$)")
    fig.text(-0.015, 0.5, r"number density (atoms/$\AA$)", rotation=90,
             va="center", ha="center", fontsize=S.fs(18))
    fig.suptitle(f"Element distribution along z — 0 vs {t_ps:.0f} ps "
                 f"(each averaged over {N_AVG} frames)", fontweight="bold")
    fig.savefig(outdir / f"zprofile_{system}_{T}K.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)



# --------------------------------------- 5. z-MSD, Dy film-stayers vs escapees
FILM_MARGIN_A = 2.5            # film band = initial Dy z-span +- this margin
DY_FILM_COLOR = "#9B93FB"      # light tone of the Dy blue (#3106FC)

# Optional film-band reference: if set (via --film-ref), the band uses the
# AS-BUILT Dy slab THICKNESS (from a pre-MD reference POSCAR, cell-independent
# fractional half-width), re-centred on the current frame-0 Dy median -- so
# equilibration-stage film swelling no longer inflates the band and Dy that
# left the ORIGINAL film footprint are counted. None -> legacy frame-0 span.
_FILM_REF_HALFWIDTH_FRAC = None


def set_film_ref(path):
    """Load a reference structure and store the Dy slab fractional half-width."""
    global _FILM_REF_HALFWIDTH_FRAC
    from ase.io import read as _read
    a = _read(str(path), format="vasp")
    sym = np.array(a.get_chemical_symbols())
    Lz = a.get_cell()[2, 2]
    fz = a.get_positions()[:, 2][sym == "Dy"] / Lz
    _FILM_REF_HALFWIDTH_FRAC = 0.5 * (fz.max() - fz.min())
    print(f"[film-ref] {path}: Dy slab half-width {_FILM_REF_HALFWIDTH_FRAC:.4f} "
          f"frac ({_FILM_REF_HALFWIDTH_FRAC * Lz:.1f} A in ref cell)")
    return _FILM_REF_HALFWIDTH_FRAC


def dy_film_groups(pos, cells, sym):
    """Classify Dy atoms by whether the WRAPPED z ever leaves the film band
    (initial Dy z-span +- FILM_MARGIN_A) within the analysis window.
    Returns (dy_idx, left_mask, (lo, hi)) or None if no Dy."""
    if "Dy" not in sym:
        return None
    Lz = cells[0][2, 2]
    dy = np.where(sym == "Dy")[0]
    zw = pos[:, dy, 2] % Lz                      # wrapped z (fixed NVT cell)
    z0 = zw[0]
    if _FILM_REF_HALFWIDTH_FRAC is not None:
        # as-built slab thickness, re-centred on the current film (median is
        # robust to a few Dy that already left during equilibration)
        centre = float(np.median(z0))
        half = _FILM_REF_HALFWIDTH_FRAC * Lz + FILM_MARGIN_A
        lo, hi = centre - half, centre + half
    else:
        lo, hi = z0.min() - FILM_MARGIN_A, z0.max() + FILM_MARGIN_A
    left = ((zw < lo) | (zw > hi)).any(axis=0)
    return dy, left, (float(lo), float(hi))


def plot_msd_z_groups(pos, cells, sym, dt, system, T, outdir):
    """z-only MSD with Dy categorised into Dy_diff (escaped the film band at
    some point) vs Dy_film (never left), tone-split on the Dy blue; Mg overlaid
    when present. Band test on wrapped z, MSD on the unwrapped paths.
    Smooth version only (10 ps running mean)."""
    g = dy_film_groups(pos, cells, sym)
    if g is None:
        return None
    dy, left, (lo, hi) = g
    t = np.arange(len(pos)) * dt
    disp2z = (pos[:, :, 2] - pos[0, :, 2]) ** 2
    win = max(3, int(round(SMOOTH_PS / dt)))
    groups = []
    if left.any():
        groups.append((f"Dy$_{{diff}}$ (n={int(left.sum())})",
                       dy[left], H.SPECIES_COLOR.get("Dy", "#3106FC")))
    if (~left).any():
        groups.append((f"Dy$_{{film}}$ (n={int((~left).sum())})",
                       dy[~left], DY_FILM_COLOR))
    if "Mg" in sym:
        mg = np.where(sym == "Mg")[0]
        groups.append((f"Mg (n={len(mg)})", mg,
                       H.SPECIES_COLOR.get("Mg", "#FB7B15")))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.11, right=0.78, bottom=0.14, top=0.90)
    for label, idx, color in groups:
        ax.plot(t, runmean(disp2z[:, idx].mean(axis=1), win),
                color=color, lw=2.6, label=label)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(r"MSD$_z$ ($\AA^2$)")
    ax.set_title("z-direction MSD", fontweight="bold")
    ax.set_xlim(t[0], t[-1])
    ax.text(0.02, 0.97, f"film band {lo:.1f}-{hi:.1f} $\\AA$ "
            f"(initial Dy span $\\pm${FILM_MARGIN_A:g} $\\AA$)",
            transform=ax.transAxes, va="top",
            fontsize=plt.rcParams["font.size"] * 0.62, color="#666666")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    ax.text(0.02, 0.89, f"{int(left.sum())} / {len(dy)} Dy crossed into BTO",
            transform=ax.transAxes, va="top", fontweight="bold",
            fontsize=plt.rcParams["font.size"] * 0.75,
            color=H.SPECIES_COLOR.get("Dy", "#3106FC"))
    fig.savefig(outdir / f"msd_z_dopantgroups_{system}_{T}K_smooth.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"n_diff": int(left.sum()), "n_film": int((~left).sum()),
            "band_A": (round(lo, 2), round(hi, 2))}


# ------------------------- 5b. dopant z-position paths (direction-resolving)
def plot_dopant_z_paths(pos, cells, sym, dt, system, T, outdir):
    """z-position vs time for the dopant groups (Dy_diff / Dy_film / Mg),
    expressed as SIGNED distance from the film centre (nearest-image), so the
    direction of out-of-plane motion is visible (MSD loses the sign):
      each atom  -> thin semi-transparent trace
      group mean -> solid line (10 ps smooth)
      group span -> filled min..max band
    The film band is shaded grey. + = drift toward +z BTO, - = toward -z BTO."""
    g = dy_film_groups(pos, cells, sym)
    if g is None:
        return
    dy, left, (lo, hi) = g
    t = np.arange(len(pos)) * dt
    Lz = cells[0][2, 2]
    centre = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    zw = pos[:, :, 2] % Lz
    # nearest image to the film centre -> signed offset in [-Lz/2, Lz/2]
    zrel = ((zw - centre + Lz / 2.0) % Lz) - Lz / 2.0
    win = max(3, int(round(SMOOTH_PS / dt)))
    groups = []
    if left.any():
        groups.append((f"Dy$_{{diff}}$ (n={int(left.sum())})",
                       dy[left], H.SPECIES_COLOR.get("Dy", "#3106FC")))
    if (~left).any():
        groups.append((f"Dy$_{{film}}$ (n={int((~left).sum())})",
                       dy[~left], DY_FILM_COLOR))
    if "Mg" in sym:
        mg = np.where(sym == "Mg")[0]
        groups.append((f"Mg (n={len(mg)})", mg,
                       H.SPECIES_COLOR.get("Mg", "#FB7B15")))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.11, right=0.78, bottom=0.14, top=0.90)
    ax.axhspan(-half, half, color="#d9d9d9", alpha=0.6, zorder=0, lw=0)
    ax.axhline(0.0, color="#999999", lw=0.8, ls="--", zorder=1)
    for label, idx, color in groups:
        Z = zrel[:, idx]                                  # (nfr, n)
        # Dy_film stays put -> mean line only (no raw traces / range band,
        # they just clutter the plot); Dy_diff and Mg get the full detail.
        if color != DY_FILM_COLOR:
            for j in range(Z.shape[1]):
                ax.plot(t, Z[:, j], color=color, lw=0.6, alpha=0.22, zorder=2)
            lo_b = runmean(Z.min(axis=1), win)
            hi_b = runmean(Z.max(axis=1), win)
            ax.fill_between(t, lo_b, hi_b, color=color, alpha=0.12, zorder=2, lw=0)
        ax.plot(t, runmean(Z.mean(axis=1), win), color=color, lw=2.6,
                label=label, zorder=4)
    ax.text(0.02, 0.97, f"film band $\\pm${half:.1f} $\\AA$ about its centre "
            f"(z={centre:.1f} $\\AA$)", transform=ax.transAxes, va="top",
            fontsize=plt.rcParams["font.size"] * 0.62, color="#666666")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(r"z $-$ film centre ($\AA$)")
    ax.set_title("Dopant z-position vs time", fontweight="bold")
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    fig.savefig(outdir / f"zpath_dopantgroups_{system}_{T}K.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- 6. diffusivity bar plot
D_FLOOR = 1e-9                 # cm^2/s display floor for log bars


def plot_diffusivity_bar(pos, cells, sym, dt, system, T, outdir):
    """Tracer diffusivity per species as TWO log-scale bar panels: in-plane
    (x+y, Einstein D = slope/4) and z (D = slope/2), Dy split into
    Dy_diff / Dy_film via the film band. Linear fit over the second half of
    the window. Values labelled in cm^2/s (1 A^2/ps = 1e-4 cm^2/s)."""
    t = np.arange(len(pos)) * dt
    if len(t) < 10:
        return
    d = pos - pos[0]
    panels = (("in-plane (x+y),  D = slope/4",
               d[:, :, 0] ** 2 + d[:, :, 1] ** 2, 4.0),
              ("z,  D = slope/2", d[:, :, 2] ** 2, 2.0))
    fit = t >= t[-1] / 2
    g = dy_film_groups(pos, cells, sym)
    bars = []                                          # (label, idx, color)
    for sp in ("Ba", "Ti", "O"):
        if sp in sym:
            bars.append((sp, np.where(sym == sp)[0],
                         H.SPECIES_COLOR.get(sp, "k")))
    if g is not None:
        dy, left, _ = g
        if left.any():
            bars.append(("Dy$_{diff}$", dy[left],
                         H.SPECIES_COLOR.get("Dy", "#3106FC")))
        if (~left).any():
            bars.append(("Dy$_{film}$", dy[~left], DY_FILM_COLOR))
    if "Mg" in sym:
        bars.append(("Mg", np.where(sym == "Mg")[0],
                     H.SPECIES_COLOR.get("Mg", "#FB7B15")))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.82,
                        wspace=0.06)
    x = np.arange(len(bars))
    for ax, (lab, disp2, denom) in zip(axes, panels):
        values = []
        for label, idx, color in bars:
            msd = disp2[:, idx].mean(axis=1)
            slope = np.polyfit(t[fit], msd[fit], 1)[0]     # A^2/ps
            values.append(slope / denom * 1e-4)            # cm^2/s
        ax.bar(x, [max(v, D_FLOOR) for v in values],
               color=[c for _, _, c in bars], width=0.62)
        for xi, v in zip(x, values):
            ax.text(xi, max(v, D_FLOOR) * 1.15,
                    f"{v:.2g}" if v > D_FLOOR else f"<{D_FLOOR:g}",
                    ha="center", va="bottom",
                    fontsize=plt.rcParams["font.size"] * 0.62)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in bars],
                           fontsize=plt.rcParams["font.size"] * 0.85)
        ax.set_title(lab)
        ax.set_ylim(bottom=D_FLOOR)
    axes[0].set_ylabel(r"D (cm$^2$/s)")
    axes[0].text(0.02, 0.97, f"fit {t[fit][0]:.0f}-{t[-1]:.0f} ps",
                 transform=axes[0].transAxes, ha="left", va="top",
                 fontsize=plt.rcParams["font.size"] * 0.62, color="#666666")
    fig.suptitle("Tracer Diffusivity", fontweight="bold", y=1.02)
    fig.savefig(outdir / f"diffusivity_{system}_{T}K.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- driver
def discover():
    runs = []
    for sd in sorted(ROOT.glob("0[0-9]_hetero_*")):
        if not sd.is_dir():
            continue
        for tr in sorted(sd.glob("*/prod*/traj.traj")):
            rd = tr.parent.resolve()          # 1463K symlink == 01_uncompensated
            if rd not in runs:
                runs.append(rd)
    return runs


def analyze(rd, partial=False, tmax_ps=None, xyz=False, film_ref=None):
    rd = rd.resolve()
    if film_ref is not None:
        ref = rd.parent / "input.POSCAR" if str(film_ref) == "auto" else Path(film_ref)
        if ref.exists():
            set_film_ref(ref)
        else:
            print(f"  [film-ref] {ref} not found -- falling back to frame-0 band")
    struct_dir = rd.parent.parent          # <struct>/<01_uncompensated>/<prod_*>
    # plots go in the SUBDIR (01_uncompensated/, 02_compensated_8VO/, ...), not
    # the structure dir -- 03_vba has BOTH an uncompensated and a compensated
    # run with the same prod name, which collided in <struct>/plot/ (2026-07-16).
    outdir = rd.parent / "plot"
    if tmax_ps is not None:
        # --tmax/--tlim N: cut plots live in their own plot/<N>ps/ subdir
        # (e.g. plot/3000ps/); the top-level plot/ keeps full-trajectory figures
        outdir = outdir / f"{tmax_ps:g}ps"
    outdir.mkdir(parents=True, exist_ok=True)
    system = rd.name                       # e.g. prod_1ns; struct name is the dir
    meta_p = rd / "run_meta.json"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            if meta.get("continuation_from"):
                # stitched part1+part2 analysis, named after the source run
                system = Path(meta["continuation_from"]).name + "_stitched"
            status = meta.get("status")
            if status != "complete":
                if not partial:
                    print(f"  SKIP (status={status}): {rd}")
                    return
                system += "_partial"
                print(f"  [partial] status={status}: analysing traj so far")
        except Exception:
            pass
    if tmax_ps is not None:
        system += f"_to{tmax_ps / 1000:g}ns"
    print(f"== {struct_dir.name} / {system}")
    pos, cells, sym, dt, T, _ = load_run(rd, tmax_ps=tmax_ps)
    H.plot_msd(rd, system, T, outdir)
    if xyz:
        plot_msd_xyz(pos, sym, dt, system, T, outdir)   # opt-in since 2026-07-23
    plot_msd_par_perp(pos, sym, dt, system, T, outdir)
    zg = plot_msd_z_groups(pos, cells, sym, dt, system, T, outdir)
    if zg:
        print(f"   Dy groups: {zg['n_diff']} escaped / {zg['n_film']} stayed "
              f"(film band {zg['band_A'][0]}-{zg['band_A'][1]} A)")
    plot_dopant_z_paths(pos, cells, sym, dt, system, T, outdir)
    plot_diffusivity_bar(pos, cells, sym, dt, system, T, outdir)
    cn_info = plot_cn(pos, cells, sym, dt, system, T, outdir)
    plot_zprofile(pos, cells, sym, dt, system, T, outdir)
    for sp, v in cn_info.items():
        print(f"   {sp}-O  rc={v['cutoff_A']:.2f} A  "
              f"CN(last 100 ps)={v['cn_mean_last100ps']:.2f}")
    print(f"   -> plots in {outdir}/")


if __name__ == "__main__":
    argv = sys.argv[1:]
    partial = "--partial" in argv
    xyz = "--xyz" in argv
    film_ref = None
    tmax_ps = None
    args = []
    it = iter([a for a in argv if a not in ("--partial", "--xyz")])
    for a in it:
        if a in ("--tmax", "--tlim"):
            tmax_ps = float(next(it))      # ps; cut the trajectory here
        elif a == "--film-ref":
            film_ref = next(it)            # POSCAR path or "auto"
        else:
            args.append(Path(a))
    runs = args or discover()
    if not runs:
        print("no production runs found")
    for r in runs:
        analyze(r, partial=partial, tmax_ps=tmax_ps, xyz=xyz, film_ref=film_ref)
