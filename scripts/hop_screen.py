#!/usr/bin/env python3
"""
Hop-detection SCREENER over finished MLIP-MD productions (implements
hop_detection_methods.md, methods 1-4 + Plot A diagnostic panel).

For every COMPLETED production (traj.traj + run_meta status=complete) under the
given roots, for each dopant (Dy on Ba-sublattice, Mg on Ti-sublattice):

  M1 per-atom displacement   delta_r(t) = |r_unwrap(t) - r(0)|   (MIC-unwrapped)
  M2 reference-site index    nearest frame-0 like-cation site each frame
                             (lattice-free: ref sites + d_NN MEASURED from frame 0,
                              so it works in bulk / strain / GB alike)
  M3 running std of position  sliding 10 ps window (vibration ~flat, hop spikes)
  M4 sustained threshold      hop CONFIRMED if the assigned site changes and the new
                              site is held >= SUSTAIN_PS; an excursion past
                              0.8*a0 that returns = ATTEMPT
  a0 = measured median nearest-neighbour like-cation spacing (~3.9-4.0 A);
       thresholds: hop 0.8*a0, halfway 0.5*a0, vibration 0.15*a0 (doc: 3.2/2.0/0.6 A).

Per trajectory: Plot A panel (M1/M2/M3 stacked, M4 hop lines) + hop_events CSV.
Master: hop_screen_summary.csv + printed table = WHERE hops actually happened.

NOTE: a displacement+site "confirmed hop" can still be a large static DISTORTION
(e.g. bulk 03 eps_5: the dopant moved but filled no vacancy). The summary flags
confirmed hops with "verify:vacancy_tracker" -- cross-check with
scripts/vacancy_tracker.py (the occupancy arbiter) before final claims.

Usage (login node, single-thread BLAS):
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      python scripts/hop_screen.py --discover            # all finished runs
  python scripts/hop_screen.py <run_dir> [<run_dir> ...] # specific runs
  options: --roots 01_BULK 02_GB   --no-plot   --outdir 04_analysis/hop_detection
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, csv, argparse, glob, re
from pathlib import Path
import numpy as np
from ase.io.trajectory import Trajectory
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _analysis_style as S        # noqa: E402  shared plot style (Arial, fontsizes); S.fs()

HOST = {"Dy": ("Ba", "Dy"), "Mg": ("Ti", "Mg")}     # mover -> like-cation sublattice
COLOR = {"Dy": "#1f3a8a", "Mg": "#ff7f0e"}
# project-fixed species colours for MSD plots (Ba green/Ti sky/O red/Dy indigo/Mg orange)
SPECIES_COLOR = {"Ba": "#2CA02C", "Ti": "#56B4E9", "O": "#D62728", "Dy": "#4338CA", "Mg": "#FF7F0E"}
DOPANTS = ("Dy", "Mg")
MSD_PER_HOP = 2.75                                   # A^2, ~a0^2/6 for a single nn hop
HOP_F, HALF_F, VIB_F = 0.65, 0.5, 0.15               # x a0; 0.65 (not 0.8) so short
#                                                    # GB cross-GB hops (~2.4 A) register
SUSTAIN_PS = 5.0
STD_WIN_PS = 10.0
ROOT = Path(__file__).resolve().parent.parent


def mic(d, cell):
    inv = np.linalg.inv(cell); f = d @ inv; f -= np.round(f); return f @ cell


def median_nn(P, cell):
    n = len(P)
    if n < 2: return None
    nn = np.empty(n)
    for i in range(n):
        r = np.linalg.norm(mic(P - P[i], cell), axis=1); r[i] = np.inf; nn[i] = r.min()
    return float(np.median(nn))


def parse_id(run_dir: Path):
    """-> (system, T_K, replica) from path, handling the several dir layouts."""
    rep = run_dir.name.replace("r", "") if run_dir.name.startswith("r") else "0"
    p = run_dir.parent
    rep = int(rep) if rep.isdigit() else 0
    p = run_dir.parent
    if p.name.endswith("K") and p.name[:-1].isdigit():     # legacy <sys>/<T>K/r0
        return p.parent.name, int(p.name[:-1]), rep
    T = 1463
    mp = run_dir / "run_meta.json"
    if mp.exists():
        try: T = int(json.loads(mp.read_text()).get("temperature_K", 1463))
        except Exception: pass
    if re.fullmatch(r"eps(0|_th|_3|_5)", p.name):          # strain: <structure>/<eps>/r0
        return f"{p.parent.name}_{p.name}", T, rep
    return p.name, T, rep                                   # flattened: <structure>/r0


def analyse(run_dir: Path):
    meta = {}
    mp = run_dir / "run_meta.json"
    if mp.exists(): meta = json.loads(mp.read_text())
    traj = Trajectory(str(run_dir / "traj.traj"))
    nfr = len(traj)
    if nfr < 10: return None
    dt = float(meta.get("traj_dt_ps", 1.0))
    equil_ps = float(meta.get("equil_ps", meta.get("prod_equil_ps", 0.0)))
    a0frame = traj[0]
    sym = np.array(a0frame.get_chemical_symbols())
    cell0 = a0frame.get_cell().array
    movers = [m for m in ("Dy", "Mg") if m in sym]
    if not movers:
        return {"movers": {}, "note": "no Dy/Mg"}
    midx = {m: int(np.where(sym == m)[0][0]) for m in movers}
    refidx = {m: np.where(np.isin(sym, HOST[m]))[0] for m in movers}
    a0 = {m: median_nn(a0frame.get_positions()[refidx[m]], cell0) for m in movers}
    # vacancy count per sublattice (perovskite ABO3: N_A = N_B = N_O/3 sites).
    # A real hop REQUIRES a vacancy on the mover's sublattice (Dy->V_Ba, Mg->V_Ti);
    # without one, a sustained site-change is a DISTORTION, not a migration.
    n_sites = int(np.sum(sym == "O")) // 3
    vac = {"Dy": n_sites - int(np.sum(np.isin(sym, ("Ba", "Dy")))),    # V_Ba
           "Mg": n_sites - int(np.sum(np.isin(sym, ("Ti", "Mg"))))}    # V_Ti

    # walk trajectory: unwrapped dopant pos + wrapped-pos site assignment
    U = {m: a0frame.get_positions()[midx[m]].copy() for m in movers}
    disp = {m: np.zeros(nfr) for m in movers}
    site = {m: np.zeros(nfr, int) for m in movers}
    pos3 = {m: np.zeros((nfr, 3)) for m in movers}
    ref0 = {m: a0frame.get_positions()[refidx[m]] for m in movers}
    prev = a0frame.get_positions(); prevc = cell0
    for f in range(nfr):
        at = traj[f]; cur = at.get_positions(); cell = at.get_cell().array
        if f > 0:
            for m in movers:
                U[m] = U[m] + mic((cur[midx[m]] - prev[midx[m]])[None], 0.5 * (prevc + cell))[0]
        for m in movers:
            disp[m][f] = np.linalg.norm(U[m] - a0frame.get_positions()[midx[m]])
            pos3[m][f] = U[m]
            d = np.linalg.norm(mic(ref0[m] - cur[midx[m]], cell), axis=1)
            site[m][f] = int(refidx[m][int(np.argmin(d))])
        prev = cur; prevc = cell

    win = max(1, int(round(STD_WIN_PS / dt)))
    sus = max(1, int(round(SUSTAIN_PS / dt)))
    t = np.arange(nfr) * dt
    out = {"movers": {}, "dt": dt, "equil_ps": equil_ps, "t": t, "sym": sym}
    for m in movers:
        thr, half, vib = HOP_F * a0[m], HALF_F * a0[m], VIB_F * a0[m]
        # running std of position (magnitude of per-window coordinate std)
        rstd = np.zeros(nfr)
        for f in range(nfr):
            a, b = max(0, f - win // 2), min(nfr, f + win // 2 + 1)
            rstd[f] = float(np.sqrt(np.mean(np.var(pos3[m][a:b], axis=0))))
        # M4 confirmed hops: assigned-site change HELD >= SUSTAIN_PS.
        events = []
        cur_site = site[m][0]; f = 1
        while f < nfr:
            if site[m][f] != cur_site:
                cand = site[m][f]; held = 1; g = f + 1
                while g < nfr and site[m][g] == cand:
                    held += 1; g += 1
                if held >= sus and t[f] >= equil_ps:
                    status = "confirmed" if disp[m][f:g].max() > thr else "shift"
                    events.append((f, t[f], int(cur_site), int(cand), held * dt, status))
                    cur_site = cand; f = g; continue
            f += 1
        # attempts = excursions where |dr| (from frame 0, the PLOTTED signal) CROSSES the
        # hop bar but does NOT sustain (returns below within < SUSTAIN_PS). Detecting on
        # the same signal that is plotted guarantees every attempt marker sits AT/ABOVE the
        # bar (no anchor-relative artifact placing markers low). A run >= sus above the bar
        # is the hop itself, not an attempt.
        attempts = []
        reset = HALF_F * a0[m]                       # must fall back below "halfway" to re-arm
        i = 0
        while i < nfr:
            if disp[m][i] > reset and t[i] >= equil_ps:
                j = i                                # one excursion ABOVE home (the reset level)
                while j < nfr and disp[m][j] > reset:
                    j += 1
                seg = disp[m][i:j]
                if seg.max() > thr:                  # the excursion actually reached the hop bar
                    ab = seg > thr; best = run_ = 0  # longest sustained stretch above the bar
                    for v in ab:
                        run_ = run_ + 1 if v else 0
                        best = max(best, run_)
                    if best < sus:                   # reached the bar but did NOT sustain -> attempt
                        pk = i + int(np.argmax(seg))
                        attempts.append((pk, float(t[pk]), float(disp[m][pk])))
                i = j
            else:
                i += 1
        prod = t >= equil_ps
        dmax = float(disp[m][prod].max())
        last100 = float(disp[m][-max(1, int(100/dt)):].mean())
        conf = [e for e in events if e[5] == "confirmed"]
        # collapse guard (doc): if the mover wanders many lattice spacings the
        # structure has melted -> reference lattice invalid, exclude from hops.
        vm = int(vac.get(m, 0))
        # sustained = displacement held past the bar for >=5 ps at a STRETCH (a settled
        # occupancy), not just on-average at the end -- so a reversible hop (goes then
        # returns) still registers as a real hop event. Matches verify_hop's dwell logic.
        _need = max(1, int(round(SUSTAIN_PS / dt)))
        _best = _run = 0
        for _v in (disp[m] > thr):
            _run = _run + 1 if _v else 0
            _best = max(_best, _run)
        sustained = _best >= _need
        if dmax > 4.0 * a0[m]:
            verdict = "MELT"
        elif sustained and vm > 0:
            verdict = "HOP"            # sustained displacement past the bar + a vacancy to fill
        elif sustained and vm == 0:
            verdict = "DISTORTION"     # stayed displaced but NO vacancy on this sublattice
        elif attempts or dmax > thr:
            verdict = "ATTEMPT"        # crossed the bar but returned to its site
        else:
            verdict = "RATTLE"
        out["movers"][m] = dict(a0=a0[m], thr=thr, half=half, vib=vib, n_vac=vm,
                                disp=disp[m], site=site[m], rstd=rstd, events=events,
                                attempts=attempts, pos3=pos3[m], d_max=dmax,
                                last100=last100, verdict=verdict)
    return out


def _mpl():
    """matplotlib (Agg) routed through the SHARED project style (_analysis_style:
    Arial/Liberation Sans + larger title/axis fonts), so these plots match the
    campaign figures."""
    import matplotlib; matplotlib.use("Agg")
    import _analysis_style as S   # noqa: F401  applies shared rcParams on import
    import matplotlib.pyplot as plt
    return plt


def _sub(name):
    """Format a system name with subscripts for plot titles:
    nVTi -> 'n V_{Ti}', VBa -> 'V_{Ba}', underscores tidied."""
    s = re.sub(r"(\d?)V(Ba|Ti)",
               lambda m: (m.group(1) + " " if m.group(1) else "") + r"V$_{" + m.group(2) + r"}$",
               name)
    return s.replace("__", "  ").replace("_", " ")


def plot_disp(system, T, res, outdir):
    """|Delta r| vs time for BOTH movers in ONE plot (species colours matching the
    MSD/other plots). Confirmed hop = red dot; attempt = pink star at the peak."""
    plt = _mpl()
    t = res["t"]; movers = res["movers"]
    fig, ax = plt.subplots(figsize=(12, 5))   # fixed margins (below) -> deterministic export size
    a0s = []
    win = max(3, int(round(5.0 / res.get("dt", 1.0))))   # 5 ps window == the sustain criterion
    sus = max(1, int(round(SUSTAIN_PS / res.get("dt", 1.0))))
    # AUTHORITATIVE verdict per mover = verify_hop's site-map call (COMMITTED/REVERSIBLE/
    # ATTEMPT/KICK-OUT/no-move), read from the per-run verify_events CSV in this analysis dir.
    # Falls back to hop_screen's own verdict for bulk (no verify_events / no full lattice).
    auth = {}
    for m in movers:
        vf = outdir / f"verify_events_{system}_{T}K_{m}.csv"
        if vf.exists():
            try:
                with open(vf) as fh:
                    row = next(csv.DictReader(fh), None)
                if row and row.get("kind", "").strip():
                    auth[m] = row["kind"].strip()
            except Exception:
                pass
    for m, r in movers.items():
        c = SPECIES_COLOR[m]
        av = auth.get(m)                       # authoritative (verify_hop) verdict or None
        vlabel = av if av else r["verdict"]    # the verdict actually shown on the plot
        ax.plot(t, r["disp"], color=c, lw=0.8, alpha=0.20, zorder=2)         # raw (faded jitter)
        ax.plot(t, S.running_mean(r["disp"], win), color=c, lw=2.4, alpha=0.95, zorder=3,
                label=f"{m}: {vlabel}")                                       # 5 ps-smoothed (the "stay")
        ab = r["disp"] > HOP_F * r["a0"]
        # hop ONSET = first >=5 ps sustained crossing of this mover's bar (works for a
        # committed hop that stays AND a reversible hop that later returns)
        onset = next((f for f in range(len(t) - sus + 1) if ab[f:f + sus].all()), None)
        committed = av == "COMMITTED" or (av is None and r["verdict"] == "HOP")
        if committed and onset is not None:
            ax.plot(t[onset], r["disp"][onset], "o", color="#d62728", mec=c, mew=1.3, ms=12, zorder=6)
            ax.annotate("hop", (t[onset], r["disp"][onset]), textcoords="offset points",
                        xytext=(7, 7), color="#d62728", fontsize=S.fs(11), fontweight="bold",
                        zorder=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        elif av == "REVERSIBLE" and onset is not None:
            ax.plot(t[onset], r["disp"][onset], "o", color="#f59e0b", mec=c, mew=1.3, ms=12, zorder=6)
            ax.annotate("reversible", (t[onset], r["disp"][onset]), textcoords="offset points",
                        xytext=(7, 7), color="#b45309", fontsize=S.fs(11), fontweight="bold",
                        zorder=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        elif vlabel in ("ATTEMPT", "attempt") and r["attempts"] and r.get("n_vac", 0) > 0:
            at = np.array([(tp, dr) for (_, tp, dr) in r["attempts"]])         # (mark attempts only
            ax.plot(at[:, 0], at[:, 1], "*", color="#ff8fa3", mec="#c2185b", mew=0.5,  # where a hop is
                    ms=12, ls="none", zorder=5)                                # POSSIBLE: vacancy present)
        # KICK-OUT / DISTORTION / no-move / RATTLE -> no marker (no real vacancy hop)
        a0s.append(r["a0"])
    # hop bar is PER MOVER (d_NN differs by sublattice); label in the RIGHT MARGIN
    # (x in axes-fraction, y in data) so labels never sit on the data
    yt = ax.get_yaxis_transform()
    for i, (m, r) in enumerate(movers.items()):
        c = SPECIES_COLOR[m]
        ax.axhline(HOP_F * r["a0"], color=c, ls="--", lw=1.3, alpha=.5)
        ax.text(1.012, HOP_F * r["a0"], f"{m} hop", color=c, fontsize=S.fs(12),
                va="top" if i == 0 else "bottom", ha="left", alpha=.95,
                transform=yt, clip_on=False)
    a0 = float(np.mean(a0s))
    ax.axhline(VIB_F * a0, color="#9e9e9e", ls=":", lw=1.0, alpha=.5)
    ax.text(1.012, VIB_F * a0, "vibration", color="#9e9e9e", fontsize=S.fs(9),
            va="center", ha="left", transform=yt, clip_on=False)
    ax.set_xlabel("Time (ps)"); ax.set_ylabel(r"$|\Delta r|$ ($\AA$)")
    ax.set_title("Cation Displacement (5 ps-window averaged)")
    # SHARED export geometry (must match coord_number.py so the two figures are the
    # same pixel size): fixed margins + NO bbox_inches="tight" -> 11x5.2 in @ 300 dpi.
    fig.subplots_adjust(left=0.11, right=0.83, top=0.88, bottom=0.15)
    fig.savefig(outdir / f"disp_{system}_{T}K.png", dpi=300)
    plt.close(fig)


def plot_sitestd(system, T, rep, res, outdir):
    """Site-index + running-std (2 stacked panels), per mover, species colour."""
    plt = _mpl()
    t = res["t"]
    for m, r in res["movers"].items():
        c = SPECIES_COLOR[m]
        fig, ax = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, constrained_layout=True)
        ax[0].step(t, r["site"], where="post", color=c, lw=1.8)
        ax[0].set_ylabel("assigned site idx")
        ax[0].set_title(f"Site + running-std: {_sub(system)}, T={T} K  [{m}: {r['verdict']}]")
        ax[1].plot(t, r["rstd"], color=c, lw=1.8)
        ax[1].axhline(1.2, color="#9e9e9e", ls="--", lw=1.0, alpha=.5)
        ax[1].set_ylabel(r"running std ($\AA$)"); ax[1].set_xlabel("Time (ps)")
        for (f, tp, so, sn, dur, st) in r["events"]:
            if st == "confirmed":
                for a in ax: a.axvline(tp, color="#d62728", lw=1.8, alpha=.85)
        fig.savefig(outdir / f"sitestd_{system}_{T}K_r{rep}_{m}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_vanhove(system, T, rep, res, outdir):
    """Plot C: van Hove self-correlation 4 pi r^2 G_s(r,t) per (system, mover)."""
    plt = _mpl()
    dt = res["dt"]; lags = [1, 10, 100, 500]
    bins = np.arange(0, 6.01, 0.1); rc = 0.5 * (bins[1:] + bins[:-1])
    for m, r in res["movers"].items():
        if r["verdict"] == "MELT":
            continue
        U = r["pos3"]; nfr = len(U)
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for i, lag_ps in enumerate(lags):
            lag = max(1, int(round(lag_ps / dt)))
            if lag >= nfr: continue
            o = np.linspace(0, nfr - 1 - lag, min(200, nfr - lag)).astype(int)
            d = np.linalg.norm(U[o + lag] - U[o], axis=1)
            h, _ = np.histogram(d, bins=bins, density=True)
            ax.plot(rc, h, color=plt.cm.viridis(i / max(1, len(lags) - 1)), lw=1.8, label=f"t={lag_ps} ps")
        ax.axvline(0.5, color="#7f7f7f", ls="--", lw=1.0, alpha=.4)
        ax.axvline(4.0, color="#7f7f7f", ls="--", lw=1.1, alpha=.6)
        ax.set_xlim(0, 6); ax.set_xlabel(r"r ($\AA$)"); ax.set_ylabel(r"$4\pi r^2\,G_s(r,t)$")
        ax.set_title(f"van Hove $G_s(r,t)$: {_sub(system)}, T={T} K  [{m}]")
        ax.legend(title="time lag")
        fig.savefig(outdir / f"vanhove_{system}_{T}K_{m}.png", dpi=300); plt.close(fig)


def _runmean(y, w=50):
    if len(y) < w: return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def plot_msd(run_dir, system, T, outdir):
    """Per-species MSD vs time from msd.dat -> raw + smooth PNGs (project colours,
    dopant movers emphasised, hosts faded)."""
    f = run_dir / "msd.dat"
    if not f.exists(): return
    hdr = open(f).readline().lstrip("# ").split()           # t_ps MSD_Ba_A2 ...
    data = np.loadtxt(f, comments="#", ndmin=2)
    if data.shape[0] < 2: return
    t = data[:, 0]
    species = [h.replace("MSD_", "").replace("_A2", "") for h in hdr[1:]]
    plt = _mpl()
    for mode in ("raw", "smooth"):
        fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
        for j, sp in enumerate(species):
            y = data[:, j + 1]
            if mode == "smooth": y = _runmean(y)
            dop = sp in DOPANTS
            ax.plot(t, y, color=SPECIES_COLOR.get(sp, "k"),
                    lw=2.6 if dop else 1.3, alpha=1.0 if dop else 0.5,
                    zorder=3 if dop else 1, label=sp)
        ax.axhline(MSD_PER_HOP, color="#7f7f7f", ls="--", lw=1.0, alpha=.6)
        ax.text(t[-1], MSD_PER_HOP, "MSD/hop ~2.75  ", color="#7f7f7f",
                fontsize=8, va="bottom", ha="right")
        ax.set_xlabel("time (ps)"); ax.set_ylabel(r"MSD ($\AA^2$)")
        ax.set_title("Mean Squared Displacement")
        ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
        fig.savefig(outdir / f"msd_{system}_{T}K_{mode}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--roots", nargs="+", default=["01_BULK", "02_GB"])
    ap.add_argument("--outdir", default="04_analysis/hop_detection")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--msd-only", action="store_true",
                    help="only (re)plot MSD from msd.dat into each analysis/ dir; skip trajectory analysis")
    a = ap.parse_args()
    outdir = ROOT / a.outdir; outdir.mkdir(parents=True, exist_ok=True)

    runs = list(a.runs)
    if a.discover or not runs:
        for root in a.roots:
            for mp in glob.glob(str(ROOT / root / "**" / "r0" / "run_meta.json"), recursive=True):
                try:
                    if json.loads(Path(mp).read_text()).get("status") == "complete" and (Path(mp).parent / "traj.traj").exists():
                        runs.append(Path(mp).parent)
                except Exception: pass
        runs = sorted(set(runs))

    summary = []
    print(f"screening {len(runs)} finished production run(s) -> {outdir}\n")
    for rd in runs:
        if not (rd / "traj.traj").exists(): continue
        system, T, rep = parse_id(rd)
        # per-structure analysis dir, next to equil/ and r0/  (run_dir = .../r0)
        run_out = rd.parent / "analysis"; run_out.mkdir(parents=True, exist_ok=True)
        if a.msd_only:
            try: plot_msd(rd, system, T, run_out); print(f"  {system:<46} msd -> {run_out}")
            except Exception as e: print(f"  [msd fail {system}] {e}")
            continue
        res = analyse(rd)
        if not res or not res.get("movers"):
            continue
        if not a.no_plot:
            try:
                plot_disp(system, T, res, run_out)             # |dr| both movers in ONE fig
                plot_sitestd(system, T, rep, res, run_out)     # site idx + running std (per mover)
                plot_vanhove(system, T, rep, res, run_out)     # van Hove G_s(r,t)
                plot_msd(rd, system, T, run_out)               # per-species MSD (raw + smooth)
            except Exception as e: print(f"  [plot fail {system}] {e}")
        for m, r in res["movers"].items():
            nconf = sum(1 for e in r["events"] if e[5] == "confirmed")
            natt = len(r["attempts"])
            flag = {"HOP": "  <== HOP (sustained + vacancy on sublattice)",
                    "DISTORTION": "  [distortion -- no vacancy on this sublattice]",
                    "MELT": "  [MELT/collapse -- excluded]"}.get(r["verdict"], "")
            print(f"  {system:<46} {m:2}  a0={r['a0']:.2f} d_max={r['d_max']:5.2f} last100={r['last100']:5.2f} "
                  f"conf={nconf} att={natt} -> {r['verdict']}{flag}")
            summary.append([system, T, rep, m, round(r['a0'],3), round(r['d_max'],3),
                            round(r['last100'],3), nconf, natt, r['verdict']])
            # per-traj hop events CSV (in the structure's own analysis/ dir).
            # site labels = nearest-host "<species>#<atom-index>" (readable, not a bare int);
            # disp_A and sustained_ps are now SEPARATE columns (attempts carry disp, not dwell).
            sym = res.get("sym")
            def _slab(idx):
                idx = int(idx)
                return f"{sym[idx]}#{idx}" if sym is not None else str(idx)
            with open(run_out / f"hop_events_{system}_{T}K_r{rep}_{m}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["hop_index","time_ps","old_site","new_site","disp_A","sustained_ps","status"])
                for i,(f,tp,so,sn,dur,st) in enumerate(r["events"]):
                    w.writerow([i, round(tp,2), _slab(so), _slab(sn),
                                round(float(r["disp"][f]),2), round(dur,2), st])
                for (f,tp,dr) in r["attempts"]:
                    w.writerow(["", round(tp,2), "", "", round(dr,2), "", "attempt"])
    with open(outdir / "hop_screen_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["system","T_K","replica","mover","a0_A","d_max_A","last100_A","n_confirmed","n_attempt","verdict"])
        w.writerows(summary)
    hops = [s for s in summary if s[-1] == "HOP"]
    melts = [s for s in summary if s[-1] == "MELT"]
    print(f"\n=== {len(hops)} trajectory-mover(s) flagged HOP (cross-check with vacancy_tracker.py) ===")
    for s in hops: print(f"  {s[0]}  {s[3]}  d_max={s[5]}  last100={s[6]}")
    if melts:
        print(f"\n[{len(melts)} mover(s) excluded as MELT/collapse: "
              + ", ".join(sorted(set(s[0] for s in melts))) + "]")
    print(f"\nsummary -> {outdir}/hop_screen_summary.csv")


if __name__ == "__main__":
    main()
