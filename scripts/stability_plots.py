#!/usr/bin/env python3
"""Simulation-stability plots for the 07/08 compensated productions, campaign
style: TWO SEPARATE figures per run -- total energy vs time and temperature vs
time -- from the MDLogger md.log (0.1 ps sampling; columns t_ps, Etot, Epot,
Ekin, T). Continuation runs are stitched part1+part2 via run_meta
continuation_from (part2's log carries absolute times). Cut at 3 ns.
Files: {etot,temp}_<system>_<T>K_{raw,smooth}.png in the run's plot/ dir;
smooth = 10 ps running mean, same window as prod_cation_analysis."""
import json
import sys
from pathlib import Path

ROOT = Path.home() / "SEM_MLCC/02_RUN/03_SURF_HETERO"
sys.path.insert(0, str(ROOT / "scripts"))
import _analysis_style as S            # noqa: F401,E402  BLAS pin + Arial
import numpy as np                     # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402

TMAX_PS = 3000.0
SMOOTH_PS = 10.0

RUNS = {
    "07_lattice_matched_vba":
        ROOT / "07_lattice_matched_vba/02_compensated_8VO/prod_5ns_nvt_local_part2",
    "08_lattice_matched_mg_vba":
        ROOT / "08_lattice_matched_mg_vba/02_compensated_12VO/prod_5ns_nvt",
}


def read_log(path):
    rows = []
    for line in open(path):
        try:
            v = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(v) >= 5:
            rows.append(v[:5])
    return np.array(rows)


def runmean(y, w):
    if len(y) < w:
        return y
    k = np.ones(w)
    return np.convolve(y, k, mode="same") / np.convolve(np.ones(len(y)), k,
                                                        mode="same")


for struct, rd in RUNS.items():
    meta = json.loads((rd / "run_meta.json").read_text())
    d = read_log(rd / "md.log")
    src = meta.get("continuation_from")
    system = rd.name
    if src:
        d = np.concatenate([read_log(Path(src) / "md.log"), d])
        system = Path(src).name + "_stitched"
    if meta.get("status") != "complete":
        system += "_partial"
    system += f"_to{TMAX_PS / 1000:g}ns"
    d = d[d[:, 0] <= TMAX_PS]
    t, etot, T_inst = d[:, 0], d[:, 1], d[:, 4]
    T = int(round(float(meta.get("temperature_K", 1463))))
    outdir = rd.parent / "plot"
    outdir.mkdir(exist_ok=True)
    win = max(3, int(round(SMOOTH_PS / (t[1] - t[0]))))

    # combined figure: E_tot on the left axis, T on the right (twin y)
    TCOL = "#858585"
    for tag in ("raw", "smooth"):
        ye = etot if tag == "raw" else runmean(etot, win)
        yt = T_inst if tag == "raw" else runmean(T_inst, win)
        fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
        ax.plot(t, ye, color="k", lw=0.8 if tag == "raw" else 1.8)
        ax.set_xlabel("time (ps)")
        ax.set_ylabel(r"$E_{\mathrm{tot}}$ (eV)")
        ax2 = ax.twinx()
        ax2.plot(t, yt, color=TCOL, lw=0.8 if tag == "raw" else 1.8,
                 alpha=0.85)
        ax2.axhline(T, color=TCOL, ls="--", lw=1.2, alpha=0.5)
        # stacked bands in one box: energy occupies the TOP half (left axis),
        # temperature the BOTTOM half (right axis), so the curves never overlap
        es = ye.max() - ye.min()
        ax.set_ylim(ye.min() - 1.15 * es, ye.max() + 0.10 * es)
        tlo, thi = min(yt.min(), T), max(yt.max(), T)
        ts = thi - tlo
        ax2.set_ylim(tlo - 0.10 * ts, thi + 1.15 * ts)
        ax2.set_ylabel(r"$T$ (K)", color=TCOL)
        ax2.tick_params(axis="y", colors=TCOL)
        ax2.text(0.99, 0.02, f"target {T} K", color=TCOL, alpha=0.7,
                 ha="right", va="bottom", transform=ax2.transAxes,
                 fontsize=S.fs(14))
        ax.set_title("Total Energy and Temperature Evolution")
        out = outdir / f"stability_{system}_{T}K_{tag}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print("->", out)
