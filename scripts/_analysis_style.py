"""
Shared style + data loaders for the 02_RUN campaign analysis scripts
(analyze_strain.py / analyze_gb.py / analyze_vacancy.py).

Mirrors the look of 01_BENCHMARK/06_diffusivity/03_benchmark (Arial, large fonts,
anti-overlap, MSD-per-hop reference line, TST hop-time). Login-node friendly:
BLAS/OMP threads are pinned to 1 so every analysis script runs single-threaded
in plain bash (no SLURM, no GPU):

    env OMP_NUM_THREADS=1 python scripts/analyze_strain.py

IMPORT THIS FIRST (before numpy) in every analysis script.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
for _p in ("/home/jamie/fonts/Arial.TTF", "/home/jamie/fonts/Arial.ttf"):
    try:
        fm.fontManager.ttflist.insert(0, fm.FontEntry(fname=_p, name="Arial"))
        break
    except Exception:
        continue
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import (TextArea, HPacker,  # noqa: E402
                                  AnnotationBbox)

# ----------------------------------------------------------------------------
# Style (anti-overlap, large Arial). SC scales every font.
# ----------------------------------------------------------------------------
SC = 1.4


def fs(pt):
    return pt * SC


plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.default": "regular",
    "axes.unicode_minus": False,
    "font.size": fs(16.5),         # body/other text bigger (user 2026-06-15b)
    "axes.titlesize": fs(18),      # title a bit smaller (was 24)
    "axes.labelsize": fs(18),      # axis labels a bit smaller (was 22)
    "axes.titleweight": "bold",
    "axes.titlepad": fs(12),
    "xtick.labelsize": fs(16),     # tick labels bigger
    "ytick.labelsize": fs(16),
    "legend.fontsize": fs(16),     # legend bigger (was 13)
    "figure.titlesize": fs(21),
    "lines.linewidth": 2.4,
    "lines.markersize": 8,
    "axes.linewidth": 1.5,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})

BBOX = dict(boxstyle="round,pad=0.4", facecolor="white",
            edgecolor="gray", alpha=0.9)

# ----------------------------------------------------------------------------
# Physical constants (TST hop time)
# ----------------------------------------------------------------------------
kB = 8.617333e-5          # eV/K
NU0 = 1.0e13              # attempt frequency, s^-1
A0 = 20.3058557292151676 / 5      # 5x5x5 lattice constant ~4.0612 A
MSD_PER_HOP = A0**2 / 6.0          # ~2.75 A^2 per single nn hop
T_SINTER = 1463                   # K
T_MELT = 1898                     # K (BTO)

# ----------------------------------------------------------------------------
# Colour maps for the campaign axes
# ----------------------------------------------------------------------------
# Strain (viridis-like by magnitude)
STRAIN_COLORS = {"eps0": "#440154", "eps_th": "#3B528B",
                 "eps_3": "#21918C", "eps_5": "#FDE725"}
STRAIN_PCT = {"eps0": 0.0, "eps_th": 1.6, "eps_3": 3.0, "eps_5": 5.0}
STRAIN_ORDER = ["eps0", "eps_th", "eps_3", "eps_5"]

# GB rung
RUNG_COLORS = {"gb": "#D62728", "near": "#FF7F0E", "int": "#1F77B4"}
RUNG_ORDER = ["gb", "near", "int"]
RUNG_LABEL = {"gb": "GB plane", "near": "near-GB", "int": "interior"}

# species colours (user-fixed 2026-06-11):
#   Ba green | Ti sky-blue | O red | Dy indigo-blue | Mg orange
SPECIES_COLOR = {"Ba": "#2CA02C", "Ti": "#56B4E9", "O": "#D62728",
                 "Dy": "#4338CA", "Mg": "#FF7F0E"}

# generic categorical palette (e.g. V1..V5)
CAT10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

NBSP = " "          # non-breaking space: kept (not trimmed) inside TextArea


# ----------------------------------------------------------------------------
# Smoothing + multi-colour title helpers (for the MSD figures)
# ----------------------------------------------------------------------------
def running_mean(y, win=50):
    """Centred boxcar running mean; win=50 pts ~ 5 ps at 0.1 ps sampling."""
    y = np.asarray(y, float)
    if len(y) < win:
        return y
    return np.convolve(y, np.ones(win) / win, mode="same")


def vac_seg(sym, n=1):
    """Vacancy title segment, coloured by species:
    ('Ba', 2) -> ('2 V$_{Ba}$', green); n==1 drops the count -> ('V$_{Ba}$', ...)."""
    pre = "" if n in (1, None) else f"{n} "
    return (f"{pre}V$_{{{sym}}}$", SPECIES_COLOR.get(sym, "black"))


def dop_seg(sym, n=1):
    """Dopant title segment coloured by species ('Dy' -> indigo, 'Mg' -> orange)."""
    pre = "" if n in (1, None) else f"{n}{NBSP}"
    return (f"{pre}{sym}", SPECIES_COLOR.get(sym, "black"))


def colored_title(ax, parts, y=1.02, fontsize=None, weight="bold"):
    """Multi-colour title above `ax`. `parts` = list of (text, colour) segments,
    packed left-to-right on a shared baseline and centred over the axes. Each
    text may use mathtext (e.g. 'V$_{Ba}$'); use NBSP for spaces that must be
    kept between coloured segments. A blank set_title reserves the layout space."""
    if fontsize is None:
        fontsize = plt.rcParams["axes.titlesize"]
    ax.set_title(" ")          # reserve vertical space for constrained_layout
    boxes = [TextArea(txt, textprops=dict(color=col, fontsize=fontsize,
                                          fontweight=weight))
             for txt, col in parts]
    pack = HPacker(children=boxes, align="baseline", pad=0, sep=0)
    ab = AnnotationBbox(pack, (0.5, y), xycoords="axes fraction",
                        box_alignment=(0.5, 0.0), frameon=False, pad=0,
                        annotation_clip=False)
    ax.add_artist(ab)
    return ab


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_msd(path):
    """Read an msd.dat written by PerSpeciesMSDTracker.
    Returns (t_ps, {species: msd_array}) or (None, None) if absent.
    Header form: '# t_ps  MSD_Ba_A2  MSD_Dy_A2  MSD_O_A2 ...'."""
    path = Path(path)
    if not path.exists():
        return None, None
    header = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                header = line
                break
    data = np.loadtxt(str(path), comments="#")
    if data.ndim == 1:
        data = data[None, :]
    species = {}
    if header:
        toks = header.lstrip("#").split()
        # toks[0] == 't_ps'; the rest are MSD_<sym>_A2
        for col, tok in enumerate(toks):
            if tok.startswith("MSD_"):
                sym = tok[4:].rsplit("_A2", 1)[0]
                if col < data.shape[1]:
                    species[sym] = data[:, col]
    return data[:, 0], species


def load_dopant_disp(path):
    """dopant_disp.dat -> (t_ps, max_dopant_disp_A) or (None, None)."""
    path = Path(path)
    if not path.exists():
        return None, None
    d = np.loadtxt(str(path), comments="#")
    if d.ndim == 1:
        d = d[None, :]
    return d[:, 0], d[:, 2]


def load_barrier(neb_out_dir):
    """Parse a gb_neb.py barrier.txt. Returns dict with floats for
    barrier_forward / barrier_reverse / dE_reaction (keys w/o units), or None."""
    p = Path(neb_out_dir) / "barrier.txt"
    if not p.exists():
        return None
    out = {}
    for line in p.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower().replace(" (ev)", "").replace(" ", "_")
        v = v.strip()
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def load_neb_curve(neb_out_dir):
    """neb_curve.txt -> (reaction_coord 0..1, dE_eV) or (None, None)."""
    p = Path(neb_out_dir) / "neb_curve.txt"
    if not p.exists():
        return None, None
    d = np.loadtxt(str(p), comments="#")
    if d.ndim == 1:
        d = d[None, :]
    rc = d[:, 0]
    rc = rc / rc[-1] if rc[-1] > 0 else rc
    return rc, d[:, 1]


def tst_tau(Ea_eV, T):
    """TST single-hop time tau = nu0^-1 exp(Ea / kB T)  [s]."""
    return (1.0 / NU0) * np.exp(Ea_eV / (kB * np.asarray(T, float)))


# ----------------------------------------------------------------------------
# Trajectory helpers (max cation displacement)
# ----------------------------------------------------------------------------
def mic_delta(delta, cell):
    inv = np.linalg.inv(cell)
    frac = delta @ inv
    frac -= np.round(frac)
    return frac @ cell


def cation_max_disp(frames, species=("Ba", "Ti")):
    """Max |dr| (min-image vs frame 0) for every atom of `species`."""
    sym = np.array(frames[0].get_chemical_symbols())
    idx = np.where(np.isin(sym, list(species)))[0]
    p0 = frames[0].get_positions()[idx]
    maxd = np.zeros(len(idx))
    for f in frames:
        cell = np.array(f.get_cell())
        d = mic_delta(f.get_positions()[idx] - p0, cell)
        maxd = np.maximum(maxd, np.linalg.norm(d, axis=1))
    return maxd


def humanize_time(s):
    """Seconds -> human-readable string."""
    if not np.isfinite(s):
        return "inf"
    if s < 1e-9:      return f"{s*1e12:.1f} ps"
    if s < 1e-6:      return f"{s*1e9:.1f} ns"
    if s < 1e-3:      return f"{s*1e6:.1f} us"
    if s < 60:        return f"{s:.2g} s"
    if s < 3600:      return f"{s/60:.1f} min"
    if s < 86400:     return f"{s/3600:.1f} h"
    if s < 3.15576e7: return f"{s/86400:.1f} days"
    yr = s / 3.15576e7
    if yr < 1e4:
        return f"{yr:.1f} yr"
    n = int(np.floor(np.log10(yr)))
    return rf"${yr/10**n:.1f}\times10^{{{n}}}$ yr"


def save(fig, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{stem}.png"
    fig.savefig(str(fp))
    plt.close(fig)
    print(f"  saved {fp}")
    return fp
