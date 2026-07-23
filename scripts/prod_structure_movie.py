#!/usr/bin/env python3
"""
OVITO/Tachyon structure animation for 03_SURF_HETERO production runs, saved next
to the static figures in <struct>/<subdir>/rendering/ as

    structure_ovito_<run>_<T>K.mp4

Frame k renders traj frame k*stride, i.e. the SAME frames (and the same time
labels) prod_dopant_movie.py block-averages into its distribution animation --
play the two side by side and they line up frame-for-frame.

The cell is rotated so z runs HORIZONTALLY, matching the z axis of the
distribution plot, so a Dy atom moving right in this movie is the same Dy moving
right in the profile beside it.

Element colours/radii come from _species_colors.py, the same module hop_screen
feeds its matplotlib SPECIES_COLOR from, so renders and plots cannot drift apart.

REQUIRES the dedicated env (base's pip-installed ovito is broken -- PySide6 vs
system Qt libs):
    /home/jamie/anaconda3/envs/ovito-render/bin/python

Run:
  OVITO_THREAD_COUNT=2 QT_QPA_PLATFORM=offscreen \
  /home/jamie/anaconda3/envs/ovito-render/bin/python scripts/prod_structure_movie.py \
      [--partial] [--stride 10] [--fps 25] [--size 1600x760] [--limit N]
      [--keep-png] [--repeat 1x10x2] [--no-cell] [--zvert] [run_dir ...]
--repeat AxBxC tiles the wrapped cell; --no-cell hides the box; --zvert uses
the reference-image orientation (z vertical, films horizontal) instead of the
movie default (z horizontal).
No args: discovers every 0?_hetero_*/*/prod*/traj.traj like prod_cation_analysis.py.
"""
import os, sys, json, shutil, subprocess
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OVITO_THREAD_COUNT", "2")   # ulimit -u is 500 on this box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _species_colors import SPECIES_COLOR, SPECIES_RADIUS, rgb   # noqa: E402

import numpy as np                                               # noqa: E402
from ase.io.trajectory import Trajectory                         # noqa: E402
from ovito.io.ase import ase_to_ovito                            # noqa: E402
from ovito.pipeline import Pipeline, StaticSource                # noqa: E402
from ovito.vis import (Viewport, TachyonRenderer,                # noqa: E402
                       ParticlesVis, SimulationCellVis)

ROOT = HERE.parent
STRIDE = 10
FPS = 25
SIZE = (1600, 760)


def run_meta(rd):
    meta = {}
    mp = rd / "run_meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = {}
    return meta


def frame_dt(rd, nfr, meta):
    total_ps = float(meta.get("time_ps", nfr - 1))
    if meta.get("continuation_from"):
        total_ps += float(meta.get("t0_ps", 0.0))
    dt = total_ps / max(1, nfr - 1)
    if meta.get("status") != "complete" and "dt_fs" in meta:
        dt = 1000 * float(meta["dt_fs"]) * 1e-3
    return dt


class _ConcatFrames:
    """Continuation runs (16_continue_nvt.py) stitched into one frame list:
    source trajectory + continuation with its frame 0 dropped (it duplicates
    the source's last frame). Supports len() and integer indexing only."""

    def __init__(self, paths):
        self.trajs = [Trajectory(str(p)) for p in paths]
        self.index = [(j, i)
                      for j, tr in enumerate(self.trajs)
                      for i in range(1 if j else 0, len(tr))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        j, i = self.index[k]
        return self.trajs[j][i]


def styled_data(atoms, repeat=(1, 1, 1), zvert=False, show_cell=True):
    """ASE Atoms -> styled OVITO DataCollection (campaign palette applied)."""
    at = atoms.copy()
    # newer runs (07/08 lattice-matched prod) write UNWRAPPED positions --
    # wrap here so every render sits inside the box; no-op for the older
    # already-wrapped trajectories
    at.wrap()
    if tuple(repeat) != (1, 1, 1):
        at = at.repeat(tuple(repeat))
    if zvert:
        # reference-image orientation (hetero_1x10x2.png): b horizontal,
        # z(c) VERTICAL. OVITO's Front viewport looks along scene +y with
        # z up, so aligning b -> x is all that's needed: c stays on z (up)
        # and the remaining in-plane axis provides the view depth.
        at.rotate(at.cell[1], (1, 0, 0), rotate_cell=True)   # b -> screen x
    else:
        # movie default: z horizontal, matching the distribution plot's z axis
        at.rotate(90, "y", rotate_cell=True)
    data = ase_to_ovito(at)
    if show_cell:
        data.cell.vis.line_width = 0.25
        data.cell.vis.rendering_color = (0.25, 0.25, 0.25)
    else:
        data.cell.vis.enabled = False
    ptypes = data.particles_.particle_types_
    for t in ptypes.types:
        t.color = rgb(t.name)
        t.radius = SPECIES_RADIUS.get(t.name, 0.8)
    data.particles.vis.radius = 0.8
    return data


def styled_pipeline(atoms, **kw):
    """ASE Atoms -> OVITO pipeline with the campaign palette applied."""
    return Pipeline(source=StaticSource(data=styled_data(atoms, **kw)))


def make_movie(rd, system, stride=STRIDE, fps=FPS, size=SIZE, keep_png=False,
               limit=None, outdir=None, repeat=(1, 1, 1), zvert=False,
               show_cell=True):
    meta = run_meta(rd)
    src = meta.get("continuation_from")
    if src:
        traj = _ConcatFrames([Path(src) / "traj.traj", rd / "traj.traj"])
    else:
        traj = Trajectory(str(rd / "traj.traj"))
    nfr = len(traj)
    dt = frame_dt(rd, nfr, meta)
    T = int(round(float(meta.get("temperature_K", 1463))))
    idx = list(range(0, nfr, stride))          # same blocks as prod_dopant_movie
    if limit:
        idx = idx[:limit]                      # --limit N: cheap render check
    dt_anim = dt * stride
    if tuple(repeat) != (1, 1, 1):
        system += "_{}x{}x{}".format(*repeat)  # own png dir + mp4 name
    style = dict(repeat=tuple(repeat), zvert=zvert, show_cell=show_cell)

    png_dir = (outdir / f"_png_{system}")
    if png_dir.exists():
        shutil.rmtree(png_dir)
    png_dir.mkdir(parents=True)

    pipeline = styled_pipeline(traj[0], **style)
    pipeline.add_to_scene()
    vp = Viewport(type=Viewport.Type.Front)
    vp.zoom_all(size=size)
    # freeze the frame-0 camera: zoom_all per frame would rescale as atoms
    # wander and the movie would breathe
    cam_pos, cam_dir, fov = vp.camera_pos, vp.camera_dir, vp.fov
    renderer = TachyonRenderer(shadows=False, direct_light_intensity=1.1,
                               ambient_occlusion=True,
                               ambient_occlusion_samples=12)
    try:
        for k, f in enumerate(idx):
            # OVITO >= 3.15: in-place mutation via particles_ raises
            # make_mutable(), and reassigning source.data after the first
            # render is silently ignored (cached evaluation, verified by
            # identical frame md5s) -- swap in a fresh pipeline per frame,
            # which the cache cannot survive; cheap next to the Tachyon render
            pipeline.remove_from_scene()
            pipeline = styled_pipeline(traj[f], **style)
            pipeline.add_to_scene()
            vp.camera_pos, vp.camera_dir, vp.fov = cam_pos, cam_dir, fov
            vp.render_image(size=size, filename=str(png_dir / f"f{k:05d}.png"),
                            renderer=renderer, background=(1, 1, 1),
                            alpha=False)
            if k and k % 25 == 0:
                print(f"     frame {k}/{len(idx)}", flush=True)
    finally:
        pipeline.remove_from_scene()

    out = outdir / f"structure_ovito_{system}_{T}K.mp4"
    # This user slice is capped at pids.max=100 (shared with every other
    # session), so ffmpeg's default frame-threading fails to spawn: libx264's 15
    # threads die with "ff_frame_thread_encoder_init failed". The PNG *decoder*
    # threads the same way, so -threads 1 is needed BEFORE -i as well as after.
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1",
           "-framerate", str(fps),
           "-i", str(png_dir / "f%05d.png"), "-vcodec", "libx264",
           "-threads", "1", "-crf", "20", "-pix_fmt", "yuv420p",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)]
    subprocess.run(cmd, check=True)
    if not keep_png:
        shutil.rmtree(png_dir)
    print(f"   -> {out}  ({len(idx)} frames from {nfr} traj frames, stride "
          f"{stride} = {dt_anim:g} ps/frame, {len(idx) / fps:.0f} s at {fps} fps)")


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
    outdir = rd.parent / "rendering"
    outdir.mkdir(exist_ok=True)
    system = rd.name
    meta = run_meta(rd)
    src = meta.get("continuation_from")
    if src:
        # stitched part1+part2 movie, named after the source run
        system = Path(src).name + "_stitched"
    status = meta.get("status")
    if status and status != "complete":
        if not partial:
            print(f"  SKIP (status={status}): {rd}")
            return
        system += "_partial"
        print(f"  [partial] status={status}: rendering traj so far")
    print(f"== {rd.parent.parent.name} / {system}")
    make_movie(rd, system, outdir=outdir, **kw)


if __name__ == "__main__":
    argv = sys.argv[1:]
    partial = "--partial" in argv
    keep_png = "--keep-png" in argv
    zvert = "--zvert" in argv
    show_cell = "--no-cell" not in argv
    stride, fps, size, limit, repeat = STRIDE, FPS, SIZE, None, (1, 1, 1)
    rest = []
    it = iter([a for a in argv
               if a not in ("--partial", "--keep-png", "--zvert", "--no-cell")])
    for a in it:
        if a == "--stride":
            stride = int(next(it))
        elif a == "--fps":
            fps = int(next(it))
        elif a == "--limit":
            limit = int(next(it))
        elif a == "--size":
            size = tuple(int(v) for v in next(it).lower().split("x"))
        elif a == "--repeat":
            repeat = tuple(int(v) for v in next(it).lower().split("x"))
        else:
            rest.append(Path(a))
    runs = rest or discover()
    if not runs:
        print("no production runs found")
    for r in runs:
        animate(r, partial=partial, stride=stride, fps=fps, size=size,
                limit=limit, keep_png=keep_png, repeat=repeat, zvert=zvert,
                show_cell=show_cell)
