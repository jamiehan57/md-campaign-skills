#!/bin/bash
# sem-render [--tlim PS] [--fps N] [--ppf PS] [--traj2xyz] [--partial] [--yes] [run_dir ...]
# Paired production renders (campaign convention) into <subdir>/rendering/:
#   - dopant z-profile movie (prod_dopant_movie.py, --zsigma 1.0)
#   - OVITO structure movie  (prod_structure_movie.py, --repeat 1x10x2 --no-cell --zvert)
#   - [--traj2xyz] extxyz export of traj.traj (complete runs only)
# Flow: shows video length + ESTIMATED RENDER TIME first, then lets you adjust
#   ps-per-frame (timescale) and fps before launching. Defaults: cover --tlim
#   (or the full traj) in ~151 frames at 25 fps  (3000 ps -> 20 ps/frame, 6 s video).
# At the prompt:  [Enter]=render   'PPF'   'PPF FPS'  (e.g. '10 25')   s=skip   q=quit
# Renders run sequentially via nohup (log path printed) so you can close the terminal.
# --partial : render still-running trajs (suffix _partial; skips xyz export)
# --yes     : accept defaults, no prompt.  Continued runs: pass the part2 dir ->
#             stitched automatically; a part1 dir is dropped when its part2 is listed.
PY=$HOME/anaconda3/envs/sem-neq/bin/python
OVPY=$HOME/anaconda3/envs/ovito-render/bin/python
SCR=$HOME/SEM_MLCC/02_RUN/03_SURF_HETERO/scripts
OVSPF=${SEM_RENDER_OVSPF:-16}      # OVITO s/frame (calibrated 2026-07-23, tiled 1x10x2)
OVSTART=30                          # OVITO env+traj startup s
tlim=""; fps=25; ppf=""; xyz=0; partial=""; yes=0; runs=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tlim|--tmax) tlim="$2"; shift 2;;
    --fps)         fps="$2"; shift 2;;
    --ppf)         ppf="$2"; shift 2;;
    --traj2xyz)    xyz=1; shift;;
    --partial)     partial="--partial"; shift;;
    --yes)         yes=1; shift;;
    -h|--help)     sed -n '2,15p' "$0"; exit 0;;
    *)             runs+=("$1"); shift;;
  esac
done

if [ ${#runs[@]} -eq 0 ]; then
  while IFS= read -r t; do runs+=("$(dirname "$t")"); done \
      < <(find . -maxdepth 3 -name traj.traj -path '*prod*' | sort)
fi
[ ${#runs[@]} -eq 0 ] && { echo "sem-render: no prod*/traj.traj under $(pwd)"; exit 1; }

# drop continuation SOURCES when their part2 is also listed (stitched covers them)
mapfile -t runs < <("$PY" - "${runs[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
runs = [Path(a).resolve() for a in sys.argv[1:]]
src = set()
for r in runs:
    mp = r / "run_meta.json"
    if mp.exists():
        try:
            s = json.loads(mp.read_text()).get("continuation_from")
            if s: src.add(Path(s).resolve())
        except Exception: pass
for r in runs:
    if r in src:
        print(f"  [patch-up] skip {r.name}: covered by its continuation", file=sys.stderr)
    else:
        print(r)
PYEOF
)

PLAN=~/claude/render_jobs/render_$(date +%m%d_%H%M%S).sh
mkdir -p ~/claude/render_jobs
{ echo '#!/bin/bash'; echo 'set -x'; echo "cd $HOME/SEM_MLCC/02_RUN/03_SURF_HETERO"; } > "$PLAN"
nplan=0

for rd in "${runs[@]}"; do
  info=$("$PY" - "$rd" <<'PYEOF'
import json, sys
from pathlib import Path
from ase.io.trajectory import Trajectory
rd = Path(sys.argv[1]).resolve()
meta = {}
mp = rd / "run_meta.json"
if mp.exists():
    try: meta = json.loads(mp.read_text())
    except Exception: pass
n = len(Trajectory(str(rd / "traj.traj")))
src = meta.get("continuation_from")
if src:
    n += len(Trajectory(str(Path(src) / "traj.traj"))) - 1
total = float(meta.get("time_ps", n - 1))
if src: total += float(meta.get("t0_ps", 0.0))
dt = total / max(1, n - 1)
if meta.get("status") != "complete" and "dt_fs" in meta:
    dt = float(meta["dt_fs"])           # ps/frame = 1000 steps * dt_fs fs
    total = (n - 1) * dt
print(f"{dt:g} {n} {total:g} {meta.get('status','?')}")
PYEOF
) || { echo "  ERROR reading $rd -- skipped"; continue; }
  read -r dt nfr total status <<< "$info"
  eff_tlim=${tlim:-$total}
  awk -v a="$eff_tlim" -v b="$total" 'BEGIN{exit !(a>b)}' && eff_tlim=$total
  if [ "$status" != "complete" ] && [ -z "$partial" ]; then
    echo "  SKIP (status=$status, no --partial): $rd"; continue
  fi
  cur_ppf=${ppf:-$(awk -v t="$eff_tlim" 'BEGIN{printf "%g", t/150}')}
  cur_fps=$fps
  while true; do
    read -r stride limit appf video ovmin zpmin <<< "$(awk \
      -v ppf="$cur_ppf" -v dt="$dt" -v tl="$eff_tlim" -v fps="$cur_fps" \
      -v nfr="$nfr" -v ovspf="$OVSPF" -v ovst="$OVSTART" 'BEGIN{
        s = int(ppf/dt + 0.5); if (s < 1) s = 1
        ap = s*dt
        l = int(tl/ap) + 1
        printf "%d %d %g %.1f %.0f %.0f", s, l, ap, l/fps,
               (ovst + ovspf*l)/60 + 0.5, (10 + 0.03*nfr + 0.3*l)/60 + 0.5}')"
    echo ""
    echo "== $(basename "$(dirname "$rd")") / $(basename "$rd")  [$status]"
    echo "   traj: $nfr frames x ${dt} ps = ${total} ps   |   render span: ${eff_tlim} ps"
    echo "   ${appf} ps/frame (stride $stride) -> $limit frames @ ${cur_fps} fps = ${video} s video"
    echo "   est. render: OVITO ~${ovmin} min  +  z-profile ~${zpmin} min"
    if [ "$yes" = 1 ]; then ans=""; else
      read -r -p "   [Enter]=render | 'PPF' or 'PPF FPS' | s=skip | q=quit > " ans || exit 0
    fi
    case "$ans" in
      "") break;;
      q)  echo "quit."; exit 0;;
      s)  continue 2;;
      *)  cur_ppf=$(awk -v a="$ans" 'BEGIN{split(a,x," "); printf "%g", x[1]}')
          f2=$(awk -v a="$ans" 'BEGIN{split(a,x," "); printf "%s", x[2]}')
          [ -n "$f2" ] && cur_fps=$f2;;
    esac
  done
  {
    echo "env OMP_NUM_THREADS=1 $PY $SCR/prod_dopant_movie.py --stride $stride --limit $limit --fps $cur_fps --zsigma 1.0 $partial $rd"
    echo "QT_QPA_PLATFORM=offscreen OVITO_THREAD_COUNT=2 $OVPY $SCR/prod_structure_movie.py --stride $stride --limit $limit --fps $cur_fps --repeat 1x10x2 --no-cell --zvert $partial $rd"
  } >> "$PLAN"
  if [ "$xyz" = 1 ]; then
    if [ "$status" = complete ]; then
      echo "env OMP_NUM_THREADS=1 $PY -c \"from pathlib import Path; from ase.io.trajectory import Trajectory; from ase.io import write; rd=Path('$rd'); out=rd.parent/'rendering'; out.mkdir(exist_ok=True); write(str(out/('traj_'+rd.name+'.xyz')), list(Trajectory(str(rd/'traj.traj'))), format='extxyz')\"" >> "$PLAN"
    else
      echo "  (xyz export skipped: run not complete)"
    fi
  fi
  nplan=$((nplan+1))
done

[ "$nplan" = 0 ] && { echo "nothing to render."; rm -f "$PLAN"; exit 0; }
echo 'echo ALL_RENDERS_DONE' >> "$PLAN"
LOG=${PLAN%.sh}.log
nohup bash "$PLAN" > "$LOG" 2>&1 &
echo ""
echo "launched $nplan run(s) in background (sequential)."
echo "  progress : tail -5 $LOG"
echo "  done when: grep ALL_RENDERS_DONE $LOG"
