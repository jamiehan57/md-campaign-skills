#!/bin/bash
# sem-analysis [--tlim PS] [--partial] [--sbatch[=PARTITION]] [--local] [run_dir ...]
# Campaign analysis suite (prod_cation_analysis.py: msd, msd_xyz, parperp,
# cn_cationO, zprofile) in the shared style.
#   default        : full trajectory -> <subdir>/plot/          (created if missing)
#   --tlim 3000    : first 3000 ps   -> <subdir>/plot/3000ps/   (created if missing)
#   --local        : run here (login node, single-thread BLAS)  [default]
#   --sbatch       : submit as a CPU-only SLURM job instead (default partition
#                    snu-g1 CPU nodes); --sbatch=snu-gpu2 to override
#   --partial      : include still-running runs (outputs suffixed _partial)
# CONTINUED ("patched-up") runs like 07: analysing the part2 dir automatically
# STITCHES part1+part2 (run_meta continuation_from); this wrapper additionally
# drops a part1 dir from the run list when its part2 is also listed, so the
# stitched analysis is not duplicated by a lone part1 pass.
# No run dir args: analyses every prod*/traj.traj under the CURRENT directory.
# Uses the sem-neq python explicitly, so it works from any active env (sem-lmp etc).
PY=$HOME/anaconda3/envs/sem-neq/bin/python
SCRIPT=$HOME/SEM_MLCC/02_RUN/03_SURF_HETERO/scripts/prod_cation_analysis.py
WRAP=$HOME/SEM_MLCC/02_RUN/scripts/sem_analysis.sh
args=(); runs=(); mode=local; part=snu-g1
while [ $# -gt 0 ]; do
  case "$1" in
    --tlim|--tmax)  args+=(--tmax "$2"); shift 2;;
    --partial)      args+=(--partial); shift;;
    --local)        mode=local; shift;;
    --sbatch)       mode=sbatch; shift;;
    --sbatch=*)     mode=sbatch; part="${1#--sbatch=}"; shift;;
    -h|--help)      sed -n '2,15p' "$0"; exit 0;;
    *)              runs+=("$1"); shift;;
  esac
done

if [ ${#runs[@]} -eq 0 ]; then
  while IFS= read -r t; do runs+=("$(dirname "$t")"); done \
      < <(find . -maxdepth 3 -name traj.traj -path '*prod*' | sort)
fi
if [ ${#runs[@]} -eq 0 ]; then
  echo "sem-analysis: no prod*/traj.traj under $(pwd) -- pass run dir(s) explicitly"
  exit 1
fi

# drop continuation SOURCES whose part2 is also in the list (stitched covers them)
mapfile -t runs < <("$PY" - "${runs[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
runs = [Path(a).resolve() for a in sys.argv[1:]]
sources = set()
for r in runs:
    mp = r / "run_meta.json"
    if mp.exists():
        try:
            src = json.loads(mp.read_text()).get("continuation_from")
            if src:
                sources.add(Path(src).resolve())
        except Exception:
            pass
for r in runs:
    if r in sources:
        print(f"  [patch-up] skip {r.name}: covered by its continuation's "
              "stitched analysis", file=sys.stderr)
    else:
        print(r)
PYEOF
)

if [ "$mode" = sbatch ]; then
  mkdir -p ~/claude/slurm
  JOB=~/claude/slurm/sem_analysis_$$.sh
  {
    echo '#!/bin/bash'
    echo "#SBATCH -p $part"
    echo '#SBATCH -N 1 -n 1 -c 2'
    echo '#SBATCH -t 06:00:00'
    echo '#SBATCH -J sem_analysis'
    echo "#SBATCH -o $HOME/claude/slurm/sem_analysis.%j.out"
    echo "cd $(pwd)"
    echo "bash $WRAP --local ${args[*]} ${runs[*]}"
  } > "$JOB"
  sbatch "$JOB"
  echo "log: ~/claude/slurm/sem_analysis.<jobid>.out"
  exit 0
fi

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$PY" -u "$SCRIPT" "${args[@]}" "${runs[@]}"
