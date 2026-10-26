#!/bin/bash
# Usage (from repo root): hpc/submit_mc.sh [CHUNKS=20] [PATHS=250]   -> CHUNKS·PATHS paths per β / γ point
set -e
CHUNKS=${1:-20}; PATHS=${2:-250}
POINTS=$(python3 -c "import ast;t=ast.parse(open('monte_carlo.py').read());v={n.targets[0].id:ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in ('BETAS','GAMMAS')};print(len(v['BETAS'])+len(v['GAMMAS']))")
rm -rf mc_parts
A=$(sbatch --parsable --array=0-$((POINTS*CHUNKS-1)) --export=ALL,CHUNKS=$CHUNKS,PATHS=$PATHS hpc/mc_array.sb)
B=$(sbatch --parsable --dependency=afterok:$A hpc/mc_aggregate.sb)
echo "array $A ($((POINTS*CHUNKS)) tasks, $((CHUNKS*PATHS)) paths/point), aggregate $B"
