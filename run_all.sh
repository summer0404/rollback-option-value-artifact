#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper.
#   ./run_all.sh            full run (E1, E2, theory checks, executed testbed E3, render, Fig. 1, supplement)
#   ./run_all.sh --no-e3    skip the executed testbed (reuses results/e3_heldout.csv, e3_tune.csv, e3_extra.csv)
#
# Requirements: python3 with numpy, scipy, matplotlib (Fig. 1), sympy (post_a10.py); git; pdflatex (TeX Live 2020 or later).
# The testbed creates temporary git repositories under $ROV_TMP (default: the system temp dir)
# with an isolated git configuration, and deletes them after each episode.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_E3=1
for a in "$@"; do
  case "$a" in
    --no-e3) RUN_E3=0 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done
cd "$HERE/src"
python3 exp_e1.py                 # E1: exact DP grid, sensitivity, misspecification  (~5 min)
python3 exp_e2.py                 # E2: Stackelberg allocation                          (~1 min)
python3 exp_theory.py             # numerical checks of Theorems 1, 3, 4 and Propositions 2, 5 (~45 min)
if [ "$RUN_E3" = 1 ]; then
  python3 exp_e3.py --mode all            # E3: tuning on development streams, held-out run (~2.5 h, 14 workers)
  python3 exp_e3.py --mode extra          # E3 exploratory: Knowledge timing and rate, gate-aware ROVER (~1.2 h)
fi
python3 analyze_e3.py             # cluster bootstrap, paired tests, sweeps
# exploratory analyses after the confirmatory run (results/exploratory/): model-level ones always, testbed ones with E3
export ROV_TMP="${ROV_TMP:-$(mktemp -d)}"
python3 post_a10.py                # exact sign of dU/dx for A10 (needs sympy)
python3 post_grid.py               # belief-grid convergence at T = 24
python3 post_onestep.py            # one-step controllers in E1 (~4 min)
python3 post_stats.py              # mean-aligned tests, A12, split-sample adaptive attacker
if [ "$RUN_E3" = 1 ]; then
  python3 post_grid_e3.py --grid 2001 --tmp "$ROV_TMP"   # ROVER with 2,001-point tables (~7 min)
  python3 post_costsem.py --tmp "$ROV_TMP"               # planner cost semantics (~7 min)
  python3 post_publication.py --tmp "$ROV_TMP"           # releases that publish contaminated work (~40 min)
else
  python3 post_grid_e3.py --grid 2001 --analyze-only; python3 post_costsem.py --analyze-only
  python3 post_publication.py --analyze-only
fi
python3 exp_overhead.py           # planner overhead (machine dependent)
python3 render.py                 # results -> generated/*.tex (paper), supplement/generated/*.tex, summary.json
python3 "$HERE/figures/fig1/make_fig1.py"      # Fig. 1 walkthrough + generated/fig1_macros.tex
cd "$HERE/supplement"
pdflatex -interaction=nonstopmode proofs >/dev/null && pdflatex -interaction=nonstopmode proofs >/dev/null
echo "done: results/*.csv, results/summary.json, supplement/proofs.pdf, generated/*.tex, figures/fig1/fig1_overview.pdf"
