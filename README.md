# Artifact: The Option Value of Rollback

Code, data, full proofs and run records for the paper "The Option Value of Rollback:
Irreversibility-Aware Self-Protection for Coding-Agent Workflows". Every number, table and figure of
the paper is generated from this folder, and `supplement/proofs.pdf` holds the full proofs that the
paper refers to as the supplementary material.

**Scope.** The evaluation is (i) exact dynamic programming and Monte Carlo simulation of a restricted
model (E1), (ii) a closed-form Stackelberg allocation study with Monte Carlo validation (E2), and
(iii) an executed testbed in which scripted (non-LLM) workflow streams drive real git repositories
built from synthetic fixtures (E3). No language-model agent, real agent harness, real repository or
real poisoned content is used. Sensors (alarms, audit outcomes, validator and review detections) are
simulated draws on the ground-truth taint. The attacker writes an inert canary tag.

## Quick check (about 2 minutes after installation)

```sh
python3 -m pip install -r requirements.txt
cd src
python3 render.py                                  # 1. every table and macro from the committed results
python3 -B check_reproduce.py --tmp /tmp/rov-check # 2. re-execute committed testbed episodes with real git
python3 exp_e2.py                                  # 3. recompute E2 (Table III, Fig. 3) from scratch
```

1. `render.py` (about 1 s) rewrites `generated/` (the tables, plot coordinates and inline numbers the
   paper inputs) and `supplement/generated/` from `results/`. The files come out byte-identical to the
   committed ones; check with `git status` or `diff`.
2. `check_reproduce.py` (about 15 s with 4 workers) re-runs 33 held-out episodes (3 cells, all 11
   policies, r1 = 1) and compares loss and decision counts with `results/e3_heldout.csv`. Use
   `--cells` for a larger sample. The directory given to `--tmp` must lie outside any git repository.
3. `exp_e2.py` (about 1 min) recomputes `results/e2_*.csv` from scratch, with identical values.

## Full reproduction

```sh
./run_all.sh           # everything, about 4.5 hours on an 18-core laptop
./run_all.sh --no-e3   # skip the executed testbed and reuse its committed episodes
```

`run_all.sh` runs E1, E2, the theorem checks, the executed testbed (development tuning, the
confirmatory held-out run and the exploratory runs), the analyses, the exploratory analyses added
after the confirmatory run, the rendering, Fig. 1 and the supplement. Per-step run times are noted in
`run_all.sh`. Every random draw is seeded and testbed counts are deterministic. Wall-clock timings
(`rec_seconds`, `wall`, `overhead.json`) depend on the machine and are reported only as a cost
indication.

**Requirements.** Python 3.10 or later with numpy, scipy, matplotlib and sympy (`requirements.txt`);
git 2.30 or later; pdflatex for the supplement. Peak memory is about 1.1 GB (theory checks), and the
testbed needs about 0.1 GB per worker. Set `ROV_TMP` to choose where temporary repositories are
created, `ROV_GIT` to choose the git binary (on macOS the `/usr/bin/git` shim is slow; `xcrun --find
git` gives the real binary), and `ROV_GEN` to redirect the paper's generated files.

**Tested with.** The committed results come from Python 3.14, numpy 2.4, scipy 1.17 and git 2.50
(18-core macOS laptop; development, confirmatory and Phase 3 runs) and from Python 3.10.8, numpy
1.26.4, scipy 1.15.3, sympy 1.14.0 and git 2.50.1 (8 GB Apple M1; Phase 4 analyses). The quick check
was last run with Python 3.12.6, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, sympy 1.14.0, git 2.55
and TeX Live 2026; with these versions E1, E2 and the exploratory model-level analyses reproduce the
committed files exactly, except that the t-test p-values in `results/exploratory/statistics/primary.json`
differ in the 15th significant digit (floating-point summation order), which no reported number uses.

## Layout

| Path | Content |
|---|---|
| `src/model.py` | Restricted recovery-audit POMDP: grid DP, exact recursion, policy evaluation, Monte Carlo (also with a misspecified planner belief), myopic rule, closed forms of Theorem 3 |
| `src/structure.py` | Structural checks (Theorem 1, Proposition 2) and exhaustive enumeration of all two-period plans |
| `src/game.py` | Multi-carrier audit game: closed forms of Theorem 4, minimax (water filling), best response, derivative formula, A10 check |
| `src/testbed.py` | Executed git testbed: fixtures, scripted agent, carriers, attacker, activation, recovery executors, policies |
| `src/exp_e1.py`, `exp_e2.py`, `exp_e3.py`, `exp_theory.py`, `exp_overhead.py` | Experiments E1, E2, E3 (development tuning, confirmatory held-out run, exploratory runs), the theorem checks and the planner overhead |
| `src/analyze_e3.py`, `stats.py` | Cluster bootstrap, Wilcoxon signed-rank with Holm correction, Vargha-Delaney A12 |
| `src/post_*.py`, `render_post.py` | Exploratory analyses added after the confirmatory run (supplement, Section 8): one-step controllers, publication harm, cost semantics, grid convergence at T = 24, estimands, split-sample adaptive attacker, exact A10 check |
| `src/render.py`, `render_extra.py` | Results to LaTeX tables, pgfplots coordinates and macros; writes `results/summary.json` |
| `src/check_reproduce.py` | Quick re-execution check of committed testbed episodes |
| `figures/fig1/` | Fig. 1: `make_fig1.py` re-runs one testbed episode with tracing and draws the figure; icons with licence and `ATTRIBUTION.md` |
| `generated/` | The tables, plot coordinates and macros that the paper inputs, as rendered from `results/` |
| `results/` | All CSV and JSON outputs, run logs, `RUNLOG.md` (chronological record of every run, deviations from the protocol) |
| `results/exploratory/` | Outputs of the exploratory analyses (`post_*.py`) |
| `protocol/protocol-E3-heldout.md` | Protocol of the confirmatory E3 run, written before the held-out run |
| `supplement/proofs.tex`, `proofs.pdf` | Supplementary material: full proofs, exact assumptions, verification protocol, development history and exploratory analyses |

## Mapping from paper to artifact

| Paper element | Script | Output |
|---|---|---|
| Table I (numerical checks), Theorems 1, 3, 4, Propositions 2 and 5 | `exp_theory.py` | `results/theory_checks.json`, `prop2_instance.csv`, `t2_tightness.csv` |
| Table II, Fig. 2 (E1) | `exp_e1.py` | `results/e1_grid.csv`, `e1_grid_seeds.csv`, `e1_sens.csv`, `e1_misspec.csv` |
| Table III, Fig. 3 (E2) | `exp_e2.py` | `results/e2_coverage.csv`, `e2_commitments.csv` |
| Table IV and H1 to H4 (E3, confirmatory held-out run) | `exp_e3.py --mode all`, `analyze_e3.py` | `results/e3_tune.csv`, `e3_thresholds.json`, `e3_heldout.csv`, `e3_main.csv`, `e3_components.csv`, `e3_sweep.csv`, `e3_tests.csv`, `e3_robust.csv`, `e3_gate.csv`, `e3_adaptive.csv`, `e3_alloc.csv`, `e3_family.csv`, `e3_carrier.csv` |
| Tables V and VI (E3, exploratory: attack timing and rate, sensors, gate prices) | `exp_e3.py --mode extra`, `analyze_e3.py`, `render_extra.py` | `results/e3_extra.csv`, `e3_extra_meta.json`, `e3_knowledge.csv`, `e3_gate_price.csv`, `e3_extra_summary.csv` |
| Fig. 1 | `figures/fig1/make_fig1.py` | `figures/fig1/fig1_overview.pdf`, `generated/fig1_macros.tex` |
| Statements marked "(supplement)" in the paper | `post_*.py`, `render_post.py` | `results/exploratory/*`, supplement Tables 3 and 4 |
| Planner overhead (Section V) | `exp_overhead.py` | `results/overhead.json` |
| Development runs (supplement, Section 7) | development version of `exp_e3.py` | `results/e3_episodes_v1_replayonly_consthazard.csv`, `e3_episodes.csv`, `e3_main_development.csv` (kept unchanged, see `RUNLOG.md`) |
| Every inline number of the paper | `render.py` | `generated/macros.tex` |

Supplement tables: 1, exhaustive enumeration for Proposition 2 (`generated/tab_prop2.tex`); 2, E1
sensitivity grid (`tab_esens.tex`); 3, one-step controllers (`tab_post_onestep.tex`); 4, publication harm
(`tab_post_pub.tex`), all in `supplement/generated/`. Theorem, proposition, definition and assumption
numbers in the supplement match the paper; Lemma 2 of the supplement is the reduction to
first-intervention policies used by Theorem 3.

## Protocol of the confirmatory run

`protocol/protocol-E3-heldout.md` fixes the configuration, tests and hypotheses H1 to H4 of the held-out
E3 run. It was written before any held-out episode was executed and is kept byte-identical, so that

```sh
shasum -a 256 protocol/protocol-E3-heldout.md
```

prints the hash recorded in `results/RUNLOG.md`
(`a5ccdd1b8749246f037a6d8c023263eb09836c55410d9229dbd29a2870e1e6c5`). The protocol has no external
registry or timestamp. Because it must stay byte-identical, it keeps its original wording (the title
"Pre-registration", "round 1" for the development phase); `protocol/README.md` explains these terms.
`RUNLOG.md` lists the steps of the run that departed from it.

## Testbed safety and anonymity

All repositories live in a fresh temporary directory outside any user repository, with
`GIT_CONFIG_GLOBAL=/dev/null`, `GIT_CONFIG_NOSYSTEM=1`, `GIT_CEILING_DIRECTORIES` set to the parent of
the temporary directory, hooks disabled, a fixed author `agent <agent@example.invalid>` and fixed
commit dates. The "remote" is a local bare repository that rejects non-fast-forward updates and
deletions, which models an irreversible push. Nothing leaves the machine.

## License

The code and data are released under the MIT License (`LICENSE`). The Fig. 1 icons are Microsoft Fluent
UI Emoji under their own MIT License (`figures/fig1/icons/LICENSE-fluentui-emoji.txt`, sources in
`figures/fig1/ATTRIBUTION.md`).
