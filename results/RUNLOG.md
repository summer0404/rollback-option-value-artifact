# RUNLOG

Chronological record of every run behind the paper, in four phases: development, the confirmatory
held-out run, exploratory runs on the held-out cells, and further exploratory analyses. Phases 1 to 3
ran on one laptop (18 cores, macOS, Python 3.14, numpy 2.4, scipy 1.17, git 2.50), 2026-10-13/14.

## Phase 1: development

| When | Script | What | Outcome |
|---|---|---|---|
| 2026-10-13 23:55 | `exp_e2.py` | E2 minimax sweep and four commitments, 5 seeds x 20,000 Monte Carlo episodes per cell | `e2_coverage.csv`, `e2_commitments.csv` |
| 2026-10-14 00:00 | `exp_theory.py` | Theorem checks: 1,000 random instances (T1, P2), 600 (T2), 1,500 + 400 + 1,500 random games (T3, P3) | `theory_checks.json`; 0 violations of every proved statement; P2 counterexamples confirmed exactly |
| 2026-10-14 00:00 | `exp_e1.py` | First E1 run with 4,000 episodes per seed | MC vs exact z-scores had mean -0.56 over 60 cells; rerun with more episodes |
| 2026-10-14 00:40 | `exp_e1.py` | E1 with 5 seeds x 10,000 episodes per cell | MC agrees with exact grid evaluation (max abs z 2.03 over 60 cells) |
| 2026-10-13 23:57 | `exp_e3.py` | First E3 attempt | Stopped by PID after 80 min: the macOS `/usr/bin/git` shim cost about 18 ms per call; switched to the real git binary (`xcrun --find git`) |
| 2026-10-14 01:15 | `exp_e3.py` | E3 v1: 5,040 executed episodes, constant insertion hazard in the Knowledge base, replay-only executor | Kept as `e3_episodes_v1_replayonly_consthazard.csv`. Two negative results, see below |
| 2026-10-14 02:00 | `exp_e3.py` | E3 v2 after the two changes below | `e3_episodes.csv` |

### Negative results of E3 v1 and the changes they led to

1. **ROVER lost to rollback-only at r1 = 0** (5.70 vs 5.28 per episode, Holm p = 0.019). Diagnosis by
   tracing 36 episodes: 16 of 22 ROVER audits happened after milestone 8 on carriers that were never
   attacked. The planner's Knowledge assumed a constant insertion hazard over the whole stream while the
   testbed attacker only acts in milestones 1 to 8. Change: the Knowledge base holds the exposure window
   (assumption A1 allows a time-varying hazard), for every belief-based policy. The original planner
   is kept as the ablation `rover_const` ("ROVER, no window").
2. **ROVER lost to reset-all at r1 = 10** (42.4 vs 28.4). Diagnosis: at r1 = 10 replaying a dependent
   commit costs five times the value of the work it preserves (w = 2), so discarding beats replaying.
   Every replay-based policy lost to reset-all for the same reason. Change: the shared recovery executor
   replays a dependent commit only when r1 <= w and discards it otherwise, for every selective-recovery
   policy, and ROVER plans with min(r1, w). The original executor is kept as the ablation
   `rover_replay_only`.

The v1 file is retained unchanged.

## Phase 2: confirmatory held-out run

After an internal review of the development results, the following runs prepared and executed the
confirmatory run.

| When | Script | What |
|---|---|---|
| 2026-10-14 03:10 | `exp_e2.py` | E2 with h = 10 (A9 requires h >= r1 over the sweep); A10 derivative condition checked per r1 |
| 2026-10-14 03:30 | `exp_e1.py` | E1 with misspecified planners that track their own belief (lambda, sigma_1, d, alpha) |
| 2026-10-14 03:35 | `exp_theory.py` | adds Theorem 1(v), exact Proposition 2(b) and 2(c) instances with enumeration, the tightness family of Theorem 3(iv), the Proposition 5(b) instance with h = 2, the kink of Proposition 5(c) |
| 2026-10-14 03:42 | `exp_e3.py --mode all` | rollback-only threshold tuned on development streams (0.7 at r1 = 1 and 10), then the confirmatory held-out run on fresh streams 10-19 and seeds 8-15 |

Changes frozen before the held-out run (protocol in `protocol/protocol-E3-heldout.md`): Knowledge floor
rho/4 after the exposure window; random audit recovers on a positive audit; periodic review and the
immutable validator audit the carriers a flagged or blocked commit read; new policies rollback_tuned,
rover_noaudit, gate, rover_gate; scenarios late attacker, sensor misspecification, gate study.

### Protocol record and deviations of the confirmatory run

- The protocol file `protocol/protocol-E3-heldout.md` (copy of the file written at 2026-10-14 03:37:59,
  SHA-256 a5ccdd1b8749246f037a6d8c023263eb09836c55410d9229dbd29a2870e1e6c5) has no external registry or
  timestamp. It is kept byte-identical so that the hash can be checked (`shasum -a 256`). In its text,
  "round 1" denotes the development phase above and its internal review. `testbed.py` and `exp_e3.py`
  were finalised at 03:39 and 03:40, after the protocol text and before the run started at 03:42. The
  edits implemented the frozen configuration listed above.
- Deviation 1: the protocol says the threshold of rollback only was tuned on the development file
  `e3_episodes.csv`. In fact `exp_e3.py --mode all` re-ran the threshold sweep on the same development
  streams and seeds (`e3_tune.csv`, 3,360 episodes) in the same invocation, before any held-out episode.
- Deviation 2: the protocol states H4 as "within each other's interval". This criterion is weak, and the
  paper also reports H4 with paired tests (Holm) and the paired interval.
- The protocol promised paired intervals against ROVER for each robustness condition. The paper reports
  them for every policy that was run (Table V).
- The floor rho/4 after the window was chosen after the development data and its internal review,
  without a sweep of other floors.

### Go/no-go criterion of E1

Fixed in the study plan before any run: "stop if the audit rate does not react to r1". Operationalised
before the E1 runs as: with phi = 0 the optimal audit rate at r1 = 10 must exceed the rate at r1 = 0.
Outcome on 2026-10-14: 8.1 against 1.5 audits per 100 periods, met.

## Phase 3: exploratory runs on the held-out cells (2026-10-14)

Exploratory runs after the confirmatory run, on the same held-out cells (streams 10-19, seeds 8-15) with
the same random draws. None of them is covered by the protocol.

| When | Script | What |
|---|---|---|
| 2026-10-14 06:48 | `exp_e3.py --mode extra` | graded insertion shifts of 2, 4 and 6 milestones (the late attacker is 8); attack rates 0.05 and 0.2 with the default prior (0.6) and a calibrated prior; ROVER with a gate-aware table (phi (1 - gamma d)) in the four gate conditions; rollback only with the threshold re-tuned for r1 <= 0.1 on the development runs (re-priced, theta = 0.5) -> `e3_extra.csv` |
| 2026-10-14 06:49 | `exp_theory.py` | A10 requires a positive attacker level: games with u* <= 0 are excluded from the Theorem 4(iii) check; the Proposition 5(b) count is restricted to games with h >= 2 and a positive level at r1 = 0 and 2; exact Proposition 2(b) thresholds from the plan envelopes; grid-versus-exact error; Proposition 5(a) payoffs at h = 1 |

Analysis (`analyze_e3.py`, `extra_analysis`): gate prices are re-priced exactly from the logged counts of
gate checks and blocked benign pushes (gate decisions do not depend on the price); periodic review and
the validator are re-priced at a zero monitoring price; the joint worst case over carrier and insertion
timing uses the main and late conditions.

## Phase 4: further exploratory analyses (2026-10-15 and 2026-10-16)

None of these runs is covered by the protocol. They ran on an 8 GB Apple M1 (Python 3.10.8, numpy
1.26.4, scipy 1.15.3, sympy 1.14.0, git 2.50.1) with at most three testbed workers. Outputs are in
`results/exploratory/`.

| Script | What | Episodes |
|---|---|---|
| `post_onestep.py` | One-step controllers (terminal-cost and rollout continuation, with and without blind recovery) on the E1 grid and the sensitivity grid, exact grid evaluation plus a Monte Carlo cross-check | model |
| `post_publication.py` | Ungated releases (nu = 1/24, 1/4) added to the main condition at r1 = 1 for nine policies; distinct leaked canary lines; re-pricing per leaked line and per contaminated release | 4,320 |
| `post_costsem.py` | Instrumented ROVER and tuned rollback only at r1 in {0.1, 1}: planned against executed recovery prices, predicted against surviving residual harm, growth of D | 960 |
| `post_grid.py`, `post_grid_e3.py` | ROVER tables on grids of 126 to 4,001 points against 8,001 points at T = 24; ROVER with 2,001-point tables on the held-out main cells | 960 |
| `post_stats.py` | Paired t-test and sign-flip permutation test of the mean next to the Wilcoxon test, paired win share next to A12, carrier selection and scoring on separate seed halves | none |
| `post_a10.py` | Exact rational root counting of dU/dx for the E2 game and the Proposition 5 instance | none |

Every testbed run reproduced the decision counts and losses of the corresponding committed episodes
exactly (releases and instrumentation only add pushes and logging). Independently, 12,110 of the 25,200
executed episodes of `e3_tune.csv`, `e3_heldout.csv` and `e3_extra.csv` were re-executed with identical
results, including every episode behind H1 to H4, Table IV, the robustness rows of Table V and the gate
rows. `testbed.py` and `exp_e3.py` were edited at 06:47 and 06:48 on 2026-10-14, after the confirmatory
run, to add the Phase 3 conditions. The re-executions show that these edits did not change the
confirmatory configurations.

`stats.boot_ci` draws the bootstrap index matrix in blocks of 50 rows from the same generator. The
results are bit-identical (E1 and E2 outputs unchanged) and the peak memory of E2 falls from about 3.2 GB
to about 0.1 GB.

## Housekeeping after the runs (no numbers changed)

- The exploratory scripts and outputs were renamed (`post_*.py`, `render_post.py`, `results/exploratory/`)
  and the Fig. 1 script moved into the artifact (`figures/fig1/`). Table V's main-run row now prints the
  intervals of Table IV (`e3_main.csv`) instead of a second bootstrap draw of the same comparisons
  (`render_extra.py`). `render.py` regenerates every table and macro from the committed results.
