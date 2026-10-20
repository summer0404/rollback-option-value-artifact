# Pre-registration of the held-out E3 run

Written 2026-10-14, after the round-1 review and before any held-out episode was executed.
The development run (streams 0 to 9 per family, seeds 0 to 7, file `e3_episodes.csv` of round 1) was
used to diagnose the first run and to tune the rollback-only threshold. The held-out run uses fresh
workflow streams 10 to 19 per family (30 streams) and fresh seeds 8 to 15.

## Frozen configuration

- ROVER: per-carrier backward-induction tables (belief grid of 501 points), effective replay price
  min(r1, w) with w = 2, Knowledge = exposure window over milestones 1 to 8 with per-carrier hazard
  rho = 0.6/4/8 inside the window and a floor rho/4 after it (never zero).
- Shared recovery executor: replay a dependent commit when r1 <= w, discard it otherwise.
- Baselines: no audit; random audit at ROVER's realised rate, recover on a positive audit; periodic
  review every 4 milestones, then audit each carrier read by a flagged commit and recover positives;
  rollback only with the threshold tuned on the development streams (grid 0.2 to 0.8 in steps of 0.1,
  chosen separately for r1 <= 1 and r1 = 10); reset all; immutable validator that blocks a contaminated
  commit and then audits the carriers it read and recovers positives; myopic rule.
- Ablations: ROVER phi-blind (phi = 0 in planning); ROVER without the exposure window (constant hazard);
  ROVER without audits (DP restricted to wait and recover); ROVER replay-only (r1 = 10 only).
- Sensors, carriers and costs as in round 1.

## Hypotheses and tests

Unit of analysis: per-stream mean over the 8 seeds (30 clusters). Tests: two-sided Wilcoxon signed-rank
on paired per-stream means. Multiplicity: Holm over all comparisons against ROVER at the same r1 (the
family is stated with each result). Intervals: 95% cluster-bootstrap intervals of the paired difference
(2,000 resamples of streams).

- H1 (primary). At r1 = 1, ROVER has a lower mean loss than tuned rollback-only.
- H2 (primary). At r1 = 1, ROVER has fewer irreversible activations than ROVER without audits.
- H3 (secondary). At r1 = 0, ROVER has a lower mean loss than ROVER phi-blind.
- H4 (secondary, reported whatever the outcome). At r1 = 0 ROVER and tuned rollback-only are within
  each other's interval, and at r1 = 10 ROVER and no audit are within each other's interval.
- Irreversible harm against rollback-only is reported as a paired difference with interval, without a
  directional claim.

## Robustness conditions (descriptive, paired intervals against ROVER, r1 = 1)

- Late attacker: insertion at milestones 9 to 16 (same random draws shifted by 8), outside the assumed window.
- Sensor misspecification: true alarm rate s1 = 0.25 (planner assumes 0.4); true audit power d = 0.7 and
  false-alarm rate alpha = 0.1 (planner assumes 0.9 and 0.05).
- Gate: benign push requests at rate nu in {1/24, 1/4} per milestone, fraction gamma in {0.5, 1} of
  irreversible activation paths passing through the harness gate; policies gate-only and ROVER plus gate.

Any deviation from this plan will be listed in `artifact/results/RUNLOG.md`.
