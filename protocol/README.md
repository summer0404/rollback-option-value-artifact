# Protocol of the confirmatory E3 run

`protocol-E3-heldout.md` is the protocol of the confirmatory held-out run (Section VI-A of the paper).
It was written at 2026-10-14 03:37:59, before any held-out episode was executed (the run started at
03:42), and it is kept exactly as written. Check it with

```sh
shasum -a 256 protocol-E3-heldout.md
# a5ccdd1b8749246f037a6d8c023263eb09836c55410d9229dbd29a2870e1e6c5
```

The hash is the one recorded in `../results/RUNLOG.md`. The file was not filed with an external
registry, so the hash shows that the file is unchanged since it was recorded, not when it was written.

No script reads this file: the configuration it fixes is implemented in `../src/testbed.py` and
`../src/exp_e3.py`, and the results do not depend on its wording.

## Terms used in the file

Because the file must stay byte-identical, it keeps the wording it was written with:

| In the file | Meaning |
|---|---|
| "Pre-registration" (title) | This protocol, written before the held-out run; it has no external registry |
| "round 1", "the development run" | The development phase on streams 0 to 9 and seeds 0 to 7 (`../results/RUNLOG.md`, Phase 1) |
| "the round-1 review" | The authors' internal review of the development results, before the protocol was written |
| "e3_episodes.csv of round 1" | `../results/e3_episodes.csv` |

## Departures from the protocol

Listed in `../results/RUNLOG.md` (Phase 2): the rollback-only threshold came from a fresh sweep over the
same development streams, and H4 is also reported with paired tests.
