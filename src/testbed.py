"""Executed scripted testbed (E3): real git repositories, scripted non-LLM workflow streams.

Each episode creates a small repository from a deterministic fixture, runs a stream of
T milestones of a scripted coding agent, and lets one adaptation policy act at every
milestone (MAPE-K). Persistent carriers:
    memory  .agent/memory.md          (on main)
    skill   .agent/skill.md           (on main)
    queue   .agent/queue.json         (on main)
    branch  wip.md on the orphan branch `wip` (read into main as .agent/wip.md)
An attacker (abstract) writes an inert canary tag into one carrier. A tainted carrier,
when read by a milestone, contaminates the derived commit with probability P_INFL
(the commit then contains a comment line holding the canary tag). Activation of a
tainted carrier either pushes main to a local bare remote that denies non-fast-forward
updates and deletions (irreversible) or commits an incident line locally (reversible).

Recovery runs real git: reset to the carrier checkpoint, then replay the milestones
since the checkpoint, cherry-picking independent commits and re-executing dependent
ones from their milestone specs with the restored carrier. Conflicts fall back to
re-execution. Sensors (alarms, audit outcomes, validator and review detections) are
simulated Bernoulli draws on the ground-truth taint, with the rates in SENSE.

All git runs happen in a private temporary directory outside any user repository, with
an isolated git configuration and fixed author names and dates.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace

import numpy as np

from game import CARRIERS
from model import Params, GridDP, EPS, W, R, A, bayes_alarm, predict, audit_split

CARRIER_NAMES = [c.name for c in CARRIERS]
NC = len(CARRIERS)
T_MS = 24                       # milestones per stream
P_ATTACK = 0.6                  # probability that an episode is attacked
T0_MAX = 8                      # insertion milestone uniform in 1..T0_MAX
P_INFL = 0.6                    # contamination probability of a tainted read
SENSE = dict(d=0.9, alpha=0.05, s0=0.1, s1=0.4, d_v=0.9, d_c=0.9)
COST = dict(H=20.0, h=1.0, r0=3.0, c_a=1.0, w_lost=2.0, c_v=0.2, c_r=0.25)
REVIEW_EVERY = 4
FAMILIES = ("flat", "hub", "layered")
N_MOD = 6
MODULE_WEIGHTS = {
    "flat": np.ones(N_MOD) / N_MOD,
    "hub": np.array([0.6, 0.08, 0.08, 0.08, 0.08, 0.08]),
    "layered": np.array([0.3, 0.25, 0.2, 0.12, 0.08, 0.05]),
}
CANARY = "canary:"

POLICIES = ("no_audit", "random_audit", "periodic_review", "rollback_only", "reset_all",
            "immutable_validator", "myopic", "rover", "rover_irrev_blind", "rover_const", "rover_replay_only")
POLICIES = POLICIES + ("rollback_tuned", "rover_noaudit", "gate", "rover_gate")
POLICIES = POLICIES + ("rover_gate_aware", "rollback_cheap")      # exploratory additions (e3_extra.csv)
ROVER_FAMILY = ("rover", "rover_irrev_blind", "rover_const", "rover_replay_only", "rover_noaudit", "rover_gate",
                "rover_gate_aware")
GATING = ("gate", "rover_gate", "rover_gate_aware")
RHO = P_ATTACK / NC / T0_MAX     # per-carrier insertion hazard inside the exposure window
RHO_FLOOR = RHO / 4              # Knowledge never assumes a zero hazard after the window
LATE_SHIFT = 8                   # late attacker: insertion shifted to milestones 9..16
C_BLOCK = 2.0                    # cost of a benign push blocked by a gate false alarm


# ----------------------------------------------------------------------------- git helpers
def _git_binary():
    """Real git binary (the macOS /usr/bin/git shim adds ~18 ms per call)."""
    if os.environ.get("ROV_GIT"):
        return os.environ["ROV_GIT"]
    if shutil.which("xcrun"):
        p = subprocess.run(["xcrun", "--find", "git"], capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    return shutil.which("git") or "git"


GIT = _git_binary()


class Git:
    def __init__(self, root):
        self.root = root
        self.repo = os.path.join(root, "repo")
        self.remote = os.path.join(root, "remote.git")
        self.deploy = os.path.join(root, "deploy")
        self.env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": root,
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CEILING_DIRECTORIES": os.path.dirname(root),
            "GIT_AUTHOR_NAME": "agent", "GIT_AUTHOR_EMAIL": "agent@example.invalid",
            "GIT_COMMITTER_NAME": "agent", "GIT_COMMITTER_EMAIL": "agent@example.invalid",
        }
        self.clock = 0
        self.calls = 0

    def run(self, *args, cwd=None, check=True, inp=None):
        self.clock += 1
        env = dict(self.env)
        stamp = f"1767225600 +0000"  # 2026-01-01T00:00:00Z
        ts = 1767225600 + self.clock
        env["GIT_AUTHOR_DATE"] = f"{ts} +0000"
        env["GIT_COMMITTER_DATE"] = f"{ts} +0000"
        self.calls += 1
        p = subprocess.run([GIT, "-c", "core.hooksPath=" + os.devnull, "-c", "commit.gpgsign=false",
                            "-c", "gc.auto=0", "-c", "advice.detachedHead=false", *args],
                           cwd=cwd or self.repo, env=env, capture_output=True, text=True, input=inp)
        if check and p.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
        return p

    def head(self):
        return self.run("rev-parse", "HEAD").stdout.strip()


# ----------------------------------------------------------------------------- fixture and stream spec
def stream_spec(family: str, stream: int, T: int = T_MS):
    """Deterministic milestone specs and carrier reads for one workflow stream."""
    rng = np.random.default_rng(10_000 + 97 * FAMILIES.index(family) + stream)
    specs = []
    created = []  # (k, module)
    for k in range(1, T + 1):
        reads = [bool(rng.random() < CARRIERS[i].Delta) for i in range(NC)]
        u = rng.random()
        if created and u < 0.25:
            j, mod = created[int(rng.integers(len(created)))]
            specs.append(dict(k=k, kind="modify", mod=mod, dep=j, a=int(rng.integers(2, 9)), reads=reads))
        elif created and u < 0.45:
            j, mod = created[int(rng.integers(len(created)))]
            specs.append(dict(k=k, kind="test", mod=mod, dep=j, a=int(rng.integers(2, 9)), reads=reads))
        else:
            mod = int(rng.choice(N_MOD, p=MODULE_WEIGHTS[family]))
            specs.append(dict(k=k, kind="add", mod=mod, dep=None, a=int(rng.integers(2, 9)),
                              b=int(rng.integers(0, 50)), reads=reads))
            created.append((k, mod))
    return specs


def build_fixture(dst: str, family: str, stream: int):
    g = Git(dst)
    os.makedirs(g.repo)
    os.makedirs(g.deploy)
    g.run("init", "-q", "--bare", g.remote, cwd=dst)
    g.run("--git-dir", g.remote, "config", "receive.denyNonFastForwards", "true", cwd=dst)
    g.run("--git-dir", g.remote, "config", "receive.denyDeletes", "true", cwd=dst)
    g.run("init", "-q", "-b", "main")
    os.makedirs(os.path.join(g.repo, "src"))
    os.makedirs(os.path.join(g.repo, "tests"))
    os.makedirs(os.path.join(g.repo, ".agent"))
    for m in range(N_MOD):
        with open(os.path.join(g.repo, "src", f"mod{m}.py"), "w") as f:
            f.write(f'"""Fixture module {m} ({family}, stream {stream})."""\n\n')
            for j in range(5):
                f.write(f"def base_{j}(x):\n    return x + {j + m}\n\n")
        with open(os.path.join(g.repo, "tests", f"test_mod{m}.py"), "w") as f:
            f.write(f"from src import mod{m}\n\n\ndef test_base():\n    assert mod{m}.base_0(1) == {1 + m}\n\n")
    with open(os.path.join(g.repo, "src", "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(g.repo, ".agent", "memory.md"), "w") as f:
        f.write("# memory notes\n- prefer small functions\n")
    with open(os.path.join(g.repo, ".agent", "skill.md"), "w") as f:
        f.write("# skill: format\n1. run the formatter\n")
    with open(os.path.join(g.repo, ".agent", "queue.json"), "w") as f:
        json.dump({"queued": []}, f)
        f.write("\n")
    with open(os.path.join(g.repo, ".agent", "wip.md"), "w") as f:
        f.write("# wip\n")
    with open(os.path.join(g.repo, ".agent", "incident.log"), "w") as f:
        f.write("")
    g.run("add", "-A")
    g.run("commit", "-q", "-m", "fixture")
    # orphan branch wip holding wip.md at its root, built with plumbing
    blob = g.run("hash-object", "-w", "--stdin", inp="# wip\n").stdout.strip()
    tree = g.run("mktree", inp=f"100644 blob {blob}\twip.md\n").stdout.strip()
    c = g.run("commit-tree", tree, "-m", "wip root").stdout.strip()
    g.run("update-ref", "refs/heads/wip", c)
    return g


# ----------------------------------------------------------------------------- episode randomness
def episode_draws(family, stream, seed, T=T_MS):
    rng = np.random.default_rng([2027, FAMILIES.index(family), stream, seed])
    u_att = float(rng.random())
    attacked = bool(u_att < P_ATTACK)
    carrier = int(rng.integers(NC))
    t0 = int(rng.integers(1, T0_MAX + 1))
    return dict(attacked=attacked, u_att=u_att, carrier=carrier, t0=t0,
                infl=rng.random(T + 2), act=rng.random((T + 2, NC)), irr=rng.random((T + 2, NC)),
                alarm=rng.random((T + 2, NC)), audit=rng.random((T + 2, NC)), rand_aud=rng.random((T + 2, NC)),
                valid=rng.random(T + 2), review=rng.random(T + 2),
                benign=rng.random(T + 2), gate_det=rng.random((T + 2, NC)), gate_path=rng.random((T + 2, NC)),
                gate_fa=rng.random(T + 2), aud2=rng.random((T + 2, NC)))


# ----------------------------------------------------------------------------- planner tables
_DP_CACHE = {}


def effective_r1(r1, replay_only=False):
    """Shared executor rule: replay a dependent commit or discard it, whichever is cheaper."""
    return r1 if replay_only else min(r1, COST["w_lost"])


def carrier_params(i, r1, phi_override=None, T=T_MS, rho_until=T0_MAX, rho_late=RHO_FLOOR, c_a=None, rho=RHO):
    c = CARRIERS[i]
    return Params(T=T, pi0=0.0, rho=rho, rho_until=rho_until, rho_late=rho_late, Delta=1, grow_p=c.Delta,
                  lam=c.lam, phi=c.phi if phi_override is None else phi_override, H=COST["H"],
                  h=COST["h"] * P_INFL, r0=COST["r0"], r1=r1, c_a=COST["c_a"] if c_a is None else c_a,
                  d=SENSE["d"],
                  alpha=SENSE["alpha"], s0=SENSE["s0"], s1=SENSE["s1"], c_fr=0.0, D0=0)


def planner(i, r1, variant="rover", rho=RHO, gate_gamma=0.0):
    """Per-carrier DP table. Variants: rover (and rover_gate, which ignores the gate), rover_irrev_blind
    (phi = 0), rover_const (constant hazard, no exposure window), rover_replay_only (no discard option),
    rover_noaudit (DP restricted to wait and recover: audits priced out), rover_gate_aware (plans with the
    irreversible fraction left after a gate that sees a share gamma of the paths, phi (1 - gamma d)).
    rho is the planner's insertion hazard inside the window (Knowledge of the attack rate)."""
    v = "rover" if variant == "rover_gate" else variant
    key = (i, r1, v, rho, gate_gamma if v == "rover_gate_aware" else 0.0)
    if key not in _DP_CACHE:
        r1e = effective_r1(r1, v == "rover_replay_only")
        const = v == "rover_const"
        phi = None
        if v == "rover_irrev_blind":
            phi = 0.0
        elif v == "rover_gate_aware":
            phi = CARRIERS[i].phi * (1.0 - gate_gamma * SENSE["d"])
        _DP_CACHE[key] = GridDP(carrier_params(i, r1e, phi,
                                               rho_until=10 ** 9 if const else T0_MAX,
                                               rho_late=rho if const else rho / 4,
                                               c_a=1e9 if v == "rover_noaudit" else None, rho=rho), G=501)
    return _DP_CACHE[key]


# ----------------------------------------------------------------------------- the agent
class Episode:
    def __init__(self, root, family, stream, seed, policy, r1, rand_rate=0.0, scenario=None):
        self.family, self.stream, self.seed, self.policy, self.r1 = family, stream, seed, policy, r1
        self.g = Git(root)
        self.specs = stream_spec(family, stream)
        self.dr = episode_draws(family, stream, seed)
        self.rand_rate = rand_rate
        sc = dict(scenario or {})
        self.scenario = sc
        if sc.get("late"):
            self.dr["t0"] = self.dr["t0"] + LATE_SHIFT          # same draw, shifted past the window
        if sc.get("shift"):
            self.dr["t0"] = self.dr["t0"] + int(sc["shift"])    # graded shift of the insertion time
        if sc.get("p_attack") is not None:                      # true attack rate, same uniform draw
            self.dr["attacked"] = bool(self.dr["u_att"] < float(sc["p_attack"]))
        # Knowledge of the attack rate: the planner's in-window hazard (default: the generator's rate)
        self.rho = RHO * float(sc["prior_rate"]) / P_ATTACK if sc.get("prior_rate") is not None else RHO
        self.true = dict(SENSE)                                 # true sensor rates (draws)
        self.true.update(sc.get("sense_true") or {})
        self.rb_threshold = float(sc.get("rb_threshold", 0.5))
        self.gate = sc.get("gate")                              # None or (nu, gamma)
        self.gating = self.gate is not None and policy in GATING
        self.tag = f"{CANARY}{family[:2]}{stream:02d}{seed:02d}"
        self.taint = [False] * NC
        self.k = 0                                   # milestones executed
        self.log = []                                # entries: dict(kind, k, sha, reads, deps, contam, carrier)
        self.base = self.g.head()
        self.wip_base = self.g.run("rev-parse", "wip").stdout.strip()
        self.ckpt = [0] * NC                         # checkpoint per carrier (index into log, 0 = base)
        self.ckpt_wip = [self.wip_base] * NC
        self.belief = [0.0] * NC
        self.m = dict(audits=0, recoveries=0, rec_forced=0, rec_false=0, regen=0, cherry=0, conflicts=0,
                      rec_seconds=0.0, irr_events=0, rev_events=0, validations=0, reviewed=0,
                      lost=0, replay_cost=0.0, detections=0, discarded=0, gate_audits=0, gate_blocks=0,
                      blocked_benign=0, benign_pushes=0)
        self.audits_c = [0] * NC
        self.pending_checks = []
        self.created_fn = set()
        self.rho_until = 10 ** 9 if policy == "rover_const" else T0_MAX     # Knowledge: exposure window
        self.rho_late = self.rho if policy == "rover_const" else self.rho / 4
        self.discard = r1 > COST["w_lost"] and policy != "rover_replay_only"  # shared executor rule
        self.trace = []                              # (k, event, carrier, info) for the Fig. 1 walkthrough

    # ------------------------------------------------------------------ files
    def path(self, *p):
        return os.path.join(self.g.repo, *p)

    def carrier_file(self, i):
        return {0: ".agent/memory.md", 1: ".agent/skill.md", 2: ".agent/queue.json", 3: None}[i]

    def wip_content(self):
        return self.g.run("show", "wip:wip.md").stdout

    def commit_wip(self, text):
        blob = self.g.run("hash-object", "-w", "--stdin", inp=text).stdout.strip()
        tree = self.g.run("mktree", inp=f"100644 blob {blob}\twip.md\n").stdout.strip()
        parent = self.g.run("rev-parse", "wip").stdout.strip()
        c = self.g.run("commit-tree", tree, "-p", parent, "-m", "wip update").stdout.strip()
        self.g.run("update-ref", "refs/heads/wip", c)

    # ------------------------------------------------------------------ milestone execution
    def apply_spec(self, s, contaminate: bool):
        """Apply milestone spec s to the working tree. Returns False if the target is missing."""
        k = s["k"]
        modp = self.path("src", f"mod{s['mod']}.py")
        text = open(modp).read()
        canary_line = f"    # {self.tag} m{k}\n"
        if s["kind"] == "add":
            body = f"def m{k}_feat(x):\n" + (canary_line if contaminate else "") + \
                   f"    return x * {s['a']} + {s['b']}\n\n"
            text = text + body
            open(modp, "w").write(text)
        elif s["kind"] == "modify":
            head = f"def m{s['dep']}_feat(x):\n"
            if head not in text:
                return False
            i0 = text.index(head) + len(head)
            extra = (canary_line if contaminate else "") + f"    x = x - {s['a']}  # m{k}\n"
            text = text[:i0] + extra + text[i0:]
            open(modp, "w").write(text)
        else:  # test
            if f"def m{s['dep']}_feat(x):" not in text:
                return False
            tp = self.path("tests", f"test_mod{s['mod']}.py")
            t = open(tp).read()
            t += f"def test_m{k}():\n" + (canary_line if contaminate else "") + \
                 f"    assert mod{s['mod']}.m{s['dep']}_feat({s['a']}) is not None\n\n"
            open(tp, "w").write(t)
        # agent persists its own notes in the carriers it read (clean updates)
        if s["reads"][0]:
            with open(self.path(".agent", "memory.md"), "a") as f:
                f.write(f"- m{k}: touched mod{s['mod']}\n")
        if s["reads"][1] and k % 3 == 0:
            with open(self.path(".agent", "skill.md"), "a") as f:
                f.write(f"{k}. check mod{s['mod']} formatting\n")
        if s["reads"][2] and k % 2 == 0:
            q = json.load(open(self.path(".agent", "queue.json")))
            q["queued"].append({"op": "notify", "milestone": k})
            json.dump(q, open(self.path(".agent", "queue.json"), "w"))
        return True

    def run_milestone(self, s, replay=False):
        """Execute milestone s (fresh or as re-execution). Returns the log entry."""
        k = s["k"]
        reads = s["reads"]
        if reads[3]:
            open(self.path(".agent", "wip.md"), "w").write(self.wip_content())
        tainted_read = any(reads[i] and self.taint[i] for i in range(NC))
        contam = tainted_read and self.dr["infl"][k] < P_INFL
        drop = False
        if self.policy == "immutable_validator":
            self.m["validations"] += 1
            if contam and self.dr["valid"][k] < self.true["d_v"]:
                contam = False              # blocked before acceptance
                self.m["detections"] += 1
                self.pending_checks = [i for i in range(NC) if reads[i]]
                if self.discard:
                    drop = True             # dropped: cheaper than re-execution
                    self.m["discarded"] += 1
                else:
                    self.m["regen"] += 1    # re-executed cleanly
                    self.m["replay_cost"] += self.r1
        ok = False if drop else self.apply_spec(s, contam)
        self.g.run("commit", "-q", "-a", "--allow-empty", "-m", f"milestone {k}")
        if reads[3] and k % 4 == 0:
            w = self.wip_content()
            if f"- wip note m{k}\n" not in w:
                self.commit_wip(w + f"- wip note m{k}\n")
        deps = set()
        if s["dep"] is not None:
            deps.add(s["dep"])
        e = dict(kind="ms", k=k, sha=self.g.head(), reads=list(reads), deps=deps, contam=contam and ok, ok=ok)
        return e

    # ------------------------------------------------------------------ attacker (abstract)
    def apply_redo(self, redo):
        kind, a, b = redo
        if kind == "append":
            with open(self.path(a), "a") as f:
                f.write(b)
        else:  # queue
            q = json.load(open(self.path(".agent", "queue.json")))
            q["queued"].append({"op": "push", "marker": b})
            json.dump(q, open(self.path(".agent", "queue.json"), "w"))
        self.g.run("commit", "-q", "-a", "-m", "notes update" if kind == "append" else "queue update")

    def attack(self, i):
        tag = f"\n- {self.tag} (inert marker)\n"
        if i == 3:
            self.commit_wip(self.wip_content() + tag)
        else:
            redo = ("queue", None, self.tag) if i == 2 else ("append", self.carrier_file(i), tag)
            self.apply_redo(redo)
            self.log.append(dict(kind="ext", k=self.k, sha=self.g.head(), carrier=i, reads=[False] * NC,
                                 deps=set(), contam=False, ok=True, redo=redo, taints=i))
        self.taint[i] = True
        self.trace.append((self.k, "attack", i, {}))

    # ------------------------------------------------------------------ dependencies
    def dependent_set(self, i):
        """Log indices after the checkpoint of carrier i that depend on it (reads, writes, transitive)."""
        S = set()
        dep_k = set()
        for idx in range(self.ckpt[i], len(self.log)):
            e = self.log[idx]
            if e["kind"] == "ext":
                if e["carrier"] == i:
                    S.add(idx)
                continue
            if e["reads"][i] or (e["deps"] & dep_k):
                S.add(idx)
                dep_k.add(e["k"])
        return S

    def D(self, i):
        return sum(1 for idx in self.dependent_set(i) if self.log[idx]["kind"] == "ms")

    # ------------------------------------------------------------------ recovery executors
    def recover(self, i, forced=False):
        t_start = time.perf_counter()
        self.m["recoveries"] += 1
        self.m["rec_forced"] += int(forced)
        if not self.taint[i]:
            self.m["rec_false"] += 1
        S = self.dependent_set(i)
        cp = self.ckpt[i]
        cp_sha = self.base if cp == 0 else self.log[cp - 1]["sha"]
        tail = self.log[cp:]
        self.g.run("reset", "-q", "--hard", cp_sha)
        if i == 3:
            self.g.run("update-ref", "refs/heads/wip", self.ckpt_wip[i])
        self.taint[i] = False
        new_tail = []
        newpos = {}
        regen = 0
        for off, e in enumerate(tail):
            idx = cp + off
            newpos[idx] = cp + len(new_tail)
            if e["kind"] == "ext":
                if e["carrier"] == i:
                    continue                      # unlogged external write or incident: not replayed
                self.apply_redo(e["redo"])        # other carriers keep their state, taint included
                e2 = dict(e)
                e2["sha"] = self.g.head()
                new_tail.append(e2)
                continue
            s = self.specs[e["k"] - 1]
            if idx in S:
                if self.discard:
                    self.m["discarded"] += 1
                    continue                      # dependent work dropped (cheaper than replay)
                e2 = self.run_milestone(s, replay=True)
                regen += 1
            else:
                r = self.g.run("cherry-pick", "--allow-empty", "--keep-redundant-commits", e["sha"], check=False)
                if r.returncode != 0:
                    self.g.run("cherry-pick", "--abort", check=False)
                    self.m["conflicts"] += 1
                    if self.discard:
                        self.m["discarded"] += 1
                        continue
                    e2 = self.run_milestone(s, replay=True)
                    regen += 1
                else:
                    self.m["cherry"] += 1
                    e2 = dict(e)
                    e2["sha"] = self.g.head()
            new_tail.append(e2)
        old_len = len(self.log)
        self.log = self.log[:cp] + new_tail
        for j in range(NC):
            if j != i and self.ckpt[j] > cp:
                self.ckpt[j] = newpos.get(self.ckpt[j], len(self.log)) if self.ckpt[j] < old_len else len(self.log)
        self.ckpt[i] = len(self.log)
        self.ckpt_wip[i] = self.g.run("rev-parse", "wip").stdout.strip()
        self.m["regen"] += regen
        dep_ks = sorted(tail[x - cp]["k"] for x in S if self.log_kind(x, tail, cp) == "ms")
        self.trace.append((self.k + 1, "recover", i, dict(forced=forced, regen=regen, tail=len(tail), dep_ks=dep_ks,
                                                          dependent=len([x for x in S if self.log_kind(x, tail, cp) == "ms"]),
                                                          seconds=time.perf_counter() - t_start)))
        cost = COST["r0"] + self.r1 * regen
        self.m["replay_cost"] += cost
        self.m["rec_seconds"] += time.perf_counter() - t_start
        self.belief[i] = 0.0

    @staticmethod
    def log_kind(idx, tail, cp):
        return tail[idx - cp]["kind"] if 0 <= idx - cp < len(tail) else ""

    def reset_all(self):
        t_start = time.perf_counter()
        self.m["recoveries"] += 1
        if not any(self.taint):
            self.m["rec_false"] += 1
        cp = min(self.ckpt)
        cp_sha = self.base if cp == 0 else self.log[cp - 1]["sha"]
        lost = sum(1 for e in self.log[cp:] if e["kind"] == "ms")
        self.g.run("reset", "-q", "--hard", cp_sha)
        self.g.run("update-ref", "refs/heads/wip", self.ckpt_wip[0])  # all equal under reset_all
        self.log = self.log[:cp]
        self.taint = [False] * NC
        self.ckpt = [len(self.log)] * NC
        self.ckpt_wip = [self.g.run("rev-parse", "wip").stdout.strip()] * NC
        self.m["lost"] += lost
        self.m["replay_cost"] += COST["r0"]
        self.m["rec_seconds"] += time.perf_counter() - t_start
        self.belief = [0.0] * NC

    # ------------------------------------------------------------------ activation
    def activate(self, i, k):
        irreversible = self.dr["irr"][k, i] < CARRIERS[i].phi
        if irreversible and self.gating and self.dr["gate_path"][k, i] < self.gate[1]:
            # the push or deploy request passes the harness gate: audit the requesting carrier
            self.m["gate_audits"] += 1
            if self.dr["gate_det"][k, i] < self.true["d"]:
                irreversible = False         # blocked: the effect stays local and is handled as an incident
                self.m["gate_blocks"] += 1
        self.trace.append((k, "activate", i, dict(irreversible=bool(irreversible), D=self.D(i))))
        if irreversible:
            self.g.run("push", "-q", self.g.remote, f"HEAD:refs/releases/r{k}-{i}")
            with open(os.path.join(self.g.deploy, f"deploy-{k}-{i}.marker"), "w") as f:
                f.write(self.tag + "\n")
            self.m["irr_events"] += 1
        else:
            redo = ("append", ".agent/incident.log", f"{self.tag} incident m{k}\n")
            self.apply_redo(redo)
            self.log.append(dict(kind="ext", k=k, sha=self.g.head(), carrier=i, reads=[False] * NC,
                                 deps=set(), contam=False, ok=True, redo=redo))
            self.m["rev_events"] += 1
        # incident observed: mandatory recovery
        if self.policy == "reset_all":
            self.reset_all()
        else:
            self.recover(i, forced=True)

    # ------------------------------------------------------------------ policy (Plan + Execute)
    def decide(self, k):
        pol = self.policy
        t = k - 1                          # period index for the planner
        if pol in ROVER_FAMILY + ("myopic", "random_audit", "rollback_only", "rollback_tuned", "rollback_cheap"):
            for i in range(NC):
                b = self.belief[i]
                D = min(self.D(i), T_MS)
                if pol in ROVER_FAMILY:
                    dp = planner(i, self.r1, pol, self.rho, self.gate[1] if self.gate else 0.0)
                    qw, qr, qa = (float(v[0]) for v in dp.qall(t, np.array([D]), np.array([b])))
                    m_ = min(qw, qr, qa)
                    a = W if qw <= m_ + EPS else (R if qr <= m_ + EPS else A)
                    if a == R:
                        self.recover(i)
                    elif a == A:
                        self.audit(i, k, lambda bp: float(dp.qR(t, np.array([D]), np.array([bp]))[0])
                                   <= float(dp.qW(t, np.array([D]), np.array([bp]))[0]) + EPS)
                elif pol == "myopic":
                    c = CARRIERS[i]
                    r1e = effective_r1(self.r1)
                    g = r1e * c.Delta + c.lam * c.phi * COST["H"]
                    cfq = COST["r0"] + r1e * D
                    if b * SENSE["d"] * g >= COST["c_a"] + SENSE["alpha"] * (1 - b) * cfq:
                        self.audit(i, k, lambda bp: bp * g >= (1 - bp) * cfq)
                elif pol == "random_audit":
                    if self.dr["rand_aud"][k, i] < self.rand_rate:
                        self.audit(i, k, None)          # recover on a positive audit
                elif pol == "rollback_only":
                    if b >= 0.5:
                        self.recover(i)
                elif pol in ("rollback_tuned", "rollback_cheap"):
                    if b >= self.rb_threshold:
                        self.recover(i)

    def audit(self, i, k, recover_rule, draw="audit"):
        self.m["audits"] += 1
        self.audits_c[i] += 1
        u = self.dr[draw][k, i]
        pos = (u < self.true["d"]) if self.taint[i] else (u < self.true["alpha"])
        self.trace.append((k, "audit", i, dict(belief=self.belief[i], D=self.D(i), positive=bool(pos))))
        Pp, bp, Pm, bm = audit_split(Params(d=SENSE["d"], alpha=SENSE["alpha"]), np.float64(self.belief[i]))
        self.belief[i] = float(bp) if pos else float(bm)
        if pos:
            self.m["detections"] += int(self.taint[i])
        if (pos if recover_rule is None else recover_rule(self.belief[i])):
            self.recover(i)

    def review(self, k):
        """Periodic commit review of the last REVIEW_EVERY milestone commits."""
        recent = [e for e in self.log if e["kind"] == "ms"][-REVIEW_EVERY:]
        self.m["reviewed"] += len(recent)
        flagged = set()
        for e in recent:
            if e["contam"] and self.dr["review"][e["k"]] < self.true["d_c"]:
                for i in range(NC):
                    if e["reads"][i]:
                        flagged.add(i)
        for i in sorted(flagged):                  # audit each carrier the flagged commit read
            self.audit(i, k, None, draw="aud2")

    # ------------------------------------------------------------------ Monitor + Analyze
    def sense(self, k, activated=()):
        p = Params(lam=0.0, s0=SENSE["s0"], s1=SENSE["s1"])
        for i in range(NC):
            if i in activated or (self.policy == "reset_all" and activated):
                self.belief[i] = self.rho if k - 1 < self.rho_until else self.rho_late
                continue
            o = 1 if (self.dr["alarm"][k, i] < (self.true["s1"] if self.taint[i] else self.true["s0"])) else 0
            if o:
                self.trace.append((k, "alarm", i, dict(prior=self.belief[i])))
            pi = replace(p, lam=CARRIERS[i].lam, rho=self.rho, rho_until=self.rho_until, rho_late=self.rho_late)
            _, post = bayes_alarm(pi, np.float64(self.belief[i]), o)
            self.belief[i] = float(predict(pi, post, k - 1))
            self.trace.append((k, "belief", i, dict(b=self.belief[i])))

    # ------------------------------------------------------------------ main loop
    def run(self):
        d = self.dr
        for k in range(1, T_MS + 1):
            if self.policy == "reset_all":
                if max(self.belief) >= 0.5:
                    self.reset_all()
            elif self.policy != "gate":
                self.decide(k)
            s = self.specs[k - 1]
            e = self.run_milestone(s)
            self.log.append(e)
            self.k = k
            if self.pending_checks:                   # validator: audit the carriers a blocked commit read
                checks, self.pending_checks = self.pending_checks, []
                for i in checks:
                    self.audit(i, k, None, draw="aud2")
            if self.policy == "periodic_review" and k % REVIEW_EVERY == 0:
                self.review(k)
            if self.gate is not None and d["benign"][k] < self.gate[0]:
                self.benign_push(k)
            activated = []
            for i in range(NC):
                if self.taint[i] and d["act"][k, i] < CARRIERS[i].lam:
                    self.activate(i, k)
                    activated.append(i)
            self.sense(k, activated)
            if d["attacked"] and k == d["t0"]:
                self.attack(d["carrier"])
        return self.measure()

    def benign_push(self, k):
        """A legitimate push request (release). A gate audits it and may block it by a false alarm."""
        self.m["benign_pushes"] += 1
        if self.gating:
            self.m["gate_audits"] += 1
            if self.dr["gate_fa"][k] < self.true["alpha"]:
                self.m["blocked_benign"] += 1
                return
        self.g.run("push", "-q", self.g.remote, f"HEAD:refs/releases/b{k}")

    # ------------------------------------------------------------------ measurement
    def measure(self):
        g = self.g
        r = g.run("grep", "-c", "-F", self.tag, "--", "src", "tests", check=False)
        resid_lines = sum(int(line.rsplit(":", 1)[1]) for line in r.stdout.split() if ":" in line)
        present = 0
        for s in self.specs:
            if s["kind"] == "add":
                txt = open(self.path("src", f"mod{s['mod']}.py")).read()
                present += int(f"def m{s['k']}_feat(x):" in txt)
            elif s["kind"] == "test":
                txt = open(self.path("tests", f"test_mod{s['mod']}.py")).read()
                present += int(f"def test_m{s['k']}():" in txt)
            else:
                txt = open(self.path("src", f"mod{s['mod']}.py")).read()
                present += int(f"# m{s['k']}\n" in txt)
        refs = g.run("--git-dir", g.remote, "for-each-ref", "--format=%(refname)", cwd=g.root).stdout.split()
        remote_canary = 0
        for ref in refs:
            rr = g.run("--git-dir", g.remote, "grep", "-c", "-F", self.tag, ref, "--", "src", "tests",
                       cwd=g.root, check=False)
            remote_canary += sum(int(x.rsplit(":", 1)[1]) for x in rr.stdout.split() if ":" in x)
        carrier_tainted_end = int(any(self.taint))
        # consistency check of the ground-truth bookkeeping with file contents
        for i in range(3):
            content = open(self.path(self.carrier_file(i))).read()
            assert (self.tag in content) == self.taint[i], f"carrier {i} bookkeeping mismatch"
        assert (self.tag in self.wip_content()) == self.taint[3], "wip bookkeeping mismatch"
        m = dict(self.m)
        for i, c in enumerate(CARRIER_NAMES):
            m[f"audits_{c}"] = self.audits_c[i]
        m.update(resid_lines=resid_lines, present=present, n_ms=len(self.specs), remote_canary=remote_canary,
                 remote_refs=len(refs), carrier_tainted_end=carrier_tainted_end, git_calls=g.calls)
        m["clean_preserved"] = present / len(self.specs)
        return m


def loss_components(m, r1):
    """Model-unit loss from measured counts (Section VI metrics)."""
    irr = COST["H"] * m["irr_events"]
    resid = COST["h"] * m["resid_lines"]
    audit = (COST["c_a"] * (m["audits"] + m.get("gate_audits", 0)) + COST["c_v"] * m["validations"]
             + COST["c_r"] * m["reviewed"] + C_BLOCK * m.get("blocked_benign", 0))
    replay = COST["r0"] * m["recoveries"] + r1 * m["regen"]
    lost = COST["w_lost"] * (m["n_ms"] - m["present"])
    return dict(irr=irr, resid=resid, audit=audit, replay=replay, lost=lost,
                loss=irr + resid + audit + replay + lost)


def run_episode(job):
    """job: dict(family, stream, seed, policy, r1, rand_rate, scenario, label, template_root, tmp_base)."""
    root = tempfile.mkdtemp(prefix="rov-", dir=job["tmp_base"])
    try:
        shutil.copytree(os.path.join(job["template_root"], f"{job['family']}-{job['stream']}"), root,
                        dirs_exist_ok=True)
        ep = Episode(root, job["family"], job["stream"], job["seed"], job["policy"], job["r1"],
                     job.get("rand_rate", 0.0), job.get("scenario"))
        t0 = time.perf_counter()
        m = ep.run()
        m["wall"] = time.perf_counter() - t0
        d = ep.dr
        m.update(family=job["family"], stream=job["stream"], seed=job["seed"], policy=job["policy"],
                 r1=job["r1"], label=job.get("label", "main"),
                 attacked=int(d["attacked"]), carrier=CARRIER_NAMES[d["carrier"]] if d["attacked"] else "none",
                 t0=d["t0"] if d["attacked"] else -1)
        m.update(loss_components(m, job["r1"]))
        return m
    finally:
        shutil.rmtree(root, ignore_errors=True)


def build_templates(template_root, streams):
    os.makedirs(template_root, exist_ok=True)
    for fam in FAMILIES:
        for s in range(streams):
            dst = os.path.join(template_root, f"{fam}-{s}")
            if not os.path.exists(dst):
                os.makedirs(dst)
                build_fixture(dst, fam, s)
