#!/usr/bin/env python3
"""make_fig1.py -- Fig. 1 overview and walkthrough of ROVER on one executed testbed episode.

Deterministic. Selects ONE episode of the executed git testbed (main setting r1 = 1) in
which the skill carrier is tainted, ROVER audits and recovers before any activation, and
rollback without audits ends in an irreversible deployment. It re-runs that episode with
ROVER and with event tracing (same seeds, same git fixture) to read the step-by-step
numbers, and reads the baseline numbers on the same episode from results/e3_heldout.csv
and the pooled numbers from results/e3_main.csv.

Writes fig1_overview.pdf, fig1_preview.png (200 dpi) next to this file and fig1_macros.tex into the
generated/ directory of render.py (render.gen_dir). No number is typed by hand.

Run from anywhere:  python3 artifact/figures/fig1/make_fig1.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ART, "results")
ICONS = os.path.join(HERE, "icons")
sys.path.insert(0, os.path.join(ART, "src"))
from render import gen_dir  # noqa: E402  (same output directory as render.py)
GEN = gen_dir(ART)

R1 = 1.0
CARRIER = "skill"

# palette
INK, NAVY, BRICK, OCHRE = "#22262E", "#1F3A5F", "#A23B2A", "#C8901F"
TEAL, PLUM, STONE = "#23726B", "#5E3A6E", "#8A8F98"
PANEL, BORDER, WHITE = "#F4EFE6", "#E6DFD2", "#FFFFFF"

NAMES = {"no_audit": "no audit", "random_audit": "random audit", "periodic_review": "periodic review",
         "rollback_tuned": "rollback only", "reset_all": "reset all", "immutable_validator": "immutable validator",
         "myopic": "myopic rule"}
BASELINES = ("no_audit", "random_audit", "periodic_review", "rollback_tuned", "reset_all",
             "immutable_validator", "myopic")
SELECTION = ("first held-out episode, in a fixed order, in which the skill is tagged, ROVER audits and "
             "recovers before any activation with no false recovery and no residual harm, and both rollback "
             "only and no audit suffer an irreversible activation")


# ------------------------------------------------------------------ data
def load_rows(name):
    return list(csv.DictReader(open(os.path.join(RES, name))))


COUNTS = {}


def pick_episode(eps):
    """First episode (in a fixed order) that shows the mechanism of the paper (rule: SELECTION)."""
    idx = {}
    for r in eps:
        if float(r["r1"]) == R1 and r.get("label", "main") == "main":
            idx[(r["family"], int(r["stream"]), int(r["seed"]), r["policy"])] = r
    keys = sorted({(f, s, sd) for (f, s, sd, p) in idx})

    def shows(rv, rb, na):
        return (float(rv["irr_events"]) == 0 and float(rv["audits"]) >= 1 and float(rv["recoveries"]) >= 1
                and float(rv["rec_forced"]) == 0 and float(rv["rec_false"]) == 0
                and float(rb["irr_events"]) >= 1 and float(na["irr_events"]) >= 1 and float(rv["resid_lines"]) == 0)
    COUNTS.update(cells=0, match=0, match_any=0, reverse=0)
    for (f, s, sd) in keys:
        rv, rb, na = (idx.get((f, s, sd, p)) for p in ("rover", "rollback_tuned", "no_audit"))
        if not (rv and rb and na):
            continue
        COUNTS["cells"] += 1
        m = shows(rv, rb, na)
        COUNTS["match_any"] += int(m)
        COUNTS["match"] += int(m and rv["carrier"] == CARRIER)
        COUNTS["reverse"] += int(float(rv["irr_events"]) >= 1 and float(rb["irr_events"]) == 0
                                 and float(na["irr_events"]) == 0)
    for (f, s, sd) in keys:
        rv = idx.get((f, s, sd, "rover"))
        rb = idx.get((f, s, sd, "rollback_tuned"))
        na = idx.get((f, s, sd, "no_audit"))
        if not (rv and rb and na):
            continue
        if rv["carrier"] != CARRIER:
            continue
        if (float(rv["irr_events"]) == 0 and float(rv["audits"]) >= 1 and float(rv["recoveries"]) >= 1
                and float(rv["rec_forced"]) == 0 and float(rv["rec_false"]) == 0
                and float(rb["irr_events"]) >= 1 and float(na["irr_events"]) >= 1
                and float(rv["resid_lines"]) == 0):
            n_match = sum(1 for (f2, s2, sd2) in keys
                          if idx.get((f2, s2, sd2, "rover")) and idx[(f2, s2, sd2, "rover")]["carrier"] == CARRIER)
            return (f, s, sd), {p: idx[(f, s, sd, p)] for p in list(BASELINES) + ["rover"] if (f, s, sd, p) in idx}
    raise SystemExit("no episode matches the walkthrough criteria")


def trace_rover(key, threshold):
    import testbed as TB
    f, s, sd = key
    base = tempfile.mkdtemp(prefix="rov-fig1-", dir=os.environ.get("ROV_TMP") or None)
    try:
        tpl = os.path.join(base, "tpl")
        os.makedirs(os.path.join(tpl, f"{f}-{s}"))
        TB.build_fixture(os.path.join(tpl, f"{f}-{s}"), f, s)
        root = os.path.join(base, "ep")
        shutil.copytree(os.path.join(tpl, f"{f}-{s}"), root)
        ep = TB.Episode(root, f, s, sd, "rover", R1, scenario=dict(rb_threshold=threshold))
        m = ep.run()
        return ep, m
    finally:
        shutil.rmtree(base, ignore_errors=True)


def load_data():
    eps = load_rows("e3_heldout.csv")
    key, rowsd = pick_episode(eps)
    th = json.load(open(os.path.join(RES, "e3_thresholds.json")))["1.0"]["threshold"]
    ep, m = trace_rover(key, th)
    rv = rowsd["rover"]
    for k in ("audits", "recoveries", "regen", "irr_events", "resid_lines"):  # replay is deterministic
        assert abs(float(rv[k]) - float(m[k])) < 1e-9, (k, rv[k], m[k])     # replay is deterministic
    tr = ep.trace
    attack = [t for t in tr if t[1] == "attack"][0]
    audit = [t for t in tr if t[1] == "audit" and t[3]["positive"]][0]
    rec = [t for t in tr if t[1] == "recover" and t[2] == attack[2]][0]
    after = [k for k in rec[3]["dep_ks"] if k > attack[0]]
    alarms = [t for t in tr if t[1] == "alarm" and t[2] == attack[2] and attack[0] < t[0] <= audit[0]]
    bhist = [(t[0], t[3]["b"]) for t in tr if t[1] == "belief" and t[2] == attack[2] and t[0] < audit[0]]
    main = {r["policy"]: r for r in load_rows("e3_main.csv")}
    best = min(BASELINES, key=lambda p: float(main[p]["loss"]))
    return dict(key=key, rows=rowsd, ep=ep, attack=attack, audit=audit, rec=rec, alarms=alarms, main=main, best=best,
                bhist=bhist, after=len(after))


def f2(x):
    return f"{float(x):.2f}"


def f1(x):
    return f"{float(x):.1f}"


def i0(x):
    return f"{int(round(float(x)))}"


# ------------------------------------------------------------------ drawing helpers
def setup_fonts():
    # STIX (Times-like, matching the body text) ships with matplotlib, so no external font files are needed
    plt.rcParams.update({"font.family": "STIXGeneral", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "image.composite_image": False, "mathtext.fontset": "stix",
                         "text.color": INK, "axes.unicode_minus": False})


_ICON = {}


def icon(ax, name, cx, cy, size, z=5, alpha=1.0):
    if name not in _ICON:
        _ICON[name] = mpimg.imread(os.path.join(ICONS, name + ".png"))
    h = size / 2
    ax.imshow(_ICON[name], extent=(cx - h, cx + h, cy - h, cy + h), zorder=z, alpha=alpha,
              interpolation="none")


def panel(ax, x0, y0, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.09",
                                fc=PANEL, ec=BORDER, lw=1.4, zorder=0))
    tw = min(0.075 * len(title) + 0.2, w - 0.16)
    ax.add_patch(FancyBboxPatch((x0 + 0.08, y0 + h - 0.13), tw, 0.24,
                                boxstyle="round,pad=0,rounding_size=0.07", fc=color, ec="none", zorder=1))
    ax.text(x0 + 0.08 + tw / 2, y0 + h - 0.01, title, ha="center", va="center", fontsize=8.5,
            weight="bold", color=WHITE, zorder=2)


def step(ax, x, y, k):
    ax.add_patch(Circle((x, y), 0.085, fc=OCHRE, ec=INK, lw=0.8, zorder=8))
    ax.text(x, y - 0.005, str(k), ha="center", va="center", fontsize=7.5, weight="bold", color=INK, zorder=9)


def bubble(ax, x, y, w, h, text, tail=None, fc=WHITE, ec=INK, fs=7.2, color=INK, weight="normal",
           ha="center", lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                fc=fc, ec=ec, lw=lw, zorder=6))
    if tail is not None:
        tx, ty = tail
        if tx < x:
            by = min(max(ty + 0.05, y + 0.1), y + h - 0.1)
            base = [(x, by - 0.05), (x, by + 0.05)]
            cover = ([x, x], [by - 0.045, by + 0.045])
        elif tx > x + w:
            by = min(max(ty + 0.05, y + 0.1), y + h - 0.1)
            base = [(x + w, by - 0.05), (x + w, by + 0.05)]
            cover = ([x + w, x + w], [by - 0.045, by + 0.045])
        else:
            bx = min(max(tx, x + 0.12), x + w - 0.12)
            by = y if ty < y else y + h
            base = [(bx - 0.05, by), (bx + 0.05, by)]
            cover = ([bx - 0.045, bx + 0.045], [by, by])
        ax.add_patch(plt.Polygon(base + [tail], closed=True, fc=fc, ec=ec, lw=lw, zorder=5.9))
        ax.plot(*cover, color=fc, lw=1.6, zorder=6.1, solid_capstyle="butt")
    tx = x + w / 2 if ha == "center" else x + 0.06
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=color, weight=weight,
            zorder=7, linespacing=1.1)


def arrow(ax, p, q, color=INK, lw=1.4, ls="-", z=3, rad=0.0, ms=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 linestyle=ls, zorder=z, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


def commit_box(ax, cx, cy, label, ec, fc=WHITE, w=0.2, h=0.17, fs=6.5, lw=1.0, color=INK):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=fc, ec=ec, lw=lw, zorder=4))
    ax.text(cx, cy - 0.003, label, ha="center", va="center", fontsize=fs, color=color, zorder=5)


def cross(ax, x, y, s=0.06, color=BRICK):
    ax.plot([x - s, x + s], [y - s, y + s], color=color, lw=2.2, zorder=7, solid_capstyle="round")
    ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=2.2, zorder=7, solid_capstyle="round")


# ------------------------------------------------------------------ figure
def draw(d):
    W, H = 7.0, 2.7
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    PH = 2.5
    rv, rb, rs = d["rows"]["rover"], d["rows"]["rollback_tuned"], d["rows"].get("reset_all")
    pr = d["rows"].get("periodic_review")
    t0 = d["attack"][0]
    ka = d["audit"][0]
    dep = d["rec"][3]["dependent"]
    regen = d["rec"][3]["regen"]

    # ================= panel A: the threat (steps 1-2)
    panel(ax, 0.04, 0.06, 2.16, PH, "A  the threat", BRICK)
    icon(ax, "supervillain", 0.32, 2.03, 0.42)
    bubble(ax, 0.58, 1.93, 0.95, 0.38, "attacker writes a\ntag into the skill", tail=(0.48, 2.08), fs=7)
    step(ax, 0.58, 2.33, 1)
    icon(ax, "toolbox", 0.42, 1.38, 0.34)
    icon(ax, "lady_beetle", 0.55, 1.28, 0.14, z=7)
    ax.text(0.42, 1.13, f"skill, tagged at step {t0}", ha="left", va="center", fontsize=7)
    arrow(ax, (0.36, 1.82), (0.4, 1.58), color=BRICK, lw=1.3)
    icon(ax, "robot", 1.12, 1.42, 0.34)
    bubble(ax, 1.36, 1.38, 0.78, 0.4, f"agent: {d['after']} later\ncommits read it", tail=(1.28, 1.5), fs=7)
    step(ax, 1.5, 1.86, 2)
    # commit chain: dependent commits since the last checkpoint, tagged ones (after the tag) in brick
    ks = d["rec"][3]["dep_ks"]
    show = ks if len(ks) <= 5 else ks[:4] + [None] + ks[-1:]
    labs = ["..." if k is None else f"m{k}" for k in show]
    xs = [0.32 + 0.255 * j for j in range(len(labs))]
    for j, (x, lab, k) in enumerate(zip(xs, labs, show)):
        if k is None:
            ax.text(x, 0.78, "...", ha="center", va="center", fontsize=8, color=INK)
        else:
            after = k > t0
            commit_box(ax, x, 0.78, lab, ec=BRICK if after else STONE, lw=1.3 if after else 0.9,
                       color=INK if after else STONE)
        if j and show[j - 1] is not None and k is not None:
            arrow(ax, (xs[j - 1] + 0.1, 0.78), (x - 0.1, 0.78), color=INK, lw=0.9, ms=7)
    first_after = next((j for j, k in enumerate(show) if k is not None and k > t0), 0)
    arrow(ax, (0.42, 1.04), (xs[first_after], 0.9), color=BRICK, lw=1.1, ls=(0, (2, 1.5)))
    icon(ax, "package", 1.94, 0.78, 0.3)
    arrow(ax, (xs[-1] + 0.1, 0.78), (1.78, 0.78), color=BRICK, lw=1.1, ls=(0, (2, 1.5)))
    ax.text(1.12, 0.35, "deploy path armed: once it ships,\nno rollback can undo it", ha="center",
            va="center", fontsize=7, color=BRICK, weight="bold", linespacing=1.05)

    # ================= panel B: baselines (step 3)
    arrow(ax, (2.22, 1.3), (2.32, 1.3), color=INK, lw=1.8, ms=13)
    panel(ax, 2.34, 0.06, 1.5, PH, "B  baselines", STONE)
    rowsB = [("warning", "rollback only", "waits for an incident",
              f"deploy shipped: harm {i0(rb['irr'])}", BRICK)]
    if pr is not None:
        rowsB.append(("magnifying_glass_tilted_left", "periodic review", "reads every commit",
                      f"audit bill {i0(pr['audit'])}, harm {i0(pr['irr'])}", BRICK if float(pr['irr']) > 0 else NAVY))
    if rs is not None:
        rowsB.append(("broom", "reset all", "restores everything",
                      f"clean work kept {float(rs['clean_preserved']) * 100:.0f}%", BRICK))
    for i, (ic, name, quip, numtxt, col) in enumerate(rowsB):
        y = 1.95 - 0.66 * i
        icon(ax, ic, 2.55, y, 0.3)
        ax.text(2.74, y + 0.17, name, ha="left", va="center", fontsize=7.3, weight="bold")
        ax.text(2.74, y + 0.0, quip, ha="left", va="center", fontsize=7)
        ax.text(2.74, y - 0.16, numtxt, ha="left", va="center", fontsize=7, color=col, weight="bold")
    step(ax, 3.72, 2.37, 3)

    # ================= panel C: ROVER loop (steps 4-6)
    arrow(ax, (3.86, 1.3), (3.96, 1.3), color=INK, lw=1.8, ms=13)
    panel(ax, 3.98, 0.06, 2.06, PH, "C  ROVER loop", NAVY)
    icon(ax, "detective", 4.24, 2.03, 0.42)
    b_audit = d["audit"][3]["belief"]
    na = len(d["alarms"])
    bubble(ax, 4.5, 2.0, 1.42, 0.36, f"{na} alarm{'s' if na != 1 else ''}, belief {b_audit:.2f},\ndeploy path ahead: audit",
           tail=(4.42, 2.1), fs=7)
    step(ax, 4.52, 2.36, 4)
    # belief of the tagged carrier per step until the audit, alarms marked
    bx0, bx1, by0, by1 = 4.2, 5.75, 1.52, 1.82
    ax.plot([bx0, bx1], [by0, by0], color=STONE, lw=0.8, zorder=3)
    ks = [k for k, _ in d["bhist"]]
    kmax = max(ka, max(ks) if ks else ka)
    sx = lambda k: bx0 + (bx1 - bx0) * (k - 1) / max(kmax - 1, 1)
    sy = lambda b: by0 + (by1 - by0) * min(b / 0.5, 1.0)
    ax.plot([sx(k) for k, _ in d["bhist"]], [sy(b) for _, b in d["bhist"]], color=NAVY, lw=1.3, zorder=4)
    for (k, ev, i, info) in d["alarms"]:
        ax.plot([sx(k)], [by0], marker="^", ms=3.2, color=OCHRE, zorder=5)
    if d["bhist"]:
        kl, bl = d["bhist"][-1]
        ax.plot([sx(kl), sx(ka)], [sy(bl), sy(d["audit"][3]["belief"])], color=NAVY, lw=1.0, ls=":", zorder=4)
    ax.plot([sx(ka)], [sy(d["audit"][3]["belief"])], marker="o", ms=3.5, color=TEAL, zorder=5)
    ax.text(bx1 + 0.03, by0, f"step {kmax}", ha="left", va="center", fontsize=6.3, color=INK)
    ax.text(bx0, by0 - 0.08, "belief in the skill, alarms as triangles", ha="left", va="top", fontsize=6.3,
            color=INK)
    icon(ax, "balance_scale", 4.3, 1.12, 0.3)
    ax.text(4.5, 1.2, f"replay {d['audit'][3]['D']} commits", ha="left", va="center", fontsize=7)
    ax.text(4.5, 1.03, "vs. the deploy risk ahead", ha="left", va="center", fontsize=7)
    step(ax, 5.86, 1.12, 5)
    icon(ax, "scissors", 4.3, 0.68, 0.3)
    ax.text(4.5, 0.77, f"git reset, replay {regen} commits", ha="left", va="center", fontsize=7)
    ax.text(4.5, 0.59, f"at step {ka}, before any deploy", ha="left", va="center", fontsize=7,
            color=TEAL, weight="bold")
    step(ax, 5.86, 0.68, 6)
    ax.text(5.01, 0.26, "audit where it cannot be undone,\nroll back where it can", ha="center",
            va="center", fontsize=7, color=NAVY, weight="bold", linespacing=1.05)

    # ================= panel D: result (step 7)
    arrow(ax, (6.06, 1.3), (6.14, 1.3), color=INK, lw=1.8, ms=13)
    panel(ax, 6.16, 0.06, 0.8, PH, "D  result", TEAL)
    icon(ax, "trophy", 6.47, 2.06, 0.36)
    step(ax, 6.86, 2.22, 7)
    cx = 6.56
    ax.text(cx, 1.68, "this episode:\nirreversible harm", ha="center", va="center", fontsize=7, weight="bold",
            linespacing=1.0)
    ax.text(cx, 1.42, i0(rv["irr"]), ha="center", va="center", fontsize=11, weight="bold", color=TEAL)
    ax.text(cx, 1.22, f"vs {i0(rb['irr'])}", ha="center", va="center", fontsize=8, weight="bold")
    ax.text(cx, 1.08, "rollback only", ha="center", va="center", fontsize=6.5)
    ax.plot([6.28, 6.84], [0.96, 0.96], color=BORDER, lw=1.0, zorder=1)
    main, best = d["main"], d["best"]
    ax.text(cx, 0.84, "mean loss,\nall streams", ha="center", va="center", fontsize=7, linespacing=1.0)
    ax.text(cx, 0.6, f2(main["rover"]["loss"]), ha="center", va="center", fontsize=11, weight="bold", color=TEAL)
    ax.text(cx, 0.4, f"vs {f2(main[best]['loss'])}", ha="center", va="center", fontsize=8, weight="bold")
    ax.text(cx, 0.25, NAMES[best], ha="center", va="center", fontsize=6.5)

    fig.savefig(os.path.join(HERE, "fig1_overview.pdf"), dpi=600, metadata={"CreationDate": None})
    fig.savefig(os.path.join(HERE, "fig1_preview.png"), dpi=200)
    plt.close(fig)


def write_macros(d):
    rv, rb, na = d["rows"]["rover"], d["rows"]["rollback_tuned"], d["rows"]["no_audit"]
    f, s, sd = d["key"]
    m = {
        "FigEpisode": f"{f}-{s}, seed {sd}",
        "FigTzero": str(d["attack"][0]),
        "FigAuditStep": str(d["audit"][0]),
        "FigBelief": f"{d['audit'][3]['belief']:.2f}",
        "FigAlarms": str(len(d["alarms"])),
        "FigDependent": str(d["rec"][3]["dependent"]),
        "FigAfterTag": str(d["after"]),
        "FigAuditD": str(d["audit"][3]["D"]),
        "FigRegen": str(d["rec"][3]["regen"]),
        "FigRoverIrr": i0(rv["irr"]),
        "FigBaseIrr": i0(rb["irr"]),
        "FigNoAuditIrr": i0(na["irr"]),
        "FigRoverLoss": f2(rv["loss"]),
        "FigBaseLoss": f2(rb["loss"]),
        "FigPooledRover": f2(d["main"]["rover"]["loss"]),
        "FigPooledBest": f2(d["main"][d["best"]]["loss"]),
        "FigPooledBestName": NAMES[d["best"]],
        "FigCells": str(COUNTS["cells"]),
        "FigMatches": str(COUNTS["match"]),
        "FigMatchesAny": str(COUNTS["match_any"]),
        "FigReverse": str(COUNTS["reverse"]),
        "FigBeforeTag": str(sum(1 for k in d["rec"][3]["dep_ks"] if k <= d["attack"][0])),
    }
    if "reset_all" in d["rows"]:
        m["FigResetClean"] = f"{float(d['rows']['reset_all']['clean_preserved']) * 100:.0f}"
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, "fig1_macros.tex"), "w") as fh:
        fh.write("% generated by figures/fig1/make_fig1.py, do not edit\n")
        for k, v in sorted(m.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    setup_fonts()
    d = load_data()
    draw(d)
    write_macros(d)
    print("fig1: episode", d["key"], "audit at", d["audit"][0], "regen", d["rec"][3]["regen"])
