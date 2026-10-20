"""Rendering of the exploratory E3 additions (e3_extra.csv, analysed by analyze_e3.extra_analysis).

Writes Table V (Knowledge: timing, attack rate, sensors), Table VI (gate prices) and their macros.
Every difference is other policy minus ROVER (positive: ROVER has the lower loss).
"""
from __future__ import annotations

import json
import os

KCOND = (("shift0", "SZero", r"In the window (main run)"), ("shift2", "STwo", r"Shifted by 2 milestones"),
         ("shift4", "SFour", r"Shifted by 4 milestones"), ("shift6", "SSix", r"Shifted by 6 milestones"),
         ("shift8", "SEight", r"Shifted by 8 (late attacker)"),
         ("rate0.20_fix", "RTwentyFix", r"Attack rate 0.2, prior 0.6"),
         ("rate0.20_cal", "RTwentyCal", r"Attack rate 0.2, prior 0.2"),
         ("rate0.05_fix", "RFiveFix", r"Attack rate 0.05, prior 0.6"),
         ("rate0.05_cal", "RFiveCal", r"Attack rate 0.05, prior 0.05"),
         ("s1low", "SOneLow", r"$\sigma_1=0.25$ (model 0.4)"),
         ("dalpha", "DAl", r"$d,\alpha=0.7,0.1$ (model 0.9, 0.05)"))
KPOL = (("no_audit", "No", "No audit"), ("rover_noaudit", "Na", "ROVER, no audits"),
        ("rover_const", "Co", "ROVER, no window"), ("rollback_tuned", "Rb", "Rollback only"),
        ("myopic", "My", "Myopic"))
CNAME = {"queue": "queued call", "branch": "work branch", "memory": "memory note", "skill": "skill"}


def render_e3_extra(M, S, rows, f, pct, num, prel, GEN, RES):
    if not os.path.exists(os.path.join(RES, "e3_knowledge.csv")):
        print("E3 extra results missing, skipping")
        return
    kn = {r["condition"]: r for r in rows("e3_knowledge.csv")}
    # The main-run row repeats the confirmatory comparisons of Table IV on the same episodes. Report the
    # intervals of Table IV (e3_main.csv) instead of a second bootstrap draw with another seed.
    main = {r["policy"]: r for r in rows("e3_main.csv")}
    for pol, _, _ in KPOL:
        if kn["shift0"].get(f"{pol}_d") and pol in main:
            for k, mk in (("_d", "d_loss"), ("_lo", "d_lo"), ("_hi", "d_hi")):
                kn["shift0"][pol + k] = main[pol][mk]
    ext = rows("e3_extra.csv")
    M["EThreeExtraEpisodes"] = num(len([r for r in ext if r["repriced"] == "0"]))
    meta = json.load(open(os.path.join(RES, "e3_extra_meta.json")))
    M["EThreeThrCheap"] = f(meta["cheap_threshold"]["0.0"]["threshold"], 1)

    def cell(r, pol):
        if not r.get(f"{pol}_d"):
            return r"\multicolumn{2}{c}{--}"
        return f"{f(r[pol + '_d'])} & [{f(r[pol + '_lo'])}, {f(r[pol + '_hi'])}]"

    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for _, _, name in KPOL)
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{2.6pt}",
             r"\begin{tabular}{@{}l r r " + " ".join([r"r@{\,}l"] * len(KPOL)) + r"@{}}", r"\toprule",
             r"Condition & Att. & ROVER & " + head + r" \\", r"\midrule"]
    for cond, tag, lab in KCOND:
        r = kn[cond]
        lines.append(f"{lab} & {pct(r['attacked'], 0)}\\% & {f(r['rover'])} & "
                     + " & ".join(cell(r, pol) for pol, _, _ in KPOL) + r" \\")
        if cond in ("shift8", "rate0.05_cal"):
            lines.append(r"\midrule")
        M[f"EThreeK{tag}Rover"] = f(r["rover"])
        M[f"EThreeK{tag}RoverIrr"] = f(r["rover_irr100"], 1)
        M[f"EThreeK{tag}Att"] = pct(r["attacked"], 0)
        for pol, pt, _ in KPOL:
            if not r.get(f"{pol}_d"):
                continue
            M[f"EThreeK{tag}{pt}"] = f(r[f"{pol}_d"])
            M[f"EThreeK{tag}{pt}Lo"] = f(r[f"{pol}_lo"])
            M[f"EThreeK{tag}{pt}Hi"] = f(r[f"{pol}_hi"])
            M[f"EThreeK{tag}{pt}P"] = prel(r[f"{pol}_p"])
            M[f"EThreeK{tag}{pt}Wins"] = int(float(r[f"{pol}_rover_lower"]))
            M[f"EThreeK{tag}{pt}Loses"] = int(float(r[f"{pol}_other_lower"]))
            M[f"EThreeK{tag}{pt}Irr"] = f(r[f"{pol}_dirr100"], 1)
            M[f"EThreeK{tag}{pt}IrrLo"] = f(r[f"{pol}_dirr100_lo"], 1)
            M[f"EThreeK{tag}{pt}IrrHi"] = f(r[f"{pol}_dirr100_hi"], 1)
            M[f"EThreeK{tag}{pt}IrrP"] = prel(r[f"{pol}_pirr"])
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_e3b.tex"), "w").write("\n".join(lines) + "\n")

    # Table VI: gate prices
    gp = sorted(rows("e3_gate_price.csv"), key=lambda x: (float(x["nu"]), float(x["gamma"])))
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l r rr rr r@{}}", r"\toprule",
             r"& & \multicolumn{2}{c}{Gate only} & \multicolumn{2}{c}{ROVER and gate} & Break-even \\",
             r"$\nu$, $\gamma$ & Checks & $c_g{=}1$ & $c_g{=}0.2$ & $c_g{=}1$ & $c_g{=}0.2$ & $c_g$ \\", r"\midrule"]
    for g in gp:
        nu, ga = float(g["nu"]), float(g["gamma"])
        nus = "1/24" if nu < 0.1 else "1/4"
        lines.append(f"${nus}$, ${ga:g}$ & {f(g['checks_gate'], 1)} & {f(g['gate_1'])} & {f(g['gate_0.2'])} & "
                     f"{f(g['rover_gate_1'])} & {f(g['rover_gate_0.2'])} & "
                     f"{f(g['breakeven_gate_price']) if float(g['breakeven_gate_price']) > 0 else 'none'} \\\\")
        key = f"{'Rare' if nu < 0.1 else 'Freq'}{'Full' if ga == 1.0 else 'Half'}"
        for c in ("0.1", "0.2", "0.5", "1"):
            ct = {"0.1": "Tenth", "0.2": "Fifth", "0.5": "Half", "1": "One"}[c]
            for name, nt in (("gate", "G"), ("aware", "A"), ("rover_gate", "U")):
                M[f"EThreeGP{key}{nt}{ct}"] = f(g[f"{name}_{c}"])
                M[f"EThreeGP{key}{nt}{ct}D"] = f(g[f"{name}_{c}_d"])
                M[f"EThreeGP{key}{nt}{ct}Lo"] = f(g[f"{name}_{c}_lo"])
                M[f"EThreeGP{key}{nt}{ct}Hi"] = f(g[f"{name}_{c}_hi"])
            M[f"EThreeGP{key}AmG{ct}"] = f(g[f"aware_minus_gate_{c}"])
            M[f"EThreeGP{key}AmG{ct}Lo"] = f(g[f"aware_minus_gate_{c}_lo"])
            M[f"EThreeGP{key}AmG{ct}Hi"] = f(g[f"aware_minus_gate_{c}_hi"])
        M[f"EThreeGP{key}Breakeven"] = f(g["breakeven_gate_price"])
        M[f"EThreeGP{key}Checks"] = f(g["checks_gate"], 1)
        M[f"EThreeGP{key}CanaryGate"] = f(g["canary_remote_gate"], 2)
        M[f"EThreeGP{key}CanaryAware"] = f(g["canary_remote_aware"], 2)
        M[f"EThreeGP{key}CanaryRG"] = f(g["canary_remote_rover_gate"], 2)
        M["EThreeCanaryRoverMain"] = f(g["canary_remote_rover_main"], 2)
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_e3c.tex"), "w").write("\n".join(lines) + "\n")

    # summary items
    sm = {r["item"]: r for r in rows("e3_extra_summary.csv")}
    for r1, tag in (("0", "Zero"), ("0.1", "Tenth")):
        r = sm[f"rollback_cheap_r{r1}"]
        M[f"EThreeRbCheap{tag}"] = f(r["value"])
        M[f"EThreeRbCheap{tag}D"] = f(r["d"])
        M[f"EThreeRbCheap{tag}Lo"] = f(r["lo"])
        M[f"EThreeRbCheap{tag}Hi"] = f(r["hi"])
        M[f"EThreeRbCheap{tag}P"] = prel(r["p"])
    for pol, tag in (("periodic_review", "Pe"), ("immutable_validator", "Va")):
        r = sm[f"{pol}_free_monitoring"]
        M[f"EThreeFree{tag}"] = f(r["value"])
        M[f"EThreeFree{tag}D"] = f(r["d"])
        M[f"EThreeFree{tag}Lo"] = f(r["lo"])
        M[f"EThreeFree{tag}Hi"] = f(r["hi"])
    for h, tag in (("primary_H1", "One"), ("primary_H2", "Two")):
        M[f"EThreePrimH{tag}P"] = prel(sm[h]["p"])
        M[f"EThreePrimH{tag}Atw"] = f(sm[h]["ref"], 2)
    for r1, tag in (("0", "Zero"), ("0.1", "Tenth"), ("1", "One"), ("10", "Ten")):
        r = sm[f"H2_irr_r{r1}"]
        M[f"EThreeHTwoIrr{tag}"] = f(r["d"], 1)
        M[f"EThreeHTwoIrr{tag}Lo"] = f(r["lo"], 1)
        M[f"EThreeHTwoIrr{tag}Hi"] = f(r["hi"], 1)
        M[f"EThreeHTwoIrr{tag}P"] = prel(r["p"])
    for k, tag in (("rec_false", "Fr"), ("recoveries", "Rec"), ("regen", "Regen")):
        r = sm[f"rollback_{k}"]
        nd = 3 if k == "rec_false" else 2
        M[f"EThreeRb{tag}"] = f(r["value"], nd)
        M[f"EThreeRb{tag}Rover"] = f(r["ref"], nd)
        M[f"EThreeRb{tag}D"] = f(r["d"], nd)
        M[f"EThreeRb{tag}Lo"] = f(r["lo"], nd)
        M[f"EThreeRb{tag}Hi"] = f(r["hi"], nd)
        M[f"EThreeRb{tag}P"] = prel(r["p"])
    for pol, tag in (("rover", "Rover"), ("rover_const", "Co"), ("rover_noaudit", "Na"), ("rollback_tuned", "Rb"),
                     ("no_audit", "No"), ("myopic", "My")):
        r = sm[f"joint_worst_{pol}"]
        M[f"EThreeJW{tag}"] = f(r["value"], 1)
        c, lab = r["ref"].split("/")
        M[f"EThreeJW{tag}C"] = CNAME[c] + (", late" if lab == "late" else "")
    rec, regen = sm["rollback_recoveries"], sm["rollback_regen"]
    M["EThreeRbRegenPer"] = f(float(regen["value"]) / float(rec["value"]), 1)
    M["EThreeRoverRegenPer"] = f(float(regen["ref"]) / float(rec["ref"]), 1)
    aw = [float(g["aware_1"]) - float(g["rover_gate_1"]) for g in gp]
    M["EThreeGPAwareMin"] = f(min(aw), 2)
    M["EThreeGPAwareMax"] = f(max(aw), 2)
    alloc = {float(r["r1"]): r for r in rows("e3_alloc.csv")}
    for r1, tag in ((0.0, "Zero"), (10.0, "Ten")):
        M[f"EThreeAllocTot{tag}"] = f(alloc[r1]["total"], 2)
        for c in ("memory", "skill", "queue", "branch"):
            M[f"EThreeAlloc{c.capitalize()}{tag}"] = f(alloc[r1][f"audits_{c}"], 2)
    adapt = {r["policy"]: r for r in rows("e3_adaptive.csv")}
    for pol, tag in (("rollback_tuned", "Rb"), ("no_audit", "No"), ("rover_const", "Co")):
        M[f"EThreeAd{tag}"] = f(adapt[pol]["payoff"], 1)
        M[f"EThreeAd{tag}C"] = CNAME[adapt[pol]["best_carrier"]]
    M["EThreeAdRoverCN"] = CNAME[adapt["rover"]["best_carrier"]]
    S["E3_extra"] = dict(knowledge={c: {k: float(kn[c][k]) for k in ("rover", "attacked")} for c in kn},
                         executed=len([r for r in ext if r["repriced"] == "0"]))
