"""Rendering of the exploratory analyses (results/exploratory/, written by the post_*.py scripts).

All of them are exploratory and post hoc: they were added after the confirmatory run and are reported
as such. Writes macros (prefix Post) and two supplement tables
(supplement/generated/tab_post_onestep.tex, tab_post_pub.tex).
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

PUB_POL = (("rover", "ROVER"), ("rover_noaudit", "ROVER, no audits"), ("rollback_tuned", "Rollback only"),
           ("no_audit", "No audit"), ("random_audit", "Random audit"), ("myopic", "Myopic rule"),
           ("periodic_review", "Periodic review"), ("immutable_validator", "Immutable validator"),
           ("reset_all", "Reset all"))


def render_post(M, S, f, num, prel, SUPGEN, RES):
    R = os.path.join(RES, "exploratory")
    if not os.path.isdir(R):
        print("exploratory results missing, skipping")
        return
    load = lambda p: json.load(open(os.path.join(R, p)))
    rows = lambda p: list(csv.DictReader(open(os.path.join(R, p))))
    out = {}

    # ------------------------------------------------------------------ one-step controllers (E1)
    os_rows = rows("onestep/onestep.csv")
    g = lambda r, k: float(r[f"gap_{k}"])
    e1a = [r for r in os_rows if r["grid"] == "E1a"]
    worst = max(e1a, key=lambda r: g(r, "myopic"))
    M["PostOsMyGap"] = f(g(worst, "myopic"), 1)
    M["PostOsTermGap"] = f(g(worst, "onestep_term_wra"), 1)
    M["PostOsRollGap"] = f(g(worst, "rollout_na_wra"), 1)
    M["PostOsCells"] = len(os_rows)
    M["PostOsRollBeats"] = sum(1 for r in os_rows if g(r, "rollout_na_wra") < g(r, "myopic") - 1e-9)
    M["PostOsBlindHelps"] = sum(1 for r in os_rows if g(r, "rollout_na_wra") < g(r, "rollout_na_wa") - 1e-9)
    M["PostOsBlindHurts"] = sum(1 for r in os_rows if g(r, "rollout_na_wra") > g(r, "rollout_na_wa") + 1e-9)
    M["PostOsRollHighMax"] = f(max(g(r, "rollout_na_wra") for r in e1a if float(r["r1"]) >= 1), 1)
    M["PostOsMcZ"] = f(load("onestep/onestep_summary.json")["mc_max_abs_z"], 2)
    lines = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
             r"$\phi$ & $r_1$ & DP loss & Myopic & Myopic, DP rec. & One-step & Rollout W/A & Rollout W/R/A \\",
             r"\midrule"]
    for r in e1a:
        lines.append(f"{float(r['phi']):g} & {float(r['r1']):g} & {f(r['dp'])} & " + " & ".join(
            f(g(r, k), 1) for k in ("myopic", "myopic_dprec", "onestep_term_wra", "rollout_na_wa", "rollout_na_wra")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(SUPGEN, "tab_post_onestep.tex"), "w").write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ publication harm (E3)
    pub = load("publication/publication_analysis.json")
    assert pub["decisions_match_committed_main"]["ok"]
    rare, freq = pub["by_nu"]["nu0.042"], pub["by_nu"]["nu0.250"]
    M["PostPubEpisodes"] = num(pub["episodes"])
    for tag, res in (("Rare", rare), ("Freq", freq)):
        lk = res["leak"]
        M[f"PostPub{tag}RoLines"] = f(lk["rover"]["rel_lines_new"])
        M[f"PostPub{tag}RoLeak"] = f(100 * lk["rover"]["share_episodes_leaking"], 0)
        M[f"PostPub{tag}NoLines"] = f(lk["no_audit"]["rel_lines_new"])
        M[f"PostPub{tag}NoLeak"] = f(100 * lk["no_audit"]["share_episodes_leaking"], 0)
        M[f"PostPub{tag}VaLines"] = f(lk["immutable_validator"]["rel_lines_new"])
        M[f"PostPub{tag}VaLeak"] = f(100 * lk["immutable_validator"]["share_episodes_leaking"], 0)
        pr = res["pricing"]
        M[f"PostPub{tag}WorstRank"] = max(v["rover_rank"] for v in pr.values())
    M["PostPubRareAllFirst"] = "yes" if all(v["rover_rank"] == 1 for v in rare["pricing"].values()) else "no"
    pf = freq["pricing"]
    for key, tag in (("release_H20", "RelTw"), ("line_h20", "LineTw"), ("line_h5", "LineFive"), ("release_H5", "RelFive")):
        t = pf[key]["table"]
        M[f"PostPubFreq{tag}Ro"] = f(t["rover"]["mean"])
        M[f"PostPubFreq{tag}Va"] = f(t["immutable_validator"]["mean"])
        M[f"PostPubFreq{tag}Pe"] = f(t["periodic_review"]["mean"])
        M[f"PostPubFreq{tag}Rank"] = pf[key]["rover_rank"]
        v = t["immutable_validator"]
        M[f"PostPubFreq{tag}VaD"] = f(v["d_other_minus_rover"])
        M[f"PostPubFreq{tag}VaLo"] = f(v["d_lo"])
        M[f"PostPubFreq{tag}VaHi"] = f(v["d_hi"])
    # repeated exposure: occurrences over all remote refs against unique lines over all remote refs
    occ = defaultdict(lambda: [0.0, 0.0])
    for r in rows("publication/publication.csv"):
        if r["label"] == "pub_nu0.250":
            a = occ[r["policy"]]
            a[0] += float(r["remote_canary"])
            a[1] += float(r["rel_lines_new"]) + float(r["act_lines_unique"])
    ratios = {p: a[0] / a[1] for p, a in occ.items() if a[1] > 0}
    M["PostPubRatioMax"] = f(max(ratios.values()), 1)
    M["PostPubRatioRo"] = f(ratios["rover"], 1)
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{$\nu=1/24$} & \multicolumn{2}{c}{$\nu=1/4$} & \multicolumn{3}{c}{loss, $\nu=1/4$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r"Policy & lines & leak \% & lines & leak \% & none & 20/line & 20/release \\", r"\midrule"]
    for p, name in PUB_POL:
        lr, lf = rare["leak"][p], freq["leak"][p]
        lines.append(f"{name} & {f(lr['rel_lines_new'])} & {f(100 * lr['share_episodes_leaking'], 0)} & "
                     f"{f(lf['rel_lines_new'])} & {f(100 * lf['share_episodes_leaking'], 0)} & "
                     f"{f(pf['base']['table'][p]['mean'])} & {f(pf['line_h20']['table'][p]['mean'])} & "
                     f"{f(pf['release_H20']['table'][p]['mean'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(SUPGEN, "tab_post_pub.tex"), "w").write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ cost semantics (E3)
    cs = load("cost_semantics/cost_semantics_analysis.json")
    assert cs["decisions_match_committed"]["ok"]
    cells = cs["cells"].values()
    err = [100 * (c["executed_cost_mean"] - c["planned_cost_mean"]) / c["planned_cost_mean"] for c in cells]
    M["PostCsCostErrMax"] = f(max(err), 1)
    M["PostCsConfMin"] = f(min(c["conflicts_per_recovery"] for c in cells))
    M["PostCsConfMax"] = f(max(c["conflicts_per_recovery"] for c in cells))
    low = [100 * (1 - c["resid_ratio_actual_over_predicted"]) for c in cells]
    M["PostCsResLowMin"] = f(min(low), 0)
    M["PostCsResLowMax"] = f(max(low), 0)
    dr = [c["resid_ratio_actual_over_direct"] for c in cells]
    M["PostCsDirectMin"] = f(min(dr))
    M["PostCsDirectMax"] = f(max(dr))
    gr = cs["growth_D_vs_model"]
    grow = [100 * (gr[c]["24"]["mean_D"] / gr[c]["24"]["model"] - 1) for c in ("skill", "queue", "branch")]
    M["PostCsGrowMin"] = f(min(grow), 0)
    M["PostCsGrowMax"] = f(max(grow), 0)
    M["PostCsEpisodes"] = num(cs["episodes"])

    # ------------------------------------------------------------------ belief-grid convergence (T = 24)
    gc = load("grid/grid_convergence.json")
    s5 = gc["summary"]["501"]
    M["PostGridRef"] = num(gc["G_ref"])
    M["PostGridMaxErr"] = f(s5["max_abs_err"], 4)
    M["PostGridMaxRel"] = f(100 * s5["max_rel_err"], 2)
    M["PostGridActDis"] = f(100 * s5["max_action_disagree_share"], 3)
    ge = load("grid/rover_G2001_analysis.json")
    M["PostGridDense"] = num(ge["grid"])
    M["PostGridEps"] = num(sum(v["episodes"] for v in ge["by_r1"].values()))
    M["PostGridIdent"] = num(sum(v["identical_decisions"] for v in ge["by_r1"].values()))

    # ------------------------------------------------------------------ estimands and adaptive attacker
    pr = load("statistics/primary.json")
    M["PostStHOneSf"] = prel(pr["holm_p_signflip"]["H1"])
    M["PostStHTwoSf"] = prel(pr["holm_p_signflip"]["H2"])
    M["PostStHOneT"] = prel(pr["holm_p_ttest"]["H1"])
    M["PostStHTwoT"] = prel(pr["holm_p_ttest"]["H2"])
    M["PostStHOneWin"] = f(pr["H1"]["paired_prob_rover_lower"], 2)
    M["PostStHOneAtw"] = f(pr["H1"]["a12_shipped_rover_lower"], 2)
    est = rows("statistics/estimands.csv")
    flips = [e for e in est if (float(e["p_holm_shipped"]) < 0.05) != (float(e["p_signflip_holm"]) < 0.05)]
    M["PostStTests"] = len(est)
    M["PostStFlips"] = len(flips)
    ad = {r["policy"]: r for r in rows("statistics/adaptive_split.csv")}
    M["PostAdRoSplit"] = f(ad["rover"]["split_eval_mean"], 1)
    M["PostAdVaSplit"] = f(ad["immutable_validator"]["split_eval_mean"], 1)
    n = [int(ad["rover"][f"n_{c}"]) for c in ("memory", "skill", "queue", "branch")]
    M["PostAdNMin"], M["PostAdNMax"] = min(n), max(n)

    # ------------------------------------------------------------------ exact A10 monotonicity
    cert = load("a10/a10_certificate.json")
    M["PostATenOK"] = "yes" if all(r["all_carriers_certified"] for r in cert["results"]) else "no"
    out.update(onestep_cells=len(os_rows), publication_episodes=pub["episodes"], flips=len(flips))
    S["exploratory"] = out
