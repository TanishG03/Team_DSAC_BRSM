"""
REPORT 2 — STEP 3: t-tests, Level Trends, Concurrent Validity
Requires outputs_r2/_r2_cache.pkl from Steps 1 & 2
"""

import pandas as pd
import numpy as np
import pickle, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import (ttest_rel, ttest_ind, mannwhitneyu, wilcoxon,
                         spearmanr, pearsonr, shapiro)
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")
OUT = Path("outputs_r2"); OUT.mkdir(exist_ok=True)

with open(OUT / "_r2_cache.pkl","rb") as f:
    cache = pickle.load(f)

lab_all         = cache["lab_all"]
game_all        = cache["game_all"]
game_comparable = cache["game_comparable"]
game_lvl1       = cache["game_lvl1"]
ptpt_lab        = cache["ptpt_lab"]
ptpt_game_all   = cache["ptpt_game_all"]
ptpt_game_comp  = cache["ptpt_game_comp"]
ptpt_game_lvl1  = cache["ptpt_game_lvl1"]
wide            = cache["wide"]
wide_comp       = cache["wide_comp"]
max_comp_level  = cache["max_comparable_level"]

PAL = {"Single":"#2D6A9F","Multiple":"#C05621"}

def stars(p):
    if p is None or (isinstance(p,float) and np.isnan(p)): return ""
    return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

def cohens_d(a, b):
    na,nb = len(a),len(b)
    sp = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(a)-np.mean(b))/sp if sp else np.nan

def cohens_dz(diff):
    s = np.std(diff, ddof=1)
    return np.mean(diff)/s if s else np.nan

def ci95(arr):
    n  = len(arr)
    se = np.std(arr,ddof=1)/np.sqrt(n)
    t  = t_dist.ppf(.975, n-1)
    return np.mean(arr)-t*se, np.mean(arr)+t*se

def fisher_r_ci(r, n):
    if n < 4: return np.nan, np.nan
    z  = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    return np.tanh(z-1.96*se), np.tanh(z+1.96*se)

def save(name):
    plt.savefig(OUT/name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  {name}")

def section(title):
    print("\n" + "─"*65)
    print(f"  {title}")
    print("─"*65)

print("="*65)
print("STEP 3: t-TESTS, LEVEL TRENDS, CONCURRENT VALIDITY")
print("="*65)


# =============================================================================
# SECTION A: H2 — PAIRED t-TESTS (Modality effect within each group)
# Tests whether Game RT > Lab RT within Single and within Multiple
# This is the within-group modality comparison
# =============================================================================

section("A. H2/H3 — PAIRED t-TESTS: Game vs Lab RT (within each group)")

paired_results = []

print("\n  DV: Log RT (primary)")
for grp in ["Single","Multiple"]:
    sub  = wide[wide["group"]==grp].dropna(subset=["logRT_Lab","logRT_Game"])
    n    = len(sub)
    lab  = sub["logRT_Lab"].values
    game = sub["logRT_Game"].values
    diff = game - lab

    t, p   = ttest_rel(game, lab)
    dz     = cohens_dz(diff)
    ci_d   = ci95(diff)
    sw_d, p_sw = shapiro(diff)

    # Also run Wilcoxon as robustness check
    W, p_wil = wilcoxon(diff, alternative="two-sided")

    interp = ("negligible" if abs(dz)<.2 else "small" if abs(dz)<.5
              else "medium" if abs(dz)<.8 else "large")

    print(f"\n  {grp} group  (n={n})")
    print(f"    Lab  logRT:  M={lab.mean():.4f}  SD={lab.std(ddof=1):.4f}  "
          f"95%CI[{ci95(lab)[0]:.4f},{ci95(lab)[1]:.4f}]")
    print(f"    Game logRT:  M={game.mean():.4f}  SD={game.std(ddof=1):.4f}  "
          f"95%CI[{ci95(game)[0]:.4f},{ci95(game)[1]:.4f}]")
    print(f"    Diff:        M={diff.mean():.4f}  SD={diff.std(ddof=1):.4f}  "
          f"95%CI[{ci_d[0]:.4f},{ci_d[1]:.4f}]")
    print(f"    Paired t({n-1})={t:.4f}  p={p:.4f}{stars(p)}  dz={dz:.3f} [{interp}]")
    print(f"    Wilcoxon W={W:.1f}  p={p_wil:.4f}{stars(p_wil)}  "
          f"(robustness check — diff scores SW p={p_sw:.4f})")

    # Raw RT version for report table
    lab_r  = sub["RT_Lab"].values
    game_r = sub["RT_Game"].values
    diff_r = game_r - lab_r
    t_r,p_r = ttest_rel(game_r, lab_r)
    dz_r    = cohens_dz(diff_r)
    print(f"    Raw RT diff: M={diff_r.mean():.1f} ms  "
          f"t({n-1})={t_r:.3f}  p={p_r:.4f}{stars(p_r)}  dz={dz_r:.3f}")

    paired_results.append({
        "Group":grp,"n":n,
        "Lab_logRT_M":lab.mean(),"Game_logRT_M":game.mean(),
        "Diff_logRT_M":diff.mean(),"Diff_logRT_SD":diff.std(ddof=1),
        "CI_lo":ci_d[0],"CI_hi":ci_d[1],
        "t":t,"df":n-1,"p":p,"sig":stars(p),"dz":dz,
        "Wilcoxon_W":W,"p_wilcoxon":p_wil,
        "Raw_diff_M":diff_r.mean(),"t_raw":t_r,"p_raw":p_r,"dz_raw":dz_r,
    })


# =============================================================================
# SECTION B: H2 — INDEPENDENT t-TESTS (Load effect: Single vs Multiple)
# =============================================================================

section("B. H2 — INDEPENDENT t-TESTS: Single vs Multiple")

indep_results = []

dvs_indep = [
    (ptpt_lab,      "logRT_mean",  "Lab Log RT"),
    (ptpt_lab,      "RT_mean",     "Lab Raw RT (ms)"),
    (ptpt_game_all, "logRT_mean",  "Game Log RT (all levels)"),
    (ptpt_game_all, "RT_mean",     "Game Raw RT ms (all levels)"),
    (ptpt_game_comp,"RT_mean_comp","Game Raw RT ms (levels 1–10)"),
    (ptpt_game_all, "success_rate","Game Success Rate (%)"),
    (ptpt_game_all, "hit_rate",    "Game Hit Rate (%)"),
    (ptpt_game_all, "false_alarms","Game False Alarms"),
    (ptpt_game_all, "max_level",   "Game Max Level Reached"),
]

for df, col, label in dvs_indep:
    if col not in df.columns: continue
    s = df[df["group"]=="Single"][col].dropna().values
    m = df[df["group"]=="Multiple"][col].dropna().values
    if len(s)<2 or len(m)<2: continue

    # Normality check on each group
    sw_s, p_sw_s = shapiro(s) if len(s)>=3 else (np.nan, np.nan)
    sw_m, p_sw_m = shapiro(m) if len(m)>=3 else (np.nan, np.nan)
    normal = (p_sw_s > .05 and p_sw_m > .05)

    # Welch t-test (primary — doesn't assume equal variance)
    t, p   = ttest_ind(s, m, equal_var=False)
    d      = cohens_d(s, m)
    ci_s   = ci95(s)
    ci_m   = ci95(m)

    # Mann-Whitney U (non-parametric robustness check)
    U, p_mw = mannwhitneyu(s, m, alternative="two-sided")
    r_mw    = 1 - 2*U/(len(s)*len(m))

    interp_d = ("negligible" if abs(d)<.2 else "small" if abs(d)<.5
                else "medium" if abs(d)<.8 else "large")

    print(f"\n  {label}:")
    print(f"    Single   M={s.mean():.3f}  SD={s.std(ddof=1):.3f}  "
          f"n={len(s)}  95%CI[{ci_s[0]:.3f},{ci_s[1]:.3f}]")
    print(f"    Multiple M={m.mean():.3f}  SD={m.std(ddof=1):.3f}  "
          f"n={len(m)}  95%CI[{ci_m[0]:.3f},{ci_m[1]:.3f}]")
    print(f"    Welch t({len(s)+len(m)-2})={t:.3f}  p={p:.4f}{stars(p)}  "
          f"d={d:.3f} [{interp_d}]")
    print(f"    Mann-Whitney U={U:.0f}  p={p_mw:.4f}{stars(p_mw)}  r={r_mw:.3f}  "
          f"({'parametric & non-param agree' if (p<.05)==(p_mw<.05) else '⚠ DISAGREE'})")

    indep_results.append({
        "DV":label,"n_Single":len(s),"n_Multiple":len(m),
        "Single_M":s.mean(),"Single_SD":s.std(ddof=1),
        "Multiple_M":m.mean(),"Multiple_SD":m.std(ddof=1),
        "t":t,"p":p,"sig":stars(p),"d":d,
        "U_MW":U,"p_MW":p_mw,"sig_MW":stars(p_mw),"r_MW":r_mw,
    })

pd.DataFrame(indep_results).to_csv(OUT/"results_r2_indep_ttests.csv", index=False)
print(f"\n  ✔  Saved results_r2_indep_ttests.csv")


# =============================================================================
# SECTION C: H4 — SPEARMAN LEVEL TRENDS (within each group)
# Tests whether RT increases and accuracy decreases as level increases
# Run on: (1) all levels per group, (2) levels 1–10 only (comparable window)
# =============================================================================

section("C. H4 — SPEARMAN LEVEL TRENDS: RT and Accuracy ~ Level")

level_results = []

dvs_level = [
    ("RT_ms",              "RT (ms)"),
    ("log_RT",             "Log RT"),
    ("success_rate",       "Success Rate (%)"),
    ("hit_rate",           "Hit Rate (%)"),
    ("false_alarms",       "False Alarms"),
    ("avg_inter_target_ms","Avg Inter-Target Time (ms)"),
]

for grp in ["Single","Multiple"]:
    print(f"\n  {grp} group:")

    for window_label, game_df in [
        ("All levels",         game_all[game_all["group"]==grp]),
        (f"Levels 1–{max_comp_level}", game_comparable[game_comparable["group"]==grp]),
    ]:
        print(f"\n    Window: {window_label}  "
              f"(n_obs={len(game_df)}, n_ptpt={game_df['participant'].nunique()})")

        for col, label in dvs_level:
            if col not in game_df.columns: continue
            tmp = game_df[["level",col]].dropna()
            if len(tmp) < 5: continue

            rho, p_sp = spearmanr(tmp["level"], tmp[col])
            slope, intercept, r_lr, p_lr, se_s = stats.linregress(
                tmp["level"], tmp[col])
            direction = "↑ increasing" if rho>0 else "↓ decreasing"
            expected  = {
                "RT (ms)":"↑","Log RT":"↑",
                "Success Rate (%)":"↓","Hit Rate (%)":"↓",
                "False Alarms":"↑","Avg Inter-Target Time (ms)":"↑"
            }.get(label,"?")
            match = "✓" if direction[0]==expected else "✗"

            print(f"      {label:<28}  ρ={rho:+.3f}  p={p_sp:.4f}{stars(p_sp)}"
                  f"  β={slope:+.3f}/level  {direction}  {match}expected")

            level_results.append({
                "Group":grp,"Window":window_label,"DV":label,
                "rho":rho,"p_spearman":p_sp,"sig":stars(p_sp),
                "slope":slope,"p_slope":p_lr,
                "direction":direction[0],
                "matches_hypothesis": direction[0]==expected,
            })

pd.DataFrame(level_results).to_csv(OUT/"results_r2_level_trends.csv", index=False)
print(f"\n  ✔  Saved results_r2_level_trends.csv")


# =============================================================================
# SECTION D: H1 — CONCURRENT VALIDITY
# Pearson r + Spearman rho, with Fisher 95% CIs
# Three comparisons:
#   (a) Lab RT vs Game RT (all levels)
#   (b) Lab RT vs Game RT (levels 1–10)
#   (c) Lab RT vs Level-1 Game RT (most structurally comparable)
# =============================================================================

section("D. H1 — CONCURRENT VALIDITY: Lab RT vs Game RT")

# Build merged tables
merged_all = pd.merge(
    ptpt_lab[["participant","group","RT_mean","logRT_mean"]].rename(
        columns={"RT_mean":"lab_RT","logRT_mean":"lab_logRT"}),
    ptpt_game_all[["participant","group","RT_mean","logRT_mean",
                   "success_rate","hit_rate","false_alarms","RT_sd"]].rename(
        columns={"RT_mean":"game_RT","logRT_mean":"game_logRT","RT_sd":"game_RT_sd"}),
    on=["participant","group"]
).dropna()

merged_comp = pd.merge(
    ptpt_lab[["participant","group","RT_mean","logRT_mean"]].rename(
        columns={"RT_mean":"lab_RT","logRT_mean":"lab_logRT"}),
    ptpt_game_comp[["participant","group","RT_mean_comp","logRT_mean_comp"]].rename(
        columns={"RT_mean_comp":"game_RT","logRT_mean_comp":"game_logRT"}),
    on=["participant","group"]
).dropna()

merged_lvl1 = pd.merge(
    ptpt_lab[["participant","group","RT_mean","logRT_mean"]].rename(
        columns={"RT_mean":"lab_RT","logRT_mean":"lab_logRT"}),
    ptpt_game_lvl1[["participant","group","RT_lvl1","logRT_lvl1"]].rename(
        columns={"RT_lvl1":"game_RT","logRT_lvl1":"game_logRT"}),
    on=["participant","group"]
).dropna()

validity_results = []

print("\n  Three validity comparisons:")
print("  (a) Lab RT vs Game RT — all levels")
print("  (b) Lab RT vs Game RT — levels 1–10 (comparable window)")
print("  (c) Lab RT vs Level-1 Game RT (most structurally comparable to lab)")

for comp_label, merged_df in [
    ("(a) All levels",    merged_all),
    (f"(b) Lvls 1–{max_comp_level}", merged_comp),
    ("(c) Level 1 only",  merged_lvl1),
]:
    print(f"\n  ── {comp_label} ────────────────────────────────────────")

    for grp in ["Single","Multiple","All"]:
        sub = merged_df if grp=="All" else merged_df[merged_df["group"]==grp]
        n   = len(sub)
        print(f"\n    {grp}  (n={n})")
        if n < 3:
            print(f"      ⚠  n<3 — insufficient for correlation")
            continue

        # Pearson on raw RT
        r_raw, p_raw = pearsonr(sub["game_RT"], sub["lab_RT"])
        ci_lo_raw, ci_hi_raw = fisher_r_ci(r_raw, n)

        # Pearson on log RT
        r_log, p_log = pearsonr(sub["game_logRT"], sub["lab_logRT"])
        ci_lo_log, ci_hi_log = fisher_r_ci(r_log, n)

        # Spearman (non-parametric robustness)
        rho, p_rho = spearmanr(sub["game_RT"], sub["lab_RT"])

        print(f"      Pearson  r (raw RT)  = {r_raw:+.3f}  "
              f"p={p_raw:.4f}{stars(p_raw)}  "
              f"95%CI[{ci_lo_raw:.3f},{ci_hi_raw:.3f}]")
        print(f"      Pearson  r (log RT)  = {r_log:+.3f}  "
              f"p={p_log:.4f}{stars(p_log)}  "
              f"95%CI[{ci_lo_log:.3f},{ci_hi_log:.3f}]")
        print(f"      Spearman ρ (raw RT)  = {rho:+.3f}  "
              f"p={p_rho:.4f}{stars(p_rho)}")

        validity_results.append({
            "Comparison":comp_label,"Group":grp,"n":n,
            "r_raw":r_raw,"p_raw":p_raw,"sig_raw":stars(p_raw),
            "ci_lo_raw":ci_lo_raw,"ci_hi_raw":ci_hi_raw,
            "r_log":r_log,"p_log":p_log,"sig_log":stars(p_log),
            "ci_lo_log":ci_lo_log,"ci_hi_log":ci_hi_log,
            "rho":rho,"p_rho":p_rho,"sig_rho":stars(p_rho),
        })

# Attenuation ceiling
print("\n  ── Reliability-Corrected Validity Ceiling ──────────────────")
print("  (Maximum observable r given Lab reliability from Report 1)")

# Use Cronbach's alpha from R1
alpha_single   = 0.744
alpha_multiple = 0.621
print(f"\n    Single   Lab α={alpha_single:.3f} → √α={np.sqrt(alpha_single):.3f}"
      f" (max observable r if game perfectly reliable)")
print(f"    Multiple Lab α={alpha_multiple:.3f} → √α={np.sqrt(alpha_multiple):.3f}"
      f" (max observable r if game perfectly reliable)")

# Check if our observed r exceeds what would be expected by chance
# Power: minimum r to reach p=.05 for each n
for grp, n in [("Single",21),("Multiple",16),("All",37)]:
    t_crit  = t_dist.ppf(.975, n-2)
    r_crit  = t_crit / np.sqrt(t_crit**2 + n - 2)
    print(f"    {grp:<10} n={n}  Minimum r for p<.05: r={r_crit:.3f}")

pd.DataFrame(validity_results).to_csv(OUT/"results_r2_validity.csv", index=False)
print(f"\n  ✔  Saved results_r2_validity.csv")


# =============================================================================
# SECTION E: EXPLORATORY — H_E1 Red vs White RT (lab)
# =============================================================================

section("E. H_E1 — Search Termination: Red vs White Trial RT (Lab)")

for grp in ["Single","Multiple"]:
    sub = lab_all[lab_all["group"]==grp].copy()
    red   = sub[sub["target_col"]=="red"]["RT_ms"].dropna().values
    white = sub[sub["target_col"]=="white"]["RT_ms"].dropna().values
    n_r, n_w = len(red), len(white)

    print(f"\n  {grp} group  (red n={n_r}, white n={n_w}):")
    print(f"    Red   (target):     M={red.mean():.1f}  SD={red.std(ddof=1):.1f}  "
          f"Mdn={np.median(red):.1f}")
    print(f"    White (distractor): M={white.mean():.1f}  SD={white.std(ddof=1):.1f}  "
          f"Mdn={np.median(white):.1f}")
    print(f"    Difference:         M={red.mean()-white.mean():.1f} ms  "
          f"({'Red faster ✓' if red.mean()<white.mean() else 'White faster'})")

    # Mann-Whitney U (trial-level, non-parametric — large n)
    U, p_u = mannwhitneyu(red, white, alternative="two-sided")
    r_eff  = 1 - 2*U/(n_r*n_w)
    print(f"    Mann-Whitney U={U:.0f}  p={p_u:.4f}{stars(p_u)}  r={r_eff:.3f}")

    # Variance comparison (Levene) — especially relevant for Multiple group
    from scipy.stats import levene as lev
    lev_stat, lev_p = lev(red, white, center="median")
    print(f"    Levene (variance equality): W={lev_stat:.4f}  p={lev_p:.4f}  "
          f"{'✓ equal variance' if lev_p>.05 else '✗ unequal variance'}")
    print(f"    SD ratio (red/white): {red.std(ddof=1)/white.std(ddof=1):.3f}")


# =============================================================================
# SECTION F: EXPLORATORY — H_E3 RTV predicts Game accuracy
# =============================================================================

section("F. H_E3 — RT Variability Predicts Game Accuracy")

rtv_results = []

for grp in ["Single","Multiple"]:
    sub = merged_all[merged_all["group"]==grp].copy()
    sub = sub.merge(
        ptpt_game_all[["participant","group","RT_sd"]].rename(
            columns={"RT_sd":"game_RT_sd"}),
        on=["participant","group"], how="left"
    )
    sub = sub.merge(
        ptpt_lab[["participant","group","RT_sd"]].rename(
            columns={"RT_sd":"lab_RT_sd"}),
        on=["participant","group"], how="left"
    )

    print(f"\n  {grp} group (n={len(sub)}):")

    for pred_col, pred_label in [
        ("game_RT_sd","Game RT SD (variability)"),
        ("lab_RT_sd", "Lab RT SD (variability)"),
    ]:
        for outcome_col, outcome_label in [
            ("success_rate","Success Rate"),
            ("false_alarms","False Alarms"),
        ]:
            if pred_col not in sub.columns or outcome_col not in sub.columns:
                continue
            tmp = sub[[pred_col, outcome_col]].dropna()
            if len(tmp) < 4: continue
            rho, p_r = spearmanr(tmp[pred_col], tmp[outcome_col])
            print(f"    {pred_label:<35} → {outcome_label:<18}"
                  f"  ρ={rho:+.3f}  p={p_r:.4f}{stars(p_r)}")
            rtv_results.append({
                "Group":grp,"Predictor":pred_label,"Outcome":outcome_label,
                "rho":rho,"p":p_r,"sig":stars(p_r)
            })

pd.DataFrame(rtv_results).to_csv(OUT/"results_r2_rtv.csv", index=False)
print(f"\n  ✔  Saved results_r2_rtv.csv")


# =============================================================================
# SECTION G: PLOTS — t-tests and validity
# =============================================================================

print("\n  Generating plots...")

# ── Fig: Paired t-test — Lab vs Game per group (raw RT, individual lines) ───
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("H3 — Paired Comparison: Lab vs Game RT per Group\n"
             "(Participant means; bold diamond = group mean ± SE)",
             fontsize=12, fontweight="bold")

for ax, grp in zip(axes, ["Single","Multiple"]):
    sub   = wide[wide["group"]==grp].dropna(subset=["RT_Lab","RT_Game"])
    color = PAL[grp]
    for _, row in sub.iterrows():
        ax.plot(["Lab","Game"],[row["RT_Lab"],row["RT_Game"]],
                "o-", color=color, alpha=0.28, linewidth=1.2, markersize=5)
    lab_m, game_m   = sub["RT_Lab"].mean(), sub["RT_Game"].mean()
    lab_se, game_se = (sub["RT_Lab"].std(ddof=1)/np.sqrt(len(sub)),
                       sub["RT_Game"].std(ddof=1)/np.sqrt(len(sub)))
    ax.errorbar(["Lab","Game"],[lab_m,game_m],
                yerr=[lab_se,game_se],
                fmt="D-", color=color, linewidth=3, markersize=11,
                capsize=8, zorder=5,
                label=f"M: Lab={lab_m:.0f}, Game={game_m:.0f} ms")
    # Annotate effect size from paired_results
    pr = [r for r in paired_results if r["Group"]==grp][0]
    ax.set_title(f"{grp}  (n={len(sub)})\n"
                 f"t({pr['df']})={pr['t']:.3f}, p={pr['p']:.4f}{stars(pr['p'])}, "
                 f"dz={pr['dz']:.3f}")
    ax.set_xlabel("Modality"); ax.set_ylabel("RT (ms)")
    ax.legend(fontsize=9)

plt.tight_layout()
save("fig_s3_01_paired_ttest.png")


# ── Fig: Independent t-test forest plot (Cohen's d, key DVs) ────────────────
key_dvs = [r for r in indep_results if r["DV"] in [
    "Lab Log RT","Game Log RT (all levels)",
    "Game False Alarms","Game Max Level Reached",
    "Game Success Rate (%)","Game Hit Rate (%)"
]]

if key_dvs:
    fig, ax = plt.subplots(figsize=(10, max(4, len(key_dvs)*1.1)))
    y_pos   = range(len(key_dvs))
    d_vals  = [r["d"] for r in key_dvs]
    labels_ = [r["DV"] for r in key_dvs]
    sigs_   = [r["sig"] for r in key_dvs]
    colors_ = [PAL["Single"] if d>0 else PAL["Multiple"] for d in d_vals]

    bars = ax.barh(list(y_pos), d_vals, color=colors_, alpha=0.78,
                   edgecolor="white", height=0.55)
    for i, (r, y, d, sig) in enumerate(zip(key_dvs, y_pos, d_vals, sigs_)):
        offset = 0.05 if d >= 0 else -0.05
        ha     = "left" if d >= 0 else "right"
        ax.text(d+offset, y, f"d={d:.2f} {sig}", va="center",
                ha=ha, fontsize=9, fontweight="bold")
    ax.axvline(0, color="black", lw=1)
    for v, ls, lbl in [(0.2,":","small"),(0.5,"--","medium"),(0.8,"-.","large")]:
        ax.axvline(v, color="#718096", ls=ls, lw=1, alpha=0.6)
        ax.axvline(-v, color="#718096", ls=ls, lw=1, alpha=0.6)
    ax.set_yticks(list(y_pos)); ax.set_yticklabels(labels_, fontsize=10)
    ax.set_xlabel("Cohen's d  (positive = Single > Multiple)")
    ax.set_title("H2 — Independent t-tests: Single vs Multiple\n"
                 "Cohen's d Forest Plot", fontweight="bold")
    plt.tight_layout()
    save("fig_s3_02_forest_plot.png")


# ── Fig: Concurrent validity scatterplots (3 comparisons × 2 groups) ────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("H1 — Concurrent Validity: Lab RT vs Game RT\n"
             "(Each dot = one participant mean; dashed = regression; "
             "dotted = identity line)",
             fontsize=12, fontweight="bold")

comp_configs = [
    ("(a) All levels",   merged_all,  "game_RT","lab_RT","All-Level Game RT (ms)"),
    (f"(b) Lvls 1–{max_comp_level}", merged_comp, "game_RT","lab_RT",
                                     f"Level 1–{max_comp_level} Game RT (ms)"),
    ("(c) Level 1 only", merged_lvl1, "game_RT","lab_RT","Level-1 Game RT (ms)"),
]

for col_i, (comp_label, merged_df, xcol, ycol, xlabel) in enumerate(comp_configs):
    for row_i, grp in enumerate(["Single","Multiple"]):
        ax    = axes[row_i, col_i]
        sub   = merged_df[merged_df["group"]==grp].dropna(subset=[xcol,ycol])
        color = PAL[grp]
        n     = len(sub)

        ax.scatter(sub[xcol], sub[ycol], color=color, s=60, alpha=0.8, zorder=3)

        # Participant labels
        for _, row in sub.iterrows():
            ax.annotate(str(row["participant"]),
                        (row[xcol], row[ycol]),
                        xytext=(4,3), textcoords="offset points",
                        fontsize=7, alpha=0.7)

        if n >= 3:
            # Regression line
            z  = np.polyfit(sub[xcol], sub[ycol], 1)
            xs = np.linspace(sub[xcol].min()*0.95, sub[xcol].max()*1.05, 200)
            ax.plot(xs, np.poly1d(z)(xs), "--", color=color, lw=1.8, alpha=0.8)

            # Identity line
            lim = [min(sub[[xcol,ycol]].min()), max(sub[[xcol,ycol]].max())]
            ax.plot(lim, lim, ":", color="gray", lw=1, alpha=0.5)

            r, p = pearsonr(sub[xcol], sub[ycol])
            ci_l, ci_h = fisher_r_ci(r, n)
            ax.set_title(f"{grp} — {comp_label}\n"
                         f"r={r:.3f}{stars(p)} 95%CI[{ci_l:.3f},{ci_h:.3f}]",
                         fontsize=8.5)
        else:
            ax.set_title(f"{grp} — {comp_label}\n(n={n})", fontsize=8.5)

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Lab RT (ms)", fontsize=8)

plt.tight_layout()
save("fig_s3_03_validity_scatter.png")


# ── Fig: Spearman level trends — RT and accuracy by level ───────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("H4 — Level Trends: RT and Accuracy by Game Level\n"
             "(Lines = mean per level; shading = ±SE; dashed = linear trend)",
             fontsize=12, fontweight="bold")

plot_dvs = [
    ("RT_ms",        "RT (ms)"),
    ("success_rate", "Success Rate (%)"),
    ("hit_rate",     "Hit Rate (%)"),
    ("false_alarms", "False Alarms"),
]

for ax, (col, ylabel) in zip(axes.flat, plot_dvs):
    for grp, color in [("Single",PAL["Single"]),("Multiple",PAL["Multiple"])]:
        sub = game_all[game_all["group"]==grp]
        lv  = sub.groupby("level")[col].agg(["mean","sem","count"])
        ax.plot(lv.index, lv["mean"], "o-", color=color,
                label=f"{grp}", linewidth=2, markersize=5, alpha=0.85)
        ax.fill_between(lv.index,
                        lv["mean"]-lv["sem"],
                        lv["mean"]+lv["sem"],
                        alpha=0.15, color=color)
        # Linear trend line
        tmp = sub[["level",col]].dropna()
        if len(tmp) >= 3:
            slope, intercept, *_ = stats.linregress(tmp["level"],tmp[col])
            xs = np.linspace(tmp["level"].min(), tmp["level"].max(), 100)
            ax.plot(xs, slope*xs+intercept, "--", color=color, alpha=0.5, lw=1.5)
    # Attrition marker
    ax.axvline(10, color="gray", ls=":", lw=1.5, alpha=0.7)
    ax.text(10.2, ax.get_ylim()[1]*0.97, "Attrition\n(Multiple)",
            fontsize=7, color="gray", va="top")
    ax.set_xlabel("Level"); ax.set_ylabel(ylabel)
    ax.set_title(ylabel); ax.legend(fontsize=9)

plt.tight_layout()
save("fig_s3_04_level_trends.png")


# =============================================================================
# SECTION H: SAVE + CACHE UPDATE
# =============================================================================

pd.DataFrame(paired_results).to_csv(OUT/"results_r2_paired_ttests.csv", index=False)
print(f"\n  ✔  Saved results_r2_paired_ttests.csv")

cache.update({
    "paired_results":  paired_results,
    "indep_results":   indep_results,
    "validity_results":validity_results,
    "level_results":   level_results,
    "rtv_results":     rtv_results,
    "merged_all":      merged_all,
    "merged_comp":     merged_comp,
    "merged_lvl1":     merged_lvl1,
})
with open(OUT/"_r2_cache.pkl","wb") as f:
    pickle.dump(cache, f)
print("  ✔  Updated _r2_cache.pkl")

print("\n  STEP 3 COMPLETE — paste full output before Step 4.\n")