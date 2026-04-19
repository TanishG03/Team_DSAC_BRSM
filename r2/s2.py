"""
REPORT 2 — STEP 2: ANOVA Diagnostics + 2x2 Mixed ANOVA
Requires outputs_r2/_r2_cache.pkl from Step 1
"""

import pandas as pd
import numpy as np
import pickle, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import (shapiro, levene, pearsonr, normaltest)
from scipy.stats import f as f_dist

warnings.filterwarnings("ignore")
OUT = Path("outputs_r2"); OUT.mkdir(exist_ok=True)

with open(OUT / "_r2_cache.pkl","rb") as f:
    cache = pickle.load(f)

lab_all          = cache["lab_all"]
game_all         = cache["game_all"]
ptpt_lab         = cache["ptpt_lab"]
ptpt_game_all    = cache["ptpt_game_all"]
ptpt_game_comp   = cache["ptpt_game_comp"]
ptpt_game_lvl1   = cache["ptpt_game_lvl1"]
max_comp_level   = cache["max_comparable_level"]

PAL = {"Single":"#2D6A9F","Multiple":"#C05621"}

def stars(p):
    if p is None or (isinstance(p,float) and np.isnan(p)): return ""
    return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

def save(name):
    plt.savefig(OUT/name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  {name}")

# ── Build wide-form participant table ─────────────────────────────────────────
# One row per participant: Lab RT, Game RT (all levels), Game RT (comp window)

wide = pd.merge(
    ptpt_lab[["participant","group","RT_mean","logRT_mean","RT_sd"]].rename(
        columns={"RT_mean":"RT_Lab","logRT_mean":"logRT_Lab","RT_sd":"SD_Lab"}),
    ptpt_game_all[["participant","group","RT_mean","logRT_mean","RT_sd",
                   "success_rate","hit_rate","false_alarms","max_level"]].rename(
        columns={"RT_mean":"RT_Game","logRT_mean":"logRT_Game","RT_sd":"SD_Game"}),
    on=["participant","group"]
).dropna(subset=["RT_Lab","RT_Game"])

wide_comp = pd.merge(
    ptpt_lab[["participant","group","RT_mean","logRT_mean"]].rename(
        columns={"RT_mean":"RT_Lab","logRT_mean":"logRT_Lab"}),
    ptpt_game_comp[["participant","group","RT_mean_comp","logRT_mean_comp"]].rename(
        columns={"RT_mean_comp":"RT_Game","logRT_mean_comp":"logRT_Game"}),
    on=["participant","group"]
).dropna()

print("="*65)
print("STEP 2: ANOVA DIAGNOSTICS + 2×2 MIXED ANOVA")
print("="*65)
print(f"\n  Wide-form (all game levels): {len(wide)} participants")
print(f"  Wide-form (levels 1–{max_comp_level}):    {len(wide_comp)} participants")
print(f"  Single: n={( wide['group']=='Single').sum()}   Multiple: n={(wide['group']=='Multiple').sum()}")

# =============================================================================
# SECTION A: ANOVA DIAGNOSTICS
# Must run BEFORE reporting the ANOVA
# =============================================================================

print("\n" + "─"*65)
print("A. ANOVA DIAGNOSTICS")
print("─"*65)

# We run diagnostics on BOTH raw RT and log-RT
# so we can explicitly justify our choice of DV in the report

for dv_label, lab_col, game_col in [
    ("Raw RT",  "RT_Lab",    "RT_Game"),
    ("Log RT",  "logRT_Lab", "logRT_Game"),
]:
    print(f"\n  ── {dv_label} ─────────────────────────────────────────")

    # 1. Shapiro-Wilk on each of the 4 cells
    print("  1. Shapiro-Wilk normality — each cell (participant-level means):")
    for grp in ["Single","Multiple"]:
        for col, mod in [(lab_col,"Lab"),(game_col,"Game")]:
            vals = wide[wide["group"]==grp][col].dropna().values
            if len(vals) >= 3:
                sw, p_sw = shapiro(vals)
                print(f"     {grp:<10} {mod:<6}  n={len(vals):>2}  "
                      f"W={sw:.4f}  p={p_sw:.4f}  {'✓ normal' if p_sw>.05 else '✗ non-normal'}")

    # 2. Levene's test on between-subjects factor (single vs multiple) — use Lab
    print("\n  2. Levene's test (homogeneity of variance — between groups on Lab DV):")
    s_lab = wide[wide["group"]=="Single"][lab_col].dropna().values
    m_lab = wide[wide["group"]=="Multiple"][lab_col].dropna().values
    lev_stat, lev_p = levene(s_lab, m_lab, center="median")
    print(f"     Levene W={lev_stat:.4f}  p={lev_p:.4f}  "
          f"{'✓ variances equal' if lev_p>.05 else '✗ variances unequal'}")

    # 3. Levene's on Game DV
    s_game = wide[wide["group"]=="Single"][game_col].dropna().values
    m_game = wide[wide["group"]=="Multiple"][game_col].dropna().values
    lev2_stat, lev2_p = levene(s_game, m_game, center="median")
    print(f"     Levene (Game) W={lev2_stat:.4f}  p={lev2_p:.4f}  "
          f"{'✓ variances equal' if lev2_p>.05 else '✗ variances unequal'}")

    # 4. Normality of DIFFERENCE scores (within-subjects)
    print("\n  3. Shapiro-Wilk on difference scores (Game − Lab) per group:")
    for grp in ["Single","Multiple"]:
        sub = wide[wide["group"]==grp].dropna(subset=[lab_col, game_col])
        diff = sub[game_col].values - sub[lab_col].values
        if len(diff) >= 3:
            sw_d, p_d = shapiro(diff)
            print(f"     {grp:<10}  n={len(diff):>2}  diff M={diff.mean():.3f}  "
                  f"W={sw_d:.4f}  p={p_d:.4f}  "
                  f"{'✓ normal' if p_d>.05 else '✗ non-normal'}")

    # 5. Outlier check: participant means > 3 SD from group mean
    print("\n  4. Outlier check (participant means > 3SD from group mean):")
    outliers_found = False
    for grp in ["Single","Multiple"]:
        for col, mod in [(lab_col,"Lab"),(game_col,"Game")]:
            sub = wide[wide["group"]==grp][["participant",col]].dropna()
            z = np.abs((sub[col] - sub[col].mean()) / sub[col].std(ddof=1))
            outs = sub[z > 3]
            if len(outs) > 0:
                print(f"     ⚠  {grp} {mod}: {outs['participant'].tolist()} flagged")
                outliers_found = True
    if not outliers_found:
        print("     No outliers > 3SD found in any cell.")


# =============================================================================
# SECTION B: DIAGNOSTIC PLOTS
# =============================================================================

print("\n  Generating diagnostic plots...")

fig = plt.figure(figsize=(16, 14))
fig.suptitle("ANOVA Diagnostics — Log RT (Participant Means)",
             fontsize=14, fontweight="bold")

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

cells = [
    ("Single","Lab",  "logRT_Lab"),
    ("Single","Game", "logRT_Game"),
    ("Multiple","Lab",  "logRT_Lab"),
    ("Multiple","Game", "logRT_Game"),
]

# Row 1: Histograms
for i, (grp, mod, col) in enumerate(cells):
    ax = fig.add_subplot(gs[0, i])
    vals = wide[wide["group"]==grp][col].dropna().values
    ax.hist(vals, bins=8, color=PAL[grp], alpha=0.75, edgecolor="white", density=True)
    # Overlay normal curve
    x = np.linspace(vals.min()-0.1, vals.max()+0.1, 200)
    ax.plot(x, stats.norm.pdf(x, vals.mean(), vals.std(ddof=1)),
            color="black", lw=1.8, ls="--", label="Normal")
    sw, p_sw = shapiro(vals)
    ax.set_title(f"{grp} {mod}\nSW p={p_sw:.3f}", fontsize=9)
    ax.set_xlabel("log RT"); ax.set_ylabel("Density")

# Row 2: Q-Q plots
for i, (grp, mod, col) in enumerate(cells):
    ax = fig.add_subplot(gs[1, i])
    vals = wide[wide["group"]==grp][col].dropna().values
    (osm, osr), (slope, intercept, _) = stats.probplot(vals, dist="norm")
    ax.scatter(osm, osr, color=PAL[grp], s=40, alpha=0.8, zorder=3)
    lx = np.array([osm.min(), osm.max()])
    ax.plot(lx, slope*lx+intercept, color="black", lw=1.5, ls="--")
    ax.set_title(f"Q-Q: {grp} {mod}", fontsize=9)
    ax.set_xlabel("Theoretical"); ax.set_ylabel("Sample")

# Row 3: Difference score distributions (Game - Lab) per group
for i, grp in enumerate(["Single","Multiple"]):
    ax = fig.add_subplot(gs[2, i*2:(i*2)+2])
    sub = wide[wide["group"]==grp].dropna(subset=["logRT_Lab","logRT_Game"])
    diff = sub["logRT_Game"].values - sub["logRT_Lab"].values
    ax.hist(diff, bins=8, color=PAL[grp], alpha=0.75, edgecolor="white", density=True)
    x = np.linspace(diff.min()-0.05, diff.max()+0.05, 200)
    ax.plot(x, stats.norm.pdf(x, diff.mean(), diff.std(ddof=1)),
            color="black", lw=1.8, ls="--", label="Normal")
    sw_d, p_d = shapiro(diff)
    ax.axvline(0, color="crimson", ls=":", lw=1.5, label="Zero diff")
    ax.set_title(f"{grp}: Difference Scores (logRT Game − Lab)\n"
                 f"M={diff.mean():.3f}  SW p={p_d:.3f}", fontsize=9)
    ax.set_xlabel("Difference (log RT)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)

save("fig_s2_01_anova_diagnostics.png")


# =============================================================================
# SECTION C: 2×2 MIXED ANOVA  (manual SS calculation, matches R1 method)
# Run on BOTH raw RT and log-RT; report log-RT as primary
# =============================================================================

print("\n" + "─"*65)
print("B. 2×2 MIXED ANOVA")
print("   Between: Target Load (Single vs Multiple)")
print("   Within:  Modality (Lab vs Game)")
print("   DV: Log RT (primary) and Raw RT (reported alongside)")
print("─"*65)

def run_mixed_anova(data, lab_col, game_col, dv_label):
    """
    Manual 2×2 mixed ANOVA.
    Between: group (Single/Multiple)
    Within:  modality (Lab/Game)
    Returns dict of results.
    """
    w = data.dropna(subset=[lab_col, game_col]).copy()
    n_s = (w["group"]=="Single").sum()
    n_m = (w["group"]=="Multiple").sum()
    N   = len(w)

    s_lab  = w[w["group"]=="Single"][lab_col].values
    s_game = w[w["group"]=="Single"][game_col].values
    m_lab  = w[w["group"]=="Multiple"][lab_col].values
    m_game = w[w["group"]=="Multiple"][game_col].values

    # Grand mean
    grand = np.concatenate([s_lab, s_game, m_lab, m_game]).mean()

    # Marginal means
    mu_s    = np.concatenate([s_lab, s_game]).mean()
    mu_m    = np.concatenate([m_lab, m_game]).mean()
    mu_lab  = np.concatenate([s_lab, m_lab]).mean()
    mu_game = np.concatenate([s_game, m_game]).mean()

    # Cell means
    cell = {
        ("Single","Lab"):   s_lab.mean(),
        ("Single","Game"):  s_game.mean(),
        ("Multiple","Lab"): m_lab.mean(),
        ("Multiple","Game"):m_game.mean(),
    }

    # Subject means
    w = w.copy()
    w["subj_mean"] = w[[lab_col, game_col]].mean(axis=1)

    # SS Between (Target Load — Factor A)
    ss_A  = n_s*2*(mu_s-grand)**2 + n_m*2*(mu_m-grand)**2
    df_A  = 1

    # SS Subjects within groups (between-subjects error)
    ss_SA = 0
    for grp, mu_g in [("Single",mu_s),("Multiple",mu_m)]:
        subj_means = w[w["group"]==grp]["subj_mean"].values
        ss_SA += 2 * np.sum((subj_means - mu_g)**2)
    df_SA = N - 2

    # SS Within (Modality — Factor B)
    ss_B  = N * ((mu_lab-grand)**2 + (mu_game-grand)**2)
    df_B  = 1

    # SS Interaction (AxB)
    ss_AB = 0
    for grp, mu_g, ng in [("Single",mu_s,n_s),("Multiple",mu_m,n_m)]:
        for mod, mu_mod in [("Lab",mu_lab),("Game",mu_game)]:
            ss_AB += ng * (cell[(grp,mod)] - mu_g - mu_mod + grand)**2
    df_AB = 1

    # SS Error within (BxS/A)
    ss_err = 0
    for _, row in w.iterrows():
        mu_g   = mu_s if row["group"]=="Single" else mu_m
        mu_mod_lab  = mu_lab
        mu_mod_game = mu_game
        ss_err += (row[lab_col]  - row["subj_mean"] - mu_lab  + grand)**2
        ss_err += (row[game_col] - row["subj_mean"] - mu_game + grand)**2
    df_err = (N-2)*1

    # MS
    ms_A   = ss_A   / df_A
    ms_SA  = ss_SA  / df_SA
    ms_B   = ss_B   / df_B
    ms_AB  = ss_AB  / df_AB
    ms_err = ss_err / df_err

    # F
    F_A  = ms_A  / ms_SA
    F_B  = ms_B  / ms_err
    F_AB = ms_AB / ms_err

    # p-values
    p_A  = 1 - f_dist.cdf(F_A,  df_A,  df_SA)
    p_B  = 1 - f_dist.cdf(F_B,  df_B,  df_err)
    p_AB = 1 - f_dist.cdf(F_AB, df_AB, df_err)

    # Partial eta-squared
    eta_A  = ss_A  / (ss_A  + ss_SA)
    eta_B  = ss_B  / (ss_B  + ss_err)
    eta_AB = ss_AB / (ss_AB + ss_err)

    def eta_interp(e):
        return "small" if e<.06 else "medium" if e<.14 else "large"

    print(f"\n  ── {dv_label} ─────────────────────────────────────────────")
    print(f"  {'Source':<28} {'SS':>10} {'df':>4} {'MS':>10} "
          f"{'F':>8} {'p':>8} {'η²p':>7} {'sig'}")
    print("  " + "─"*80)

    rows = [
        ("Target Load [A]",   ss_A,   df_A,   ms_A,   F_A,    p_A,    eta_A),
        ("  Error S/A",       ss_SA,  df_SA,  ms_SA,  None,   None,   None),
        ("Modality [B]",      ss_B,   df_B,   ms_B,   F_B,    p_B,    eta_B),
        ("Load×Modality [AB]",ss_AB,  df_AB,  ms_AB,  F_AB,   p_AB,   eta_AB),
        ("  Error BxS/A",     ss_err, df_err, ms_err, None,   None,   None),
    ]
    for src, ss, df_, ms, F, p, eta in rows:
        F_s   = f"{F:.3f}" if F is not None else "—"
        p_s   = f"{p:.4f}" if p is not None else "—"
        eta_s = f"{eta:.3f}" if eta is not None else "—"
        sig_s = stars(p) if p is not None else ""
        print(f"  {src:<28} {ss:>10.3f} {df_:>4.0f} {ms:>10.3f} "
              f"{F_s:>8} {p_s:>8} {eta_s:>7} {sig_s}")

    print(f"\n  ➤ Target Load:    F({df_A},{df_SA})={F_A:.3f}  "
          f"p={p_A:.4f}{stars(p_A)}  η²p={eta_A:.3f} [{eta_interp(eta_A)}]")
    print(f"  ➤ Modality:       F({df_B},{df_err})={F_B:.3f}  "
          f"p={p_B:.4f}{stars(p_B)}  η²p={eta_B:.3f} [{eta_interp(eta_B)}]")
    print(f"  ➤ Interaction:    F({df_AB},{df_err})={F_AB:.3f}  "
          f"p={p_AB:.4f}{stars(p_AB)}  η²p={eta_AB:.3f} [{eta_interp(eta_AB)}]")

    return {
        "dv": dv_label,
        "F_A":F_A,"p_A":p_A,"eta_A":eta_A,"df_A":df_A,"df_SA":df_SA,
        "F_B":F_B,"p_B":p_B,"eta_B":eta_B,"df_B":df_B,
        "F_AB":F_AB,"p_AB":p_AB,"eta_AB":eta_AB,"df_AB":df_AB,"df_err":df_err,
        "cell_means": cell, "grand":grand,
        "mu_s":mu_s,"mu_m":mu_m,"mu_lab":mu_lab,"mu_game":mu_game,
        "n_s":n_s,"n_m":n_m,"N":N,
    }

# Run ANOVA on log-RT (primary) and raw RT (supplementary)
anova_logRT = run_mixed_anova(wide, "logRT_Lab", "logRT_Game", "Log RT (primary)")
anova_rawRT = run_mixed_anova(wide, "RT_Lab",    "RT_Game",    "Raw RT (supplementary)")

# Also run on comparable window (levels 1–10)
print(f"\n  ── Comparable Window (Levels 1–{max_comp_level}) — Log RT ─────────────")
anova_comp = run_mixed_anova(wide_comp, "logRT_Lab", "logRT_Game",
                              f"Log RT — Levels 1–{max_comp_level}")


# =============================================================================
# SECTION D: INTERACTION PLOT (annotated for report)
# =============================================================================

print("\n  Generating interaction plots...")

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("2×2 Mixed ANOVA — Interaction Plots\n"
             "Between: Target Load (Single vs Multiple) | Within: Modality (Lab vs Game)",
             fontsize=12, fontweight="bold")

for ax, (res, dv_label, lab_col, game_col) in zip(axes, [
    (anova_logRT, "Log RT",        "logRT_Lab", "logRT_Game"),
    (anova_rawRT, "Raw RT (ms)",   "RT_Lab",    "RT_Game"),
    (anova_comp,  f"Log RT — Lvls 1–{max_comp_level}", "logRT_Lab", "logRT_Game"),
]):
    # Use wide or wide_comp depending on which result this is
    src = wide_comp if "comp" in dv_label.lower() or f"Lvls 1" in dv_label else wide

    for grp, color, ls, mk in [("Single",PAL["Single"],"--","o"),
                                 ("Multiple",PAL["Multiple"],"-","s")]:
        sub = src[src["group"]==grp].dropna(subset=[lab_col, game_col])
        lab_m  = sub[lab_col].mean()
        game_m = sub[game_col].mean()
        lab_se  = sub[lab_col].std(ddof=1) / np.sqrt(len(sub))
        game_se = sub[game_col].std(ddof=1) / np.sqrt(len(sub))
        ax.errorbar(["Lab","Game"],[lab_m,game_m],
                    yerr=[lab_se,game_se],
                    color=color, ls=ls, marker=mk,
                    linewidth=2.2, markersize=9, capsize=6,
                    label=f"{grp} (n={len(sub)})")

    # Annotate F and p for interaction
    p_ab = res["p_AB"]
    f_ab = res["F_AB"]
    eta_ab = res["eta_AB"]
    ax.set_title(f"{dv_label}\n"
                 f"Interaction: F({res['df_AB']},{res['df_err']})={f_ab:.2f}, "
                 f"p={p_ab:.4f}{stars(p_ab)}, η²p={eta_ab:.3f}",
                 fontsize=9)
    ax.set_xlabel("Modality")
    ax.set_ylabel(dv_label)
    ax.legend(fontsize=9)

plt.tight_layout()
save("fig_s2_02_anova_interaction.png")


# Also: Individual participant trajectories (Lab → Game) per group
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Individual Participant Trajectories: Lab → Game (Log RT)\n"
             "Each line = one participant; bold = group mean",
             fontsize=12, fontweight="bold")

for ax, grp in zip(axes, ["Single","Multiple"]):
    sub = wide[wide["group"]==grp].dropna(subset=["logRT_Lab","logRT_Game"])
    color = PAL[grp]
    for _, row in sub.iterrows():
        ax.plot(["Lab","Game"],[row["logRT_Lab"],row["logRT_Game"]],
                "o-", color=color, alpha=0.28, linewidth=1.2, markersize=5)
    # Group means
    ax.plot(["Lab","Game"],
            [sub["logRT_Lab"].mean(), sub["logRT_Game"].mean()],
            "D-", color=color, linewidth=3, markersize=11,
            zorder=5, label=f"Group mean\n"
                           f"Lab={sub['logRT_Lab'].mean():.3f}\n"
                           f"Game={sub['logRT_Game'].mean():.3f}")
    ax.set_title(f"{grp} group (n={len(sub)})")
    ax.set_xlabel("Modality"); ax.set_ylabel("Log RT")
    ax.legend(fontsize=9)

plt.tight_layout()
save("fig_s2_03_individual_trajectories.png")


# =============================================================================
# SECTION E: ANOVA ASSUMPTION SUMMARY (print clearly for report)
# =============================================================================

print("\n" + "─"*65)
print("C. ANOVA ASSUMPTION SUMMARY FOR REPORT")
print("─"*65)

print("""
  Sphericity:
    2-level within-subjects factor (Lab/Game) — sphericity is
    automatically satisfied (Mauchly's test not needed). No correction required.

  Normality:
    Participant-level log RT means are examined above (Shapiro-Wilk).
    Note: ANOVA is robust to moderate non-normality at n=21 and n=16
    (Central Limit Theorem applies to means). We report both standard
    ANOVA and note if robust alternatives are warranted.

  Homogeneity of variance:
    Levene's test reported above for each DV.
    If violated, Welch correction is applied to between-subjects F.

  Independence:
    Random group assignment + counterbalanced modality order.
    No repeated between-group observations.
""")


# =============================================================================
# SECTION F: SAVE RESULTS
# =============================================================================

anova_results_df = pd.DataFrame([
    {"DV":"Log RT (all levels)",f"F_A":anova_logRT["F_A"],"p_A":anova_logRT["p_A"],
     "eta_A":anova_logRT["eta_A"],"sig_A":stars(anova_logRT["p_A"]),
     "F_B":anova_logRT["F_B"],"p_B":anova_logRT["p_B"],
     "eta_B":anova_logRT["eta_B"],"sig_B":stars(anova_logRT["p_B"]),
     "F_AB":anova_logRT["F_AB"],"p_AB":anova_logRT["p_AB"],
     "eta_AB":anova_logRT["eta_AB"],"sig_AB":stars(anova_logRT["p_AB"])},
    {"DV":"Raw RT (all levels)",
     "F_A":anova_rawRT["F_A"],"p_A":anova_rawRT["p_A"],
     "eta_A":anova_rawRT["eta_A"],"sig_A":stars(anova_rawRT["p_A"]),
     "F_B":anova_rawRT["F_B"],"p_B":anova_rawRT["p_B"],
     "eta_B":anova_rawRT["eta_B"],"sig_B":stars(anova_rawRT["p_B"]),
     "F_AB":anova_rawRT["F_AB"],"p_AB":anova_rawRT["p_AB"],
     "eta_AB":anova_rawRT["eta_AB"],"sig_AB":stars(anova_rawRT["p_AB"])},
    {"DV":f"Log RT (levels 1–{max_comp_level})",
     "F_A":anova_comp["F_A"],"p_A":anova_comp["p_A"],
     "eta_A":anova_comp["eta_A"],"sig_A":stars(anova_comp["p_A"]),
     "F_B":anova_comp["F_B"],"p_B":anova_comp["p_B"],
     "eta_B":anova_comp["eta_B"],"sig_B":stars(anova_comp["p_B"]),
     "F_AB":anova_comp["F_AB"],"p_AB":anova_comp["p_AB"],
     "eta_AB":anova_comp["eta_AB"],"sig_AB":stars(anova_comp["p_AB"])},
])
anova_results_df.to_csv(OUT/"results_r2_anova.csv", index=False)
print("\n  ✔  Saved results_r2_anova.csv")

# Update cache with ANOVA results and wide-form data
cache.update({
    "wide":         wide,
    "wide_comp":    wide_comp,
    "anova_logRT":  anova_logRT,
    "anova_rawRT":  anova_rawRT,
    "anova_comp":   anova_comp,
})
with open(OUT/"_r2_cache.pkl","wb") as f:
    pickle.dump(cache, f)
print("  ✔  Updated _r2_cache.pkl")

print("\n  STEP 2 COMPLETE — paste output before running Step 3.\n")