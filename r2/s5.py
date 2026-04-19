"""
REPORT 2 — STEP 5: Exploratory Hypotheses H_E2 and H_E4
  H_E2 — ICI Slowing Across Click Rank (Kruskal-Wallis + Dunn's + Red/White ICI)
  H_E4 — Accuracy Ceiling Limits Validity (RT vs SR/HR as validity criterion)
Requires outputs_r2/_r2_cache.pkl from Steps 1–4
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
from scipy.stats import (kruskal, mannwhitneyu, spearmanr, pearsonr,
                          shapiro, levene)
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")
OUT = Path("outputs_r2"); OUT.mkdir(exist_ok=True)

# ── Install scikit-posthocs for Dunn's test ───────────────────────────────────
import subprocess, sys
try:
    import scikit_posthocs as sp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "scikit-posthocs", "--quiet"])
    import scikit_posthocs as sp

with open(OUT / "_r2_cache.pkl", "rb") as f:
    cache = pickle.load(f)

lab_all          = cache["lab_all"]
game_all         = cache["game_all"]
ptpt_lab         = cache["ptpt_lab"]
ptpt_game_all    = cache["ptpt_game_all"]
ptpt_game_comp   = cache["ptpt_game_comp"]
ptpt_game_lvl1   = cache["ptpt_game_lvl1"]
merged_all       = cache["merged_all"]
merged_comp      = cache["merged_comp"]
merged_lvl1      = cache["merged_lvl1"]
validity_results = cache["validity_results"]
max_comp_level   = cache["max_comparable_level"]

PAL = {"Single": "#2D6A9F", "Multiple": "#C05621"}

def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

def fisher_r_ci(r, n):
    if n < 4: return np.nan, np.nan
    z  = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96*se), np.tanh(z + 1.96*se)

def save(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  {name}")

def section(title):
    print("\n" + "─" * 65)
    print(f"  {title}")
    print("─" * 65)

print("=" * 65)
print("STEP 5: EXPLORATORY HYPOTHESES H_E2 AND H_E4")
print("=" * 65)


# =============================================================================
# H_E2 — ICI SLOWING ACROSS CLICK RANK (Multiple Lab only)
# Hypothesis: ICI increases monotonically across click-pair rank (1→2 ... 4→5)
# consistent with FIT: most salient targets found first, serial search for rest.
# Also: ICI shorter on red trials than white trials.
#
# Method:
#   (1) Extract ICI per click transition per trial
#   (2) Kruskal-Wallis across 4 rank groups
#   (3) Dunn's post-hoc with Bonferroni correction
#   (4) Mann-Whitney U: red ICI vs white ICI
# =============================================================================

section("H_E2 — ICI Slowing Across Click Rank (Multiple Lab)")

print("""
  Hypothesis: ICI increases significantly across click-pair ranks
  (1→2 < 2→3 < 3→4 < 4→5), consistent with FIT serial search.
  ICI shorter on red (target-present) vs white (distractor) trials.
""")

# ── Build ICI dataset ─────────────────────────────────────────────────────────
ici_rows = []

multiple_lab = lab_all[lab_all["group"] == "Multiple"].copy()

for _, trial in multiple_lab.iterrows():
    clicks = trial["all_click_ms"]
    if not isinstance(clicks, list) or len(clicks) < 2:
        continue
    for k in range(len(clicks) - 1):
        ici_ms = clicks[k + 1] - clicks[k]
        if ici_ms <= 0:
            continue  # skip non-positive ICIs (data artefact)
        rank_label = f"{k+1}→{k+2}"
        ici_rows.append({
            "participant":  trial["participant"],
            "trial_n":      trial["trial_n"],
            "target_col":   trial["target_col"],
            "click_rank":   k + 1,       # 1=first→second, 2=second→third …
            "rank_label":   rank_label,
            "ICI_ms":       ici_ms,
            "log_ICI":      np.log(ici_ms),
        })

ici_df_all = pd.DataFrame(ici_rows)

# Restrict to ranks 1–4 (the four theoretically meaningful transitions
# in a 5-target trial: 1st→2nd click through 4th→5th click).
# Ranks 5+ arise from extra/erroneous clicks (n=3 observations total)
# and are excluded from all inferential tests.
ici_df = ici_df_all[ici_df_all["click_rank"] <= 4].copy()

print(f"  ICI observations (all ranks) : {len(ici_df_all)}")
print(f"  ICI observations (ranks 1–4) : {len(ici_df)}  "
      f"(excluded {len(ici_df_all)-len(ici_df)} extra-click artefacts)")
print(f"  Participants:                  {ici_df['participant'].nunique()}")
print(f"  Click ranks analysed:          {sorted(ici_df['click_rank'].unique())}")

# ── Descriptives by rank ──────────────────────────────────────────────────────
print("\n  ICI Descriptives by Click-Pair Rank:")
print(f"  {'Rank':<8} {'n':>6} {'M (ms)':>10} {'SD':>8} {'Mdn':>8}")
print("  " + "─" * 45)

rank_groups = {}
for rank in sorted(ici_df["click_rank"].unique()):
    sub = ici_df[ici_df["click_rank"] == rank]["ICI_ms"].values
    rank_groups[rank] = sub
    label = f"{rank}→{rank+1}"
    print(f"  {label:<8} {len(sub):>6} {sub.mean():>10.1f} "
          f"{sub.std(ddof=1):>8.1f} {np.median(sub):>8.1f}")

# ── Kruskal-Wallis test ───────────────────────────────────────────────────────
print("\n  ── Kruskal-Wallis Test (ICI ~ Click Rank) ───────────────────")

groups_list = [rank_groups[r] for r in sorted(rank_groups.keys())]
H, p_kw = kruskal(*groups_list)
df_kw   = len(rank_groups) - 1

# Effect size: eta-squared from H
n_total = sum(len(g) for g in groups_list)
eta2_kw = (H - df_kw + 1) / (n_total - df_kw)  # epsilon-squared approximation
eta2_kw = max(0, eta2_kw)

print(f"\n  H({df_kw}, N={n_total}) = {H:.3f}  p = {p_kw:.4f}{stars(p_kw)}")
print(f"  ε² (effect size) = {eta2_kw:.3f}  "
      f"({'small' if eta2_kw<.06 else 'medium' if eta2_kw<.14 else 'large'})")

if p_kw < .05:
    print("\n  ── Dunn's Post-Hoc (Bonferroni correction) ──────────────────")
    # scikit-posthocs Dunn's test
    ici_ph = ici_df[["click_rank", "ICI_ms"]].copy()
    ici_ph["click_rank"] = ici_ph["click_rank"].astype(str)

    dunn_p = sp.posthoc_dunn(
        ici_ph, val_col="ICI_ms", group_col="click_rank", p_adjust="bonferroni"
    )
    print("\n  Dunn p-values (Bonferroni-corrected):")
    print("  Rows/Cols = Click rank (1=1→2, 2=2→3, 3=3→4, 4=4→5)")
    print(dunn_p.round(4).to_string())

    # Pairwise direction check (adjacent ranks)
    print("\n  Adjacent-rank comparisons:")
    for r1, r2 in [(1,2),(2,3),(3,4)]:
        g1 = rank_groups[r1]
        g2 = rank_groups[r2]
        U, p_u = mannwhitneyu(g1, g2, alternative="less")  # H: rank r1 < rank r2
        r_eff  = 1 - 2*U / (len(g1)*len(g2))
        direction = "✓ increasing" if g2.mean() > g1.mean() else "✗ decreasing"
        print(f"    Rank {r1}→{r2}: M={g1.mean():.1f} → M={g2.mean():.1f}  "
              f"MW U={U:.0f}  p(one-tail)={p_u:.4f}{stars(p_u)}  "
              f"r={r_eff:.3f}  {direction}")
else:
    print("  Kruskal-Wallis ns — post-hoc not warranted.")

# ── Red vs White ICI ──────────────────────────────────────────────────────────
print("\n  ── Red vs White ICI (Mann-Whitney U) ────────────────────────")

red_ici   = ici_df[ici_df["target_col"] == "red"]["ICI_ms"].values
white_ici = ici_df[ici_df["target_col"] == "white"]["ICI_ms"].values

print(f"\n  Red   (target present): n={len(red_ici)}  "
      f"M={red_ici.mean():.1f}  SD={red_ici.std(ddof=1):.1f}  "
      f"Mdn={np.median(red_ici):.1f}")
print(f"  White (distractor):     n={len(white_ici)}  "
      f"M={white_ici.mean():.1f}  SD={white_ici.std(ddof=1):.1f}  "
      f"Mdn={np.median(white_ici):.1f}")
print(f"  Difference:  M={red_ici.mean()-white_ici.mean():.1f} ms  "
      f"({'Red faster ✓' if red_ici.mean() < white_ici.mean() else 'White faster'})")

U_rw, p_rw = mannwhitneyu(red_ici, white_ici, alternative="two-sided")
r_rw = 1 - 2*U_rw / (len(red_ici)*len(white_ici))
print(f"  Mann-Whitney U={U_rw:.0f}  p={p_rw:.4f}{stars(p_rw)}  r={r_rw:.3f}")

# Variance comparison
lev_stat, lev_p = levene(red_ici, white_ici, center="median")
print(f"  Levene (variance): W={lev_stat:.4f}  p={lev_p:.4f}  "
      f"{'✗ unequal' if lev_p < .05 else '✓ equal'} variance")
print(f"  SD ratio (red/white): {red_ici.std(ddof=1)/white_ici.std(ddof=1):.3f}")


# =============================================================================
# H_E4 — ACCURACY CEILING LIMITS VALIDITY
# Hypothesis: SR and HR show weaker concurrent validity than RT
# because they are ceiling-constrained (Mdn=100%).
# Test: compare validity correlations (Lab RT ~ Game RT) vs
#       (Lab RT ~ Game SR) and (Lab RT ~ Game HR)
# =============================================================================

section("H_E4 — Accuracy Ceiling Limits Validity")

print("""
  Hypothesis: Because Game SR and HR are ceiling-constrained (Mdn=100%),
  they will show weaker concurrent validity with Lab RT than Game RT does.
  False Alarms (not ceiling-constrained) may be more discriminating.
""")

# ── Descriptive: variance in each accuracy metric ────────────────────────────
print("  ── Accuracy Metric Variance (ceiling evidence) ──────────────")
print(f"\n  {'Metric':<20} {'Group':<10} {'M':>7} {'SD':>7} {'Mdn':>7} "
      f"{'% at ceiling':>14} {'skew':>7}")
print("  " + "─" * 72)

for grp in ["Single", "Multiple"]:
    sub = ptpt_game_all[ptpt_game_all["group"] == grp]
    for col, label, ceiling in [
        ("success_rate", "Success Rate",  100.0),
        ("hit_rate",     "Hit Rate",      100.0),
        ("false_alarms", "False Alarms",  0.0),
    ]:
        vals = sub[col].dropna().values
        pct_at_ceiling = 100 * (vals == ceiling).mean() if len(vals) else np.nan
        skew_val = stats.skew(vals) if len(vals) > 2 else np.nan
        print(f"  {label:<20} {grp:<10} {vals.mean():>7.2f} "
              f"{vals.std(ddof=1):>7.2f} {np.median(vals):>7.1f} "
              f"{pct_at_ceiling:>14.1f}% {skew_val:>7.3f}")

# ── Concurrent validity: Lab RT ~ Game accuracy metrics ──────────────────────
print("\n  ── Concurrent Validity: Lab RT vs Game Accuracy Metrics ─────")
print("  (Compare these r values to Lab RT vs Game RT validity above)")
print()

he4_results = []

# Build merged table with all accuracy metrics
merged_acc = pd.merge(
    ptpt_lab[["participant", "group", "RT_mean", "logRT_mean"]].rename(
        columns={"RT_mean": "lab_RT", "logRT_mean": "lab_logRT"}),
    ptpt_game_all[["participant", "group",
                   "RT_mean", "logRT_mean",
                   "success_rate", "hit_rate", "false_alarms"]].rename(
        columns={"RT_mean": "game_RT", "logRT_mean": "game_logRT"}),
    on=["participant", "group"]
).dropna()

print(f"  {'Criterion':<22} {'Group':<10} {'n':>4}  "
      f"{'r':>7}  {'p':>8}  {'sig':>5}  {'95%CI':>18}  {'interpretation'}")
print("  " + "─" * 85)

for grp in ["Single", "Multiple"]:
    sub = merged_acc[merged_acc["group"] == grp].copy()
    n   = len(sub)

    comparisons = [
        ("Game RT (raw)",    "game_RT",      "lab_RT"),
        ("Game RT (log)",    "game_logRT",   "lab_logRT"),
        ("Success Rate",     "success_rate", "lab_RT"),
        ("Hit Rate",         "hit_rate",     "lab_RT"),
        ("False Alarms",     "false_alarms", "lab_RT"),
    ]

    for label, game_col, lab_col in comparisons:
        tmp = sub[[game_col, lab_col]].dropna()
        nn  = len(tmp)
        if nn < 3:
            continue
        r, p = pearsonr(tmp[game_col], tmp[lab_col])
        ci_l, ci_h = fisher_r_ci(r, nn)

        # Critical r for significance
        t_crit = t_dist.ppf(.975, nn - 2)
        r_crit = t_crit / np.sqrt(t_crit**2 + nn - 2)

        interp = "n/a" if np.isnan(r) else (
            "large" if abs(r) >= .50 else
            "medium" if abs(r) >= .30 else
            "small" if abs(r) >= .10 else "negligible"
        )

        print(f"  {label:<22} {grp:<10} {nn:>4}  "
              f"{r:>+7.3f}  {p:>8.4f}  {stars(p):>5}  "
              f"[{ci_l:>+6.3f},{ci_h:>+6.3f}]  {interp}")

        he4_results.append({
            "Group": grp, "Criterion": label, "n": nn,
            "r": r, "p": p, "sig": stars(p),
            "ci_lo": ci_l, "ci_hi": ci_h,
        })

    print()  # blank line between groups

# ── Variance-explained comparison ────────────────────────────────────────────
print("  ── R² Comparison (variance explained in Lab RT) ─────────────")
print()
print(f"  {'Criterion':<22} {'Single r²':>12} {'Multiple r²':>14}")
print("  " + "─" * 52)

for label in ["Game RT (raw)", "Game RT (log)", "Success Rate",
              "Hit Rate", "False Alarms"]:
    r_s = next((x["r"] for x in he4_results
                if x["Group"] == "Single" and x["Criterion"] == label), np.nan)
    r_m = next((x["r"] for x in he4_results
                if x["Group"] == "Multiple" and x["Criterion"] == label), np.nan)
    r2_s = r_s**2 if not np.isnan(r_s) else np.nan
    r2_m = r_m**2 if not np.isnan(r_m) else np.nan
    print(f"  {label:<22} {r2_s:>12.4f} {r2_m:>14.4f}")


# =============================================================================
# PLOTS
# =============================================================================

section("PLOTS")

# ── Fig 1: ICI by click rank (bar + spaghetti) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("H_E2 — ICI Across Click-Pair Rank (Multiple Lab)\n"
             "Hypothesis: FIT predicts monotonic increase (most salient targets first)",
             fontsize=12, fontweight="bold")

# Bar chart (group means ± SE)
ax = axes[0]
ranks = sorted(ici_df["click_rank"].unique())
labels_r = [f"{r}→{r+1}" for r in ranks]
means_r  = [rank_groups[r].mean() for r in ranks]
ses_r    = [rank_groups[r].std(ddof=1)/np.sqrt(len(rank_groups[r])) for r in ranks]

colors_bars = [PAL["Multiple"]] * len(ranks)
bars = ax.bar(labels_r, means_r, yerr=ses_r,
              color=colors_bars, alpha=0.78,
              edgecolor="white", capsize=7, width=0.55)
ax.set_xlabel("Click-Pair Transition")
ax.set_ylabel("ICI (ms)")
ax.set_title(f"Mean ICI ± SE\nKruskal-Wallis H({df_kw})={H:.2f}, "
             f"p={p_kw:.4f}{stars(p_kw)}, ε²={eta2_kw:.3f}",
             fontsize=9)
# Annotate means
for bar_, m, se in zip(bars, means_r, ses_r):
    ax.text(bar_.get_x() + bar_.get_width()/2,
            m + se + 20, f"{m:.0f}", ha="center", va="bottom", fontsize=9)

# Spaghetti: per-participant mean ICI per rank
ax2 = axes[1]
ptpt_ici = (ici_df.groupby(["participant", "click_rank"])["ICI_ms"]
            .mean().reset_index())
ptpt_ici_w = ptpt_ici.pivot(index="participant", columns="click_rank",
                             values="ICI_ms").reset_index().dropna()

for _, row in ptpt_ici_w.iterrows():
    vals = [row[r] for r in ranks if r in ptpt_ici_w.columns]
    ax2.plot(labels_r[:len(vals)], vals,
             "o-", color=PAL["Multiple"], alpha=0.25, lw=1.2, ms=4)

# Group mean spaghetti line
group_means = [ptpt_ici_w[r].mean() for r in ranks if r in ptpt_ici_w.columns]
ax2.plot(labels_r[:len(group_means)], group_means,
         "D-", color=PAL["Multiple"], lw=3, ms=10, zorder=5, label="Group mean")
ax2.set_xlabel("Click-Pair Transition")
ax2.set_ylabel("ICI (ms)")
ax2.set_title("Per-Participant ICI Trajectories\n(Grey = individual; bold = mean)",
              fontsize=9)
ax2.legend(fontsize=9)

plt.tight_layout()
save("fig_s5_01_ici_rank.png")


# ── Fig 2: Red vs White ICI violin ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("H_E2 — Red vs White Trial ICI (Multiple Lab)\n"
             "Hypothesis: ICI shorter on red (colour-guided localisation)",
             fontsize=11, fontweight="bold")

ici_plot = ici_df[["target_col", "ICI_ms"]].copy()
ici_plot["target_col"] = ici_plot["target_col"].map(
    {"red": "Red (target)", "white": "White (distractor)"}
)
colors_vio = {"Red (target)": "#C0392B", "White (distractor)": "#7F8C8D"}

vp = ax.violinplot(
    [red_ici, white_ici], positions=[1, 2],
    showmedians=True, showextrema=False
)
for pc, color in zip(vp["bodies"], ["#C0392B", "#7F8C8D"]):
    pc.set_facecolor(color); pc.set_alpha(0.65)
vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)

# Overlay jitter
for x_pos, vals, color in [(1, red_ici, "#C0392B"), (2, white_ici, "#7F8C8D")]:
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(x_pos + jitter, vals, color=color, s=8, alpha=0.25, zorder=3)

ax.set_xticks([1, 2])
ax.set_xticklabels(["Red (target)", "White (distractor)"])
ax.set_ylabel("ICI (ms)")
ax.set_title(f"Red M={red_ici.mean():.0f} ms  vs  White M={white_ici.mean():.0f} ms\n"
             f"MW U={U_rw:.0f}  p={p_rw:.4f}{stars(p_rw)}  r={r_rw:.3f}",
             fontsize=9)

plt.tight_layout()
save("fig_s5_02_ici_red_white.png")


# ── Fig 3: H_E4 — Validity comparison bar chart ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("H_E4 — Accuracy Ceiling Limits Validity\n"
             "Pearson r: Lab RT vs each Game metric (bars = |r|, ✦ = significant)",
             fontsize=12, fontweight="bold")

criteria_order = ["Game RT (raw)", "Game RT (log)",
                  "Success Rate", "Hit Rate", "False Alarms"]
x_pos = np.arange(len(criteria_order))

for ax, grp in zip(axes, ["Single", "Multiple"]):
    r_vals   = []
    r2_vals  = []
    p_vals   = []
    for label in criteria_order:
        row = next((x for x in he4_results
                    if x["Group"] == grp and x["Criterion"] == label), None)
        r_vals.append(row["r"] if row else np.nan)
        r2_vals.append(row["r"]**2 if row else np.nan)
        p_vals.append(row["p"] if row else np.nan)

    color = PAL[grp]
    bar_colors = [color if not np.isnan(r) else "#CCCCCC" for r in r_vals]
    bars = ax.bar(x_pos, [abs(r) if not np.isnan(r) else 0 for r in r_vals],
                  color=bar_colors, alpha=0.78, edgecolor="white", width=0.55)

    # Mark direction and significance
    for i, (r, p, bar_) in enumerate(zip(r_vals, p_vals, bars)):
        if np.isnan(r): continue
        direction = "+" if r >= 0 else "−"
        sig_mark  = "✦" if p < .05 else ""
        ax.text(bar_.get_x() + bar_.get_width()/2,
                abs(r) + 0.02,
                f"{direction}{abs(r):.3f}{sig_mark}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Reference lines
    for v, ls in [(0.3,"--"),(0.5,"-.")]:
        ax.axhline(v, color="gray", ls=ls, lw=1, alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(criteria_order, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("|Pearson r|")
    ax.set_title(f"{grp} group (n={len(merged_acc[merged_acc['group']==grp])})\n"
                 "✦ = p<.05   --- = medium (r=.30)   -·- = large (r=.50)",
                 fontsize=9)

plt.tight_layout()
save("fig_s5_03_he4_validity_comparison.png")


# ── Fig 4: H_E4 — Accuracy distribution showing ceiling ─────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("H_E4 — Accuracy Ceiling Effect\n"
             "Participant-level mean distributions — ceiling at 100% constrains variance",
             fontsize=12, fontweight="bold")

metrics = [
    ("success_rate", "Success Rate (%)", 100),
    ("hit_rate",     "Hit Rate (%)",     100),
    ("false_alarms", "False Alarms",     None),
]

for col_i, (col, ylabel, ceiling) in enumerate(metrics):
    for row_i, grp in enumerate(["Single", "Multiple"]):
        ax   = axes[row_i, col_i]
        vals = ptpt_game_all[ptpt_game_all["group"] == grp][col].dropna().values
        color = PAL[grp]

        ax.hist(vals, bins=12, color=color, alpha=0.75,
                edgecolor="white", density=False)
        if ceiling is not None:
            ax.axvline(ceiling, color="crimson", ls="--", lw=2,
                       label=f"Ceiling={ceiling}")
        ax.set_xlabel(ylabel)
        ax.set_ylabel("Count")
        pct_ceil = 100*(vals == ceiling).mean() if ceiling is not None else np.nan
        title_extra = (f"  {pct_ceil:.1f}% at ceiling"
                       if not np.isnan(pct_ceil) else "")
        ax.set_title(f"{grp} — {ylabel}\n"
                     f"M={vals.mean():.1f}  SD={vals.std(ddof=1):.1f}"
                     f"{title_extra}",
                     fontsize=9)
        if ceiling is not None:
            ax.legend(fontsize=8)

plt.tight_layout()
save("fig_s5_04_accuracy_ceiling.png")


# =============================================================================
# SAVE RESULTS
# =============================================================================

he2_ici_summary = pd.DataFrame([
    {"click_rank": f"{r}→{r+1}",
     "n": len(rank_groups[r]),
     "M_ICI": rank_groups[r].mean(),
     "SD_ICI": rank_groups[r].std(ddof=1),
     "Mdn_ICI": np.median(rank_groups[r])}
    for r in sorted(rank_groups.keys())
])
he2_ici_summary["KW_H"]   = H
he2_ici_summary["KW_p"]   = p_kw
he2_ici_summary["KW_sig"] = stars(p_kw)
he2_ici_summary["epsilon2"] = eta2_kw

he2_ici_summary.to_csv(OUT / "results_r2_he2_ici.csv", index=False)

pd.DataFrame(he4_results).to_csv(OUT / "results_r2_he4_validity.csv", index=False)

print(f"\n  ✔  Saved results_r2_he2_ici.csv")
print(f"  ✔  Saved results_r2_he4_validity.csv")

# Update cache
cache.update({
    "ici_df":       ici_df,
    "he4_results":  he4_results,
    "he2_summary":  he2_ici_summary,
})
with open(OUT / "_r2_cache.pkl", "wb") as f:
    pickle.dump(cache, f)
print("  ✔  Updated _r2_cache.pkl")

print("\n  STEP 5 COMPLETE — paste full output before Step 6 (write-up).\n")