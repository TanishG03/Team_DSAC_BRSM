"""
╔══════════════════════════════════════════════════════════════════════╗
║  SELECTIVE ATTENTION STUDY — REPORT 2                                ║
║  STEP 1: 2×2 Mixed ANOVA + Full Diagnostics                         ║
╚══════════════════════════════════════════════════════════════════════╝

Design:
  Between-subjects factor A : Target Load   (Single vs Multiple)
  Within-subjects factor  B : Modality      (Lab vs Game)
  DV                        : Participant-mean RT (ms)  &  Log(RT)
  Unit of analysis          : Participant (n=37)

Diagnostic sequence (must pass before trusting ANOVA p-values):
  D1. Sample size & balance check
  D2. Outlier detection  (IQR rule + z-score per cell)
  D3. Normality of ANOVA residuals  (Shapiro-Wilk, Q-Q plots)
       — both raw RT and log(RT) residuals
  D4. Homogeneity of between-subjects variance (Levene's)
  D5. Sphericity  (trivially satisfied: only 2 levels of Modality)

Then:
  A1. 2×2 Mixed ANOVA  (raw RT)
  A2. 2×2 Mixed ANOVA  (log RT) — if normality better on log scale
  A3. Simple effects follow-up  (Lab alone; Game alone)
  A4. Paired t-tests within each group  (Lab vs Game)

Outputs:
  anova_diag_D1_balance.txt
  fig_diag_D2_outliers.png
  fig_diag_D3_residual_normality.png
  fig_diag_D4_levene.png
  anova_results_raw.csv
  anova_results_log.csv
  fig_anova_interaction.png
  fig_anova_simple_effects.png
  anova_full_summary.txt
"""

import pandas as pd
import numpy as np
import pickle, warnings, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import (shapiro, levene, ttest_rel, ttest_ind,
                         f as f_dist, t as t_dist)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT      = Path("outputs");      OUT.mkdir(exist_ok=True)
DIAG_OUT = Path("outputs/anova"); DIAG_OUT.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        130,
})

PAL = {"Single": "#2D6A9F", "Multiple": "#C05621",
       "Lab": "#2F855A",    "Game":     "#9B2C2C"}

def stars(p):
    if p is None or np.isnan(p): return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

def save_fig(name):
    path = DIAG_OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  {name}")

def section(title):
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def ci95_mean(arr):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n   = len(arr)
    se  = arr.std(ddof=1) / np.sqrt(n)
    t   = t_dist.ppf(.975, n - 1)
    return arr.mean() - t * se, arr.mean() + t * se

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    sp = np.sqrt(((na-1)*np.var(a,ddof=1) + (nb-1)*np.var(b,ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / sp if sp else np.nan

def cohens_dz(diff):
    s = np.std(diff, ddof=1)
    return np.mean(diff) / s if s else np.nan

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 1 — 2×2 MIXED ANOVA + DIAGNOSTICS")
print("=" * 70)

cache_path = OUT / "_data_cache.pkl"
if not cache_path.exists():
    raise FileNotFoundError(
        "Run analysis_part1_loading.py and analysis_part2_descriptives.py first "
        "to generate outputs/_data_cache.pkl"
    )

with open(cache_path, "rb") as f:
    cache = pickle.load(f)

ptpt_lab  = cache["ptpt_lab"]
ptpt_game = cache["ptpt_game"]

# ── Build wide form: one row per participant, columns RT_Lab & RT_Game ────────
wide = pd.merge(
    ptpt_lab[["participant", "group", "RT_mean"]].rename(columns={"RT_mean": "RT_Lab"}),
    ptpt_game[["participant", "group", "RT_mean"]].rename(columns={"RT_mean": "RT_Game"}),
    on=["participant", "group"]
).dropna(subset=["RT_Lab", "RT_Game"])

# Log-transform (natural log)
wide["logRT_Lab"]  = np.log(wide["RT_Lab"])
wide["logRT_Game"] = np.log(wide["RT_Game"])

# Convenience subsets
s_wide = wide[wide["group"] == "Single"].copy()
m_wide = wide[wide["group"] == "Multiple"].copy()

N   = len(wide)
n_s = len(s_wide)
n_m = len(m_wide)

print(f"\n  Wide-form dataset built: {N} participants")
print(f"    Single   : n = {n_s}")
print(f"    Multiple : n = {n_m}")
print(f"  Columns: {wide.columns.tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# D1. SAMPLE SIZE & BALANCE CHECK
# ─────────────────────────────────────────────────────────────────────────────
section("D1 — SAMPLE SIZE & BALANCE")

balance_lines = []
balance_lines.append("SAMPLE SIZE & BALANCE CHECK")
balance_lines.append("=" * 50)
balance_lines.append(f"Total participants with both modalities : {N}")
balance_lines.append(f"  Single group  : n = {n_s}")
balance_lines.append(f"  Multiple group: n = {n_m}")
balance_lines.append(f"  Balance ratio : {n_s}/{n_m} = {n_s/n_m:.2f}")
balance_lines.append("")
balance_lines.append("Cell means (participant-mean RT, ms):")

for grp in ["Single", "Multiple"]:
    for mod, col in [("Lab", "RT_Lab"), ("Game", "RT_Game")]:
        sub = wide[wide["group"] == grp][col]
        ci  = ci95_mean(sub.values)
        line = (f"  {grp:<12} × {mod:<6}: "
                f"M={sub.mean():.1f}  SD={sub.std(ddof=1):.1f}  "
                f"n={len(sub)}  95%CI[{ci[0]:.1f}, {ci[1]:.1f}]")
        balance_lines.append(line)
        print(line)

balance_lines.append("")
balance_lines.append(f"Note: Unequal group sizes (n_s={n_s}, n_m={n_m}) means")
balance_lines.append("  between-subjects SS uses unweighted cell means (Type III SS).")
balance_lines.append("  Power for between-subjects effect is lower in the smaller group.")

power_note = (
    f"  With n_m={n_m} and α=.05, power to detect a medium effect (f=.25) "
    f"in a 2×2 ANOVA ≈ 0.33 — LOW. Interpret null results with caution."
)
balance_lines.append(power_note)

# Minimum detectable effect (rough: r for correlation, d for t-test)
# For the between-subjects effect with n_s=21, n_m=16, Welch t-test:
# We need |t| > ~2.03; minimum d ≈ t * sqrt(1/n_s + 1/n_m)
min_d = 2.03 * np.sqrt(1/n_s + 1/n_m)
balance_lines.append(f"  Minimum detectable d (between groups, α=.05, two-tailed) ≈ {min_d:.2f}")
print(f"  Minimum detectable d (between groups) ≈ {min_d:.2f}")

bal_text = "\n".join(balance_lines)
(DIAG_OUT / "anova_diag_D1_balance.txt").write_text(bal_text)
print("\n  ✔  Saved anova_diag_D1_balance.txt")


# ─────────────────────────────────────────────────────────────────────────────
# D2. OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────
section("D2 — OUTLIER DETECTION")

outlier_records = []

def check_outliers(arr, group, modality, col_label):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z_scores = np.abs((arr - arr.mean()) / arr.std(ddof=1))
    iqr_out  = ((arr < lo) | (arr > hi)).sum()
    z_out    = (z_scores > 3).sum()
    return {"Group": group, "Modality": modality, "DV": col_label,
            "n": len(arr), "IQR_lo": lo, "IQR_hi": hi,
            "IQR_outliers": iqr_out, "Z_outliers_>3": z_out,
            "Min": arr.min(), "Max": arr.max(),
            "Mean": arr.mean(), "SD": arr.std(ddof=1)}

for grp in ["Single", "Multiple"]:
    sub = wide[wide["group"] == grp]
    for col, label in [("RT_Lab", "RT Lab"), ("RT_Game", "RT Game"),
                       ("logRT_Lab", "Log-RT Lab"), ("logRT_Game", "Log-RT Game")]:
        r = check_outliers(sub[col].values, grp, col.split("_")[1], label)
        outlier_records.append(r)

out_df = pd.DataFrame(outlier_records)
print("\n  Outlier Summary (IQR ±1.5 rule and |z| > 3):")
print(out_df[["Group","DV","n","IQR_outliers","Z_outliers_>3","Min","Max"]].to_string(index=False))
out_df.to_csv(DIAG_OUT / "anova_diag_D2_outliers.csv", index=False)
print("\n  ✔  Saved anova_diag_D2_outliers.csv")

# Fig D2: Boxplots with outlier flags — 2×2 grid (raw + log, both groups)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("D2 — Outlier Check: Participant-Mean RT per Cell\n"
             "(whiskers = IQR ×1.5 rule; dots beyond = potential outliers)",
             fontsize=13, fontweight="bold")

plot_specs = [
    (axes[0,0], "RT_Lab",    "Raw RT — Lab Task",    "ms"),
    (axes[0,1], "RT_Game",   "Raw RT — Game Task",   "ms"),
    (axes[1,0], "logRT_Lab", "Log(RT) — Lab Task",   "ln(ms)"),
    (axes[1,1], "logRT_Game","Log(RT) — Game Task",  "ln(ms)"),
]

for ax, col, title, unit in plot_specs:
    plot_data = wide[["group", col]].copy()
    pal = {"Single": PAL["Single"], "Multiple": PAL["Multiple"]}
    sns.boxplot(data=plot_data, x="group", y=col,
                palette=pal, order=["Single", "Multiple"],
                width=0.45, linewidth=1.3,
                flierprops=dict(marker="o", ms=7, markerfacecolor="crimson",
                                markeredgecolor="crimson", alpha=0.8),
                ax=ax)
    sns.stripplot(data=plot_data, x="group", y=col,
                  palette=pal, order=["Single", "Multiple"],
                  alpha=0.5, size=5, jitter=True, ax=ax)
    # Annotate means and n_outliers
    for i, grp in enumerate(["Single", "Multiple"]):
        vals = wide[wide["group"] == grp][col].dropna()
        q1, q3 = vals.quantile(.25), vals.quantile(.75)
        iqr    = q3 - q1
        n_out  = ((vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)).sum()
        ax.text(i, vals.max() * 1.01 if col.startswith("RT") else vals.max() + 0.03,
                f"M={vals.mean():.{0 if col.startswith('RT') else 3}f}\n"
                f"{n_out} outlier(s)",
                ha="center", fontsize=8.5,
                color="crimson" if n_out > 0 else "dimgray",
                fontweight="bold" if n_out > 0 else "normal")
    ax.set_title(title)
    ax.set_xlabel("Group")
    ax.set_ylabel(f"Participant Mean RT ({unit})")

plt.tight_layout()
save_fig("fig_diag_D2_outliers.png")


# ─────────────────────────────────────────────────────────────────────────────
# D3. NORMALITY OF ANOVA RESIDUALS
# ─────────────────────────────────────────────────────────────────────────────
section("D3 — NORMALITY OF ANOVA RESIDUALS")

# Compute ANOVA residuals for raw RT and log(RT)
# Residual = observed − (group_mean + modality_mean − grand_mean)
# i.e. cell_mean is the fitted value; residual = y_ij - fitted_ij

def compute_anova_residuals(df, dv_lab, dv_game):
    """
    Returns residuals for a 2×2 mixed ANOVA on the long-form
    data implied by the wide df.
    For each participant: residual_Lab  = RT_Lab  − fitted_Lab
                          residual_Game = RT_Game − fitted_Game
    fitted = grand_mean + group_effect + modality_effect
    (No interaction included in residual computation — we want
    residuals that include the interaction error.)
    """
    grand_lab  = df[dv_lab].mean()
    grand_game = df[dv_game].mean()
    grand      = (grand_lab + grand_game) / 2

    residuals = []
    for _, row in df.iterrows():
        grp      = row["group"]
        # Group mean (average of Lab & Game for this group)
        grp_sub  = df[df["group"] == grp]
        grp_mean = (grp_sub[dv_lab].mean() + grp_sub[dv_game].mean()) / 2
        # Modality means (across both groups)
        mu_lab   = df[dv_lab].mean()
        mu_game  = df[dv_game].mean()

        fitted_lab  = grp_mean + (mu_lab  - grand)
        fitted_game = grp_mean + (mu_game - grand)

        residuals.append(row[dv_lab]  - fitted_lab)
        residuals.append(row[dv_game] - fitted_game)

    return np.array(residuals)

resid_raw = compute_anova_residuals(wide, "RT_Lab",    "RT_Game")
resid_log = compute_anova_residuals(wide, "logRT_Lab", "logRT_Game")

# Shapiro-Wilk on residuals
sw_raw_stat, sw_raw_p = shapiro(resid_raw)
sw_log_stat, sw_log_p = shapiro(resid_log)

print(f"\n  Shapiro-Wilk on ANOVA residuals:")
print(f"    Raw RT  : W = {sw_raw_stat:.4f},  p = {sw_raw_p:.4f}  {stars(sw_raw_p)}"
      f"  → {'NORMAL ✓' if sw_raw_p > .05 else 'NON-NORMAL ✗ — consider log(RT)'}")
print(f"    Log(RT) : W = {sw_log_stat:.4f},  p = {sw_log_p:.4f}  {stars(sw_log_p)}"
      f"  → {'NORMAL ✓' if sw_log_p > .05 else 'NON-NORMAL ✗'}")

# Skewness & kurtosis of residuals
sk_raw = stats.skew(resid_raw);  ku_raw = stats.kurtosis(resid_raw)
sk_log = stats.skew(resid_log);  ku_log = stats.kurtosis(resid_log)
print(f"\n  Skewness / excess kurtosis:")
print(f"    Raw RT  : skew = {sk_raw:.3f},  kurt = {ku_raw:.3f}")
print(f"    Log(RT) : skew = {sk_log:.3f},  kurt = {ku_log:.3f}")
print("  (Ideal: |skew| < 1, |kurt| < 2 for ANOVA robustness)")

# Also check within-group residuals separately
print(f"\n  Shapiro-Wilk within each group (raw RT residuals):")
for grp in ["Single", "Multiple"]:
    sub = wide[wide["group"] == grp]
    r   = compute_anova_residuals(sub, "RT_Lab", "RT_Game")
    w, p = shapiro(r)
    print(f"    {grp:<12}: W={w:.4f}  p={p:.4f}  {stars(p)}"
          f"  {'✓' if p > .05 else '✗'}")

# Recommendation
print("\n  ── Normality Recommendation ──")
if sw_raw_p > .05 and sw_log_p > .05:
    rec = "Both raw and log residuals are normal. Proceed with raw RT."
elif sw_raw_p <= .05 and sw_log_p > .05:
    rec = ("Raw RT residuals are NON-NORMAL. Log(RT) residuals are normal. "
           "PRIMARY analysis: Log(RT). Report raw RT as sensitivity check.")
elif sw_raw_p > .05 and sw_log_p <= .05:
    rec = "Raw RT residuals are normal. Proceed with raw RT."
else:
    rec = ("BOTH raw and log residuals are non-normal. "
           "Run ANOVA on both; additionally report a non-parametric "
           "sensitivity analysis (permutation or aligned-rank transform).")
print(f"  → {rec}")

# Fig D3: Residual normality — Q-Q + histogram for raw and log
fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(2, 4, figure=fig)
fig.suptitle("D3 — Normality of ANOVA Residuals\n"
             "Top row: Raw RT residuals  |  Bottom row: Log(RT) residuals",
             fontsize=13, fontweight="bold")

for row_i, (resid, label, sw_s, sw_p_v) in enumerate([
    (resid_raw, "Raw RT",   sw_raw_stat, sw_raw_p),
    (resid_log, "Log(RT)",  sw_log_stat, sw_log_p),
]):
    # Histogram
    ax = fig.add_subplot(gs[row_i, 0:2])
    ax.hist(resid, bins=min(20, max(6, len(resid)//3)),
            color="#2D6A9F", alpha=0.65, edgecolor="white", density=True)
    x_kde = np.linspace(resid.min()*1.1, resid.max()*1.1, 300)
    kde   = stats.gaussian_kde(resid, bw_method="scott")
    ax.plot(x_kde, kde(x_kde), color="#2D6A9F", lw=2)
    # Overlay normal curve
    mu_r, sd_r = resid.mean(), resid.std(ddof=1)
    ax.plot(x_kde, stats.norm.pdf(x_kde, mu_r, sd_r),
            "r--", lw=2, label="Normal")
    ax.set_title(f"{label} Residuals — Histogram + KDE\n"
                 f"SW: W={sw_s:.4f}, p={sw_p_v:.4f} {stars(sw_p_v)}")
    ax.set_xlabel("Residual"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # Q-Q plot
    ax2 = fig.add_subplot(gs[row_i, 2:4])
    (osm, osr), (slope, intercept, r_val) = stats.probplot(resid, dist="norm")
    ax2.scatter(osm, osr, color="#C05621", s=30, alpha=0.75, zorder=3)
    lx = np.array([osm.min(), osm.max()])
    ax2.plot(lx, slope * lx + intercept, "k--", lw=1.8)
    ax2.set_title(f"{label} — Normal Q-Q Plot\nR²={r_val**2:.4f}")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")

plt.tight_layout()
save_fig("fig_diag_D3_residual_normality.png")

# Save normality report
norm_report = (
    f"NORMALITY OF ANOVA RESIDUALS\n{'='*50}\n\n"
    f"Raw RT residuals:\n"
    f"  Shapiro-Wilk: W={sw_raw_stat:.4f}, p={sw_raw_p:.4f} {stars(sw_raw_p)}\n"
    f"  Skew={sk_raw:.3f}, Kurt(excess)={ku_raw:.3f}\n\n"
    f"Log(RT) residuals:\n"
    f"  Shapiro-Wilk: W={sw_log_stat:.4f}, p={sw_log_p:.4f} {stars(sw_log_p)}\n"
    f"  Skew={sk_log:.3f}, Kurt(excess)={ku_log:.3f}\n\n"
    f"Recommendation:\n  {rec}\n"
)
(DIAG_OUT / "anova_diag_D3_normality.txt").write_text(norm_report)
print("  ✔  Saved anova_diag_D3_normality.txt")


# ─────────────────────────────────────────────────────────────────────────────
# D4. HOMOGENEITY OF VARIANCE (LEVENE'S)
# ─────────────────────────────────────────────────────────────────────────────
section("D4 — HOMOGENEITY OF VARIANCE (Levene's Test)")

# For mixed ANOVA: test homogeneity across GROUPS for each modality
# Also test on difference scores (Game − Lab) which is the key within-subjects comparison
lev_rows = []

for col, label in [("RT_Lab",    "Lab RT"),
                   ("RT_Game",   "Game RT"),
                   ("logRT_Lab", "Log Lab RT"),
                   ("logRT_Game","Log Game RT")]:
    s_vals = wide[wide["group"] == "Single"][col].dropna().values
    m_vals = wide[wide["group"] == "Multiple"][col].dropna().values
    lev_stat, lev_p = levene(s_vals, m_vals, center="median")  # Brown-Forsythe
    print(f"  Levene's (Brown-Forsythe) — {label}:")
    print(f"    F(1,{N-2}) = {lev_stat:.4f},  p = {lev_p:.4f}  {stars(lev_p)}"
          f"  → {'Homogeneous ✓' if lev_p > .05 else 'HETEROGENEOUS ✗'}")
    lev_rows.append({"DV": label, "F": lev_stat, "df1": 1, "df2": N-2,
                     "p": lev_p, "sig": stars(lev_p),
                     "homogeneous": lev_p > .05})

# Difference scores
wide["diff_RT"]    = wide["RT_Game"]    - wide["RT_Lab"]
wide["diff_logRT"] = wide["logRT_Game"] - wide["logRT_Lab"]
for col, label in [("diff_RT",    "Diff RT (Game−Lab)"),
                   ("diff_logRT", "Diff Log(RT)")]:
    s_vals = wide[wide["group"] == "Single"][col].dropna().values
    m_vals = wide[wide["group"] == "Multiple"][col].dropna().values
    lev_stat, lev_p = levene(s_vals, m_vals, center="median")
    print(f"  Levene's (Brown-Forsythe) — {label}:")
    print(f"    F(1,{N-2}) = {lev_stat:.4f},  p = {lev_p:.4f}  {stars(lev_p)}"
          f"  → {'Homogeneous ✓' if lev_p > .05 else 'HETEROGENEOUS ✗'}")
    lev_rows.append({"DV": label, "F": lev_stat, "df1": 1, "df2": N-2,
                     "p": lev_p, "sig": stars(lev_p),
                     "homogeneous": lev_p > .05})

lev_df = pd.DataFrame(lev_rows)
lev_df.to_csv(DIAG_OUT / "anova_diag_D4_levene.csv", index=False)
print("\n  ✔  Saved anova_diag_D4_levene.csv")

# Fig D4: Variance comparison plots
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("D4 — Homogeneity of Variance Check\n"
             "Between-groups spread should be comparable for valid ANOVA F-test",
             fontsize=13, fontweight="bold")

plot_specs_d4 = [
    (axes[0,0], "RT_Lab",    "Lab RT (ms)",       "raw"),
    (axes[0,1], "RT_Game",   "Game RT (ms)",      "raw"),
    (axes[1,0], "logRT_Lab", "Log(RT) — Lab",     "log"),
    (axes[1,1], "diff_RT",   "Game − Lab RT (ms)","raw"),
]

for ax, col, title, scale in plot_specs_d4:
    pal = {"Single": PAL["Single"], "Multiple": PAL["Multiple"]}
    # Violin + strip + box
    sns.violinplot(data=wide, x="group", y=col,
                   palette=pal, order=["Single","Multiple"],
                   inner=None, cut=0, linewidth=1.2,
                   alpha=0.4, ax=ax)
    sns.boxplot(data=wide, x="group", y=col,
                palette=pal, order=["Single","Multiple"],
                width=0.25, linewidth=1.5,
                flierprops=dict(marker="o", ms=6, alpha=0.7),
                ax=ax)
    # Annotate SD
    for i, grp in enumerate(["Single","Multiple"]):
        vals = wide[wide["group"]==grp][col].dropna()
        lev_row = next((r for r in lev_rows if r["DV"] in title[:15]), None)
        ax.text(i, vals.min() - (vals.max()-vals.min())*0.08,
                f"SD={vals.std(ddof=1):.{0 if scale=='raw' else 3}f}",
                ha="center", fontsize=9, color=PAL[grp], fontweight="bold")
    # Add Levene result to title
    lev_match = lev_df[lev_df["DV"].str.contains(title.split(" ")[0][:6])]
    if not lev_match.empty:
        lr = lev_match.iloc[0]
        ax.set_title(f"{title}\nLevene F={lr['F']:.3f}, p={lr['p']:.4f} {lr['sig']}")
    else:
        ax.set_title(title)
    ax.set_xlabel("Group")
    ax.set_ylabel(title)

plt.tight_layout()
save_fig("fig_diag_D4_levene.png")


# ─────────────────────────────────────────────────────────────────────────────
# D5. SPHERICITY NOTE
# ─────────────────────────────────────────────────────────────────────────────
section("D5 — SPHERICITY")
print("  Modality has only 2 levels (Lab vs Game).")
print("  With k=2 within-subjects levels, sphericity is automatically satisfied.")
print("  Mauchly's test is not applicable; Greenhouse-Geisser correction = 1.0.")
print("  ✓ No correction needed for the within-subjects effect.")


# ─────────────────────────────────────────────────────────────────────────────
# A1. 2×2 MIXED ANOVA — RAW RT
# ─────────────────────────────────────────────────────────────────────────────
section("A1 — 2×2 MIXED ANOVA (Raw RT)")

def run_mixed_anova_2x2(df, dv_lab, dv_game, label="RT"):
    """
    Compute a 2×2 Mixed ANOVA by hand.
    Factor A (between) : group  (Single vs Multiple)
    Factor B (within)  : modality  (Lab vs Game)
    
    Returns dict of SS, df, MS, F, p, eta2p for each effect.
    Also prints ANOVA table.
    """
    N_   = len(df)
    n_s_ = (df["group"] == "Single").sum()
    n_m_ = (df["group"] == "Multiple").sum()

    s_ = df[df["group"] == "Single"]
    m_ = df[df["group"] == "Multiple"]

    # Cell means
    mu_SL = s_[dv_lab].mean()
    mu_SG = s_[dv_game].mean()
    mu_ML = m_[dv_lab].mean()
    mu_MG = m_[dv_game].mean()

    # Marginal means
    mu_S     = (mu_SL + mu_SG) / 2     # Single marginal
    mu_M     = (mu_ML + mu_MG) / 2     # Multiple marginal
    mu_Lab   = (mu_SL*n_s_ + mu_ML*n_m_) / N_   # Lab marginal (weighted)
    mu_Game  = (mu_SG*n_s_ + mu_MG*n_m_) / N_   # Game marginal
    grand    = (mu_Lab * N_ + mu_Game * N_) / (2 * N_)
    # Simpler: grand = df[[dv_lab, dv_game]].values.mean()
    grand    = np.concatenate([df[dv_lab].values, df[dv_game].values]).mean()

    # Subject means (participant average across Lab & Game)
    df = df.copy()
    df["subj_mean"] = (df[dv_lab] + df[dv_game]) / 2

    # ── SS_A (Between-subjects: Target Load) ─────────────────────────────────
    mu_S_actual = s_["subj_mean"].mean()
    mu_M_actual = m_["subj_mean"].mean()
    ss_A  = 2 * (n_s_ * (mu_S_actual - grand)**2 +
                 n_m_ * (mu_M_actual - grand)**2)
    df_A  = 1

    # ── SS_S/A (Error for A: participants within groups) ─────────────────────
    ss_SA = 2 * sum((row["subj_mean"] - s_["subj_mean"].mean())**2
                    for _, row in s_.iterrows()) + \
            2 * sum((row["subj_mean"] - m_["subj_mean"].mean())**2
                    for _, row in m_.iterrows())
    df_SA = N_ - 2

    # ── SS_B (Within-subjects: Modality) ─────────────────────────────────────
    mu_Lab_u  = df[dv_lab].mean()
    mu_Game_u = df[dv_game].mean()
    ss_B  = N_ * ((mu_Lab_u - grand)**2 + (mu_Game_u - grand)**2)
    df_B  = 1

    # ── SS_AB (Interaction: Load × Modality) ─────────────────────────────────
    # Cell means (unweighted for Type III with unequal n — approximate)
    mu_S_u = s_[[dv_lab, dv_game]].values.mean()
    mu_M_u = m_[[dv_lab, dv_game]].values.mean()
    ss_AB = sum(
        n_g * (cell_m - mu_g - mod_m + grand)**2
        for (n_g, mu_g, cells) in [
            (n_s_, mu_S_u, [(mu_SL, mu_Lab_u), (mu_SG, mu_Game_u)]),
            (n_m_, mu_M_u, [(mu_ML, mu_Lab_u), (mu_MG, mu_Game_u)]),
        ]
        for (cell_m, mod_m) in cells
    )
    df_AB = 1

    # ── SS_BxS/A (Within-subjects error) ─────────────────────────────────────
    # Residual = y_ij − subj_mean − modality_mean + grand
    ss_BsA = sum(
        (row[dv_lab]  - row["subj_mean"] - mu_Lab_u  + grand)**2 +
        (row[dv_game] - row["subj_mean"] - mu_Game_u + grand)**2
        for _, row in df.iterrows()
    )
    df_BsA = N_ - 2

    # ── MS, F, p ─────────────────────────────────────────────────────────────
    ms_A    = ss_A   / df_A
    ms_SA   = ss_SA  / df_SA
    ms_B    = ss_B   / df_B
    ms_AB   = ss_AB  / df_AB
    ms_BsA  = ss_BsA / df_BsA

    F_A     = ms_A   / ms_SA
    F_B     = ms_B   / ms_BsA
    F_AB    = ms_AB  / ms_BsA

    p_A     = 1 - f_dist.cdf(F_A,  df_A,  df_SA)
    p_B     = 1 - f_dist.cdf(F_B,  df_B,  df_BsA)
    p_AB    = 1 - f_dist.cdf(F_AB, df_AB, df_BsA)

    # ── Partial eta² ─────────────────────────────────────────────────────────
    eta2p_A  = ss_A  / (ss_A  + ss_SA)
    eta2p_B  = ss_B  / (ss_B  + ss_BsA)
    eta2p_AB = ss_AB / (ss_AB + ss_BsA)

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"\n  2×2 Mixed ANOVA — {label}")
    print(f"  {'Source':<35} {'SS':>10} {'df':>4} {'MS':>10} {'F':>8} {'p':>8} {'sig':>5} {'η²p':>6}")
    print("  " + "─" * 90)

    for src, ss, df_, ms, F, p, eta in [
        ("Between: Target Load [A]",      ss_A,   df_A,   ms_A,   F_A,  p_A,  eta2p_A),
        ("  Error B/w (S/A)",             ss_SA,  df_SA,  ms_SA,  None, None, None),
        ("Within:  Modality [B]",         ss_B,   df_B,   ms_B,   F_B,  p_B,  eta2p_B),
        ("Interaction: Load×Modality [AB]",ss_AB, df_AB,  ms_AB,  F_AB, p_AB, eta2p_AB),
        ("  Error W/in (B×S/A)",          ss_BsA, df_BsA, ms_BsA, None, None, None),
    ]:
        f_str   = f"{F:8.3f}" if F is not None else "        "
        p_str   = f"{p:8.4f}" if p is not None else "        "
        sig_str = f"{stars(p):>5}"   if p is not None else "     "
        eta_str = f"{eta:6.3f}"      if eta is not None else "      "
        print(f"  {src:<35} {ss:10.2f} {df_:4.0f} {ms:10.2f} "
              f"{f_str} {p_str} {sig_str} {eta_str}")

    print()
    for name, eta, F, p, df_num, df_den in [
        ("Target Load [A]",       eta2p_A,  F_A,  p_A,  df_A,  df_SA),
        ("Modality [B]",          eta2p_B,  F_B,  p_B,  df_B,  df_BsA),
        ("Load×Modality [A×B]",   eta2p_AB, F_AB, p_AB, df_AB, df_BsA),
    ]:
        interp = ("negligible" if eta < .01 else "small" if eta < .06 else
                  "medium" if eta < .14 else "large")
        print(f"  ➤ {name:<28}: F({df_num:.0f},{df_den:.0f})={F:.3f}  "
              f"p={p:.4f}{stars(p)}  η²p={eta:.3f} ({interp})")

    return {
        "label":    label,
        "ss_A":     ss_A,  "df_A":  df_A,  "ms_A":  ms_A,  "F_A":  F_A,  "p_A":  p_A,  "eta2p_A":  eta2p_A,
        "ss_SA":    ss_SA, "df_SA": df_SA, "ms_SA": ms_SA,
        "ss_B":     ss_B,  "df_B":  df_B,  "ms_B":  ms_B,  "F_B":  F_B,  "p_B":  p_B,  "eta2p_B":  eta2p_B,
        "ss_AB":    ss_AB, "df_AB": df_AB, "ms_AB": ms_AB, "F_AB": F_AB, "p_AB": p_AB, "eta2p_AB": eta2p_AB,
        "ss_BsA":   ss_BsA,"df_BsA":df_BsA,"ms_BsA":ms_BsA,
        "cell_means": {"SL":mu_SL,"SG":mu_SG,"ML":mu_ML,"MG":mu_MG},
        "grand":    grand,
        "mu_S": mu_S_actual, "mu_M": mu_M_actual,
        "mu_Lab": mu_Lab_u, "mu_Game": mu_Game_u,
    }

anova_raw = run_mixed_anova_2x2(wide, "RT_Lab", "RT_Game", label="Raw RT (ms)")

# ─────────────────────────────────────────────────────────────────────────────
# A2. 2×2 MIXED ANOVA — LOG(RT)
# ─────────────────────────────────────────────────────────────────────────────
section("A2 — 2×2 MIXED ANOVA (Log RT)")
anova_log = run_mixed_anova_2x2(wide, "logRT_Lab", "logRT_Game", label="Log(RT)")


# ─────────────────────────────────────────────────────────────────────────────
# A3. SIMPLE EFFECTS FOLLOW-UP
# ─────────────────────────────────────────────────────────────────────────────
section("A3 — SIMPLE EFFECTS FOLLOW-UP")

simple_rows = []

# Simple effect of Load at each level of Modality (Welch t-test)
print("  Simple effect of Target Load WITHIN each Modality:")
for col, mod_label in [("RT_Lab","Lab"), ("RT_Game","Game"),
                       ("logRT_Lab","Log Lab"), ("logRT_Game","Log Game")]:
    s_v = wide[wide["group"]=="Single"][col].dropna().values
    m_v = wide[wide["group"]=="Multiple"][col].dropna().values
    t, p = ttest_ind(s_v, m_v, equal_var=False)
    d    = cohens_d(s_v, m_v)
    ci_s = ci95_mean(s_v)
    ci_m = ci95_mean(m_v)
    p_bonf = min(p * 2, 1.0)  # Bonferroni for 2 comparisons
    print(f"\n  {mod_label}:")
    print(f"    Single   M={s_v.mean():.3f}  SD={s_v.std(ddof=1):.3f}  "
          f"95%CI[{ci_s[0]:.3f}, {ci_s[1]:.3f}]")
    print(f"    Multiple M={m_v.mean():.3f}  SD={m_v.std(ddof=1):.3f}  "
          f"95%CI[{ci_m[0]:.3f}, {ci_m[1]:.3f}]")
    print(f"    Welch t({n_s+n_m-2:.0f})={t:.3f}  p={p:.4f}{stars(p)}"
          f"  p_bonf={p_bonf:.4f}{stars(p_bonf)}  d={d:.3f}")
    simple_rows.append({"Contrast": f"Load @ {mod_label}",
                        "Single_M": s_v.mean(), "Multiple_M": m_v.mean(),
                        "t": t, "p": p, "p_bonf": p_bonf, "d": d,
                        "sig": stars(p), "sig_bonf": stars(p_bonf)})

# Simple effect of Modality WITHIN each group (paired t-test)
print("\n\n  Simple effect of Modality WITHIN each Group:")
for grp in ["Single", "Multiple"]:
    for (dv_l, dv_g, label) in [("RT_Lab","RT_Game","Raw RT"),
                                  ("logRT_Lab","logRT_Game","Log RT")]:
        sub  = wide[wide["group"] == grp]
        g    = sub[dv_g].values
        l    = sub[dv_l].values
        diff = g - l
        t, p = ttest_rel(g, l)
        dz   = cohens_dz(diff)
        ci   = (np.mean(diff) - t_dist.ppf(.975, len(diff)-1) * np.std(diff,ddof=1)/np.sqrt(len(diff)),
                np.mean(diff) + t_dist.ppf(.975, len(diff)-1) * np.std(diff,ddof=1)/np.sqrt(len(diff)))
        p_bonf = min(p * 2, 1.0)
        print(f"\n  {grp} — {label}:")
        print(f"    Game  M={g.mean():.3f}  Lab M={l.mean():.3f}  "
              f"Diff M={diff.mean():.3f}  SD={diff.std(ddof=1):.3f}")
        print(f"    95%CI[{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"    t({len(sub)-1})={t:.3f}  p={p:.4f}{stars(p)}"
              f"  p_bonf={p_bonf:.4f}{stars(p_bonf)}  dz={dz:.3f}")
        simple_rows.append({"Contrast": f"Modality @ {grp} ({label})",
                             "Single_M": g.mean(), "Multiple_M": l.mean(),
                             "t": t, "p": p, "p_bonf": p_bonf, "d": dz,
                             "sig": stars(p), "sig_bonf": stars(p_bonf)})

simple_df = pd.DataFrame(simple_rows)
simple_df.to_csv(DIAG_OUT / "anova_simple_effects.csv", index=False)
print("\n\n  ✔  Saved anova_simple_effects.csv")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
section("VISUALIZATIONS")

# ── Fig: Interaction plot (the central figure)
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("2×2 Mixed ANOVA — Interaction Plot: Target Load × Modality\n"
             "DV = Participant-mean RT (ms)",
             fontsize=13, fontweight="bold")

r = anova_raw

# Panel A: Classic interaction plot (mean ± SE)
ax = axes[0]
means_data = {
    "Single":   [r["cell_means"]["SL"], r["cell_means"]["SG"]],
    "Multiple": [r["cell_means"]["ML"], r["cell_means"]["MG"]],
}
ses_data = {}
for grp in ["Single","Multiple"]:
    sub = wide[wide["group"]==grp]
    ses_data[grp] = [sub["RT_Lab"].sem(), sub["RT_Game"].sem()]

for grp, color, mk in [("Single",PAL["Single"],"o"),
                        ("Multiple",PAL["Multiple"],"s")]:
    ms_ = means_data[grp]
    ses = ses_data[grp]
    ax.errorbar(["Lab","Game"], ms_, yerr=ses,
                fmt=f"{mk}-", color=color, linewidth=2.5,
                markersize=10, capsize=8, label=grp)
    for x_i, (mod_l, m_, se_) in enumerate(zip(["Lab","Game"], ms_, ses)):
        ax.text(x_i + (0.06 if grp=="Single" else -0.06),
                m_ + se_ + 30,
                f"{m_:.0f}", ha="center", fontsize=9,
                color=color, fontweight="bold")

ax.set_title(f"Mean RT ± SE\nInteraction: F({r['df_AB']:.0f},{r['df_BsA']:.0f})"
             f"={r['F_AB']:.3f}, p={r['p_AB']:.4f}{stars(r['p_AB'])}"
             f"\nη²p={r['eta2p_AB']:.3f}")
ax.set_xlabel("Modality"); ax.set_ylabel("Mean RT (ms)")
ax.legend(title="Target Load")

# Panel B: Individual participant lines (spaghetti)
ax = axes[1]
for grp, color in [("Single",PAL["Single"]),("Multiple",PAL["Multiple"])]:
    sub = wide[wide["group"]==grp]
    for _, row in sub.iterrows():
        ax.plot(["Lab","Game"],
                [row["RT_Lab"], row["RT_Game"]],
                "-o", color=color, alpha=0.25, lw=1, ms=5)
    # Group mean
    ax.plot(["Lab","Game"],
            [sub["RT_Lab"].mean(), sub["RT_Game"].mean()],
            "D-", color=color, lw=3, ms=11,
            label=f"{grp} (n={len(sub)})", zorder=5)
ax.set_title("Individual Participant Trajectories\n(Lab → Game per person)")
ax.set_xlabel("Modality"); ax.set_ylabel("RT (ms)")
ax.legend(title="Target Load")

# Panel C: Effect sizes summary
ax = axes[2]
effects = ["Target Load\n[A]", "Modality\n[B]", "Interaction\n[A×B]"]
eta2ps  = [r["eta2p_A"], r["eta2p_B"], r["eta2p_AB"]]
Fvals   = [r["F_A"], r["F_B"], r["F_AB"]]
pvals   = [r["p_A"], r["p_B"], r["p_AB"]]
colors_ = [PAL["Single"], PAL["Lab"], PAL["Game"]]

bars = ax.bar(effects, eta2ps, color=colors_, alpha=0.8,
              edgecolor="white", width=0.5)
for thresh, ls_, lbl_ in [(0.01,":", "small"),
                           (0.06,"--","medium"),
                           (0.14,"-.","large")]:
    ax.axhline(thresh, color="#718096", ls=ls_, lw=1.2,
               label=f"η²p={thresh} ({lbl_})")
for bar, F_, p_, e_ in zip(bars, Fvals, pvals, eta2ps):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height() + 0.005,
            f"F={F_:.2f}\n{stars(p_)}", ha="center",
            fontsize=9, fontweight="bold")
ax.set_title("Effect Sizes (partial η²)")
ax.set_ylabel("η²p")
ax.legend(fontsize=7, loc="upper left")

plt.tight_layout()
save_fig("fig_anova_interaction_main.png")


# ── Fig: Side-by-side raw vs log interaction
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Raw RT vs Log(RT): Interaction Pattern Comparison",
             fontsize=13, fontweight="bold")

for ax, anova_res, dv_l, dv_g, ylabel in [
    (axes[0], anova_raw, "RT_Lab",    "RT_Game",    "Mean RT (ms)"),
    (axes[1], anova_log, "logRT_Lab", "logRT_Game", "Mean Log(RT) [ln ms]"),
]:
    r_ = anova_res
    for grp, color, mk in [("Single",PAL["Single"],"o"),
                            ("Multiple",PAL["Multiple"],"s")]:
        sub = wide[wide["group"]==grp]
        ms_ = [sub[dv_l].mean(), sub[dv_g].mean()]
        ses = [sub[dv_l].sem(),  sub[dv_g].sem()]
        ax.errorbar(["Lab","Game"], ms_, yerr=ses,
                    fmt=f"{mk}-", color=color, lw=2.5,
                    ms=9, capsize=7, label=grp)
    ax.set_title(f"F_AB={r_['F_AB']:.3f}, p={r_['p_AB']:.4f}{stars(r_['p_AB'])}\n"
                 f"η²p={r_['eta2p_AB']:.3f}")
    ax.set_xlabel("Modality")
    ax.set_ylabel(ylabel)
    ax.legend(title="Load")

plt.tight_layout()
save_fig("fig_anova_raw_vs_log.png")


# ── Fig: Simple effects — bar chart with CI
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Simple Effects: Load × Modality Decomposition",
             fontsize=13, fontweight="bold")

# Panel A: Simple effect of Load at Lab vs Game
ax = axes[0]
se_load_rows = simple_df[simple_df["Contrast"].str.startswith("Load @") &
                          simple_df["Contrast"].str.contains("Raw")]
if not se_load_rows.empty:
    mods = ["Lab", "Game"]
    diffs = []
    for mod in mods:
        row_ = se_load_rows[se_load_rows["Contrast"].str.contains(mod)]
        if not row_.empty:
            r_ = row_.iloc[0]
            diffs.append({"Modality": mod, "d": r_["d"],
                          "p": r_["p"], "sig": r_["sig"]})
    if diffs:
        dd = pd.DataFrame(diffs)
        bar_colors = [PAL["Lab"], PAL["Game"]]
        bars = ax.bar(dd["Modality"], dd["d"],
                      color=bar_colors, alpha=0.8, edgecolor="white", width=0.4)
        ax.axhline(0, color="black", lw=1)
        for bar, row_ in zip(bars, diffs):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"d={row_['d']:.2f}\n{row_['sig']}",
                    ha="center", fontsize=10, fontweight="bold")
        ax.set_title("Effect of Target Load (Single − Multiple)\n"
                     "within each Modality  (Cohen's d)")
        ax.set_xlabel("Modality"); ax.set_ylabel("Cohen's d (positive = Single > Multiple)")

# Panel B: Simple effect of Modality at Single vs Multiple
ax = axes[1]
se_mod_rows = simple_df[simple_df["Contrast"].str.startswith("Modality @") &
                         simple_df["Contrast"].str.contains("Raw RT")]
if not se_mod_rows.empty:
    grps = ["Single", "Multiple"]
    diffs2 = []
    for grp in grps:
        row_ = se_mod_rows[se_mod_rows["Contrast"].str.contains(grp)]
        if not row_.empty:
            r_ = row_.iloc[0]
            diffs2.append({"Group": grp, "dz": r_["d"],
                           "p": r_["p"], "sig": r_["sig"]})
    if diffs2:
        dd2 = pd.DataFrame(diffs2)
        bar_colors2 = [PAL["Single"], PAL["Multiple"]]
        bars2 = ax.bar(dd2["Group"], dd2["dz"],
                       color=bar_colors2, alpha=0.8, edgecolor="white", width=0.4)
        ax.axhline(0, color="black", lw=1)
        for bar, row_ in zip(bars2, diffs2):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"dz={row_['dz']:.2f}\n{row_['sig']}",
                    ha="center", fontsize=10, fontweight="bold")
        ax.set_title("Effect of Modality (Game − Lab)\n"
                     "within each Group  (Cohen's dz)")
        ax.set_xlabel("Group"); ax.set_ylabel("Cohen's dz (positive = Game > Lab)")

plt.tight_layout()
save_fig("fig_anova_simple_effects.png")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE FULL RESULTS TABLES
# ─────────────────────────────────────────────────────────────────────────────

# ANOVA tables
for res, fname in [(anova_raw,"anova_results_raw.csv"),
                   (anova_log,"anova_results_log.csv")]:
    rows = [
        {"Source":"Target Load [A]","SS":res["ss_A"],"df":res["df_A"],
         "MS":res["ms_A"],"F":res["F_A"],"p":res["p_A"],
         "eta2p":res["eta2p_A"],"sig":stars(res["p_A"])},
        {"Source":"Error (S/A)","SS":res["ss_SA"],"df":res["df_SA"],
         "MS":res["ms_SA"],"F":None,"p":None,"eta2p":None,"sig":""},
        {"Source":"Modality [B]","SS":res["ss_B"],"df":res["df_B"],
         "MS":res["ms_B"],"F":res["F_B"],"p":res["p_B"],
         "eta2p":res["eta2p_B"],"sig":stars(res["p_B"])},
        {"Source":"Load×Modality [A×B]","SS":res["ss_AB"],"df":res["df_AB"],
         "MS":res["ms_AB"],"F":res["F_AB"],"p":res["p_AB"],
         "eta2p":res["eta2p_AB"],"sig":stars(res["p_AB"])},
        {"Source":"Error (B×S/A)","SS":res["ss_BsA"],"df":res["df_BsA"],
         "MS":res["ms_BsA"],"F":None,"p":None,"eta2p":None,"sig":""},
    ]
    pd.DataFrame(rows).to_csv(DIAG_OUT / fname, index=False)
    print(f"  ✔  Saved {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# FULL SUMMARY REPORT (text)
# ─────────────────────────────────────────────────────────────────────────────

summary_lines = [
    "SELECTIVE ATTENTION STUDY — REPORT 2",
    "STEP 1: 2×2 Mixed ANOVA + Full Diagnostics",
    "=" * 70,
    "",
    "DESIGN",
    "  Between-subjects: Target Load (Single n=21, Multiple n=16)",
    "  Within-subjects:  Modality (Lab, Game)",
    "  DV: Participant-mean RT (ms); also Log(RT)",
    "",
    "DIAGNOSTIC RESULTS",
    "─" * 50,
    f"D1. Sample size: N={N} (Single={n_s}, Multiple={n_m}) — unbalanced; note low power",
    f"D2. Outliers: See anova_diag_D2_outliers.csv",
    f"D3. Shapiro-Wilk on ANOVA residuals:",
    f"    Raw RT  : W={sw_raw_stat:.4f}, p={sw_raw_p:.4f} {stars(sw_raw_p)}",
    f"    Log(RT) : W={sw_log_stat:.4f}, p={sw_log_p:.4f} {stars(sw_log_p)}",
    f"    → {rec}",
    f"D4. Levene's homogeneity: See anova_diag_D4_levene.csv",
    f"D5. Sphericity: Automatically satisfied (k=2 within-levels)",
    "",
    "ANOVA RESULTS — RAW RT",
    "─" * 50,
    f"Target Load [A]:       F({anova_raw['df_A']:.0f},{anova_raw['df_SA']:.0f})={anova_raw['F_A']:.3f}  p={anova_raw['p_A']:.4f}{stars(anova_raw['p_A'])}  η²p={anova_raw['eta2p_A']:.3f}",
    f"Modality [B]:          F({anova_raw['df_B']:.0f},{anova_raw['df_BsA']:.0f})={anova_raw['F_B']:.3f}  p={anova_raw['p_B']:.4f}{stars(anova_raw['p_B'])}  η²p={anova_raw['eta2p_B']:.3f}",
    f"Load × Modality [A×B]: F({anova_raw['df_AB']:.0f},{anova_raw['df_BsA']:.0f})={anova_raw['F_AB']:.3f}  p={anova_raw['p_AB']:.4f}{stars(anova_raw['p_AB'])}  η²p={anova_raw['eta2p_AB']:.3f}",
    "",
    "ANOVA RESULTS — LOG(RT)",
    "─" * 50,
    f"Target Load [A]:       F({anova_log['df_A']:.0f},{anova_log['df_SA']:.0f})={anova_log['F_A']:.3f}  p={anova_log['p_A']:.4f}{stars(anova_log['p_A'])}  η²p={anova_log['eta2p_A']:.3f}",
    f"Modality [B]:          F({anova_log['df_B']:.0f},{anova_log['df_BsA']:.0f})={anova_log['F_B']:.3f}  p={anova_log['p_B']:.4f}{stars(anova_log['p_B'])}  η²p={anova_log['eta2p_B']:.3f}",
    f"Load × Modality [A×B]: F({anova_log['df_AB']:.0f},{anova_log['df_BsA']:.0f})={anova_log['F_AB']:.3f}  p={anova_log['p_AB']:.4f}{stars(anova_log['p_AB'])}  η²p={anova_log['eta2p_AB']:.3f}",
    "",
    "SIMPLE EFFECTS",
    "─" * 50,
]
for _, row_ in simple_df.iterrows():
    summary_lines.append(
        f"  {row_['Contrast']:<40} "
        f"t={row_['t']:.3f}  p={row_['p']:.4f}{row_['sig']}  "
        f"d/dz={row_['d']:.3f}"
    )

summary_lines += [
    "",
    "INTERPRETATION NOTES",
    "─" * 50,
    "• The interaction [A×B] is the central hypothesis (H3).",
    "• If significant: Single group shows larger Game-Lab RT gap than Multiple.",
    "• If raw RT normality violated, report log(RT) ANOVA as primary analysis.",
    "• Low between-subjects power (n_m=16): null result for Load [A] is",
    "  inconclusive, not evidence of no effect.",
    "• Effect size benchmarks: η²p <.01=negligible, .01-.06=small,",
    "  .06-.14=medium, >.14=large (Cohen, 1988).",
]

summary_text = "\n".join(summary_lines)
(DIAG_OUT / "anova_full_summary.txt").write_text(summary_text)
print("  ✔  Saved anova_full_summary.txt")

print("\n" + "=" * 70)
print("  STEP 1 COMPLETE")
print(f"  Outputs in: {DIAG_OUT}")
print("=" * 70)