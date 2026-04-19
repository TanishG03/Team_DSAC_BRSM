"""
REPORT 2 — STEP 6: Concurrent Validity (H1)
  (a) Pearson + Spearman: participant-mean Lab RT vs participant-mean Game RT
      - All levels, Levels 1-10, Level 1 only
      - Bootstrap 95% CIs (n=2000 resamples)
  (b) Attenuation correction: observed r vs theoretical ceiling
  (c) H_E4: validity for RT vs SR/HR (ceiling-constrained accuracy)
Requires _r2_cache.pkl from Steps 1-4 in outputs_r2/
"""

import pandas as pd
import numpy as np
import pickle, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
OUT = Path("outputs_r2"); OUT.mkdir(exist_ok=True)

np.random.seed(42)
N_BOOT = 2000

with open(OUT / "_r2_cache.pkl", "rb") as f:
    cache = pickle.load(f)

lab_all         = cache["lab_all"]
game_all        = cache["game_all"]
game_comparable = cache["game_comparable"]
ptpt_lab        = cache["ptpt_lab"]
ptpt_game_all   = cache["ptpt_game_all"]

PAL = {"Single": "#2D6A9F", "Multiple": "#C05621"}

def stars(p):
    if p is None or np.isnan(p): return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

def save(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  {name}")

def section(title):
    print("\n" + "─" * 65)
    print(f"  {title}")
    print("─" * 65)

# ── Bootstrap CI for Pearson r ────────────────────────────────────────────────
def bootstrap_r(x, y, n_boot=N_BOOT, ci=95):
    """Bootstrap percentile CI for Pearson r."""
    x, y = np.array(x), np.array(y)
    n = len(x)
    boot_r = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        boot_r.append(pearsonr(xb, yb)[0])
    lo = np.percentile(boot_r, (100 - ci) / 2)
    hi = np.percentile(boot_r, 100 - (100 - ci) / 2)
    return lo, hi, np.array(boot_r)

# ── Participant-level summaries ───────────────────────────────────────────────
# Lab: participant mean RT
ptpt_lab_s = ptpt_lab[ptpt_lab["group"] == "Single"][["participant", "mean_RT_ms", "mean_log_RT"]].copy()
ptpt_lab_m = ptpt_lab[ptpt_lab["group"] == "Multiple"][["participant", "mean_RT_ms", "mean_log_RT"]].copy()

# Game: all levels
ptpt_game_s_all = (ptpt_game_all[ptpt_game_all["group"] == "Single"]
                   .groupby("participant")[["RT_ms", "log_RT", "success_rate", "hit_rate", "false_alarms"]]
                   .mean().reset_index())
ptpt_game_m_all = (ptpt_game_all[ptpt_game_all["group"] == "Multiple"]
                   .groupby("participant")[["RT_ms", "log_RT", "success_rate", "hit_rate", "false_alarms"]]
                   .mean().reset_index())

# Game: levels 1-10
ptpt_game_s_10 = (game_comparable[game_comparable["group"] == "Single"]
                  .groupby("participant")[["RT_ms", "log_RT", "success_rate", "hit_rate", "false_alarms"]]
                  .mean().reset_index())
ptpt_game_m_10 = (game_comparable[game_comparable["group"] == "Multiple"]
                  .groupby("participant")[["RT_ms", "log_RT", "success_rate", "hit_rate", "false_alarms"]]
                  .mean().reset_index())

# Game: level 1 only
ptpt_game_s_l1 = (game_all[(game_all["group"] == "Single") & (game_all["level"] == 1)]
                  .groupby("participant")[["RT_ms", "log_RT"]].mean().reset_index())
ptpt_game_m_l1 = (game_all[(game_all["group"] == "Multiple") & (game_all["level"] == 1)]
                  .groupby("participant")[["RT_ms", "log_RT"]].mean().reset_index())

# Reliability figures from Report 1
LAB_ALPHA = {"Single": 0.744, "Multiple": 0.621}

print("=" * 65)
print("STEP 6: CONCURRENT VALIDITY (H1)")
print("=" * 65)

# =============================================================================
# SECTION A: Lab RT vs Game RT — three windows
# =============================================================================
section("A. H1 — LAB RT vs GAME RT: PEARSON, SPEARMAN, BOOTSTRAP CI")

print("""
  H1: Participants fast in Lab should be fast in Game.
  Three comparison windows tested for each group:
    (a) All game levels
    (b) Levels 1–10 (comparable difficulty window)
    (c) Level 1 only (most structurally comparable to lab)
  Primary DV: raw RT. Secondary: log RT. Robustness: Spearman ρ.
  Bootstrap 95% CI (n=2000 resamples, seed=42).
""")

validity_rows = []

for grp, lab_df, game_dfs in [
    ("Single",   ptpt_lab_s, [("All levels",  ptpt_game_s_all),
                               ("Levels 1–10", ptpt_game_s_10),
                               ("Level 1",     ptpt_game_s_l1)]),
    ("Multiple", ptpt_lab_m, [("All levels",  ptpt_game_m_all),
                               ("Levels 1–10", ptpt_game_m_10),
                               ("Level 1",     ptpt_game_m_l1)]),
]:
    print(f"\n  ════ {grp} group ════")
    alpha   = LAB_ALPHA[grp]
    r_ceil  = np.sqrt(alpha)
    n_min_sig = None  # minimum r for p<.05 given n

    for window, game_df in game_dfs:
        merged = lab_df.merge(game_df, on="participant", suffixes=("_lab", "_game"))
        n = len(merged)

        # min r for sig at this n
        from scipy.stats import t as t_dist
        t_crit = t_dist.ppf(0.975, df=n - 2)
        r_min_sig = t_crit / np.sqrt(t_crit**2 + n - 2)

        print(f"\n  ── {window}  (n={n}) ──")

        for label, xcol, ycol in [
            ("Raw RT",  "mean_RT_ms" if "mean_RT_ms" in merged.columns else "RT_ms_lab",
                        "RT_ms_game" if "RT_ms_game" in merged.columns else "RT_ms"),
            ("Log RT",  "mean_log_RT" if "mean_log_RT" in merged.columns else "log_RT_lab",
                        "log_RT_game" if "log_RT_game" in merged.columns else "log_RT"),
        ]:
            # Fix column names depending on merge
            xc = "mean_RT_ms"  if label == "Raw RT" and "mean_RT_ms" in merged.columns else \
                 "RT_ms_lab"   if label == "Raw RT" and "RT_ms_lab"  in merged.columns else \
                 "RT_ms"       if label == "Raw RT" else None
            yc = "RT_ms_game" if label == "Raw RT" and "RT_ms_game" in merged.columns else \
                 "RT_ms"       if label == "Raw RT" and "RT_ms" in merged.columns else None

            if label == "Log RT":
                xc = "mean_log_RT" if "mean_log_RT" in merged.columns else "log_RT_lab"
                yc = "log_RT_game"  if "log_RT_game" in merged.columns else "log_RT"

            if xc not in merged.columns or yc not in merged.columns:
                continue

            x = merged[xc].dropna()
            y = merged[yc].dropna()
            valid = merged[[xc, yc]].dropna()
            x, y = valid[xc].values, valid[yc].values

            if len(x) < 5:
                continue

            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)
            ci_lo, ci_hi, _ = bootstrap_r(x, y)

            sig_flag = "✓ sig" if p_p < .05 else "✗ ns"
            print(f"    {label:<8}  r={r_p:+.3f}  p={p_p:.4f}{stars(p_p):3s}  "
                  f"95%CI[{ci_lo:+.3f},{ci_hi:+.3f}]  "
                  f"ρ={r_s:+.3f}(p={p_s:.4f})  {sig_flag}")

            validity_rows.append({
                "group": grp, "window": window, "dv": label,
                "n": n, "r": r_p, "p": p_p, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "rho": r_s, "rho_p": p_s,
                "r_ceiling": r_ceil, "r_min_sig": r_min_sig,
                "r_as_pct_ceiling": r_p / r_ceil * 100,
            })

        print(f"    Reliability ceiling: √α={r_ceil:.3f}  "
              f"Min r for p<.05 at n={n}: r={r_min_sig:.3f}")

# =============================================================================
# SECTION B: Attenuation Correction Summary
# =============================================================================
section("B. ATTENUATION CORRECTION — Observed r vs Theoretical Ceiling")

print("""
  The maximum observable correlation between Lab RT and Game RT is
  bounded by the Lab measure's reliability: r_max = √α_lab.
  A perfectly reliable Game would still only correlate at r_max.
  If game reliability < 1.0, the practical ceiling is even lower.

  Formula: r_corrected = r_observed / √(α_lab × α_game)
  Since α_game is unknown, we report r_observed / √α_lab (lower bound
  on the attenuation-corrected correlation).
""")

print(f"  {'Group':<10} {'α_lab':>6} {'√α_lab':>8} {'Window':<15} "
      f"{'r_obs':>7} {'r/√α':>7} {'% ceiling':>10}")
print("  " + "─" * 65)

for row in validity_rows:
    if row["dv"] != "Raw RT":
        continue
    pct = row["r"] / row["r_ceiling"] * 100
    print(f"  {row['group']:<10} {LAB_ALPHA[row['group']]:>6.3f} "
          f"{row['r_ceiling']:>8.3f} {row['window']:<15} "
          f"{row['r']:>7.3f} {row['r']/row['r_ceiling']:>7.3f} {pct:>9.1f}%")

# =============================================================================
# SECTION C: H_E4 — Accuracy Ceiling Limits Validity
# =============================================================================
section("C. H_E4 — ACCURACY CEILING LIMITS VALIDITY")

print("""
  H_E4: Because SR/HR are ceiling-constrained (Mdn=100%), they should
  show weaker validity correlations than RT.  False Alarms (not ceiling-
  constrained) may be a more discriminating alternative.
""")

he4_rows = []

for grp, lab_df, game_df10 in [
    ("Single",   ptpt_lab_s, ptpt_game_s_10),
    ("Multiple", ptpt_lab_m, ptpt_game_m_10),
]:
    print(f"\n  ── {grp} group (Levels 1–10) ──")
    merged = lab_df.merge(game_df10, on="participant")
    n = len(merged)

    lab_rt_col = "mean_RT_ms" if "mean_RT_ms" in merged.columns else "RT_ms_lab"

    for game_col, label in [
        ("RT_ms",        "Game RT (ms)      "),
        ("success_rate", "Success Rate (%)  "),
        ("hit_rate",     "Hit Rate (%)      "),
        ("false_alarms", "False Alarms      "),
    ]:
        if game_col not in merged.columns:
            continue
        valid = merged[[lab_rt_col, game_col]].dropna()
        if len(valid) < 5:
            continue
        x = valid[lab_rt_col].values
        y = valid[game_col].values
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        ci_lo, ci_hi, _ = bootstrap_r(x, y)
        variance = np.var(y)
        print(f"    Lab RT vs {label}  r={r_p:+.3f}  p={p_p:.4f}{stars(p_p):3s}  "
              f"95%CI[{ci_lo:+.3f},{ci_hi:+.3f}]  "
              f"ρ={r_s:+.3f}  Var(game_dv)={variance:.2f}")
        he4_rows.append({
            "group": grp, "game_dv": game_col, "label": label.strip(),
            "n": n, "r": r_p, "p": p_p,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "rho": r_s,
            "game_var": variance,
        })

# =============================================================================
# VISUALISATIONS
# =============================================================================
section("VALIDITY VISUALISATIONS")

# ── Fig 1: Scatter grid — Lab RT vs Game RT (3 windows × 2 groups) ────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Concurrent Validity: Lab RT vs Game RT\n"
             "Top: Single group | Bottom: Multiple group",
             fontsize=13, fontweight="bold")

windows_data = {
    "Single": [
        ("All levels",  ptpt_lab_s, ptpt_game_s_all),
        ("Levels 1–10", ptpt_lab_s, ptpt_game_s_10),
        ("Level 1",     ptpt_lab_s, ptpt_game_s_l1),
    ],
    "Multiple": [
        ("All levels",  ptpt_lab_m, ptpt_game_m_all),
        ("Levels 1–10", ptpt_lab_m, ptpt_game_m_10),
        ("Level 1",     ptpt_lab_m, ptpt_game_m_l1),
    ],
}

for row_i, grp in enumerate(["Single", "Multiple"]):
    color = PAL[grp]
    for col_i, (window, lab_df, game_df) in enumerate(windows_data[grp]):
        ax = axes[row_i, col_i]
        merged = lab_df.merge(game_df, on="participant")

        lab_col  = "mean_RT_ms" if "mean_RT_ms" in merged.columns else "RT_ms_lab"
        game_col = "RT_ms_game" if "RT_ms_game" in merged.columns else "RT_ms"

        if lab_col not in merged.columns or game_col not in merged.columns:
            ax.set_title(f"{grp} — {window}\n(data unavailable)")
            continue

        valid = merged[[lab_col, game_col]].dropna()
        x, y  = valid[lab_col].values, valid[game_col].values
        n     = len(x)
        r_p, p_p = pearsonr(x, y)
        ci_lo, ci_hi, _ = bootstrap_r(x, y)

        ax.scatter(x, y, color=color, s=60, alpha=0.8, edgecolors="white", lw=0.5)

        # Regression line
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, m * xr + b, color=color, lw=2, ls="--", alpha=0.8)

        ax.set_title(f"{grp} — {window}\n"
                     f"r={r_p:+.3f}{stars(p_p)}  "
                     f"95%CI[{ci_lo:+.3f},{ci_hi:+.3f}]  n={n}",
                     fontsize=9)
        ax.set_xlabel("Lab Mean RT (ms)")
        ax.set_ylabel("Game Mean RT (ms)")

plt.tight_layout()
save("fig_s6_01_validity_scatter.png")


# ── Fig 2: Forest plot — r across windows and groups ─────────────────────────
raw_rows = [r for r in validity_rows if r["dv"] == "Raw RT"]

fig, ax = plt.subplots(figsize=(10, 6))
y_ticks, y_labels = [], []
y = 0

for grp in ["Single", "Multiple"]:
    color = PAL[grp]
    grp_rows = [r for r in raw_rows if r["group"] == grp]
    for row in grp_rows:
        ax.errorbar(
            row["r"], y,
            xerr=[[row["r"] - row["ci_lo"]], [row["ci_hi"] - row["r"]]],
            fmt="o", color=color, ms=8, capsize=5, lw=1.8,
        )
        ax.axvline(row["r_ceiling"], color=color, lw=0.8, ls=":", alpha=0.5)
        y_ticks.append(y)
        y_labels.append(f"{grp} — {row['window']}")
        y += 1
    y += 0.5  # gap between groups

ax.axvline(0, color="black", lw=1.2, ls="-")
ax.axvline(0.3, color="grey", lw=0.8, ls="--", alpha=0.6, label="r=0.3 (small)")
ax.axvline(0.5, color="grey", lw=0.8, ls="-.", alpha=0.6, label="r=0.5 (medium)")
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=10)
ax.set_xlabel("Pearson r (Lab RT vs Game RT)  [error bars = 95% bootstrap CI]")
ax.set_title("Concurrent Validity Forest Plot\n"
             "Dotted vertical lines = reliability ceiling (√α) per group",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(-0.6, 1.0)
plt.tight_layout()
save("fig_s6_02_validity_forest.png")


# ── Fig 3: H_E4 — validity by DV type (RT vs accuracy) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("H_E4: Validity Correlations — RT vs Accuracy Measures\n"
             "(Lab RT vs each Game DV; Levels 1–10)",
             fontsize=12, fontweight="bold")

dv_order   = ["RT_ms", "success_rate", "hit_rate", "false_alarms"]
dv_labels  = ["Game RT", "Success Rate", "Hit Rate", "False Alarms"]

for ax, grp in zip(axes, ["Single", "Multiple"]):
    color = PAL[grp]
    grp_rows = [r for r in he4_rows if r["group"] == grp]
    dv_map   = {r["game_dv"]: r for r in grp_rows}

    rs   = [dv_map.get(d, {}).get("r",    np.nan) for d in dv_order]
    cilo = [dv_map.get(d, {}).get("ci_lo", np.nan) for d in dv_order]
    cihi = [dv_map.get(d, {}).get("ci_hi", np.nan) for d in dv_order]
    ps   = [dv_map.get(d, {}).get("p",    1.0)   for d in dv_order]

    x_pos = np.arange(len(dv_order))
    bars  = ax.bar(x_pos, rs, color=color, alpha=0.75, edgecolor="white", width=0.6)

    for xi, (r_val, lo, hi, p_val) in enumerate(zip(rs, cilo, cihi, ps)):
        if np.isnan(r_val): continue
        ax.errorbar(xi, r_val, yerr=[[r_val - lo], [hi - r_val]],
                    fmt="none", color="black", capsize=5, lw=1.5)
        ax.text(xi, hi + 0.03, stars(p_val), ha="center", fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dv_labels, fontsize=10)
    ax.set_ylabel("Pearson r (Lab RT vs Game DV)")
    ax.set_title(f"{grp} group (n={grp_rows[0]['n'] if grp_rows else 'NA'})")
    ax.set_ylim(-0.7, 0.9)
    ax.axhline(0.3,  color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-0.3, color="grey", lw=0.8, ls="--", alpha=0.5)

plt.tight_layout()
save("fig_s6_03_he4_validity_by_dv.png")


# =============================================================================
# SAVE RESULTS
# =============================================================================
df_validity = pd.DataFrame(validity_rows)
df_he4      = pd.DataFrame(he4_rows)

df_validity.to_csv(OUT / "results_r2_validity_formal.csv", index=False)
df_he4.to_csv(OUT      / "results_r2_he4_validity.csv",   index=False)

print(f"\n  ✔  Saved results_r2_validity_formal.csv")
print(f"  ✔  Saved results_r2_he4_validity.csv")

cache.update({
    "validity_rows": validity_rows,
    "he4_rows":      he4_rows,
})
with open(OUT / "_r2_cache.pkl", "wb") as f:
    pickle.dump(cache, f)
print("  ✔  Updated _r2_cache.pkl")

print("\n  STEP 6 COMPLETE — paste full output before Step 7.\n")