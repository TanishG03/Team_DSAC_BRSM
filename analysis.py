"""
=============================================================================
STATISTICAL ANALYSIS — Selective Attention Study (Full Dataset)
Directory structure expected:
  preprocessed_data/
    multiple/
      lab/    → {pid}_visual_search_*.csv
      phone/  → {pid}_attentional_spotter_results.csv
    single/
      lab/    → {pid}_visual_search_*.csv
      phone/  → {pid}_attentional_spotter_results.csv

Analyses:
  • Participant-level aggregation (Slide 15)
  • RQ1 — Pearson/Spearman concurrent validity (Game vs Lab RT)
  • RQ2 — 2×2 Mixed ANOVA + Independent t-test (Single vs Multiple)
  • RQ3 — Paired t-test (Game vs Lab within each group)
  • RQ4 — Spearman level trends in game
  • Reliability — Split-half + Spearman-Brown + Cronbach's α
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import ast, warnings, re
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_rel, ttest_ind, f as f_dist

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"Single": "#4C72B0", "Multiple": "#DD8452",
          "Lab": "#55A868", "Game": "#C44E52"}

# ── CONFIGURE THESE PATHS ─────────────────────────────────────────────────────
DATA_ROOT = Path("preprocessed_data")
OUT       = Path("outputs")
OUT.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def stars(p):
    if pd.isna(p): return ""
    if p < .001: return "***"
    if p < .01:  return "**"
    if p < .05:  return "*"
    return "ns"

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    pooled = np.sqrt(((na-1)*np.var(a,ddof=1) + (nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(a)-np.mean(b))/pooled if pooled else np.nan

def cohens_dz(diff):
    s = np.std(diff, ddof=1)
    return np.mean(diff)/s if s else np.nan

def print_section(title):
    print("\n" + "═"*70)
    print(f"  {title}")
    print("═"*70)


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS  (same as EDA script)
# ─────────────────────────────────────────────────────────────────────────────

def extract_pid(path):
    m = re.match(r"^(\d+)", path.stem)
    return m.group(1) if m else path.stem

def load_lab(path, participant_id, group):
    df = pd.read_csv(path)
    df = df.dropna(subset=["target_col"])
    df = df[df["target_col"].isin(["red","white"])].copy()
    df["participant"] = participant_id
    df["group"]       = group
    df["trial_n"]     = range(1, len(df)+1)
    def first_click_ms(val):
        try: return float(ast.literal_eval(str(val))[0]) * 1000
        except: return np.nan
    df["RT_ms"]     = df["mouse.time"].apply(first_click_ms)
    df["is_target"] = (df["target_col"] == "red").astype(int)
    df["hit"]       = 1
    return df[["participant","group","trial_n","is_target","RT_ms","hit"]]

def load_game(path, participant_id, group):
    df = pd.read_csv(path)
    df["participant"] = participant_id
    df["group"]       = group
    df = df.rename(columns={
        "Level":"level","Completed":"completed",
        "SuccessRate(%)":"success_rate","HitRate(%)":"hit_rate",
        "FalseAlarms":"false_alarms","InitialResponseTime(ms)":"RT_ms",
        "AvgInterTargetTime(ms)":"avg_inter_target_ms","FinalScore":"final_score",
    })
    df["completed"] = df["completed"].astype(str).str.lower().isin(["true","1","yes"])
    return df[["participant","group","level","completed","success_rate",
               "hit_rate","false_alarms","RT_ms","avg_inter_target_ms","final_score"]]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL PARTICIPANTS
# ─────────────────────────────────────────────────────────────────────────────

all_lab_dfs, all_game_dfs = [], []

for group_dir, group_label in [("multiple","Multiple"),("single","Single")]:
    lab_dir   = DATA_ROOT / group_dir / "lab"
    phone_dir = DATA_ROOT / group_dir / "phone"
    if not lab_dir.exists() or not phone_dir.exists():
        print(f"  ⚠  {group_label}: directory missing — skipping")
        continue
    lab_map   = {extract_pid(f): f for f in sorted(lab_dir.glob("*.csv"))}
    phone_map = {extract_pid(f): f for f in sorted(phone_dir.glob("*.csv"))}
    for pid in sorted(set(lab_map)|set(phone_map), key=lambda x: int(x) if x.isdigit() else x):
        if pid in lab_map:
            try: all_lab_dfs.append(load_lab(lab_map[pid], pid, group_label))
            except Exception as e: print(f"  ⚠  Lab load error {pid}: {e}")
        if pid in phone_map:
            try: all_game_dfs.append(load_game(phone_map[pid], pid, group_label))
            except Exception as e: print(f"  ⚠  Game load error {pid}: {e}")

lab_all  = pd.concat(all_lab_dfs,  ignore_index=True) if all_lab_dfs  else pd.DataFrame()
game_all = pd.concat(all_game_dfs, ignore_index=True) if all_game_dfs else pd.DataFrame()

print("=" * 70)
print("DATASET LOADED")
print(f"  Lab:  {len(lab_all)} trials  from {lab_all['participant'].nunique() if not lab_all.empty else 0} participants")
print(f"  Game: {len(game_all)} levels  from {game_all['participant'].nunique() if not game_all.empty else 0} participants")


# ─────────────────────────────────────────────────────────────────────────────
# PARTICIPANT-LEVEL AGGREGATION  (Slide 15)
# ─────────────────────────────────────────────────────────────────────────────

print_section("PARTICIPANT-LEVEL AGGREGATION (Slide 15)")

ptpt_lab = (lab_all.groupby(["participant","group"])
            .agg(RT_mean=("RT_ms","mean"), RT_sd=("RT_ms","std"),
                 hit_rate=("hit","mean"), n_trials=("RT_ms","count"))
            .reset_index()) if not lab_all.empty else pd.DataFrame()

ptpt_game = (game_all.groupby(["participant","group"])
             .agg(RT_mean=("RT_ms","mean"), RT_sd=("RT_ms","std"),
                  success_rate=("success_rate","mean"), hit_rate=("hit_rate","mean"),
                  false_alarms=("false_alarms","mean"), n_levels=("level","count"),
                  max_level=("level","max"))
             .reset_index()) if not game_all.empty else pd.DataFrame()

print(f"\n{'Participant':<14}{'Group':<12}{'Modality':<10}"
      f"{'RT_M (ms)':>12}{'RT_SD':>10}{'Accuracy %':>12}")
print("-" * 62)

for grp in ["Single","Multiple"]:
    for df, mod, acc_col in [
        (ptpt_lab,  "Lab",  "hit_rate"),
        (ptpt_game, "Game", "success_rate"),
    ]:
        if df.empty: continue
        sub = df[df["group"]==grp]
        for _, row in sub.iterrows():
            acc = row.get(acc_col, np.nan)
            acc_str = f"{acc*100:.1f}" if acc_col=="hit_rate" else f"{acc:.1f}"
            print(f"{row['participant']:<14}{grp:<12}{mod:<10}"
                  f"{row['RT_mean']:>12.1f}{row['RT_sd']:>10.1f}{acc_str:>12}")

# Save aggregated
ptpt_lab.to_csv(OUT  / "participant_lab_aggregated.csv",  index=False)
ptpt_game.to_csv(OUT / "participant_game_aggregated.csv", index=False)
print("\n  ✔  Saved aggregated CSVs")


# ─────────────────────────────────────────────────────────────────────────────
# RQ1 — CONCURRENT VALIDITY
# ─────────────────────────────────────────────────────────────────────────────

print_section("RQ1 — CONCURRENT VALIDITY (Pearson r, Spearman ρ)")

validity_rows = []

if not ptpt_lab.empty and not ptpt_game.empty:
    merged = pd.merge(
        ptpt_lab[["participant","group","RT_mean"]].rename(columns={"RT_mean":"lab_RT"}),
        ptpt_game[["participant","group","RT_mean"]].rename(columns={"RT_mean":"game_RT"}),
        on=["participant","group"]
    )

    for grp in ["Single","Multiple"]:
        sub = merged[merged["group"]==grp].dropna()
        n   = len(sub)
        print(f"\n  {grp} group  (n={n} participants with both modalities)")
        if n < 3:
            print(f"    ⚠  n={n} — need ≥3 for reliable correlation. "
                  "Printing available data only.")
        if n >= 2:
            r, p_r  = pearsonr(sub["game_RT"],  sub["lab_RT"])
            rho,p_s = spearmanr(sub["game_RT"], sub["lab_RT"])
            print(f"    Pearson  r = {r:+.3f}   p = {p_r:.4f} {stars(p_r)}")
            print(f"    Spearman ρ = {rho:+.3f}   p = {p_s:.4f} {stars(p_s)}")
            validity_rows.append({"Group":grp,"n":n,"r":r,"p_pearson":p_r,
                                   "rho":rho,"p_spearman":p_s})
        else:
            print("    Insufficient data for correlation.")
else:
    print("  No data available.")


# ─────────────────────────────────────────────────────────────────────────────
# 2×2 MIXED ANOVA  (Slides 15-16)
# Uses participant means (proper ANOVA input)
# ─────────────────────────────────────────────────────────────────────────────

print_section("2×2 MIXED ANOVA — RT (Modality [within] × Target Load [between])")

if not ptpt_lab.empty and not ptpt_game.empty:
    # Build participant-level wide-form: one row per participant
    wide = pd.merge(
        ptpt_lab[["participant","group","RT_mean"]].rename(columns={"RT_mean":"RT_Lab"}),
        ptpt_game[["participant","group","RT_mean"]].rename(columns={"RT_mean":"RT_Game"}),
        on=["participant","group"]
    ).dropna()

    print(f"\n  Participants with both modalities: {len(wide)}")
    print(f"  Single: {(wide['group']=='Single').sum()}    "
          f"Multiple: {(wide['group']=='Multiple').sum()}")

    # ── Manual 2×2 Mixed ANOVA ───────────────────────────────────────────────
    # Between: Target Load (A)
    # Within:  Modality   (B)
    # Each participant contributes two scores (Lab, Game)

    grand = wide[["RT_Lab","RT_Game"]].values.mean()
    N     = len(wide)
    n_s   = (wide["group"]=="Single").sum()
    n_m   = (wide["group"]=="Multiple").sum()

    # Cell means
    s_lab  = wide[wide["group"]=="Single"]["RT_Lab"].values
    s_game = wide[wide["group"]=="Single"]["RT_Game"].values
    m_lab  = wide[wide["group"]=="Multiple"]["RT_Lab"].values
    m_game = wide[wide["group"]=="Multiple"]["RT_Game"].values

    cell_mu = {
        ("Single","Lab"):    s_lab.mean(),
        ("Single","Game"):   s_game.mean(),
        ("Multiple","Lab"):  m_lab.mean(),
        ("Multiple","Game"): m_game.mean(),
    }

    # Marginal means
    mu_single   = wide[wide["group"]=="Single"][["RT_Lab","RT_Game"]].values.mean()
    mu_multiple = wide[wide["group"]=="Multiple"][["RT_Lab","RT_Game"]].values.mean()
    mu_lab      = wide["RT_Lab"].mean()
    mu_game     = wide["RT_Game"].mean()

    # Subject means
    wide["subj_mean"] = wide[["RT_Lab","RT_Game"]].mean(axis=1)

    # SS_A  (between: Target Load)
    ss_A  = (n_s  * 2 * (mu_single   - grand)**2 +
             n_m  * 2 * (mu_multiple - grand)**2)
    df_A  = 1

    # SS_S/A  (subjects within groups = between-subjects error)
    ss_SA = sum((row["subj_mean"] - mu_single)**2 * 2
                for _, row in wide[wide["group"]=="Single"].iterrows()) + \
            sum((row["subj_mean"] - mu_multiple)**2 * 2
                for _, row in wide[wide["group"]=="Multiple"].iterrows())
    df_SA = N - 2   # N subjects - 2 groups

    # SS_B  (within: Modality)
    ss_B  = N * ((mu_lab  - grand)**2 + (mu_game - grand)**2)
    df_B  = 1

    # SS_AB (interaction)
    ss_AB = sum(
        n_grp * ((cell_mu[(grp,mod)] - mu_grp - mu_mod + grand)**2)
        for grp, n_grp, mu_grp in [
            ("Single", n_s, mu_single), ("Multiple", n_m, mu_multiple)
        ]
        for mod, mu_mod in [("Lab", mu_lab), ("Game", mu_game)]
    )
    df_AB = 1

    # SS_BxS/A  (within-subjects error)
    ss_BsA = 0
    for _, row in wide.iterrows():
        grp    = row["group"]
        mu_grp = mu_single if grp == "Single" else mu_multiple
        for val, mu_mod in [(row["RT_Lab"], mu_lab), (row["RT_Game"], mu_game)]:
            predicted = mu_grp + mu_mod - grand
            ss_BsA   += (val - row["subj_mean"] - mu_mod + grand)**2

    df_BsA = (N - 2) * 1   # (N - n_groups) * (n_within - 1)

    ms_A   = ss_A   / df_A
    ms_SA  = ss_SA  / df_SA  if df_SA  > 0 else np.nan
    ms_B   = ss_B   / df_B
    ms_AB  = ss_AB  / df_AB
    ms_BsA = ss_BsA / df_BsA if df_BsA > 0 else np.nan

    F_A   = ms_A  / ms_SA  if ms_SA  else np.nan
    F_B   = ms_B  / ms_BsA if ms_BsA else np.nan
    F_AB  = ms_AB / ms_BsA if ms_BsA else np.nan

    p_A  = 1 - f_dist.cdf(F_A,  df_A,  df_SA)  if not np.isnan(F_A)  else np.nan
    p_B  = 1 - f_dist.cdf(F_B,  df_B,  df_BsA) if not np.isnan(F_B)  else np.nan
    p_AB = 1 - f_dist.cdf(F_AB, df_AB, df_BsA) if not np.isnan(F_AB) else np.nan

    eta2_A  = ss_A  / (ss_A  + ss_SA)  if (ss_A  + ss_SA)  else np.nan
    eta2_B  = ss_B  / (ss_B  + ss_BsA) if (ss_B  + ss_BsA) else np.nan
    eta2_AB = ss_AB / (ss_AB + ss_BsA) if (ss_AB + ss_BsA) else np.nan

    print("\n  Cell Means (participant-level, ms):")
    for (grp, mod), mu in cell_mu.items():
        print(f"    {grp:<12} × {mod:<6}  M = {mu:.1f}")

    print("\n  ANOVA Table:")
    hdr = f"  {'Source':<30}{'SS':>12}{'df':>5}{'MS':>12}{'F':>9}{'p':>10}{'η²p':>8}"
    print(hdr)
    print("  " + "-"*80)

    anova_rows = [
        ("Target Load [A]",         ss_A,   df_A,   ms_A,   F_A,   p_A,   eta2_A),
        ("  Error (S/A)",           ss_SA,  df_SA,  ms_SA,  np.nan,np.nan,np.nan),
        ("Modality [B]",            ss_B,   df_B,   ms_B,   F_B,   p_B,   eta2_B),
        ("Target Load × Modality",  ss_AB,  df_AB,  ms_AB,  F_AB,  p_AB,  eta2_AB),
        ("  Error (B×S/A)",         ss_BsA, df_BsA, ms_BsA, np.nan,np.nan,np.nan),
    ]
    for src, ss, df_, ms, F, p, eta in anova_rows:
        ss_s  = f"{ss:>12.1f}"  if not np.isnan(ss)  else f"{'—':>12}"
        df_s  = f"{df_:>5.0f}"  if not np.isnan(df_) else f"{'—':>5}"
        ms_s  = f"{ms:>12.1f}"  if not np.isnan(ms)  else f"{'—':>12}"
        F_s   = f"{F:>9.3f}"    if not np.isnan(F)   else f"{'—':>9}"
        p_s   = f"{p:>8.4f} {stars(p)}" if not np.isnan(p) else f"{'—':>10}  "
        eta_s = f"{eta:>8.3f}"  if not np.isnan(eta) else f"{'—':>8}"
        print(f"  {src:<30}{ss_s}{df_s}{ms_s}{F_s} {p_s}{eta_s}")

    print(f"\n  RQ2 — Main effect of Target Load:  F({df_A},{df_SA}) = {F_A:.3f}, "
          f"p = {p_A:.4f} {stars(p_A)}, η²p = {eta2_A:.3f}")
    print(f"  RQ3 — Main effect of Modality:     F({df_B},{df_BsA}) = {F_B:.3f}, "
          f"p = {p_B:.4f} {stars(p_B)}, η²p = {eta2_B:.3f}")
    print(f"  Interaction (Load × Modality):     F({df_AB},{df_BsA}) = {F_AB:.3f}, "
          f"p = {p_AB:.4f} {stars(p_AB)}, η²p = {eta2_AB:.3f}")

    # Save ANOVA table
    anova_df = pd.DataFrame(anova_rows,
                            columns=["Source","SS","df","MS","F","p","eta2p"])
    anova_df.to_csv(OUT / "anova_results.csv", index=False)
    print("\n  ✔  Saved anova_results.csv")

else:
    print("  Insufficient data for ANOVA.")
    wide = pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# RQ3 — PAIRED t-TEST: Game vs Lab (within each group)
# ─────────────────────────────────────────────────────────────────────────────

print_section("RQ3 — PAIRED t-TEST: Game vs Lab RT (within groups)")

paired_rows = []
if not wide.empty:
    for grp in ["Single","Multiple"]:
        sub = wide[wide["group"]==grp].dropna(subset=["RT_Lab","RT_Game"])
        n   = len(sub)
        if n < 2:
            print(f"\n  {grp}: n={n} — insufficient"); continue
        g_rt = sub["RT_Game"].values
        l_rt = sub["RT_Lab"].values
        diff = g_rt - l_rt
        t, p = ttest_rel(g_rt, l_rt)
        dz   = cohens_dz(diff)
        se   = diff.std(ddof=1) / np.sqrt(n)
        ci_l, ci_h = diff.mean() - 1.96*se, diff.mean() + 1.96*se
        print(f"\n  {grp} group  (n={n} participants)")
        print(f"    Game RT   M = {g_rt.mean():.1f} ms  SD = {g_rt.std(ddof=1):.1f}")
        print(f"    Lab  RT   M = {l_rt.mean():.1f} ms  SD = {l_rt.std(ddof=1):.1f}")
        print(f"    Diff      M = {diff.mean():.1f} ms  95% CI [{ci_l:.1f}, {ci_h:.1f}]")
        print(f"    t({n-1}) = {t:.3f}   p = {p:.4f} {stars(p)}   dz = {dz:.3f}")
        paired_rows.append({"Group":grp,"n":n,"t":t,"p":p,"dz":dz,
                             "game_M":g_rt.mean(),"lab_M":l_rt.mean(),
                             "diff_M":diff.mean()})
else:
    print("  Run ANOVA section first (requires merged wide-form data).")


# ─────────────────────────────────────────────────────────────────────────────
# RQ2 — INDEPENDENT t-TEST: Single vs Multiple
# ─────────────────────────────────────────────────────────────────────────────

print_section("RQ2 — INDEPENDENT t-TEST: Single vs Multiple Target Load")

indep_rows = []
for df, label, col in [
    (ptpt_lab,  "Lab RT (ms)",         "RT_mean"),
    (ptpt_game, "Game RT (ms)",         "RT_mean"),
    (ptpt_game, "Game Success Rate (%)", "success_rate"),
    (ptpt_game, "Game Hit Rate (%)",     "hit_rate"),
    (ptpt_game, "Game False Alarms",     "false_alarms"),
]:
    if df.empty or col not in df.columns: continue
    s = df[df["group"]=="Single"][col].dropna().values
    m = df[df["group"]=="Multiple"][col].dropna().values
    if len(s) < 2 or len(m) < 2:
        print(f"\n  {label}: insufficient data  (Single n={len(s)}, Multiple n={len(m)})")
        continue
    t, p = ttest_ind(s, m, equal_var=False)
    d    = cohens_d(s, m)
    print(f"\n  {label}:")
    print(f"    Single   M = {s.mean():.2f}  SD = {s.std(ddof=1):.2f}  n = {len(s)}")
    print(f"    Multiple M = {m.mean():.2f}  SD = {m.std(ddof=1):.2f}  n = {len(m)}")
    print(f"    Welch t = {t:.3f}   p = {p:.4f} {stars(p)}   Cohen's d = {d:.3f}")
    indep_rows.append({"DV":label,"t":t,"p":p,"d":d,
                       "Single_M":s.mean(),"Multiple_M":m.mean()})

pd.DataFrame(indep_rows).to_csv(OUT / "ttest_independent_results.csv", index=False)
print("\n  ✔  Saved ttest_independent_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# RQ4 — LEVEL EFFECTS IN GAME (Spearman trend per group)
# ─────────────────────────────────────────────────────────────────────────────

print_section("RQ4 — LEVEL EFFECTS IN GAME (Spearman trends)")

level_rows = []
if not game_all.empty:
    for grp in ["Single","Multiple"]:
        sub = game_all[game_all["group"]==grp]
        print(f"\n  {grp} group:")
        for col, label in [("RT_ms","RT (ms)"),("success_rate","Success Rate %"),
                            ("hit_rate","Hit Rate %"),("false_alarms","False Alarms")]:
            tmp = sub[["level",col]].dropna()
            if len(tmp) < 3:
                print(f"    {label:<22}  — insufficient data"); continue
            rho, p = spearmanr(tmp["level"], tmp[col])
            direction = "↑ increases" if rho > 0 else "↓ decreases"
            print(f"    {label:<22}  ρ = {rho:+.3f}   p = {p:.4f} {stars(p)}  {direction}")
            level_rows.append({"Group":grp,"DV":label,"rho":rho,"p":p})

pd.DataFrame(level_rows).to_csv(OUT / "level_trend_results.csv", index=False)
print("\n  ✔  Saved level_trend_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# RELIABILITY — Split-half + Cronbach's α
# ─────────────────────────────────────────────────────────────────────────────

print_section("RELIABILITY — Lab RT (Split-Half, Cronbach's α)")

def split_half_r(arr):
    arr = np.array(arr, dtype=float)
    n   = (len(arr)//2)*2
    odd, even = arr[:n:2], arr[1:n:2]
    if len(odd) < 3: return np.nan, np.nan
    r, _ = pearsonr(odd, even)
    return r, (2*r)/(1+r)

def cronbach_alpha_from_participants(ptpt_series_list):
    """
    Cronbach's alpha treating each participant's lab RT trials as 'items'.
    ptpt_series_list: list of 1-D arrays (one per participant).
    """
    if len(ptpt_series_list) < 2: return np.nan
    min_len = min(len(x) for x in ptpt_series_list)
    if min_len < 2: return np.nan
    mat   = np.vstack([x[:min_len] for x in ptpt_series_list])  # participants × trials
    k     = mat.shape[0]
    item_vars  = mat.var(axis=1, ddof=1)
    total_var  = mat.sum(axis=0).var(ddof=1)
    if total_var == 0: return np.nan
    return (k/(k-1)) * (1 - item_vars.sum()/total_var)

reliability_rows = []
if not lab_all.empty:
    for grp in ["Single","Multiple"]:
        sub      = lab_all[lab_all["group"]==grp]
        all_rt   = sub["RT_ms"].dropna().values
        r, r_sb  = split_half_r(all_rt)
        ptpt_rts = [g["RT_ms"].dropna().values
                    for _, g in sub.groupby("participant") if len(g["RT_ms"].dropna()) >= 2]
        alpha    = cronbach_alpha_from_participants(ptpt_rts) if len(ptpt_rts) >= 2 else np.nan
        n_ptpt   = sub["participant"].nunique()
        print(f"\n  {grp} group  (n={n_ptpt} participants, {len(all_rt)} total trials)")
        print(f"    Split-half r (odd/even)     = {r:.3f}")
        print(f"    Spearman-Brown corrected r  = {r_sb:.3f}")
        print(f"    Cronbach's α (across ptpts) = {alpha:.3f}" if not np.isnan(alpha)
              else "    Cronbach's α: insufficient participants")
        reliability_rows.append({"Group":grp,"split_half_r":r,
                                  "spearman_brown_r":r_sb,"cronbach_alpha":alpha})

pd.DataFrame(reliability_rows).to_csv(OUT / "reliability_results.csv", index=False)
print("\n  ✔  Saved reliability_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

# Figure A — Interaction Plot
fig, ax = plt.subplots(figsize=(8,6))
ax.set_title("2×2 Interaction Plot: Mean RT by Group × Modality",
             fontsize=13, fontweight="bold")

for mod, df in [("Lab",ptpt_lab),("Game",ptpt_game)]:
    if df.empty: continue
    means, sems = [], []
    for grp in ["Single","Multiple"]:
        v = df[df["group"]==grp]["RT_mean"].dropna()
        means.append(v.mean()); sems.append(v.sem())
    ls, mk = ("--","o") if mod=="Lab" else ("-","s")
    ax.errorbar(["Single","Multiple"], means, yerr=sems, label=mod,
                linestyle=ls, marker=mk, color=COLORS[mod],
                linewidth=2.2, markersize=9, capsize=6)

ax.set_xlabel("Target Load", fontsize=12)
ax.set_ylabel("Mean RT (ms)", fontsize=12)
ax.legend(title="Modality", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "fig_A_interaction_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✔  Saved fig_A_interaction_plot.png")

# Figure B — ANOVA effect sizes
if not wide.empty:
    fig, ax = plt.subplots(figsize=(9,5))
    labels = ["Target Load\n(RQ2)","Modality\n(RQ3)","Interaction"]
    eta2s  = [eta2_A, eta2_B, eta2_AB]
    pvals  = [p_A,    p_B,    p_AB]
    bar_c  = ["#4C72B0","#DD8452","#55A868"]
    bars   = ax.bar(labels, eta2s, color=bar_c, edgecolor="white", alpha=0.85)
    ax.axhline(0.01, color="gray",  linestyle=":", linewidth=1.2, label="small (η²=.01)")
    ax.axhline(0.06, color="orange",linestyle=":", linewidth=1.2, label="medium (η²=.06)")
    ax.axhline(0.14, color="red",   linestyle=":", linewidth=1.2, label="large (η²=.14)")
    for bar, pv in zip(bars, pvals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                stars(pv), ha="center", fontsize=13, fontweight="bold")
    ax.set_title("Partial η² by Effect (ANOVA)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Partial η²")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "fig_B_effect_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved fig_B_effect_sizes.png")

# Figure C — Level effects with Spearman trend lines
if not game_all.empty:
    fig, axes = plt.subplots(2, 2, figsize=(14,9))
    fig.suptitle("RQ4 — Game Performance by Level", fontsize=14, fontweight="bold")
    for ax, (col, ylabel) in zip(axes.flat, [
        ("RT_ms","Initial RT (ms)"),("success_rate","Success Rate (%)"),
        ("hit_rate","Hit Rate (%)"),("false_alarms","False Alarms"),
    ]):
        for grp, color in [("Single",COLORS["Single"]),("Multiple",COLORS["Multiple"])]:
            sub = game_all[game_all["group"]==grp].groupby("level")[col].agg(["mean","sem"])
            ax.plot(sub.index, sub["mean"], "o-", color=color, label=grp,
                    linewidth=2, markersize=5, alpha=0.85)
            ax.fill_between(sub.index, sub["mean"]-sub["sem"],
                            sub["mean"]+sub["sem"], alpha=0.15, color=color)
            # Trend line
            tmp = game_all[game_all["group"]==grp][["level",col]].dropna()
            if len(tmp) >= 3:
                z  = np.polyfit(tmp["level"], tmp[col], 1)
                xs = np.linspace(tmp["level"].min(), tmp["level"].max(), 100)
                ax.plot(xs, np.poly1d(z)(xs), "--", color=color, alpha=0.5, linewidth=1.5)
        ax.set_xlabel("Level"); ax.set_ylabel(ylabel); ax.set_title(ylabel); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "fig_C_level_effects.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved fig_C_level_effects.png")

# Figure D — Concurrent validity scatter
if not ptpt_lab.empty and not ptpt_game.empty:
    merged = pd.merge(
        ptpt_lab[["participant","group","RT_mean"]].rename(columns={"RT_mean":"lab_RT"}),
        ptpt_game[["participant","group","RT_mean"]].rename(columns={"RT_mean":"game_RT"}),
        on=["participant","group"]
    )
    fig, axes = plt.subplots(1,2, figsize=(13,5))
    fig.suptitle("RQ1 — Concurrent Validity: Game vs Lab RT (Participant Means)",
                 fontsize=14, fontweight="bold")
    for ax, grp in zip(axes, ["Single","Multiple"]):
        sub = merged[merged["group"]==grp].dropna()
        if len(sub) < 2:
            ax.set_title(f"{grp} (n={len(sub)} — insufficient)"); continue
        ax.scatter(sub["game_RT"], sub["lab_RT"],
                   color=COLORS[grp], s=70, alpha=0.8, edgecolors="white")
        for _, row in sub.iterrows():
            ax.annotate(row["participant"],(row["game_RT"],row["lab_RT"]),
                        textcoords="offset points", xytext=(5,3), fontsize=7, alpha=0.7)
        z  = np.polyfit(sub["game_RT"],sub["lab_RT"],1)
        xs = np.linspace(sub["game_RT"].min(),sub["game_RT"].max(),100)
        ax.plot(xs,np.poly1d(z)(xs),"--",color="black",linewidth=1.5)
        r, p = pearsonr(sub["game_RT"],sub["lab_RT"])
        ax.set_title(f"{grp}  (n={len(sub)})   r = {r:.3f} {stars(p)}")
        ax.set_xlabel("Game Mean RT (ms)"); ax.set_ylabel("Lab Mean RT (ms)")
    plt.tight_layout()
    plt.savefig(OUT / "fig_D_validity_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved fig_D_validity_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print_section("RESULTS SUMMARY")
print("""
  RQ1 — Concurrent Validity
      Pearson r & Spearman ρ (Game vs Lab participant means) — see above.

  RQ2 — Target Load Effect
      ANOVA main effect A + Welch t-test (Single vs Multiple) — see above.

  RQ3 — Modality Effect
      ANOVA main effect B + Paired t-test (Game vs Lab) — see above.

  RQ4 — Level Effects
      Spearman trend per group per DV — see above.

  Reliability
      Split-half r, Spearman-Brown r, Cronbach's α — see above.

  All results saved to: outputs/
    anova_results.csv
    ttest_independent_results.csv
    level_trend_results.csv
    reliability_results.csv
    participant_lab_aggregated.csv
    participant_game_aggregated.csv
""")

print("  ✔  ANALYSIS COMPLETE")