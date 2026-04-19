"""
REPORT 2 — STEP 1: Data Loading, Log-RT, Level Selection
Run from folder containing preprocessed_data/
"""

import pandas as pd
import numpy as np
import ast, re, warnings, pickle
from pathlib import Path
from scipy.stats import shapiro, skew

warnings.filterwarnings("ignore")

DATA_ROOT = Path("../preprocessed_data")
OUT = Path("outputs_r2"); OUT.mkdir(exist_ok=True)

# ── Parsers (same as R1) ──────────────────────────────────────────────────────

def parse_list_col(val):
    try:
        return ast.literal_eval(str(val))
    except:
        return []

def extract_pid(path):
    m = re.match(r"^(\d+)", path.stem)
    return m.group(1) if m else path.stem

def load_lab_file(path, pid, group):
    df = pd.read_csv(path)
    df = df.dropna(subset=["target_col"])
    df = df[df["target_col"].isin(["red","white"])].copy().reset_index(drop=True)
    df["participant"] = pid
    df["group"] = group
    df["trial_n"] = range(1, len(df)+1)
    df["is_target"] = (df["target_col"] == "red").astype(int)
    df["mt_list"] = df["mouse.time"].apply(parse_list_col)
    df["RT_ms"] = df["mt_list"].apply(lambda x: float(x[0])*1000 if len(x) > 0 else np.nan)
    df["n_clicks"] = df["mt_list"].apply(len)
    df["trial_dur_s"] = df["trial.stopped"] - df["trial.started"]
    df["all_click_ms"] = df["mt_list"].apply(lambda x: [v*1000 for v in x] if x else [])
    df["log_RT"] = np.log(df["RT_ms"])
    return df[["participant","group","trial_n","target_col","is_target",
               "RT_ms","log_RT","n_clicks","trial_dur_s","all_click_ms"]]

def load_game_file(path, pid, group):
    df = pd.read_csv(path)
    df["participant"] = pid
    df["group"] = group
    df = df.rename(columns={
        "Level": "level", "Completed": "completed",
        "SuccessRate(%)": "success_rate", "HitRate(%)": "hit_rate",
        "FalseAlarms": "false_alarms", "InitialResponseTime(ms)": "RT_ms",
        "AvgInterTargetTime(ms)": "avg_inter_target_ms",
        "FinalScore": "final_score",
    })
    df["completed"] = df["completed"].astype(str).str.lower().isin(["true","1","yes"])
    df["log_RT"] = np.log(df["RT_ms"].replace(0, np.nan))
    return df[["participant","group","level","completed","success_rate","hit_rate",
               "false_alarms","RT_ms","log_RT","avg_inter_target_ms","final_score"]]

def load_all(data_root):
    lab_dfs, game_dfs = [], []
    for group_dir, group_label in [("multiple","Multiple"),("single","Single")]:
        lab_dir = data_root / group_dir / "lab"
        phone_dir = data_root / group_dir / "phone"
        lab_map   = {extract_pid(f): f for f in sorted(lab_dir.glob("*.csv"))}
        phone_map = {extract_pid(f): f for f in sorted(phone_dir.glob("*.csv"))}
        for pid in sorted(set(lab_map)|set(phone_map), key=lambda x: int(x) if x.isdigit() else x):
            if pid in lab_map:
                try: lab_dfs.append(load_lab_file(lab_map[pid], pid, group_label))
                except Exception as e: print(f"  Lab error PID {pid}: {e}")
            if pid in phone_map:
                try: game_dfs.append(load_game_file(phone_map[pid], pid, group_label))
                except Exception as e: print(f"  Game error PID {pid}: {e}")
    return (pd.concat(lab_dfs, ignore_index=True) if lab_dfs else pd.DataFrame(),
            pd.concat(game_dfs, ignore_index=True) if game_dfs else pd.DataFrame())

print("="*60)
print("STEP 1: Loading data...")
print("="*60)

lab_all, game_all = load_all(DATA_ROOT)

print(f"\n  Lab trials loaded  : {len(lab_all)}")
print(f"  Game levels loaded : {len(game_all)}")
print(f"  Lab participants   : {lab_all['participant'].nunique()}")
print(f"  Game participants  : {game_all['participant'].nunique()}")

# ── Log-RT check ─────────────────────────────────────────────────────────────

print("\n" + "─"*60)
print("LOG-RT NORMALITY CHECK (Shapiro-Wilk on residuals)")
print("─"*60)

for grp in ["Single","Multiple"]:
    for mod, df, col in [("Lab", lab_all, "log_RT"), ("Game", game_all, "log_RT")]:
        sub = df[df["group"]==grp][col].dropna().values
        if len(sub) < 3: continue
        raw_sub = df[df["group"]==grp]["RT_ms"].dropna().values
        sw_raw, p_raw = shapiro(raw_sub)
        sw_log, p_log = shapiro(sub)
        skew_raw = skew(raw_sub)
        skew_log = skew(sub)
        print(f"\n  {grp} {mod}  (n={len(sub)}):")
        print(f"    Raw  RT — SW p={p_raw:.4f}  skew={skew_raw:.3f}  "
              f"{'✗ non-normal' if p_raw<.05 else '✓ normal'}")
        print(f"    Log  RT — SW p={p_log:.4f}  skew={skew_log:.3f}  "
              f"{'✗ non-normal' if p_log<.05 else '✓ normal'}")

# ── Level selection analysis ─────────────────────────────────────────────────

print("\n" + "─"*60)
print("LEVEL SELECTION: How many participants per level?")
print("─"*60)

for grp in ["Single","Multiple"]:
    sub = game_all[game_all["group"]==grp]
    n_total = sub["participant"].nunique()
    print(f"\n  {grp} group (n={n_total} participants):")
    print(f"  {'Level':<8} {'N_attempts':<14} {'N_participants':<18} {'% of group':<12} {'Completion%'}")
    for lvl in sorted(sub["level"].unique()):
        lvl_data = sub[sub["level"]==lvl]
        n_ptpt = lvl_data["participant"].nunique()
        n_att = len(lvl_data)
        pct = 100 * n_ptpt / n_total
        comp = 100 * lvl_data["completed"].mean()
        print(f"  {lvl:<8} {n_att:<14} {n_ptpt:<18} {pct:<12.1f} {comp:.1f}%")

# ── Identify comparable level window ─────────────────────────────────────────

print("\n" + "─"*60)
print("COMPARABLE LEVEL WINDOW — RECOMMENDATION")
print("─"*60)

s_max_levels = game_all[game_all["group"]=="Single"].groupby("participant")["level"].max()
m_max_levels = game_all[game_all["group"]=="Multiple"].groupby("participant")["level"].max()

print(f"\n  Single group max level:   M={s_max_levels.mean():.2f}  "
      f"Min={s_max_levels.min()}  Max={s_max_levels.max()}")
print(f"  Multiple group max level: M={m_max_levels.mean():.2f}  "
      f"Min={m_max_levels.min()}  Max={m_max_levels.max()}")

# Find the level where BOTH groups have 100% participant coverage
both_full_coverage = []
for lvl in range(1, 16):
    s_cov = game_all[(game_all["group"]=="Single")&(game_all["level"]==lvl)]["participant"].nunique()
    m_cov = game_all[(game_all["group"]=="Multiple")&(game_all["level"]==lvl)]["participant"].nunique()
    s_tot = game_all[game_all["group"]=="Single"]["participant"].nunique()
    m_tot = game_all[game_all["group"]=="Multiple"]["participant"].nunique()
    if s_cov == s_tot and m_cov == m_tot:
        both_full_coverage.append(lvl)

max_comparable_level = max(both_full_coverage) if both_full_coverage else None
print(f"\n  Levels with 100% coverage in BOTH groups: {both_full_coverage}")
print(f"  ➤ Recommended comparable window: Levels 1–{max_comparable_level}")
print(f"  ➤ Level 1 is the best single-level lab equivalent (first-encounter RT)")

# ── Build participant-level aggregates ───────────────────────────────────────

print("\n" + "─"*60)
print("BUILDING PARTICIPANT-LEVEL AGGREGATES")
print("─"*60)

ptpt_lab = (
    lab_all.groupby(["participant","group"])
    .agg(
        RT_mean   = ("RT_ms",  "mean"),
        RT_sd     = ("RT_ms",  "std"),
        RT_mdn    = ("RT_ms",  "median"),
        logRT_mean= ("log_RT", "mean"),
        logRT_sd  = ("log_RT", "std"),
        n_trials  = ("RT_ms",  "count"),
    ).reset_index()
)

# Full game aggregate (all levels)
ptpt_game_all = (
    game_all.groupby(["participant","group"])
    .agg(
        RT_mean            = ("RT_ms",              "mean"),
        RT_sd              = ("RT_ms",              "std"),
        RT_mdn             = ("RT_ms",              "median"),
        logRT_mean         = ("log_RT",             "mean"),
        logRT_sd           = ("log_RT",             "std"),
        success_rate       = ("success_rate",       "mean"),
        hit_rate           = ("hit_rate",           "mean"),
        false_alarms       = ("false_alarms",       "mean"),
        n_levels           = ("level",              "count"),
        max_level          = ("level",              "max"),
        pct_complete       = ("completed",          "mean"),
        inter_tgt_M        = ("avg_inter_target_ms","mean"),
    ).reset_index()
)

# Comparable window aggregate (levels 1 to max_comparable_level only)
if max_comparable_level:
    game_comparable = game_all[game_all["level"] <= max_comparable_level].copy()
    ptpt_game_comp = (
        game_comparable.groupby(["participant","group"])
        .agg(
            RT_mean_comp   = ("RT_ms",        "mean"),
            RT_sd_comp     = ("RT_ms",        "std"),
            logRT_mean_comp= ("log_RT",       "mean"),
            SR_comp        = ("success_rate", "mean"),
            HR_comp        = ("hit_rate",     "mean"),
            FA_comp        = ("false_alarms", "mean"),
            n_levels_comp  = ("level",        "count"),
        ).reset_index()
    )

# Level 1 only aggregate (most comparable to lab)
game_lvl1 = game_all[game_all["level"] == 1].copy()
ptpt_game_lvl1 = (
    game_lvl1.groupby(["participant","group"])
    .agg(
        RT_lvl1    = ("RT_ms",        "mean"),
        logRT_lvl1 = ("log_RT",       "mean"),
        SR_lvl1    = ("success_rate", "mean"),
    ).reset_index()
)

print(f"\n  ptpt_lab shape          : {ptpt_lab.shape}")
print(f"  ptpt_game_all shape     : {ptpt_game_all.shape}")
if max_comparable_level:
    print(f"  ptpt_game_comp shape    : {ptpt_game_comp.shape}  (levels 1–{max_comparable_level})")
print(f"  ptpt_game_lvl1 shape    : {ptpt_game_lvl1.shape}  (level 1 only)")

# Print participant-level summary
print("\n  Participant lab RT means:")
print(ptpt_lab.groupby("group")[["RT_mean","logRT_mean"]].describe().round(3).to_string())
print("\n  Participant game RT means (all levels):")
print(ptpt_game_all.groupby("group")[["RT_mean","logRT_mean"]].describe().round(3).to_string())

# ── Save cache ────────────────────────────────────────────────────────────────

with open(OUT / "_r2_cache.pkl","wb") as f:
    pickle.dump({
        "lab_all":          lab_all,
        "game_all":         game_all,
        "game_comparable":  game_comparable if max_comparable_level else pd.DataFrame(),
        "game_lvl1":        game_lvl1,
        "ptpt_lab":         ptpt_lab,
        "ptpt_game_all":    ptpt_game_all,
        "ptpt_game_comp":   ptpt_game_comp if max_comparable_level else pd.DataFrame(),
        "ptpt_game_lvl1":   ptpt_game_lvl1,
        "max_comparable_level": max_comparable_level,
    }, f)

print(f"\n  ✔  Saved _r2_cache.pkl to {OUT}/")
print("\n  STEP 1 COMPLETE — paste output above before running Step 2.\n")