"""
=============================================================================
STREAMLINED EDA — Selective Attention Study
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

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"Single": "#4C72B0", "Multiple": "#DD8452"}

# ── CONFIGURE THESE PATHS ─────────────────────────────────────────────────────
DATA_ROOT = Path("preprocessed_data")
OUT       = Path("outputs")
OUT.mkdir(exist_ok=True)
EXPECTED_N = {"Single": 21, "Multiple": 16}
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def extract_pid(path: Path) -> str:
    m = re.match(r"^(\d+)", path.stem)
    return m.group(1) if m else path.stem

def load_lab(path: Path, participant_id: str, group: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["target_col"])
    df = df[df["target_col"].isin(["red", "white"])].copy()
    df["participant"] = participant_id
    df["group"]       = group
    
    def first_click_ms(val):
        try: return float(ast.literal_eval(str(val))[0]) * 1000
        except: return np.nan

    df["RT_ms"]     = df["mouse.time"].apply(first_click_ms)
    df["is_target"] = (df["target_col"] == "red").astype(int)
    df["hit"]       = 1
    return df[["participant", "group", "is_target", "RT_ms", "hit"]]

def load_game(path: Path, participant_id: str, group: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={
        "Level": "level", "Completed": "completed", "SuccessRate(%)": "success_rate",
        "HitRate(%)": "hit_rate", "FalseAlarms": "false_alarms",
        "InitialResponseTime(ms)": "RT_ms"
    })
    df["participant"] = participant_id
    df["group"]       = group
    df["completed"]   = df["completed"].astype(str).str.lower().isin(["true","1","yes"])
    return df[["participant", "group", "level", "completed", "success_rate", 
               "hit_rate", "false_alarms", "RT_ms"]]

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
all_lab_dfs, all_game_dfs, load_log = [], [], []

for group_dir, group_label in [("multiple", "Multiple"), ("single", "Single")]:
    lab_dir   = DATA_ROOT / group_dir / "lab"
    phone_dir = DATA_ROOT / group_dir / "phone"
    if not lab_dir.exists() or not phone_dir.exists(): continue

    lab_map   = {extract_pid(f): f for f in sorted(lab_dir.glob("*.csv"))}
    phone_map = {extract_pid(f): f for f in sorted(phone_dir.glob("*.csv"))}
    all_pids  = sorted(set(lab_map) | set(phone_map), key=lambda x: int(x) if x.isdigit() else x)

    for pid in all_pids:
        has_lab, has_game = pid in lab_map, pid in phone_map
        
        if has_lab: all_lab_dfs.append(load_lab(lab_map[pid], pid, group_label))
        if has_game: all_game_dfs.append(load_game(phone_map[pid], pid, group_label))
            
        load_log.append({
            "participant": pid, "group": group_label,
            "has_lab": has_lab, "has_game": has_game,
            "status": "complete" if (has_lab and has_game) else "partial"
        })

lab_all  = pd.concat(all_lab_dfs, ignore_index=True) if all_lab_dfs else pd.DataFrame()
game_all = pd.concat(all_game_dfs, ignore_index=True) if all_game_dfs else pd.DataFrame()
log_df   = pd.DataFrame(load_log)

# ─────────────────────────────────────────────────────────────────────────────
# 3. ATTRITION & MISSING DATA
# ─────────────────────────────────────────────────────────────────────────────
attrition_stats = log_df.groupby("group").apply(lambda x: pd.Series({
    "Enrolled": len(x),
    "Complete_Records": (x["status"] == "complete").sum(),
    "Dropout_Rate_%": 100 * (1 - (x["status"] == "complete").sum() / EXPECTED_N.get(x.name, len(x)))
})).reset_index()
attrition_stats.to_csv(OUT / "01_attrition_rates.csv", index=False)

# Attrition Plot
plt.figure(figsize=(8, 5))
sns.countplot(data=log_df, x="group", hue="status", palette="Set2")
plt.title("Participant Completion Status by Group")
plt.savefig(OUT / "fig_01_attrition.png", bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. DESCRIPTIVE STATISTICS (Numeric Data)
# ─────────────────────────────────────────────────────────────────────────────
def get_descriptives(df, cols):
    if df.empty: return pd.DataFrame()
    return df.groupby('group')[cols].agg(
        ['count', 'mean', 'std', 'var', 'min', 'max', lambda x: x.skew()]
    ).rename(columns={'<lambda_0>': 'skew'})

lab_desc = get_descriptives(lab_all, ["RT_ms"])
game_desc = get_descriptives(game_all, ["RT_ms", "success_rate", "false_alarms"])

lab_desc.to_csv(OUT / "02_lab_descriptives.csv")
game_desc.to_csv(OUT / "03_game_descriptives.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 5. DISTRIBUTIONS (Histograms)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Reaction Time Distributions (Lab vs Phone)", fontweight="bold")

for i, (df, title) in enumerate([(lab_all, "Lab RT (ms)"), (game_all, "Game RT (ms)")]):
    if not df.empty:
        sns.histplot(data=df, x="RT_ms", hue="group", kde=True, ax=axes[i], palette=COLORS, alpha=0.6)
        axes[i].set_title(title)

plt.tight_layout()
plt.savefig(OUT / "fig_02_rt_distributions.png", bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6. VARIANCE & OUTLIERS (Boxplots)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Variance and Outliers by Group", fontweight="bold")

if not lab_all.empty:
    sns.boxplot(data=lab_all, x="group", y="RT_ms", ax=axes[0], palette=COLORS)
    axes[0].set_title("Lab Task: RT (ms)")

if not game_all.empty:
    sns.boxplot(data=game_all, x="group", y="RT_ms", ax=axes[1], palette=COLORS)
    axes[1].set_title("Game Task: RT (ms)")
    
    sns.boxplot(data=game_all, x="group", y="success_rate", ax=axes[2], palette=COLORS)
    axes[2].set_title("Game Task: Success Rate (%)")

plt.tight_layout()
plt.savefig(OUT / "fig_03_variance_boxplots.png", bbox_inches="tight")
plt.close()

print(f"EDA Complete. Cleaned data and core plots saved to {OUT.resolve()}")