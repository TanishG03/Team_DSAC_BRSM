"""
╔══════════════════════════════════════════════════════════════════════╗
║  SELECTIVE ATTENTION STUDY — COMPLETE ANALYSIS                       ║
║  PART 1: Data Loading, Parsing & Structure Inspection                ║
╚══════════════════════════════════════════════════════════════════════╝

Covers:
  • Parsing all 4 dataset types (Single Lab, Multi Lab, Single Game, Multi Game)
  • Understanding variable semantics for each type
  • Handling structural differences (single click vs multi-click lab files)
  • Producing clean participant-level and trial-level DataFrames
  • Saving summary of loaded data

Run from the folder containing preprocessed_data/
"""

import pandas as pd
import numpy as np
import ast, re, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_ROOT = Path("preprocessed_data")
OUT       = Path("outputs"); OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1A — LAB FILE PARSER
# Key insight:
#   Single target lab → mouse.time is a 1-element list: only one target to click
#   Multi  target lab → mouse.time is a 5-element list: 5 targets to click per trial
#   RT = mouse.time[0] * 1000 ms  (time from stimulus onset to first click)
#   Trial duration = trial.stopped − trial.started (for multi: total task time)
# ─────────────────────────────────────────────────────────────────────────────

def parse_list_col(val):
    """Safely parse a stringified Python list."""
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []

def load_lab_file(path: Path, participant_id: str, group: str) -> pd.DataFrame:
    """
    Load one participant's lab (visual search) CSV.

    Returns one row per trial with:
      participant, group, trial_n, target_col (red/white),
      is_target (1=red, 0=white),
      RT_ms       → first click time in ms (= first response time)
      n_clicks    → number of clicks in that trial
      trial_dur_s → total trial duration in seconds
      hit         → 1 (participant always responded — no omission tracking in raw)
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["target_col"])
    df = df[df["target_col"].isin(["red", "white"])].copy()
    df.reset_index(drop=True, inplace=True)

    df["participant"] = participant_id
    df["group"]       = group
    df["trial_n"]     = range(1, len(df) + 1)
    df["is_target"]   = (df["target_col"] == "red").astype(int)

    # Parse mouse.time list → RT_ms (first element, convert s→ms)
    df["mt_list"]    = df["mouse.time"].apply(parse_list_col)
    df["RT_ms"]      = df["mt_list"].apply(
        lambda x: float(x[0]) * 1000 if len(x) > 0 else np.nan
    )
    df["n_clicks"]   = df["mt_list"].apply(len)
    df["trial_dur_s"]= df["trial.stopped"] - df["trial.started"]

    # For multi-target: all_click_times_ms gives the inter-target sequence
    df["all_click_ms"] = df["mt_list"].apply(
        lambda x: [v * 1000 for v in x] if x else []
    )

    df["hit"] = 1  # click always recorded (no omission in this data)

    keep = ["participant","group","trial_n","target_col","is_target",
            "RT_ms","n_clicks","trial_dur_s","all_click_ms","hit"]
    return df[keep]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1B — GAME FILE PARSER
# Key variables:
#   InitialResponseTime(ms) → RT to first target (directly usable)
#   AvgInterTargetTime(ms)  → mean time between subsequent hits (multi only)
#   SuccessRate(%)          → % of target+distractor interactions that were correct
#   HitRate(%)              → % of actual targets that were hit
#   FalseAlarms             → clicks on non-targets
#   Level                   → difficulty (higher = more targets / distractors)
#   Completed               → True = finished level, False = failed/timeout
# ─────────────────────────────────────────────────────────────────────────────

def parse_hit_positions(val: str) -> list:
    """Parse '(x1,y1);(x2,y2);...' into list of (x,y) tuples."""
    try:
        return [tuple(map(float, p.strip("()").split(",")))
                for p in str(val).split(";") if p.strip()]
    except Exception:
        return []

def load_game_file(path: Path, participant_id: str, group: str) -> pd.DataFrame:
    """
    Load one participant's game (attentional spotter) CSV.

    Returns one row per level attempt with:
      participant, group, level, completed,
      success_rate, hit_rate, false_alarms,
      RT_ms (=InitialResponseTime), avg_inter_target_ms,
      final_score, n_hits (parsed from HitPositions)
    """
    df = pd.read_csv(path)
    df["participant"] = participant_id
    df["group"]       = group

    df = df.rename(columns={
        "Level":                   "level",
        "Completed":               "completed",
        "SuccessRate(%)":          "success_rate",
        "HitRate(%)":              "hit_rate",
        "FalseAlarms":             "false_alarms",
        "InitialResponseTime(ms)": "RT_ms",
        "AvgInterTargetTime(ms)":  "avg_inter_target_ms",
        "FinalScore":              "final_score",
        "GameMode":                "game_mode",
        "Timestamp":               "timestamp",
    })

    df["completed"]  = df["completed"].astype(str).str.lower().isin(["true","1","yes"])
    df["hit_coords"] = df["HitPositions(x,y)"].apply(parse_hit_positions)
    df["n_hits"]     = df["hit_coords"].apply(len)
    df["timestamp"]  = pd.to_datetime(df["timestamp"], errors="coerce")

    keep = ["participant","group","level","completed","success_rate","hit_rate",
            "false_alarms","RT_ms","avg_inter_target_ms","final_score",
            "game_mode","timestamp","n_hits","hit_coords"]
    return df[keep]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1C — BULK LOADER (walks directory tree)
# ─────────────────────────────────────────────────────────────────────────────

def extract_pid(path: Path) -> str:
    m = re.match(r"^(\d+)", path.stem)
    return m.group(1) if m else path.stem

def load_all(data_root: Path):
    """
    Walk preprocessed_data/{multiple,single}/{lab,phone}/
    Return four DataFrames: lab_all, game_all, ptpt_log, load_errors
    """
    lab_dfs, game_dfs, log_rows, errors = [], [], [], []

    for group_dir, group_label in [("multiple","Multiple"), ("single","Single")]:
        lab_dir   = data_root / group_dir / "lab"
        phone_dir = data_root / group_dir / "phone"

        if not lab_dir.exists():
            print(f"  ⚠  Missing: {lab_dir}")
        if not phone_dir.exists():
            print(f"  ⚠  Missing: {phone_dir}")
            continue

        lab_map   = {extract_pid(f): f for f in sorted(lab_dir.glob("*.csv"))}
        phone_map = {extract_pid(f): f for f in sorted(phone_dir.glob("*.csv"))}
        all_pids  = sorted(set(lab_map) | set(phone_map),
                           key=lambda x: int(x) if x.isdigit() else x)

        for pid in all_pids:
            row = {"participant": pid, "group": group_label,
                   "has_lab": False, "has_game": False}

            if pid in lab_map:
                try:
                    df = load_lab_file(lab_map[pid], pid, group_label)
                    lab_dfs.append(df)
                    row["has_lab"]    = True
                    row["lab_trials"] = len(df)
                    row["lab_file"]   = lab_map[pid].name
                except Exception as e:
                    errors.append({"pid": pid, "type": "lab", "error": str(e)})

            if pid in phone_map:
                try:
                    df = load_game_file(phone_map[pid], pid, group_label)
                    game_dfs.append(df)
                    row["has_game"]    = True
                    row["game_levels"] = len(df)
                    row["game_file"]   = phone_map[pid].name
                except Exception as e:
                    errors.append({"pid": pid, "type": "game", "error": str(e)})

            row["status"] = ("complete" if (row["has_lab"] and row["has_game"])
                             else "lab_only" if row["has_lab"]
                             else "game_only" if row["has_game"]
                             else "missing")
            log_rows.append(row)

    lab_all  = pd.concat(lab_dfs,  ignore_index=True) if lab_dfs  else pd.DataFrame()
    game_all = pd.concat(game_dfs, ignore_index=True) if game_dfs else pd.DataFrame()
    ptpt_log = pd.DataFrame(log_rows)
    err_df   = pd.DataFrame(errors)

    return lab_all, game_all, ptpt_log, err_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1D — RUN & INSPECT
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("PART 1 — DATA LOADING & STRUCTURE")
print("=" * 65)

lab_all, game_all, ptpt_log, errors = load_all(DATA_ROOT)

# ── Participant log ──────────────────────────────────────────────────────────
print("\n── Participant Load Log ─────────────────────────────────────")
for grp in ["Single", "Multiple"]:
    sub = ptpt_log[ptpt_log["group"] == grp]
    print(f"\n  {grp} group  ({len(sub)} participants found)")
    for status, cnt in sub["status"].value_counts().items():
        print(f"    {status:<15} : {cnt}")

if not errors.empty:
    print("\n  ⚠ Load errors:")
    print(errors.to_string(index=False))

# ── Dataset shapes ───────────────────────────────────────────────────────────
print("\n── Dataset Shapes ────────────────────────────────────────────")
for name, df in [("Lab (all trials)", lab_all), ("Game (all levels)", game_all)]:
    if df.empty:
        print(f"  {name}: EMPTY")
        continue
    print(f"\n  {name}:  {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"    Columns : {df.columns.tolist()}")
    print(f"    Participants : {df['participant'].nunique()}")
    for grp in ["Single","Multiple"]:
        sub = df[df["group"] == grp]
        n_p = sub["participant"].nunique()
        print(f"    {grp:<12}: {n_p} participants, {len(sub)} rows")

# ── Variable summary per dataset ─────────────────────────────────────────────
print("\n── Variable Overview: Lab Data ───────────────────────────────")
if not lab_all.empty:
    print(lab_all[["target_col","is_target","RT_ms",
                   "n_clicks","trial_dur_s","hit"]].describe(include="all").round(3).to_string())
    print(f"\n  Target distribution:\n{lab_all.groupby(['group','target_col']).size().to_string()}")

print("\n── Variable Overview: Game Data ──────────────────────────────")
if not game_all.empty:
    num_cols = ["level","success_rate","hit_rate","false_alarms",
                "RT_ms","avg_inter_target_ms","final_score","n_hits"]
    present  = [c for c in num_cols if c in game_all.columns]
    print(game_all[present].describe().round(2).to_string())
    print(f"\n  Completion rate:\n{game_all.groupby(['group','completed']).size().to_string()}")

# ── Save load log ─────────────────────────────────────────────────────────────
ptpt_log.to_csv(OUT / "load_log.csv", index=False)
print(f"\n  ✔  Saved load_log.csv")

# ── Export for downstream parts ───────────────────────────────────────────────
import pickle
with open(OUT / "_data_cache.pkl","wb") as f:
    pickle.dump({"lab_all": lab_all, "game_all": game_all,
                 "ptpt_log": ptpt_log}, f)
print("  ✔  Saved _data_cache.pkl  (used by Parts 2–5)")
print("\n  PART 1 COMPLETE.\n")


"""
╔══════════════════════════════════════════════════════════════════════╗
║  SELECTIVE ATTENTION STUDY — COMPLETE ANALYSIS                       ║
║  PART 2: Descriptive Statistics (All 4 Dataset Types)                ║
╚══════════════════════════════════════════════════════════════════════╝

Covers (per dataset × group):
  2A. Single Lab   — RT, target type, trial duration
  2B. Multi Lab    — RT, click sequences, trial duration
  2C. Single Game  — RT, success/hit rates, score, level
  2D. Multi Game   — RT, success/hit rates, false alarms, inter-target time
  2E. 4-cell summary table (Group × Modality)
  2F. Participant-level aggregation (Slide 15)
"""

import pandas as pd
import numpy as np
import pickle, warnings
from pathlib import Path
from scipy.stats import skew, kurtosis, shapiro

warnings.filterwarnings("ignore")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ── Load cached data from Part 1 ─────────────────────────────────────────────
with open(OUT / "_data_cache.pkl","rb") as f:
    cache = pickle.load(f)
lab_all  = cache["lab_all"]
game_all = cache["game_all"]
ptpt_log = cache["ptpt_log"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Full descriptive stats for a numeric array
# ─────────────────────────────────────────────────────────────────────────────

def full_desc(arr, label=""):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {k: np.nan for k in ["n","M","SD","SE","Mdn","IQR","Min","Max","Skew","Kurt","SW_p"]}
    sw_stat, sw_p = (shapiro(arr) if len(arr) >= 3 else (np.nan, np.nan))
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "n":    len(arr),
        "M":    arr.mean(),
        "SD":   arr.std(ddof=1),
        "SE":   arr.std(ddof=1) / np.sqrt(len(arr)),
        "Mdn":  np.median(arr),
        "IQR":  q3 - q1,
        "Min":  arr.min(),
        "Max":  arr.max(),
        "Skew": skew(arr),
        "Kurt": kurtosis(arr),
        "SW_p": sw_p,   # Shapiro-Wilk p (normality check)
    }

def print_desc(d, label, indent="  "):
    sw_flag = "" if np.isnan(d["SW_p"]) else ("  ✓ normal" if d["SW_p"] > .05 else "  ✗ non-normal")
    print(f"{indent}{label}:")
    print(f"{indent}  n={d['n']}  M={d['M']:.2f}  SD={d['SD']:.2f}  SE={d['SE']:.2f}"
          f"  Mdn={d['Mdn']:.2f}  IQR={d['IQR']:.2f}")
    print(f"{indent}  Min={d['Min']:.2f}  Max={d['Max']:.2f}"
          f"  Skew={d['Skew']:.3f}  Kurt={d['Kurt']:.3f}"
          f"  SW-p={d['SW_p']:.4f}{sw_flag}")

def section(title):
    print("\n" + "═"*70)
    print(f"  {title}")
    print("═"*70)


# ─────────────────────────────────────────────────────────────────────────────
# 2A. SINGLE TARGET LAB
# ─────────────────────────────────────────────────────────────────────────────

section("2A — SINGLE TARGET LAB (visual_search, single group)")

s_lab = lab_all[lab_all["group"]=="Single"].copy() if not lab_all.empty else pd.DataFrame()

if s_lab.empty:
    print("  No data loaded.")
else:
    print(f"\n  Participants : {s_lab['participant'].nunique()}")
    print(f"  Total trials : {len(s_lab)}")
    print(f"  Trials/participant: {len(s_lab)/s_lab['participant'].nunique():.1f}")
    print(f"  n_clicks per trial: always {s_lab['n_clicks'].unique().tolist()} "
          f"(single target = one click)")

    print("\n  >> RT (ms) — all trials:")
    print_desc(full_desc(s_lab["RT_ms"]), "Overall RT")

    print("\n  >> RT by target colour:")
    for col, label in [(1,"Target (red)"),(0,"Distractor (white)")]:
        arr = s_lab[s_lab["is_target"]==col]["RT_ms"]
        print_desc(full_desc(arr), label)

    print("\n  >> Trial Duration (s):")
    print_desc(full_desc(s_lab["trial_dur_s"]), "Trial Duration")

    print("\n  >> Per-participant RT means:")
    pp = s_lab.groupby("participant")["RT_ms"].agg(["mean","std","count"])
    print(pp.round(2).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 2B. MULTI TARGET LAB
# ─────────────────────────────────────────────────────────────────────────────

section("2B — MULTIPLE TARGET LAB (visual_search, multiple group)")

m_lab = lab_all[lab_all["group"]=="Multiple"].copy() if not lab_all.empty else pd.DataFrame()

if m_lab.empty:
    print("  No data loaded.")
else:
    print(f"\n  Participants : {m_lab['participant'].nunique()}")
    print(f"  Total trials : {len(m_lab)}")
    print(f"  Clicks per trial: {m_lab['n_clicks'].value_counts().to_dict()} "
          f"(5 targets per trial)")

    print("\n  >> Initial RT (ms) — time to FIRST target click:")
    print_desc(full_desc(m_lab["RT_ms"]), "Initial RT")

    print("\n  >> Initial RT by target colour (trial type):")
    for col, label in [(1,"Target (red)"),(0,"Distractor (white)")]:
        arr = m_lab[m_lab["is_target"]==col]["RT_ms"]
        print_desc(full_desc(arr), label)

    print("\n  >> Total Trial Duration (s) [time to click all 5 targets]:")
    print_desc(full_desc(m_lab["trial_dur_s"]), "Trial Duration")

    # Inter-click intervals from multi-target
    all_ici = []
    for clicks in m_lab["all_click_ms"]:
        if len(clicks) >= 2:
            all_ici.extend(np.diff(clicks).tolist())
    print("\n  >> Inter-Click Intervals (ms) [time between successive target clicks]:")
    print_desc(full_desc(all_ici), "Inter-Click Interval")

    print("\n  >> Per-participant RT means:")
    pp = m_lab.groupby("participant")["RT_ms"].agg(["mean","std","count"])
    print(pp.round(2).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 2C. SINGLE TARGET GAME
# ─────────────────────────────────────────────────────────────────────────────

section("2C — SINGLE TARGET GAME (attentional_spotter, single group)")

s_game = game_all[game_all["group"]=="Single"].copy() if not game_all.empty else pd.DataFrame()

if s_game.empty:
    print("  No data loaded.")
else:
    print(f"\n  Participants : {s_game['participant'].nunique()}")
    print(f"  Total level attempts : {len(s_game)}")
    print(f"  Level range : {s_game['level'].min()} – {s_game['level'].max()}")
    print(f"  Completion : {s_game['completed'].sum()}/{len(s_game)} "
          f"({100*s_game['completed'].mean():.1f}%)")
    print(f"  AvgInterTargetTime : {s_game['avg_inter_target_ms'].unique().tolist()} "
          f"(always 0 — single target has no inter-target interval)")

    print("\n  >> Initial RT (ms):")
    print_desc(full_desc(s_game["RT_ms"]), "RT")

    print("\n  >> Success Rate (%):")
    print_desc(full_desc(s_game["success_rate"]), "Success Rate")

    print("\n  >> Hit Rate (%):")
    print_desc(full_desc(s_game["hit_rate"]), "Hit Rate")

    print("\n  >> False Alarms:")
    print_desc(full_desc(s_game["false_alarms"]), "False Alarms")

    print("\n  >> Final Score:")
    print_desc(full_desc(s_game["final_score"]), "Score")

    print("\n  >> Levels by participant:")
    pp = s_game.groupby("participant").agg(
        n_levels=("level","count"), max_level=("level","max"),
        mean_RT=("RT_ms","mean"), mean_SR=("success_rate","mean")
    )
    print(pp.round(2).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 2D. MULTI TARGET GAME
# ─────────────────────────────────────────────────────────────────────────────

section("2D — MULTIPLE TARGET GAME (attentional_spotter, multiple group)")

m_game = game_all[game_all["group"]=="Multiple"].copy() if not game_all.empty else pd.DataFrame()

if m_game.empty:
    print("  No data loaded.")
else:
    print(f"\n  Participants : {m_game['participant'].nunique()}")
    print(f"  Total level attempts : {len(m_game)}")
    print(f"  Level range : {m_game['level'].min()} – {m_game['level'].max()}")
    comp = m_game["completed"].value_counts()
    print(f"  Completion : {comp.get(True,0)} complete, {comp.get(False,0)} incomplete "
          f"({100*m_game['completed'].mean():.1f}%)")

    print("\n  >> Initial RT (ms) [time to first target]:")
    print_desc(full_desc(m_game["RT_ms"]), "Initial RT")

    print("\n  >> Avg Inter-Target Time (ms) [mean time between hits]:")
    print_desc(full_desc(m_game["avg_inter_target_ms"]), "Inter-Target Time")

    print("\n  >> Success Rate (%):")
    print_desc(full_desc(m_game["success_rate"]), "Success Rate")

    print("\n  >> Hit Rate (%):")
    print_desc(full_desc(m_game["hit_rate"]), "Hit Rate")

    print("\n  >> False Alarms (count per level):")
    print_desc(full_desc(m_game["false_alarms"]), "False Alarms")

    print("\n  >> Final Score:")
    print_desc(full_desc(m_game["final_score"]), "Score")

    print("\n  >> n_hits (targets clicked per level):")
    print_desc(full_desc(m_game["n_hits"]), "n_hits")

    print("\n  >> Levels by participant:")
    pp = m_game.groupby("participant").agg(
        n_levels=("level","count"), max_level=("level","max"),
        mean_RT=("RT_ms","mean"), mean_SR=("success_rate","mean"),
        mean_FA=("false_alarms","mean")
    )
    print(pp.round(2).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 2E. 4-CELL SUMMARY TABLE  (Group × Modality)
# ─────────────────────────────────────────────────────────────────────────────

section("2E — 4-CELL SUMMARY TABLE  (Group × Modality × DV)")

rows = []
for grp in ["Single","Multiple"]:
    for mod_label, df, dv_list in [
        ("Lab",  lab_all,  [("RT_ms","RT (ms)"),("trial_dur_s","Trial Dur (s)")]),
        ("Game", game_all, [("RT_ms","RT (ms)"),("success_rate","Success %"),
                             ("hit_rate","Hit Rate %"),("false_alarms","False Alarms")]),
    ]:
        if df.empty: continue
        sub = df[df["group"]==grp]
        for col, label in dv_list:
            if col not in sub.columns: continue
            d = full_desc(sub[col].dropna().values)
            rows.append({"Group":grp,"Modality":mod_label,"Variable":label,
                         "n":int(d["n"]), "M":round(d["M"],2), "SD":round(d["SD"],2),
                         "SE":round(d["SE"],2), "Mdn":round(d["Mdn"],2),
                         "Min":round(d["Min"],2), "Max":round(d["Max"],2),
                         "Skew":round(d["Skew"],3), "SW_p":round(d["SW_p"],4)})

cell_df = pd.DataFrame(rows)
print()
print(cell_df.to_string(index=False))
cell_df.to_csv(OUT / "descriptives_4cell.csv", index=False)
print(f"\n  ✔  Saved descriptives_4cell.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 2F. PARTICIPANT-LEVEL AGGREGATION  (Slide 15 requirement)
# ─────────────────────────────────────────────────────────────────────────────

section("2F — PARTICIPANT-LEVEL AGGREGATES (Slide 15)")

ptpt_lab = (
    lab_all.groupby(["participant","group"])
    .agg(
        RT_mean      = ("RT_ms",        "mean"),
        RT_sd        = ("RT_ms",        "std"),
        RT_mdn       = ("RT_ms",        "median"),
        hit_rate     = ("hit",          "mean"),
        n_trials     = ("RT_ms",        "count"),
        trial_dur_M  = ("trial_dur_s",  "mean"),
    )
    .reset_index()
) if not lab_all.empty else pd.DataFrame()

ptpt_game = (
    game_all.groupby(["participant","group"])
    .agg(
        RT_mean      = ("RT_ms",              "mean"),
        RT_sd        = ("RT_ms",              "std"),
        RT_mdn       = ("RT_ms",              "median"),
        success_rate = ("success_rate",        "mean"),
        hit_rate     = ("hit_rate",            "mean"),
        false_alarms = ("false_alarms",        "mean"),
        n_levels     = ("level",               "count"),
        max_level    = ("level",               "max"),
        pct_complete = ("completed",           "mean"),
        inter_tgt_M  = ("avg_inter_target_ms", "mean"),
    )
    .reset_index()
) if not game_all.empty else pd.DataFrame()

print("\n  Participant-level Lab aggregates (first 10 rows):")
if not ptpt_lab.empty:
    print(ptpt_lab.head(10).round(2).to_string(index=False))

print("\n  Participant-level Game aggregates (first 10 rows):")
if not ptpt_game.empty:
    print(ptpt_game.head(10).round(2).to_string(index=False))

# Group-level summary of participant means
print("\n  Group means of participant means (the values that go into ANOVA):")
for df, label in [(ptpt_lab,"Lab"),(ptpt_game,"Game")]:
    if df.empty: continue
    gm = df.groupby("group")[["RT_mean","RT_sd"]].agg(["mean","std"])
    print(f"\n  {label}:")
    print(gm.round(2).to_string())

ptpt_lab.to_csv(OUT  / "participant_lab_aggregated.csv",  index=False)
ptpt_game.to_csv(OUT / "participant_game_aggregated.csv", index=False)
print(f"\n  ✔  Saved participant_lab_aggregated.csv")
print(f"  ✔  Saved participant_game_aggregated.csv")

# Save for downstream parts
import pickle
with open(OUT / "_data_cache.pkl","wb") as f:
    pickle.dump({"lab_all":lab_all,"game_all":game_all,
                 "ptpt_log":ptpt_log,"ptpt_lab":ptpt_lab,
                 "ptpt_game":ptpt_game}, f)
print("  ✔  Updated _data_cache.pkl\n")
print("  PART 2 COMPLETE.\n")

"""
╔══════════════════════════════════════════════════════════════════════╗
║  SELECTIVE ATTENTION STUDY — COMPLETE ANALYSIS                       ║
║  PART 3: Visualizations (All 4 Dataset Types)                        ║
╚══════════════════════════════════════════════════════════════════════╝

Figures produced:
  3-01  RT histograms + KDE — all 4 cells (Group × Modality)
  3-02  RT Q-Q plots — normality check
  3-03  Lab: RT by target colour (box + strip) per group
  3-04  Lab: Trial duration distribution per group
  3-05  Multi lab: Inter-click interval distribution
  3-06  Game: RT, Success Rate, Hit Rate, False Alarms — box per group
  3-07  Game: Score progression by level per group
  3-08  Game: Accuracy metrics by level (line, mean ± SE)
  3-09  Multi game: Avg inter-target time by level
  3-10  4-cell interaction plot (mean RT ± SE, Group × Modality)
  3-11  Concurrent validity scatter (game RT vs lab RT, per participant)
  3-12  Participant-level RT heatmap
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
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       130,
})

PAL = {
    "Single":   "#2D6A9F",
    "Multiple": "#C05621",
    "Lab":      "#2F855A",
    "Game":     "#9B2C2C",
    "red_target": "#E53E3E",
    "white_distractor": "#4A5568",
}
ALPHA = 0.82

def save(name):
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  {name}")

# ── Load data ────────────────────────────────────────────────────────────────
with open(OUT / "_data_cache.pkl","rb") as f:
    cache = pickle.load(f)
lab_all   = cache["lab_all"]
game_all  = cache["game_all"]
ptpt_lab  = cache["ptpt_lab"]
ptpt_game = cache["ptpt_game"]

s_lab  = lab_all[lab_all["group"]=="Single"]   if not lab_all.empty  else pd.DataFrame()
m_lab  = lab_all[lab_all["group"]=="Multiple"] if not lab_all.empty  else pd.DataFrame()
s_game = game_all[game_all["group"]=="Single"]   if not game_all.empty else pd.DataFrame()
m_game = game_all[game_all["group"]=="Multiple"] if not game_all.empty else pd.DataFrame()

print("=" * 65)
print("PART 3 — VISUALIZATIONS")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-01  RT Distributions — 4 cells
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Reaction Time Distributions — All Four Conditions",
             fontsize=14, fontweight="bold", y=1.01)

cells = [
    (s_lab,  "Single Target — Lab",   PAL["Single"],   "Lab"),
    (m_lab,  "Multiple Target — Lab", PAL["Multiple"], "Lab"),
    (s_game, "Single Target — Game",  PAL["Single"],   "Game"),
    (m_game, "Multiple Target — Game",PAL["Multiple"], "Game"),
]

for ax, (df, title, color, mod) in zip(axes.flat, cells):
    if df.empty: ax.set_title(title + "\n(no data)"); continue
    rt = df["RT_ms"].dropna()
    ax.hist(rt, bins=min(25, max(6, len(rt)//3)), color=color,
            alpha=0.65, edgecolor="white", density=True, label="Histogram")
    # KDE overlay
    kde_x = np.linspace(rt.min()*0.8, rt.max()*1.1, 300)
    kde   = stats.gaussian_kde(rt, bw_method="scott")
    ax.plot(kde_x, kde(kde_x), color=color, linewidth=2.2, label="KDE")
    # Mean & median lines
    ax.axvline(rt.mean(),   color="black",  ls="--", lw=1.6,
               label=f"M={rt.mean():.0f}")
    ax.axvline(rt.median(), color="dimgray",ls=":",  lw=1.6,
               label=f"Mdn={rt.median():.0f}")
    ax.set_title(f"{title}\n(n={len(rt)}, SD={rt.std():.0f} ms)")
    ax.set_xlabel("RT (ms)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)

plt.tight_layout()
save("fig_3-01_rt_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-02  Q-Q Plots — normality check
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Q-Q Plots (Normal) — RT per Condition", fontsize=14, fontweight="bold")

for ax, (df, title, color) in zip(axes.flat, [
    (s_lab,  "Single Lab",   PAL["Single"]),
    (m_lab,  "Multiple Lab", PAL["Multiple"]),
    (s_game, "Single Game",  PAL["Single"]),
    (m_game, "Multiple Game",PAL["Multiple"]),
]):
    if df.empty: ax.set_title(title + " (no data)"); continue
    rt = df["RT_ms"].dropna()
    (osm, osr), (slope, intercept, _) = stats.probplot(rt, dist="norm")
    ax.scatter(osm, osr, color=color, s=25, alpha=0.7, zorder=3)
    lx = np.array([osm.min(), osm.max()])
    ax.plot(lx, slope*lx + intercept, color="black", lw=1.5, ls="--")
    sw_stat, sw_p = stats.shapiro(rt) if len(rt) >= 3 else (np.nan,np.nan)
    normal_flag = "normal" if sw_p > .05 else "non-normal"
    ax.set_title(f"{title}\nShapiro-Wilk p={sw_p:.4f} ({normal_flag})")
    ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")

plt.tight_layout()
save("fig_3-02_qq_plots.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-03  Lab: RT by target colour — box + strip
# ─────────────────────────────────────────────────────────────────────────────

if not lab_all.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Lab Task: RT by Target Type per Group",
                 fontsize=14, fontweight="bold")

    for ax, (df, grp) in zip(axes, [(s_lab,"Single"),(m_lab,"Multiple")]):
        if df.empty: ax.set_title(grp + " (no data)"); continue
        d = df.dropna(subset=["RT_ms"]).copy()
        d["Type"] = d["is_target"].map({1:"Target\n(red)", 0:"Distractor\n(white)"})
        pal = {"Target\n(red)": PAL["red_target"],
               "Distractor\n(white)": PAL["white_distractor"]}
        order = ["Target\n(red)", "Distractor\n(white)"]
        sns.boxplot(data=d, x="Type", y="RT_ms", palette=pal,
                    order=order, width=0.45, linewidth=1.3,
                    flierprops=dict(marker="o", ms=3, alpha=0.4), ax=ax)
        sns.stripplot(data=d, x="Type", y="RT_ms", palette=pal,
                      order=order, alpha=0.4, size=4, jitter=True, ax=ax)
        # Annotate means
        for i, tp in enumerate(order):
            m = d[d["Type"]==tp]["RT_ms"].mean()
            ax.text(i, d["RT_ms"].max()*1.02, f"M={m:.0f}", ha="center",
                    fontsize=9, fontweight="bold")
        ax.set_title(f"{grp} Group  (n={d['participant'].nunique()} ptpts, {len(d)} trials)")
        ax.set_xlabel(""); ax.set_ylabel("RT (ms)")

    plt.tight_layout()
    save("fig_3-03_lab_rt_by_target_type.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-04  Lab: Trial duration & number of clicks
# ─────────────────────────────────────────────────────────────────────────────

if not lab_all.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Lab Task: Trial Duration & Click Pattern", fontsize=14, fontweight="bold")

    # Panel A: trial duration by group
    ax = axes[0]
    for grp, color in [("Single",PAL["Single"]),("Multiple",PAL["Multiple"])]:
        sub = lab_all[lab_all["group"]==grp]["trial_dur_s"].dropna()
        ax.hist(sub, bins=15, color=color, alpha=0.6,
                edgecolor="white", label=grp, density=True)
    ax.set_title("Trial Duration Distribution")
    ax.set_xlabel("Duration (s)"); ax.set_ylabel("Density"); ax.legend()

    # Panel B: RT vs trial duration scatter
    ax = axes[1]
    for grp, color in [("Single",PAL["Single"]),("Multiple",PAL["Multiple"])]:
        sub = lab_all[lab_all["group"]==grp].dropna(subset=["RT_ms","trial_dur_s"])
        ax.scatter(sub["trial_dur_s"], sub["RT_ms"],
                   color=color, alpha=0.5, s=25, label=grp)
    ax.set_title("RT vs Trial Duration (Lab)")
    ax.set_xlabel("Trial Duration (s)"); ax.set_ylabel("RT — first click (ms)")
    ax.legend()

    plt.tight_layout()
    save("fig_3-04_lab_trial_duration.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-05  Multi lab: Inter-click intervals
# ─────────────────────────────────────────────────────────────────────────────

if not m_lab.empty:
    all_ici = []
    click_ranks = {1:[],2:[],3:[],4:[],5:[]}
    for _, row in m_lab.iterrows():
        clicks = row["all_click_ms"]
        if len(clicks) >= 2:
            for i, ici in enumerate(np.diff(clicks)):
                all_ici.append(ici)
                if i+1 in click_ranks:
                    click_ranks[i+1].append(ici)
        for rank, val in enumerate(clicks, 1):
            if rank in click_ranks:
                click_ranks[rank] = click_ranks.get(rank, [])

    # Re-collect by click rank (absolute time of each click)
    click_times_by_rank = {k: [] for k in range(1,6)}
    for _, row in m_lab.iterrows():
        for i, t in enumerate(row["all_click_ms"], 1):
            if i <= 5: click_times_by_rank[i].append(t)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Multiple Target Lab: Click Timing", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.hist(all_ici, bins=20, color=PAL["Multiple"],
            alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(all_ici), color="black", ls="--", lw=1.5,
               label=f"M={np.mean(all_ici):.0f} ms")
    ax.set_title("Inter-Click Interval Distribution")
    ax.set_xlabel("ICI (ms)"); ax.set_ylabel("Frequency"); ax.legend()

    ax = axes[1]
    means = [np.mean(click_times_by_rank[r]) for r in range(1,6)]
    sems  = [np.std(click_times_by_rank[r],ddof=1)/np.sqrt(len(click_times_by_rank[r]))
             for r in range(1,6)]
    ax.errorbar(range(1,6), means, yerr=sems, fmt="o-",
                color=PAL["Multiple"], linewidth=2, markersize=7, capsize=5)
    ax.set_title("Mean Time to Each Target Click\n(cumulative RT per click rank)")
    ax.set_xlabel("Click Rank (1=first target)"); ax.set_ylabel("Time from trial onset (ms)")
    ax.set_xticks(range(1,6))

    plt.tight_layout()
    save("fig_3-05_multi_lab_click_timing.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-06  Game: Key metrics per group — box + strip
# ─────────────────────────────────────────────────────────────────────────────

if not game_all.empty:
    metrics = [
        ("RT_ms",              "Initial RT (ms)"),
        ("success_rate",       "Success Rate (%)"),
        ("hit_rate",           "Hit Rate (%)"),
        ("false_alarms",       "False Alarms"),
        ("avg_inter_target_ms","Avg Inter-Target Time (ms)"),
        ("final_score",        "Final Score"),
    ]
    # Use participant means for cleaner box plots
    ptpt_game_m = game_all.groupby(["participant","group"]).agg(
        RT_ms=("RT_ms","mean"),
        success_rate=("success_rate","mean"),
        hit_rate=("hit_rate","mean"),
        false_alarms=("false_alarms","mean"),
        avg_inter_target_ms=("avg_inter_target_ms","mean"),
        final_score=("final_score","sum"),
    ).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Game Metrics by Group — Participant Means",
                 fontsize=14, fontweight="bold")

    for ax, (col, ylabel) in zip(axes.flat, metrics):
        if col not in ptpt_game_m.columns: ax.set_visible(False); continue
        data = ptpt_game_m.dropna(subset=[col])
        sns.boxplot(data=data, x="group", y=col, order=["Single","Multiple"],
                    palette={"Single":PAL["Single"],"Multiple":PAL["Multiple"]},
                    width=0.45, linewidth=1.3, ax=ax,
                    flierprops=dict(marker="o", ms=4, alpha=0.5))
        sns.stripplot(data=data, x="group", y=col, order=["Single","Multiple"],
                      palette={"Single":PAL["Single"],"Multiple":PAL["Multiple"]},
                      alpha=0.55, size=5, jitter=True, ax=ax)
        for i, grp in enumerate(["Single","Multiple"]):
            vals = data[data["group"]==grp][col].dropna()
            if len(vals):
                ax.text(i, vals.max()*1.03, f"M={vals.mean():.1f}\n(n={len(vals)})",
                        ha="center", fontsize=8)
        ax.set_title(ylabel); ax.set_xlabel(""); ax.set_ylabel(ylabel)

    plt.tight_layout()
    save("fig_3-06_game_metrics_by_group.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-07  Game: Final score progression by level
# ─────────────────────────────────────────────────────────────────────────────

# if not game_all.empty:
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
#     fig.suptitle("Game: Score Progression by Level", fontsize=14, fontweight="bold")

#     for ax, (grp, df, color) in zip(axes, [
#         ("Single",   s_game, PAL["Single"]),
#         ("Multiple", m_game, PAL["Multiple"]),
#     ]):
#         if df.empty: ax.set_title(grp + " (no data)"); continue
#         for pid, pdf in df.groupby("participant"):
#             pdf = pdf.sort_values("level")
#             ax.plot(pdf["level"], pdf["final_score"], "o-",
#                     color=color, alpha=0.3, linewidth=1, markersize=4)
#         # Group mean line
#         grp_mean = df.groupby("level")["final_score"].mean()
#         ax.plot(grp_mean.index, grp_mean.values, "o-",
#                 color=color, linewidth=2.5, markersize=7,
#                 label="Group mean", zorder=5)
#         # Mark incompletes
#         inc = df[~df["completed"]]
#         if not inc.empty:
#             ax.scatter(inc["level"], inc["final_score"],
#                        marker="X", color="crimson", s=80, zorder=6,
#                        label="Incomplete")
#         ax.set_title(f"{grp} Group  (n={df['participant'].nunique()} participants)")
#         ax.set_xlabel("Level"); ax.set_ylabel("Final Score")
#         ax.legend()

#     plt.tight_layout()
#     save("fig_3-07_game_score_progression.png")

import matplotlib.lines as mlines

# Ensure you have a 2x2 grid layout
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Individual Progression Trajectories: Lab vs. Game", 
             fontsize=15, fontweight="bold")

# Define the configurations for the 4 plots: (axis, group, dataframe, x_col, y_col, title_prefix)
plot_configs = [
    (axes[0, 0], "Single",   lab_all[lab_all["group"]=="Single"],   "trial_n", "RT_ms",       "Lab: RT Progression"),
    (axes[0, 1], "Multiple", lab_all[lab_all["group"]=="Multiple"], "trial_n", "RT_ms",       "Lab: RT Progression"),
    (axes[1, 0], "Single",   game_all[game_all["group"]=="Single"], "level",   "final_score", "Game: Score Progression"),
    (axes[1, 1], "Multiple", game_all[game_all["group"]=="Multiple"],"level",   "final_score", "Game: Score Progression")
]

for ax, grp, df, x_col, y_col, title_prefix in plot_configs:
    if df.empty: 
        ax.set_title(f"{title_prefix} — {grp} (no data)")
        continue
        
    # 1. Plot individual lines
    for pid, pdf in df.groupby("participant"):
        pdf = pdf.sort_values(x_col)
        ax.plot(pdf[x_col], pdf[y_col], "o-", 
                color=PAL[grp], alpha=0.25, linewidth=1, markersize=4)
                
    # 2. Plot group mean line
    grp_mean = df.groupby(x_col)[y_col].mean()
    ax.plot(grp_mean.index, grp_mean.values, "o-", 
            color=PAL[grp], linewidth=3, markersize=7, 
            label="Group Mean", zorder=5)
            
    # 3. Mark incompletes / errors with a red 'X'
    if "completed" in df.columns:
        inc = df[~df["completed"]]  # Game logic
        fail_label = "Incomplete Level"
    elif "hit" in df.columns:
        inc = df[df["hit"] == 0]    # Lab logic
        fail_label = "Missed Target"
    else:
        inc = pd.DataFrame()

    if not inc.empty and y_col in inc.columns:
        ax.scatter(inc[x_col], inc[y_col], 
                   marker="X", color="crimson", s=80, zorder=6, 
                   label=fail_label)

    # Formatting and Labels
    ax.set_title(f"{title_prefix} — {grp} Group (n={df['participant'].nunique()})")
    ax.set_xlabel("Trial Number" if "trial" in x_col else "Level")
    ax.set_ylabel("Reaction Time (ms)" if y_col == "RT_ms" else "Final Score")
    
    # 4. Custom Legend Generation
    handles, labels = ax.get_legend_handles_labels()
    # Create a proxy artist for the individual lines
    indiv_line = mlines.Line2D([], [], color=PAL[grp], marker='o', 
                               alpha=0.3, linewidth=1, markersize=4, 
                               label='Individual Trajectory')
    handles.insert(0, indiv_line)
    labels.insert(0, 'Individual Trajectory')
    
    # Remove duplicates from legend to keep it clean
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

plt.tight_layout()
save("fig_3-07_progression_trajectories_4panel.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-08  Game: Accuracy metrics by level (mean ± SE ribbon)
# ─────────────────────────────────────────────────────────────────────────────


if not game_all.empty:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Game: Performance Metrics by Level (Mean ± SE)",
                 fontsize=14, fontweight="bold")

    metrics_level = [
        ("RT_ms",        "Initial RT (ms)"),
        ("success_rate", "Success Rate (%)"),
        ("hit_rate",     "Hit Rate (%)"),
        ("false_alarms", "False Alarms"),
    ]
    
    for ax, (col, ylabel) in zip(axes.flat, metrics_level):
        for grp, color in [("Single",PAL["Single"]),("Multiple",PAL["Multiple"])]:
            sub = game_all[game_all["group"]==grp].groupby("level")[col].agg(["mean","sem","count"])
            ax.plot(sub.index, sub["mean"], "o-", color=color,
                    label=f"{grp} (n={sub['count'].sum():.0f})",
                    linewidth=2, markersize=5, alpha=ALPHA)
            ax.fill_between(sub.index,
                            sub["mean"] - sub["sem"],
                            sub["mean"] + sub["sem"],
                            alpha=0.15, color=color)
                            
        # Add trend line for each group
        for grp, color in [("Single",PAL["Single"]),("Multiple",PAL["Multiple"])]:
            tmp = game_all[game_all["group"]==grp][["level",col]].dropna()
            if len(tmp) >= 3:
                z  = np.polyfit(tmp["level"], tmp[col], 1)
                xs = np.linspace(tmp["level"].min(), tmp["level"].max(), 100)
                ax.plot(xs, np.poly1d(z)(xs), "--", color=color,
                        alpha=0.5, linewidth=1.5)
                        
        # --- NEW: Attrition Demarcation ---
        # Draw a dashed line at Level 10
        ax.axvline(x=10, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        
        # Shade the background from Level 10 to 15 (zorder=0 keeps it behind the data lines)
        ax.axvspan(xmin=10, xmax=15.5, color='gray', alpha=0.1, zorder=0)
        
        # Add a subtle text label to explain the shaded region
        ymin, ymax = ax.get_ylim()
        ax.text(10.2, ymax - (ymax - ymin) * 0.05, 'High Attrition\nZone',
                color='gray', fontsize=9, va='top', ha='left', alpha=0.8)
        # ----------------------------------

        ax.set_title(ylabel)
        ax.set_xlabel("Level")
        ax.set_ylabel(ylabel)
        ax.legend()
        
        # Ensure x-axis shows integer levels cleanly
        ax.set_xticks(range(1, 16))

    plt.tight_layout()
    save("fig_3-08_game_metrics_by_level.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-09  Multi game: Inter-target time by level
# ─────────────────────────────────────────────────────────────────────────────

if not m_game.empty and "avg_inter_target_ms" in m_game.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Multiple Target Game: Inter-Target Time",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    sub = m_game.dropna(subset=["avg_inter_target_ms"])
    ax.hist(sub["avg_inter_target_ms"], bins=15,
            color=PAL["Multiple"], alpha=0.75, edgecolor="white")
    ax.axvline(sub["avg_inter_target_ms"].mean(), color="black", ls="--",
               lw=1.5, label=f"M={sub['avg_inter_target_ms'].mean():.0f}")
    ax.set_title("Distribution of Avg Inter-Target Time")
    ax.set_xlabel("Avg Inter-Target Time (ms)"); ax.set_ylabel("Frequency"); ax.legend()

    ax = axes[1]
    lv = m_game.groupby("level")["avg_inter_target_ms"].agg(["mean","sem"])
    ax.plot(lv.index, lv["mean"], "s-", color=PAL["Multiple"],
            linewidth=2, markersize=6)
    ax.fill_between(lv.index, lv["mean"]-lv["sem"],
                    lv["mean"]+lv["sem"], alpha=0.2, color=PAL["Multiple"])
    ax.set_title("Avg Inter-Target Time by Level")
    ax.set_xlabel("Level"); ax.set_ylabel("Avg Inter-Target Time (ms)")

    plt.tight_layout()
    save("fig_3-09_multi_game_intertarget.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-10  4-cell Interaction Plot — mean RT ± SE
# ─────────────────────────────────────────────────────────────────────────────

if not ptpt_lab.empty and not ptpt_game.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Interaction Plot: Mean RT by Group × Modality",
                 fontsize=14, fontweight="bold")

    # Left: raw participant means plotted
    ax = axes[0]
    for mod, df, ls, mk in [("Lab", ptpt_lab,"--","o"),("Game",ptpt_game,"-","s")]:
        means, sems = [], []
        for grp in ["Single","Multiple"]:
            v = df[df["group"]==grp]["RT_mean"].dropna()
            means.append(v.mean()); sems.append(v.sem() if len(v)>1 else 0)
        ax.errorbar(["Single","Multiple"], means, yerr=sems, label=mod,
                    linestyle=ls, marker=mk, color=PAL[mod],
                    linewidth=2.2, markersize=9, capsize=7)
    ax.set_title("Mean RT ± SE (Participant Means)")
    ax.set_xlabel("Target Load"); ax.set_ylabel("Mean RT (ms)")
    ax.legend(title="Modality")

    # Right: violin + individual lines per participant (if matched)
    ax = axes[1]
    merged = pd.merge(
        ptpt_lab[["participant","group","RT_mean"]].rename(columns={"RT_mean":"RT_Lab"}),
        ptpt_game[["participant","group","RT_mean"]].rename(columns={"RT_mean":"RT_Game"}),
        on=["participant","group"]
    ).dropna()

    for grp, color in [("Single",PAL["Single"]),("Multiple",PAL["Multiple"])]:
        sub = merged[merged["group"]==grp]
        jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(sub))
        for _, row in sub.iterrows():
            ax.plot(["Lab","Game"], [row["RT_Lab"], row["RT_Game"]],
                    "-o", color=color, alpha=0.3, linewidth=1, markersize=5)
        # Group means
        ax.plot(["Lab","Game"],
                [sub["RT_Lab"].mean(), sub["RT_Game"].mean()],
                "D-", color=color, linewidth=2.5, markersize=8,
                label=f"{grp} mean")

    ax.set_title("Individual Participant Trajectories\n(Lab → Game per person)")
    ax.set_xlabel("Modality"); ax.set_ylabel("RT (ms)")
    ax.legend()

    plt.tight_layout()
    save("fig_3-10_interaction_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-11  Concurrent validity scatter — game RT vs lab RT (participant means)
# ─────────────────────────────────────────────────────────────────────────────

if not ptpt_lab.empty and not ptpt_game.empty:
    merged = pd.merge(
        ptpt_lab[["participant","group","RT_mean","RT_sd"]].rename(
            columns={"RT_mean":"lab_RT","RT_sd":"lab_RT_sd"}),
        ptpt_game[["participant","group","RT_mean","RT_sd","success_rate","hit_rate"]].rename(
            columns={"RT_mean":"game_RT","RT_sd":"game_RT_sd"}),
        on=["participant","group"]
    ).dropna(subset=["lab_RT","game_RT"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("RQ1 — Concurrent Validity: Game RT vs Lab RT (Participant Means)",
                 fontsize=14, fontweight="bold")

    for ax, grp in zip(axes, ["Single","Multiple"]):
        sub = merged[merged["group"]==grp]
        if len(sub) < 2:
            ax.set_title(f"{grp}  (n={len(sub)} — need ≥2)"); continue
        color = PAL[grp]
        ax.errorbar(sub["game_RT"], sub["lab_RT"],
                    xerr=sub["game_RT_sd"]/np.sqrt(len(sub)),
                    yerr=sub["lab_RT_sd"]/np.sqrt(len(sub)),
                    fmt="o", color=color, alpha=0.7,
                    markersize=7, capsize=3, elinewidth=0.8)
        for _, row in sub.iterrows():
            ax.annotate(row["participant"],
                        (row["game_RT"], row["lab_RT"]),
                        textcoords="offset points", xytext=(6,3),
                        fontsize=7.5, alpha=0.75)
        # Regression line
        z  = np.polyfit(sub["game_RT"], sub["lab_RT"], 1)
        xs = np.linspace(sub["game_RT"].min()*0.95, sub["game_RT"].max()*1.05, 200)
        ax.plot(xs, np.poly1d(z)(xs), "--", color=color, linewidth=1.8, alpha=0.7)
        # Diagonal reference (perfect agreement)
        lim = [min(sub[["game_RT","lab_RT"]].min()), max(sub[["game_RT","lab_RT"]].max())]
        ax.plot(lim, lim, ":", color="gray", linewidth=1, alpha=0.6, label="Identity")

        r, p = pearsonr(sub["game_RT"], sub["lab_RT"])
        sig  = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
        ax.set_title(f"{grp}  (n={len(sub)})   r = {r:.3f} {sig}")
        ax.set_xlabel("Game Mean RT (ms)"); ax.set_ylabel("Lab Mean RT (ms)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    save("fig_3-11_concurrent_validity.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3-12  Participant-level RT heatmap
# ─────────────────────────────────────────────────────────────────────────────

if not ptpt_lab.empty and not ptpt_game.empty:
    merged_heat = pd.merge(
        ptpt_lab[["participant","group","RT_mean"]].rename(columns={"RT_mean":"Lab_RT"}),
        ptpt_game[["participant","group","RT_mean","success_rate","hit_rate",
                   "false_alarms","pct_complete"]].rename(
            columns={"RT_mean":"Game_RT","pct_complete":"Pct_Complete"}),
        on=["participant","group"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Participant Profiles — Heatmap (z-scored)",
                 fontsize=14, fontweight="bold")

    for ax, grp in zip(axes, ["Single","Multiple"]):
        sub = merged_heat[merged_heat["group"]==grp].copy()
        if len(sub) < 2: ax.set_title(f"{grp} (insufficient)"); continue
        cols_h = ["Lab_RT","Game_RT","success_rate","hit_rate","false_alarms"]
        cols_h = [c for c in cols_h if c in sub.columns and sub[c].notna().sum() > 0]
        sub_h  = sub.set_index("participant")[cols_h]
        # z-score each column
        sub_z  = (sub_h - sub_h.mean()) / (sub_h.std() + 1e-9)
        cmap   = "RdBu_r"
        sns.heatmap(sub_z, ax=ax, cmap=cmap, center=0,
                    linewidths=0.5, linecolor="white",
                    annot=len(sub_z) <= 25,
                    fmt=".1f" if len(sub_z) <= 25 else "",
                    cbar_kws={"shrink":0.7, "label":"z-score"})
        ax.set_title(f"{grp} Group  (n={len(sub)})")
        ax.set_xlabel(""); ax.set_ylabel("Participant")

    plt.tight_layout()
    save("fig_3-12_participant_heatmap.png")


print("\n  PART 3 COMPLETE.\n")

"""
╔══════════════════════════════════════════════════════════════════════╗
║  SELECTIVE ATTENTION STUDY — COMPLETE ANALYSIS                       ║
║  PART 4: Inferential Statistics                                       ║
╚══════════════════════════════════════════════════════════════════════╝

Tests:
  4A. RQ1 — Concurrent Validity  (Pearson r + Spearman ρ, Game vs Lab RT)
  4B. RQ2 — Target Load effect   (2×2 Mixed ANOVA + Independent t-tests)
  4C. RQ3 — Modality effect      (Paired t-tests, Game vs Lab within groups)
  4D. RQ4 — Level effects        (Spearman trends, per group × DV)
  4E. Reliability                (Split-half + Spearman-Brown + Cronbach's α)

Statistical output:
  • Test statistic, df, p-value, effect size (d / dz / r / η²p)
  • 95% CI where applicable
  • Significance stars: * p<.05  ** p<.01  *** p<.001  ns
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
from scipy.stats import (pearsonr, spearmanr, ttest_rel, ttest_ind,
                         f as f_dist, t as t_dist, shapiro)

warnings.filterwarnings("ignore")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
with open(OUT / "_data_cache.pkl","rb") as f:
    cache = pickle.load(f)
lab_all   = cache["lab_all"]
game_all  = cache["game_all"]
ptpt_lab  = cache["ptpt_lab"]
ptpt_game = cache["ptpt_game"]

PAL = {"Single":"#2D6A9F","Multiple":"#C05621","Lab":"#2F855A","Game":"#9B2C2C"}

def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ""
    return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

def cohens_d(a, b):
    na,nb = len(a),len(b)
    if na<2 or nb<2: return np.nan
    sp = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(a)-np.mean(b))/sp if sp else np.nan

def cohens_dz(diff):
    s = np.std(diff,ddof=1)
    return np.mean(diff)/s if s else np.nan

def ci95_mean(arr):
    n  = len(arr)
    se = np.std(arr,ddof=1)/np.sqrt(n)
    t  = t_dist.ppf(.975, n-1)
    return np.mean(arr)-t*se, np.mean(arr)+t*se

def ci95_diff(diff):
    se = np.std(diff,ddof=1)/np.sqrt(len(diff))
    t  = t_dist.ppf(.975, len(diff)-1)
    return np.mean(diff)-t*se, np.mean(diff)+t*se

def section(title):
    print("\n"+"═"*70)
    print(f"  {title}")
    print("═"*70)

def save(name):
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  {name}")

print("="*70)
print("PART 4 — INFERENTIAL STATISTICS")
print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# 4A. RQ1 — CONCURRENT VALIDITY
# ─────────────────────────────────────────────────────────────────────────────

section("4A — RQ1: Concurrent Validity  (Pearson r + Spearman ρ)")

validity_rows = []

if not ptpt_lab.empty and not ptpt_game.empty:
    merged = pd.merge(
        ptpt_lab[["participant","group","RT_mean","RT_sd"]].rename(
            columns={"RT_mean":"lab_RT","RT_sd":"lab_RT_sd"}),
        ptpt_game[["participant","group","RT_mean","RT_sd","success_rate","hit_rate"]].rename(
            columns={"RT_mean":"game_RT","RT_sd":"game_RT_sd"}),
        on=["participant","group"]
    ).dropna(subset=["lab_RT","game_RT"])

    # Overall (all participants)
    for grp in ["Single","Multiple","All"]:
        if grp == "All":
            sub = merged
        else:
            sub = merged[merged["group"]==grp]
        n = len(sub)
        print(f"\n  {grp} group  (n={n} participants with both modalities)")
        if n < 3:
            print(f"    ⚠  n={n} — minimum n=3 needed for stable correlation.")
        if n >= 2:
            r,  p_r  = pearsonr(sub["game_RT"],  sub["lab_RT"])
            rho,p_s  = spearmanr(sub["game_RT"], sub["lab_RT"])
            # Fisher z CI for r
            if n >= 4:
                z_r = np.arctanh(r)
                se_z = 1/np.sqrt(n-3)
                ci_lo_z, ci_hi_z = z_r-1.96*se_z, z_r+1.96*se_z
                ci_lo, ci_hi = np.tanh(ci_lo_z), np.tanh(ci_hi_z)
            else:
                ci_lo, ci_hi = np.nan, np.nan
            print(f"    Pearson  r = {r:+.3f}   p = {p_r:.4f} {stars(p_r)}"
                  f"   95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
            print(f"    Spearman ρ = {rho:+.3f}   p = {p_s:.4f} {stars(p_s)}")
            validity_rows.append({"Group":grp,"n":n,"r":r,"p_pearson":p_r,
                                   "rho":rho,"p_spearman":p_s,
                                   "r_ci_lo":ci_lo,"r_ci_hi":ci_hi})
        else:
            print("    Insufficient participants.")

    # Also run correlations on other DVs
    print("\n  Validity: Success Rate vs Lab Trial Duration")
    for grp in ["Single","Multiple"]:
        sub2 = merged[merged["group"]==grp].dropna(subset=["success_rate"])
        if len(sub2) < 3: continue
        # success rate vs lab RT
        r2, p2 = pearsonr(sub2["success_rate"], sub2["lab_RT"])
        print(f"    {grp}: r(Success Rate, Lab RT) = {r2:.3f}  p={p2:.4f} {stars(p2)}")

pd.DataFrame(validity_rows).to_csv(OUT / "results_rq1_validity.csv", index=False)
print("\n  ✔  Saved results_rq1_validity.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4B. RQ2 — 2×2 MIXED ANOVA + INDEPENDENT t-TEST
# ─────────────────────────────────────────────────────────────────────────────

section("4B — RQ2: Target Load Effect  (2×2 Mixed ANOVA + Welch t-tests)")

# Build participant-level wide form
wide = pd.DataFrame()
if not ptpt_lab.empty and not ptpt_game.empty:
    wide = pd.merge(
        ptpt_lab[["participant","group","RT_mean"]].rename(columns={"RT_mean":"RT_Lab"}),
        ptpt_game[["participant","group","RT_mean","success_rate","hit_rate","false_alarms"]].rename(
            columns={"RT_mean":"RT_Game"}),
        on=["participant","group"]
    ).dropna()

    print(f"\n  Wide-form dataset: {len(wide)} participants with both modalities")
    print(f"  Single: {(wide['group']=='Single').sum()}  "
          f"Multiple: {(wide['group']=='Multiple').sum()}")

    # Cell means
    print("\n  Cell Means (ms):")
    for grp in ["Single","Multiple"]:
        for mod in ["RT_Lab","RT_Game"]:
            sub = wide[wide["group"]==grp][mod]
            print(f"    {grp:<12} × {mod.replace('RT_',''):<6}  "
                  f"M={sub.mean():.1f}  SD={sub.std(ddof=1):.1f}  n={len(sub)}")

if not wide.empty:
    N   = len(wide)
    n_s = (wide["group"]=="Single").sum()
    n_m = (wide["group"]=="Multiple").sum()

    s_lab_v  = wide[wide["group"]=="Single"]["RT_Lab"].values
    s_game_v = wide[wide["group"]=="Single"]["RT_Game"].values
    m_lab_v  = wide[wide["group"]=="Multiple"]["RT_Lab"].values
    m_game_v = wide[wide["group"]=="Multiple"]["RT_Game"].values

    grand    = wide[["RT_Lab","RT_Game"]].values.mean()
    mu_s     = np.hstack([s_lab_v, s_game_v]).mean()
    mu_m     = np.hstack([m_lab_v, m_game_v]).mean()
    mu_lab   = wide["RT_Lab"].mean()
    mu_game  = wide["RT_Game"].mean()
    wide_c   = wide.copy()
    wide_c["subj_mean"] = wide_c[["RT_Lab","RT_Game"]].mean(axis=1)

    # SS_A (between: Target Load)
    ss_A  = n_s*2*(mu_s-grand)**2 + n_m*2*(mu_m-grand)**2
    df_A  = 1
    # SS_S/A (subjects within groups — between-subjects error)
    ss_SA = sum((r["subj_mean"]-mu_s)**2*2
                for _,r in wide_c[wide_c["group"]=="Single"].iterrows()) + \
            sum((r["subj_mean"]-mu_m)**2*2
                for _,r in wide_c[wide_c["group"]=="Multiple"].iterrows())
    df_SA = N - 2
    # SS_B (within: Modality)
    ss_B  = N*((mu_lab-grand)**2+(mu_game-grand)**2)
    df_B  = 1
    # SS_AB (interaction)
    cell_mu = {("Single","Lab"):s_lab_v.mean(),("Single","Game"):s_game_v.mean(),
               ("Multiple","Lab"):m_lab_v.mean(),("Multiple","Game"):m_game_v.mean()}
    ss_AB = sum(ng*((cell_mu[(g,mod)]-mg-mm+grand)**2)
                for g,ng,mg in [("Single",n_s,mu_s),("Multiple",n_m,mu_m)]
                for mod,mm in [("Lab",mu_lab),("Game",mu_game)])
    df_AB = 1
    # SS_BxS/A (within-subjects error)
    ss_BsA = sum((v - r["subj_mean"] - mm + grand)**2
                 for _,r in wide_c.iterrows()
                 for v,mm in [(r["RT_Lab"],mu_lab),(r["RT_Game"],mu_game)])
    df_BsA = (N-2)*1

    ms_A,ms_SA,ms_B,ms_AB,ms_BsA = (ss/df if df>0 else np.nan
        for ss,df in [(ss_A,df_A),(ss_SA,df_SA),(ss_B,df_B),(ss_AB,df_AB),(ss_BsA,df_BsA)])

    F_A   = ms_A   / ms_SA  if ms_SA  else np.nan
    F_B   = ms_B   / ms_BsA if ms_BsA else np.nan
    F_AB  = ms_AB  / ms_BsA if ms_BsA else np.nan
    p_A   = 1-f_dist.cdf(F_A,  df_A,  df_SA)  if not np.isnan(F_A)  else np.nan
    p_B   = 1-f_dist.cdf(F_B,  df_B,  df_BsA) if not np.isnan(F_B)  else np.nan
    p_AB  = 1-f_dist.cdf(F_AB, df_AB, df_BsA) if not np.isnan(F_AB) else np.nan

    eta2_A  = ss_A  / (ss_A  + ss_SA)  if (ss_A +ss_SA ) else np.nan
    eta2_B  = ss_B  / (ss_B  + ss_BsA) if (ss_B +ss_BsA) else np.nan
    eta2_AB = ss_AB / (ss_AB + ss_BsA) if (ss_AB+ss_BsA) else np.nan

    print("\n  ── 2×2 Mixed ANOVA Table ──────────────────────────────────────")
    print(f"  {'Source':<30}{'SS':>12}{'df':>5}{'MS':>12}{'F':>9}{'p':>10}{'η²p':>8}")
    print("  " + "─"*80)
    for src,ss,df_,ms,F,p,eta in [
        ("Target Load [A — Between]",ss_A, df_A, ms_A, F_A, p_A, eta2_A),
        ("  Error (S/A)",            ss_SA,df_SA,ms_SA,np.nan,np.nan,np.nan),
        ("Modality [B — Within]",    ss_B, df_B, ms_B, F_B, p_B, eta2_B),
        ("Load × Modality [AxB]",    ss_AB,df_AB,ms_AB,F_AB,p_AB,eta2_AB),
        ("  Error (BxS/A)",          ss_BsA,df_BsA,ms_BsA,np.nan,np.nan,np.nan),
    ]:
        fmt = lambda v, d=1: f"{v:>{d}.{d-1}f}" if not (v is None or np.isnan(v)) else "—"
        p_str = f"{p:>8.4f} {stars(p)}" if not (p is None or np.isnan(p)) else "          "
        eta_s = f"{eta:>8.3f}" if not (eta is None or np.isnan(eta)) else "       —"
        print(f"  {src:<30}{ss:>12.1f}{df_:>5.0f}{ms:>12.1f}"
              f"{F:>9.3f} {p_str}{eta_s}"
              if not np.isnan(F) else
              f"  {src:<30}{ss:>12.1f}{df_:>5.0f}{ms:>12.1f}"
              f"{'':>9} {'':>10}{eta_s}")

    print(f"\n  ➤ RQ2 Target Load: F({df_A:.0f},{df_SA:.0f})={F_A:.3f}, "
          f"p={p_A:.4f}{stars(p_A)}, η²p={eta2_A:.3f}")
    print(f"  ➤ RQ3 Modality:    F({df_B:.0f},{df_BsA:.0f})={F_B:.3f}, "
          f"p={p_B:.4f}{stars(p_B)}, η²p={eta2_B:.3f}")
    print(f"  ➤ Interaction:     F({df_AB:.0f},{df_BsA:.0f})={F_AB:.3f}, "
          f"p={p_AB:.4f}{stars(p_AB)}, η²p={eta2_AB:.3f}")

    # Effect size interpretation
    for name, eta in [("Target Load",eta2_A),("Modality",eta2_B),("Interaction",eta2_AB)]:
        interp = "small" if eta<.06 else "medium" if eta<.14 else "large"
        print(f"    {name:<18}: η²p={eta:.3f} → {interp} effect")

    anova_df = pd.DataFrame([
        {"Source":"Target Load [A]","SS":ss_A,"df":df_A,"MS":ms_A,
         "F":F_A,"p":p_A,"eta2p":eta2_A,"sig":stars(p_A)},
        {"Source":"Error (S/A)",    "SS":ss_SA,"df":df_SA,"MS":ms_SA},
        {"Source":"Modality [B]",   "SS":ss_B,"df":df_B,"MS":ms_B,
         "F":F_B,"p":p_B,"eta2p":eta2_B,"sig":stars(p_B)},
        {"Source":"AxB Interaction","SS":ss_AB,"df":df_AB,"MS":ms_AB,
         "F":F_AB,"p":p_AB,"eta2p":eta2_AB,"sig":stars(p_AB)},
        {"Source":"Error (BxS/A)",  "SS":ss_BsA,"df":df_BsA,"MS":ms_BsA},
    ])
    anova_df.to_csv(OUT / "results_anova.csv", index=False)
    print("\n  ✔  Saved results_anova.csv")

# ── Independent t-tests: Single vs Multiple ──────────────────────────────────
print("\n  ── Independent t-tests (Welch): Single vs Multiple ─────────────")
indep_rows = []

for df, label, col in [
    (ptpt_lab,  "Lab RT (ms)",           "RT_mean"),
    (ptpt_game, "Game RT (ms)",           "RT_mean"),
    (ptpt_game, "Game Success Rate (%)",  "success_rate"),
    (ptpt_game, "Game Hit Rate (%)",      "hit_rate"),
    (ptpt_game, "Game False Alarms",      "false_alarms"),
    (ptpt_game, "Game Max Level",         "max_level"),
]:
    if df.empty or col not in df.columns: continue
    s = df[df["group"]=="Single"][col].dropna().values
    m = df[df["group"]=="Multiple"][col].dropna().values
    if len(s)<2 or len(m)<2:
        print(f"\n  {label}: insufficient data"); continue

    t, p  = ttest_ind(s, m, equal_var=False)
    d     = cohens_d(s, m)
    ci_s  = ci95_mean(s)
    ci_m  = ci95_mean(m)
    print(f"\n  {label}:")
    print(f"    Single   M={s.mean():.2f}  SD={s.std(ddof=1):.2f}  n={len(s)}"
          f"  95%CI[{ci_s[0]:.2f},{ci_s[1]:.2f}]")
    print(f"    Multiple M={m.mean():.2f}  SD={m.std(ddof=1):.2f}  n={len(m)}"
          f"  95%CI[{ci_m[0]:.2f},{ci_m[1]:.2f}]")
    print(f"    Welch t({len(s)+len(m)-2:.0f}) = {t:.3f}   p = {p:.4f} {stars(p)}"
          f"   Cohen's d = {d:.3f}")
    interp = "negligible" if abs(d)<.2 else "small" if abs(d)<.5 else "medium" if abs(d)<.8 else "large"
    print(f"    Effect: d={d:.3f} → {interp}")
    indep_rows.append({"DV":label,"t":t,"p":p,"d":d,
                       "Single_M":s.mean(),"Multiple_M":m.mean(),
                       "sig":stars(p)})

pd.DataFrame(indep_rows).to_csv(OUT / "results_rq2_ttest_indep.csv", index=False)
print("\n  ✔  Saved results_rq2_ttest_indep.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4C. RQ3 — PAIRED t-TEST: Game vs Lab within each group
# ─────────────────────────────────────────────────────────────────────────────

section("4C — RQ3: Modality Effect  (Paired t-test, Game vs Lab RT)")

paired_rows = []
if not wide.empty:
    for grp in ["Single","Multiple"]:
        sub = wide[wide["group"]==grp].dropna(subset=["RT_Lab","RT_Game"])
        n   = len(sub)
        if n < 2:
            print(f"\n  {grp}: n={n} — insufficient"); continue

        g = sub["RT_Game"].values
        l = sub["RT_Lab"].values
        diff = g - l
        t, p = ttest_rel(g, l)
        dz   = cohens_dz(diff)
        ci   = ci95_diff(diff)

        print(f"\n  {grp} group  (n={n} matched participants)")
        print(f"    Game RT   M={g.mean():.1f}  SD={g.std(ddof=1):.1f}"
              f"  95%CI[{ci95_mean(g)[0]:.1f},{ci95_mean(g)[1]:.1f}]")
        print(f"    Lab  RT   M={l.mean():.1f}  SD={l.std(ddof=1):.1f}"
              f"  95%CI[{ci95_mean(l)[0]:.1f},{ci95_mean(l)[1]:.1f}]")
        print(f"    Diff      M={diff.mean():.1f}  SD={diff.std(ddof=1):.1f}"
              f"  95%CI[{ci[0]:.1f},{ci[1]:.1f}]")
        print(f"    t({n-1}) = {t:.3f}   p = {p:.4f} {stars(p)}   dz = {dz:.3f}")
        interp = "negligible" if abs(dz)<.2 else "small" if abs(dz)<.5 else "medium" if abs(dz)<.8 else "large"
        print(f"    Effect: dz={dz:.3f} → {interp}")
        paired_rows.append({"Group":grp,"n":n,"t":t,"df":n-1,"p":p,"dz":dz,
                             "game_M":g.mean(),"lab_M":l.mean(),
                             "diff_M":diff.mean(),"diff_SD":diff.std(ddof=1),
                             "ci_lo":ci[0],"ci_hi":ci[1],"sig":stars(p)})

    # Also test on accuracy DVs
    print("\n  Paired t-tests on Accuracy (Game Success Rate vs Lab Hit Rate):")
    for grp in ["Single","Multiple"]:
        sub = wide[wide["group"]==grp].dropna(subset=["success_rate"])
        if len(sub) < 2: continue
        # Lab hit rate = 100% for all (every trial has a click), compare game SR
        game_sr = sub["success_rate"].values
        lab_hr  = np.full(len(sub), 100.0)  # lab = always 100% hit
        t2, p2 = ttest_rel(game_sr, lab_hr)
        print(f"    {grp}: Game SR ({game_sr.mean():.1f}%) vs Lab HR (100%) "
              f"→ t({len(sub)-1})={t2:.3f}  p={p2:.4f} {stars(p2)}")

pd.DataFrame(paired_rows).to_csv(OUT / "results_rq3_ttest_paired.csv", index=False)
print("\n  ✔  Saved results_rq3_ttest_paired.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4D. RQ4 — LEVEL EFFECTS (Spearman trends)
# ─────────────────────────────────────────────────────────────────────────────

section("4D — RQ4: Level Effects in Game  (Spearman + linear trend)")

level_rows = []
if not game_all.empty:
    dvs = [("RT_ms","RT (ms)"),("success_rate","Success Rate %"),
           ("hit_rate","Hit Rate %"),("false_alarms","False Alarms"),
           ("avg_inter_target_ms","Avg Inter-Target ms")]

    for grp in ["Single","Multiple"]:
        sub_all = game_all[game_all["group"]==grp]
        print(f"\n  {grp} group  ({sub_all['participant'].nunique()} participants):")
        for col, label in dvs:
            if col not in sub_all.columns: continue
            tmp = sub_all[["level",col]].dropna()
            if len(tmp) < 4: continue
            rho, p_s = spearmanr(tmp["level"], tmp[col])
            r_p, p_p = pearsonr( tmp["level"], tmp[col])
            # Linear regression slope
            slope, intercept, r_lr, p_lr, se_slope = stats.linregress(tmp["level"],tmp[col])
            direction = "↑" if rho>0 else "↓"
            print(f"    {label:<28}"
                  f"  ρ={rho:+.3f} p={p_s:.4f}{stars(p_s)}"
                  f"  β={slope:+.2f}/level  {direction}")
            level_rows.append({"Group":grp,"DV":label,"rho":rho,"p_spearman":p_s,
                                "r_pearson":r_p,"p_pearson":p_p,
                                "slope":slope,"p_slope":p_lr,"sig":stars(p_s)})

pd.DataFrame(level_rows).to_csv(OUT / "results_rq4_level_trends.csv", index=False)
print("\n  ✔  Saved results_rq4_level_trends.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4E. RELIABILITY
# ─────────────────────────────────────────────────────────────────────────────

section("4E — Reliability  (Split-Half + Spearman-Brown + Cronbach's α)")

def split_half_odd_even(arr):
    arr = np.array(arr, dtype=float)[~np.isnan(arr)]
    n   = (len(arr)//2)*2
    if n < 6: return np.nan, np.nan
    r, _ = pearsonr(arr[:n:2], arr[1:n:2])
    return r, (2*r)/(1+r) if r != -1 else np.nan

def cronbach_alpha(data_matrix):
    """
    data_matrix: rows = participants, cols = trials (or items)
    Cronbach's alpha = (k/(k-1)) * (1 - sum(item_var)/total_var)
    """
    mat = np.array(data_matrix, dtype=float)
    # Remove rows with NaN
    mat = mat[~np.isnan(mat).any(axis=1)]
    if mat.shape[0] < 2 or mat.shape[1] < 2: return np.nan
    k         = mat.shape[1]   # items (trials)
    item_vars  = mat.var(axis=0, ddof=1)
    total_var  = mat.sum(axis=1).var(ddof=1)
    if total_var == 0: return np.nan
    return (k/(k-1)) * (1 - item_vars.sum()/total_var)

rel_rows = []
if not lab_all.empty:
    for grp in ["Single","Multiple"]:
        sub = lab_all[lab_all["group"]==grp]
        n_p = sub["participant"].nunique()
        all_rt = sub["RT_ms"].dropna().values
        print(f"\n  {grp} group  (n={n_p} participants, {len(all_rt)} total trials):")

        # Split-half on pooled RT
        r_sh, r_sb = split_half_odd_even(all_rt)
        print(f"    Split-half r  (odd/even trials): r = {r_sh:.3f}"
              if not np.isnan(r_sh) else "    Split-half: insufficient data")
        print(f"    Spearman-Brown prophecy:         r = {r_sb:.3f}"
              if not np.isnan(r_sb) else "")

        # Cronbach's α: build participants × trials matrix
        ptpt_rts  = [g["RT_ms"].dropna().values
                     for _, g in sub.groupby("participant")
                     if g["RT_ms"].dropna().count() >= 5]
        if len(ptpt_rts) >= 2:
            min_t = min(len(x) for x in ptpt_rts)
            mat   = np.vstack([x[:min_t] for x in ptpt_rts])
            alpha = cronbach_alpha(mat)
            interp = ("poor" if alpha<.5 else "questionable" if alpha<.6 else
                      "acceptable" if alpha<.7 else "good" if alpha<.8 else "excellent")
            print(f"    Cronbach's α  (participants × {min_t} trials): "
                  f"α = {alpha:.3f}  → {interp}"
                  if not np.isnan(alpha) else
                  "    Cronbach's α: insufficient data")
        else:
            print("    Cronbach's α: need ≥2 participants with ≥5 trials")
            alpha = np.nan

        rel_rows.append({"Group":grp,"n_participants":n_p,
                         "split_half_r":r_sh,"spearman_brown":r_sb,
                         "cronbach_alpha":alpha})

pd.DataFrame(rel_rows).to_csv(OUT / "results_reliability.csv", index=False)
print("\n  ✔  Saved results_reliability.csv")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENTIAL VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,
                     "axes.titleweight":"bold","font.family":"DejaVu Sans"})

# Fig 4-01  ANOVA: Effect size bar chart
if not wide.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ANOVA Results: Effect Sizes & F-values", fontsize=13, fontweight="bold")

    labels = ["Target Load\n(RQ2)","Modality\n(RQ3)","Interaction"]
    eta2s  = [eta2_A, eta2_B, eta2_AB]
    f_vals = [F_A, F_B, F_AB]
    pvals  = [p_A, p_B, p_AB]
    colors = [PAL["Single"],PAL["Lab"],PAL["Game"]]

    ax = axes[0]
    bars = ax.bar(labels, eta2s, color=colors, edgecolor="white", alpha=0.85, width=0.5)
    for thresh, lbl, lc in [(0.01,"small","#CBD5E0"),(0.06,"medium","#A0AEC0"),(0.14,"large","#718096")]:
        ax.axhline(thresh, color=lc, ls="--", lw=1.2, label=lbl)
    for bar, p_v, e in zip(bars, pvals, eta2s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                stars(p_v), ha="center", fontsize=13, fontweight="bold")
    ax.set_title("Partial η²  (effect size)"); ax.set_ylabel("η²p"); ax.legend(fontsize=8)

    ax = axes[1]
    valid = [(l, F, p) for l, F, p in zip(labels, f_vals, pvals) if not np.isnan(F)]
    if valid:
        lbls_v, fs_v, ps_v = zip(*valid)
        bars2 = ax.bar(lbls_v, fs_v, color=colors[:len(fs_v)], edgecolor="white", alpha=0.85, width=0.5)
        # Critical F at α=.05
        f_crit = f_dist.ppf(.95, 1, df_BsA if df_BsA else df_SA)
        ax.axhline(f_crit, color="crimson", ls="--", lw=1.5, label=f"F-crit={f_crit:.2f} (p=.05)")
        for bar, pv, fv in zip(bars2, ps_v, fs_v):
            ax.text(bar.get_x()+bar.get_width()/2, fv+0.3,
                    f"F={fv:.2f}\n{stars(pv)}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title("F-values by Effect"); ax.set_ylabel("F statistic"); ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT/"fig_4-01_anova_effects.png", dpi=150, bbox_inches="tight")
    plt.close(); print("\n  ✔  fig_4-01_anova_effects.png")


# Fig 4-02  Paired t-test: Game vs Lab RT — paired dot plot
if paired_rows:
    fig, axes = plt.subplots(1, len(paired_rows), figsize=(6*len(paired_rows), 6))
    if len(paired_rows) == 1: axes = [axes]
    fig.suptitle("RQ3 — Paired Comparison: Game vs Lab RT",
                 fontsize=13, fontweight="bold")

    for ax, row in zip(axes, paired_rows):
        grp  = row["Group"]
        sub  = wide[wide["group"]==grp].dropna(subset=["RT_Lab","RT_Game"])
        for _, r in sub.iterrows():
            ax.plot(["Lab","Game"],[r["RT_Lab"],r["RT_Game"]],
                    "o-", color=PAL[grp], alpha=0.3, linewidth=1, markersize=6)
        ax.plot(["Lab","Game"],[sub["RT_Lab"].mean(),sub["RT_Game"].mean()],
                "D-", color=PAL[grp], linewidth=2.8, markersize=10,
                zorder=5, label=f"Mean")
        ax.set_title(f"{grp}  t({row['df']})={row['t']:.3f}  "
                     f"p={row['p']:.4f}{stars(row['p'])}\ndz={row['dz']:.3f}")
        ax.set_xlabel("Modality"); ax.set_ylabel("RT (ms)"); ax.legend()

    plt.tight_layout()
    plt.savefig(OUT/"fig_4-02_paired_ttest.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✔  fig_4-02_paired_ttest.png")


# Fig 4-03  Independent t-test forest plot
if indep_rows:
    fig, ax = plt.subplots(figsize=(10, max(4, len(indep_rows)*1.2)))
    y_pos = range(len(indep_rows))
    d_vals  = [r["d"] for r in indep_rows]
    labels_ = [r["DV"] for r in indep_rows]
    colors_ = [PAL["Single"] if r["d"]>0 else PAL["Multiple"] for r in indep_rows]

    ax.barh(list(y_pos), d_vals, color=colors_, alpha=0.75, edgecolor="white", height=0.55)
    for i, (r, y) in enumerate(zip(indep_rows, y_pos)):
        ax.text(r["d"] + (0.05 if r["d"]>=0 else -0.05), y,
                f"d={r['d']:.2f} {stars(r['p'])}", va="center", ha="left" if r["d"]>=0 else "right",
                fontsize=9, fontweight="bold")
    ax.axvline(0,   color="black", lw=1)
    for v, ls, lbl in [(0.2,":","small"),(0.5,"--","medium"),(0.8,"-.","large")]:
        ax.axvline( v, color="#718096", ls=ls, lw=1, alpha=0.6, label=f"|d|={v} ({lbl})")
        ax.axvline(-v, color="#718096", ls=ls, lw=1, alpha=0.6)
    ax.set_yticks(list(y_pos)); ax.set_yticklabels(labels_)
    ax.set_xlabel("Cohen's d  (positive = Single > Multiple)")
    ax.set_title("RQ2 — Independent t-tests: Single vs Multiple\n(Cohen's d Forest Plot)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT/"fig_4-03_forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✔  fig_4-03_forest_plot.png")


# Fig 4-04  Level trend heatmap (Spearman ρ per group × DV)
if level_rows:
    heat_df = pd.DataFrame(level_rows).pivot(index="DV", columns="Group", values="rho")
    fig, ax  = plt.subplots(figsize=(7, max(4, len(heat_df)*1.2)))
    sns.heatmap(heat_df, annot=True, fmt=".2f", center=0,
                cmap="RdBu_r", linewidths=0.5, linecolor="white",
                cbar_kws={"label":"Spearman ρ"}, ax=ax)
    ax.set_title("RQ4 — Spearman ρ: DV ~ Level\n(red=negative trend, blue=positive)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT/"fig_4-04_level_trend_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✔  fig_4-04_level_trend_heatmap.png")

print("\n  PART 4 COMPLETE.\n")

"""
╔══════════════════════════════════════════════════════════════════════╗
║  ADDITIONAL SNIPPET — Inter-Click Interval (ICI) Analysis            ║
║  Multiple Target Lab Data Only                                        ║
╚══════════════════════════════════════════════════════════════════════╝

What is ICI?
  Each multi-target lab trial requires the participant to click 5 targets.
  mouse.time stores the time (s) of each click relative to stimulus onset.
  ICI = time between consecutive clicks within a trial (ms).

  Click sequence per trial:  [t1, t2, t3, t4, t5]
  ICIs:  t2-t1, t3-t2, t4-t3, t5-t4  → 4 ICIs × 15 trials = 60 ICI values

This snippet adds:
  STATS : Full descriptives per click-pair rank, by trial type, Shapiro-Wilk,
          Kruskal-Wallis test across ranks, Mann-Whitney U by target colour
  PLOTS : fig_ici_01 — Distribution + KDE per click-pair (2→3, 3→4, 4→5)
          fig_ici_02 — Box + strip: ICI by click-pair rank
          fig_ici_03 — ICI by trial target colour (red vs white)
          fig_ici_04 — ICI over trial sequence (learning/fatigue trend)
          fig_ici_05 — Cumulative click time profile (mean ± SE ribbon)
          fig_ici_06 — Heatmap: ICI per trial × click rank
"""

import pandas as pd
import numpy as np
import ast, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import (shapiro, kruskal, mannwhitneyu,
                         spearmanr, f_oneway)

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("preprocessed_data")   # adjust if needed
OUT       = Path("outputs"); OUT.mkdir(exist_ok=True)

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
})

# RANK_COLORS = {
#     "1→2": "#2D6A9F",
#     "2→3": "#2F855A",
#     "3→4": "#C05621",
#     "4→5": "#7B2D8B",
# }
TARGET_COLORS = {"red": "#E53E3E", "white": "#4A5568"}

def save(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  {name}")

def stars(p):
    if p is None or np.isnan(p): return ""
    return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BUILD ICI DATAFRAME
# Load all multiple-group lab files and extract ICI for every click transition
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  ICI ANALYSIS — Multiple Target Lab")
print("=" * 65)

lab_dir = DATA_ROOT / "multiple" / "lab"
if not lab_dir.exists():
    raise FileNotFoundError(f"Directory not found: {lab_dir}")

records = []

for filepath in sorted(lab_dir.glob("*.csv")):
    pid_match = __import__("re").match(r"^(\d+)", filepath.stem)
    pid = pid_match.group(1) if pid_match else filepath.stem

    df = pd.read_csv(filepath)
    df = df.dropna(subset=["target_col"])
    df = df[df["target_col"].isin(["red","white"])].copy()
    df = df.reset_index(drop=True)

    for trial_idx, row in df.iterrows():
        # Parse mouse.time → list of click times (s) → convert to ms
        try:
            clicks_ms = [t * 1000
                         for t in ast.literal_eval(str(row["mouse.time"]))]
        except Exception:
            continue

        if len(clicks_ms) < 2:
            continue  # need at least 2 clicks for one ICI

        n_clicks = len(clicks_ms)

        for rank in range(1, n_clicks):          # rank: 1-based from-click
            ici = clicks_ms[rank] - clicks_ms[rank - 1]
            records.append({
                "participant":  pid,
                "trial_n":      trial_idx + 1,
                "target_col":   row["target_col"],
                "is_target":    int(row["target_col"] == "red"),
                "from_click":   rank,
                "to_click":     rank + 1,
                "click_pair":   f"{rank}→{rank+1}",
                "ICI_ms":       ici,
                "abs_t1_ms":    clicks_ms[rank - 1],   # cumulative time of from-click
                "abs_t2_ms":    clicks_ms[rank],        # cumulative time of to-click
                "n_clicks_trial": n_clicks,
            })

ici_df = pd.DataFrame(records)

n_ptpt   = ici_df["participant"].nunique()
n_trials = ici_df["trial_n"].nunique()
n_ici    = len(ici_df)
pairs    = sorted(ici_df["click_pair"].unique(),
                  key=lambda x: int(x.split("→")[0]))

RANK_COLORS = dict(zip(pairs, sns.color_palette("husl", len(pairs))))

print(f"\n  Participants : {n_ptpt}")
print(f"  Trials       : {n_trials}")
print(f"  Total ICI obs: {n_ici}  ({n_ici//n_trials} ICIs/trial × {n_trials} trials)")
print(f"  Click pairs  : {pairs}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*65)
print("  DESCRIPTIVE STATISTICS")
print("─"*65)

def full_desc(arr, label=""):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {}
    sw_p = shapiro(arr)[1] if len(arr) >= 3 else np.nan
    q1, q3 = np.percentile(arr, [25, 75])
    d = dict(n=len(arr), M=arr.mean(), SD=arr.std(ddof=1),
             SE=arr.std(ddof=1)/np.sqrt(len(arr)),
             Mdn=np.median(arr), IQR=q3-q1,
             Min=arr.min(), Max=arr.max(),
             Skew=stats.skew(arr), Kurt=stats.kurtosis(arr),
             SW_p=sw_p)
    return d

def print_desc(d, label, indent="  "):
    sw_flag = ("✓ normal" if d["SW_p"]>.05 else "✗ non-normal") if not np.isnan(d["SW_p"]) else ""
    print(f"{indent}{label}:  n={d['n']}  M={d['M']:.1f}  SD={d['SD']:.1f}"
          f"  SE={d['SE']:.1f}  Mdn={d['Mdn']:.1f}  IQR={d['IQR']:.1f}"
          f"  Min={d['Min']:.1f}  Max={d['Max']:.1f}"
          f"  Skew={d['Skew']:.2f}  Kurt={d['Kurt']:.2f}"
          f"  SW-p={d['SW_p']:.4f} {sw_flag}")

# Overall
print("\n  >> Overall ICI:")
d_all = full_desc(ici_df["ICI_ms"])
print_desc(d_all, "All ICIs")

# By click-pair rank
print("\n  >> ICI by Click-Pair Rank:")
rank_desc_rows = []
for pair in pairs:
    sub = ici_df[ici_df["click_pair"]==pair]["ICI_ms"]
    d   = full_desc(sub)
    print_desc(d, pair)
    rank_desc_rows.append({"click_pair": pair, **d})

# By target colour
print("\n  >> ICI by Trial Target Type:")
for col in ["red","white"]:
    sub = ici_df[ici_df["target_col"]==col]["ICI_ms"]
    print_desc(full_desc(sub), f"target={col}")

# By participant
print("\n  >> ICI by Participant:")
for pid, grp in ici_df.groupby("participant"):
    d = full_desc(grp["ICI_ms"])
    print_desc(d, f"PID {pid}", indent="    ")

# Save descriptives table
rank_desc_df = pd.DataFrame(rank_desc_rows)
rank_desc_df.to_csv(OUT / "ici_descriptives.csv", index=False)
print("\n  ✔  Saved ici_descriptives.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — INFERENTIAL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*65)
print("  INFERENTIAL STATISTICS")
print("─"*65)

# 3A. Kruskal-Wallis across click-pair ranks
#     H0: ICI distributions are equal across ranks 1→2, 2→3, 3→4, 4→5
groups_by_rank = [ici_df[ici_df["click_pair"]==p]["ICI_ms"].dropna().values
                  for p in pairs]
H, p_kw = kruskal(*groups_by_rank)
print(f"\n  3A. Kruskal-Wallis (ICI across {len(pairs)} click-pair ranks):")
print(f"      H({len(pairs)-1}) = {H:.3f}   p = {p_kw:.4f} {stars(p_kw)}")
print("      (Tests whether search speed is uniform across the click sequence)")

# Post-hoc pairwise Mann-Whitney U with Bonferroni correction
if p_kw < .05:
    print("      Post-hoc pairwise Mann-Whitney U (Bonferroni corrected):")
    n_comparisons = len(pairs)*(len(pairs)-1)//2
    for i, p1 in enumerate(pairs):
        for p2 in pairs[i+1:]:
            a = ici_df[ici_df["click_pair"]==p1]["ICI_ms"].dropna()
            b = ici_df[ici_df["click_pair"]==p2]["ICI_ms"].dropna()
            U, p_u = mannwhitneyu(a, b, alternative="two-sided")
            p_adj = min(p_u * n_comparisons, 1.0)
            r_eff = 1 - 2*U/(len(a)*len(b))   # rank-biserial r
            sig = stars(p_adj)
            print(f"        {p1} vs {p2}: U={U:.0f}  p_adj={p_adj:.4f} {sig}"
                  f"  r={r_eff:.3f}")

# 3B. Mann-Whitney U: ICI on target vs distractor trials
print(f"\n  3B. Mann-Whitney U (ICI: target=red vs distractor=white trials):")
ici_red   = ici_df[ici_df["target_col"]=="red"]["ICI_ms"].dropna()
ici_white = ici_df[ici_df["target_col"]=="white"]["ICI_ms"].dropna()
U2, p_mw  = mannwhitneyu(ici_red, ici_white, alternative="two-sided")
r_mw      = 1 - 2*U2/(len(ici_red)*len(ici_white))
print(f"      red:   M={ici_red.mean():.1f}  n={len(ici_red)}")
print(f"      white: M={ici_white.mean():.1f}  n={len(ici_white)}")
print(f"      U = {U2:.0f}   p = {p_mw:.4f} {stars(p_mw)}   r = {r_mw:.3f}")

# 3C. Spearman trend: Does ICI change with trial number?
print(f"\n  3C. Spearman ρ — ICI ~ Trial Number (fatigue / learning effect):")
trial_ici = ici_df.groupby("trial_n")["ICI_ms"].mean()
rho, p_sp = spearmanr(trial_ici.index, trial_ici.values)
slope, intercept, _, _, se_s = stats.linregress(trial_ici.index, trial_ici.values)
print(f"      ρ = {rho:+.3f}   p = {p_sp:.4f} {stars(p_sp)}")
print(f"      Linear slope = {slope:+.2f} ms/trial  "
      f"({'slowing' if slope>0 else 'speeding up'} across trials)")

# 3D. Spearman trend: Does ICI change across click ranks within trial?
print(f"\n  3D. Spearman ρ — ICI ~ Click Rank (within-trial speed change):")
rank_ici = ici_df.groupby("from_click")["ICI_ms"].mean()
rho2, p2 = spearmanr(rank_ici.index, rank_ici.values)
print(f"      ρ = {rho2:+.3f}   p = {p2:.4f} {stars(p2)}")
print(f"      (positive = getting slower at later targets within trial)")

# Save stats summary
stats_summary = pd.DataFrame([
    {"Test":"Kruskal-Wallis (rank)", "stat":H, "df":len(pairs)-1,
     "p":p_kw, "effect":"—", "sig":stars(p_kw)},
    {"Test":"Mann-Whitney (colour)", "stat":U2, "df":"—",
     "p":p_mw, "effect":f"r={r_mw:.3f}", "sig":stars(p_mw)},
    {"Test":"Spearman (trial trend)", "stat":rho, "df":"—",
     "p":p_sp, "effect":"—", "sig":stars(p_sp)},
    {"Test":"Spearman (rank trend)", "stat":rho2, "df":"—",
     "p":p2, "effect":"—", "sig":stars(p2)},
])
stats_summary.to_csv(OUT / "ici_inferential_stats.csv", index=False)
print("\n  ✔  Saved ici_inferential_stats.csv")

# ── FIG: Per-participant mean RT — dot plot (Lab and Game side by side)
# Shows individual differences within each group clearly
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Per-Participant Mean RT — Individual Differences\n"
             "Hypothesis: Greater inter-individual spread in Game than Lab, "
             "reflecting less controlled measurement environment",
             fontsize=11, fontweight="bold")

for ax, (df, title, ylabel) in zip(axes, [
    (ptpt_lab,  "Lab Task",  "Mean RT (ms)"),
    (ptpt_game, "Game Task", "Mean RT (ms)"),
]):
    rng = np.random.default_rng(42)
    for grp, color, xpos in [("Single","#2D6A9F",1), ("Multiple","#C05621",2)]:
        sub = df[df["group"]==grp]["RT_mean"].dropna().values
        jx  = rng.uniform(xpos-0.2, xpos+0.2, len(sub))
        ax.scatter(jx, sub, color=color, s=65, alpha=0.75, zorder=3)
        ax.hlines(sub.mean(), xpos-0.3, xpos+0.3, color=color, lw=2.5, zorder=4)
        ax.errorbar(xpos, sub.mean(), yerr=sub.std(ddof=1),
                    fmt="none", color=color, capsize=6, lw=1.5)
        ax.annotate(f"M={sub.mean():.0f}\nSD={sub.std(ddof=1):.0f}",
                    xy=(xpos, sub.mean()), xytext=(xpos+0.35, sub.mean()),
                    fontsize=9, color=color, fontweight="bold", va="center")
    ax.set_xticks([1,2])
    ax.set_xticklabels(["Single\n(n=21)","Multiple\n(n=16)"])
    ax.set_title(title); ax.set_ylabel(ylabel)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0],color="#2D6A9F",marker="o",ls="",label="Single"),
        Line2D([0],[0],color="#C05621",marker="o",ls="",label="Multiple"),
    ], fontsize=9)

plt.tight_layout()
plt.savefig(OUT/"fig_r1_participant_rt_dotplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔  fig_r1_participant_rt_dotplot.png")

# ── FIG: Accuracy ceiling — SR and HR distributions across all 4 cells
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Game Accuracy Distributions — Ceiling Effect\n"
             "Hypothesis: Both groups will show strong ceiling effects (Mdn=100%) "
             "suggesting the game accuracy measure has limited discriminatory power",
             fontsize=11, fontweight="bold")

cells_acc = [
    (game_all[game_all["group"]=="Single"],  "Single — Success Rate",   "success_rate", "#2D6A9F"),
    (game_all[game_all["group"]=="Multiple"],"Multiple — Success Rate", "success_rate", "#C05621"),
    (game_all[game_all["group"]=="Single"],  "Single — Hit Rate",       "hit_rate",     "#2D6A9F"),
    (game_all[game_all["group"]=="Multiple"],"Multiple — Hit Rate",     "hit_rate",     "#C05621"),
]
for ax, (df, title, col, color) in zip(axes.flat, cells_acc):
    vals = df[col].dropna().values
    below = vals[vals < 100]
    at100 = (vals == 100).sum()
    # Bar for ceiling
    ax.bar([100], [at100], width=2.5, color=color, alpha=0.85,
           label=f"100% ceiling  (n={at100}, {100*at100/len(vals):.0f}%)",
           edgecolor="white")
    if len(below):
        ax.hist(below, bins=np.arange(0,103,5), color=color,
                alpha=0.35, edgecolor="white", label=f"<100%  (n={len(below)})")
    ax.axvline(vals.mean(), color="black", ls="--", lw=1.6,
               label=f"M={vals.mean():.1f}%")
    ax.set_title(f"{title}\nM={vals.mean():.1f}%, Mdn={np.median(vals):.0f}%")
    ax.set_xlabel("Accuracy (%)"); ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT/"fig_r1_accuracy_ceiling.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔  fig_r1_accuracy_ceiling.png")

# ── FIG: RTV — within-participant SD scatter, Lab vs Game
# Requires _data_cache.pkl with ptpt_lab and ptpt_game
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs")
with open(OUT/"_data_cache.pkl","rb") as f:
    c = pickle.load(f)
ptpt_lab  = c["ptpt_lab"]
ptpt_game = c["ptpt_game"]

merged_rtv = ptpt_lab[["participant","group","RT_sd"]].rename(
    columns={"RT_sd":"lab_sd"}).merge(
    ptpt_game[["participant","group","RT_sd"]].rename(
        columns={"RT_sd":"game_sd"}),
    on=["participant","group"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Intra-Individual RT Variability (SD per Participant) — Lab vs Game\n"
             "Additional Hypothesis: Higher RT variability in the Game may indicate "
             "attentional instability not captured by mean RT alone",
             fontsize=10, fontweight="bold")

for ax, (grp, color) in zip(axes, [("Single","#2D6A9F"),("Multiple","#C05621")]):
    sub = merged_rtv[merged_rtv["group"]==grp].sort_values("lab_sd").reset_index(drop=True)
    x   = range(len(sub))
    ax.scatter(x, sub["lab_sd"].values,  marker="o", color=color,
               alpha=0.55, s=60, label=f"Lab   M={sub['lab_sd'].mean():.0f} ms")
    ax.scatter(x, sub["game_sd"].values, marker="s", color=color,
               alpha=0.90, s=60, label=f"Game M={sub['game_sd'].mean():.0f} ms")
    ax.hlines(sub["lab_sd"].mean(),  0, len(sub)-1, color=color, ls="--", lw=1.5, alpha=0.5)
    ax.hlines(sub["game_sd"].mean(), 0, len(sub)-1, color=color, ls="-",  lw=1.5, alpha=0.9)
    ax.set_title(f"{grp} Group\n"
                 f"Lab SD: {sub['lab_sd'].min():.0f}–{sub['lab_sd'].max():.0f} ms  |  "
                 f"Game SD: {sub['game_sd'].min():.0f}–{sub['game_sd'].max():.0f} ms")
    ax.set_xlabel("Participant rank (sorted by Lab SD)")
    ax.set_ylabel("Within-participant SD of RT (ms)")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT/"fig_r1_rtv_variability.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  fig_r1_rtv_variability.png")
# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*65)
print("  PLOTS")
print("─"*65)

# ── FIG ICI-01  Distribution + KDE per click-pair rank ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("ICI Distribution per Click-Pair Rank\n(Multiple Target Lab)",
             fontsize=14, fontweight="bold")

for ax, pair in zip(axes.flat, pairs):
    sub   = ici_df[ici_df["click_pair"]==pair]["ICI_ms"].dropna().values
    color = RANK_COLORS[pair]
    ax.hist(sub, bins=max(5, len(sub)//3), color=color,
            alpha=0.55, edgecolor="white", density=True, label="Histogram")
    if len(sub) >= 4:
        kde_x = np.linspace(sub.min()*0.85, sub.max()*1.1, 300)
        kde   = stats.gaussian_kde(sub, bw_method="scott")
        ax.plot(kde_x, kde(kde_x), color=color, lw=2.2, label="KDE")
    ax.axvline(sub.mean(),   color="black",  ls="--", lw=1.6,
               label=f"M={sub.mean():.0f}")
    ax.axvline(np.median(sub), color="dimgray", ls=":",  lw=1.5,
               label=f"Mdn={np.median(sub):.0f}")
    sw_p   = shapiro(sub)[1] if len(sub)>=3 else np.nan
    normal = "normal" if sw_p>.05 else "non-normal"
    ax.set_title(f"Click {pair}  (n={len(sub)}, SD={sub.std():.0f} ms)\nSW p={sw_p:.4f} — {normal}")
    ax.set_xlabel("ICI (ms)"); ax.set_ylabel("Density"); ax.legend(fontsize=8)

plt.tight_layout()
save("fig_ici_01_distributions_by_rank.png")


# ── FIG ICI-02  Box + Strip: ICI by click-pair rank ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("ICI by Click-Pair Rank — Distribution Comparison",
             fontsize=14, fontweight="bold")

# Left: boxplot
ax = axes[0]
ici_df["click_pair_ord"] = pd.Categorical(ici_df["click_pair"],
                                           categories=pairs, ordered=True)
sns.boxplot(data=ici_df, x="click_pair_ord", y="ICI_ms",
            palette=RANK_COLORS, width=0.5, linewidth=1.3,
            flierprops=dict(marker="o", ms=4, alpha=0.4), ax=ax)
sns.stripplot(data=ici_df, x="click_pair_ord", y="ICI_ms",
              palette=RANK_COLORS, alpha=0.4, size=5,
              jitter=True, ax=ax)
# Annotate group means
for i, pair in enumerate(pairs):
    m = ici_df[ici_df["click_pair"]==pair]["ICI_ms"].mean()
    ax.text(i, ici_df["ICI_ms"].max()*1.03, f"M={m:.0f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_title(f"Box + Strip  (KW: H={H:.2f}, p={p_kw:.4f} {stars(p_kw)})")
ax.set_xlabel("Click Pair (from → to)"); ax.set_ylabel("ICI (ms)")

# Right: violin
ax = axes[1]
sns.violinplot(data=ici_df, x="click_pair_ord", y="ICI_ms",
               palette=RANK_COLORS, inner="quartile",
               cut=0, linewidth=1.2, ax=ax)
ax.set_title("Violin Plot — ICI by Click Rank")
ax.set_xlabel("Click Pair (from → to)"); ax.set_ylabel("ICI (ms)")

plt.tight_layout()
save("fig_ici_02_boxplot_by_rank.png")

# ── FIG: ICI within-trial slowing — annotated bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Inter-Click Interval by Click-Pair Rank — Multiple Target Lab\n"
             "Hypothesis: ICI increases with click rank, reflecting progressive depletion "
             "of salient targets (Feature Integration Theory, Treisman & Gelade, 1980)",
             fontsize=10, fontweight="bold")

pairs_main = ["1→2","2→3","3→4","4→5"]
means_ = [ici_df[ici_df["click_pair"]==p]["ICI_ms"].mean() for p in pairs_main]
sems_  = [ici_df[ici_df["click_pair"]==p]["ICI_ms"].sem()  for p in pairs_main]
colors_= ["#2D6A9F","#2F855A","#C05621","#7B2D8B"]

# Left: annotated bar with trend arrow
ax = axes[0]
bars = ax.bar(pairs_main, means_, yerr=sems_, color=colors_,
              alpha=0.78, edgecolor="white", width=0.55, capsize=5)
for bar, m in zip(bars, means_):
    ax.text(bar.get_x()+bar.get_width()/2, m+15, f"{m:.0f} ms",
            ha="center", fontsize=10, fontweight="bold")
ax.annotate("", xy=(3.4, max(means_)+80), xytext=(0, max(means_)+80),
            arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2))
ax.text(1.7, max(means_)+100, "Within-trial slowing →",
        ha="center", fontsize=9, color="crimson", fontstyle="italic")
ax.set_title("Mean ICI ± SE per Click Transition\n(bars coloured by transition rank)")
ax.set_xlabel("Click Pair (from → to)"); ax.set_ylabel("ICI (ms)")

# Right: individual participant means per rank
ax = axes[1]
for pid, pdf in ici_df[ici_df["click_pair"].isin(pairs_main)].groupby("participant"):
    pm = [pdf[pdf["click_pair"]==p]["ICI_ms"].mean() for p in pairs_main]
    ax.plot(range(1,5), pm, "o-", color="#718096", alpha=0.35, lw=1, ms=4)
grp_means = [ici_df[ici_df["click_pair"]==p]["ICI_ms"].mean() for p in pairs_main]
ax.plot(range(1,5), grp_means, "D-", color="#C05621",
        lw=2.5, ms=9, zorder=5, label="Group mean")
ax.set_xticks(range(1,5)); ax.set_xticklabels(pairs_main)
ax.set_title("ICI per Participant × Click Rank\n(grey=individual, orange=mean)")
ax.set_xlabel("Click Pair"); ax.set_ylabel("ICI (ms)"); ax.legend()

plt.tight_layout()
plt.savefig(OUT/"fig_r1_ici_slowing.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔  fig_r1_ici_slowing.png")


# ── FIG ICI-03  ICI by trial target colour ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle("ICI by Trial Target Type (red vs white)",
             fontsize=14, fontweight="bold")

# Panel A: overall box+strip
ax = axes[0]
sns.boxplot(data=ici_df, x="target_col", y="ICI_ms",
            palette=TARGET_COLORS, order=["red","white"],
            width=0.45, linewidth=1.3, ax=ax,
            flierprops=dict(marker="o", ms=4, alpha=0.4))
sns.stripplot(data=ici_df, x="target_col", y="ICI_ms",
              palette=TARGET_COLORS, order=["red","white"],
              alpha=0.4, size=5, jitter=True, ax=ax)
ax.set_title(f"Overall ICI\nMW: U={U2:.0f}, p={p_mw:.4f} {stars(p_mw)}, r={r_mw:.3f}")
ax.set_xlabel("Trial Target Colour"); ax.set_ylabel("ICI (ms)")

# Panel B: ICI by rank AND colour
ax = axes[1]
color_order = ["red","white"]
x_pos       = np.arange(len(pairs))
width       = 0.35
for j, (col, lbl, alpha_v) in enumerate([("red","Target (red)",0.85),
                                          ("white","Distractor (white)",0.65)]):
    means = [ici_df[(ici_df["click_pair"]==p)&(ici_df["target_col"]==col)]["ICI_ms"].mean()
             for p in pairs]
    sems  = [ici_df[(ici_df["click_pair"]==p)&(ici_df["target_col"]==col)]["ICI_ms"].sem()
             for p in pairs]
    bars  = ax.bar(x_pos + j*width, means, width, label=lbl,
                   color=TARGET_COLORS[col], alpha=alpha_v, edgecolor="white")
    ax.errorbar(x_pos + j*width, means, yerr=sems, fmt="none",
                color="black", capsize=4, linewidth=1.2)
ax.set_xticks(x_pos + width/2)
ax.set_xticklabels(pairs)
ax.set_title("Mean ICI ± SE\nper Rank × Target Type")
ax.set_xlabel("Click Pair"); ax.set_ylabel("ICI (ms)"); ax.legend()

# Panel C: KDE overlay — red vs white
ax = axes[2]
for col, color, label in [("red",TARGET_COLORS["red"],"Target (red)"),
                           ("white",TARGET_COLORS["white"],"Distractor (white)")]:
    sub = ici_df[ici_df["target_col"]==col]["ICI_ms"].dropna().values
    if len(sub) >= 4:
        kde_x = np.linspace(ici_df["ICI_ms"].min()*0.8,
                            ici_df["ICI_ms"].max()*1.1, 300)
        kde   = stats.gaussian_kde(sub, bw_method="scott")
        ax.plot(kde_x, kde(kde_x), color=color, lw=2.2, label=label)
        ax.fill_between(kde_x, kde(kde_x), alpha=0.15, color=color)
    ax.axvline(sub.mean(), color=color, ls="--", lw=1.4)
ax.set_title("KDE: ICI by Target Colour")
ax.set_xlabel("ICI (ms)"); ax.set_ylabel("Density"); ax.legend()

plt.tight_layout()
save("fig_ici_03_by_target_type.png")


# ── FIG ICI-04  ICI across trial sequence (fatigue / learning) ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("ICI Trend Across Trials (Fatigue / Learning Effect)",
             fontsize=14, fontweight="bold")

# Panel A: mean ICI per trial with trend line
ax = axes[0]
trial_means = ici_df.groupby("trial_n")["ICI_ms"].agg(["mean","sem"])
ax.errorbar(trial_means.index, trial_means["mean"],
            yerr=trial_means["sem"], fmt="o-",
            color="#2D6A9F", lw=2, markersize=6, capsize=4, alpha=0.85,
            label="Mean ICI ± SE")
# Trend line
xs  = np.linspace(trial_means.index.min(), trial_means.index.max(), 200)
ax.plot(xs, slope*xs + intercept, "--", color="crimson", lw=1.8,
        label=f"Trend: β={slope:+.1f} ms/trial {stars(p_sp)}")
ax.set_title(f"Mean ICI per Trial\nSpearman ρ={rho:+.3f}, p={p_sp:.4f} {stars(p_sp)}")
ax.set_xlabel("Trial Number"); ax.set_ylabel("Mean ICI (ms)"); ax.legend()

# Panel B: ICI coloured by click rank across trials (scatter)
ax = axes[1]
for pair in pairs:
    sub = ici_df[ici_df["click_pair"]==pair]
    trial_rank_means = sub.groupby("trial_n")["ICI_ms"].mean()
    ax.plot(trial_rank_means.index, trial_rank_means.values,
            "o-", color=RANK_COLORS[pair], label=pair,
            lw=1.5, markersize=5, alpha=0.75)
ax.set_title("ICI per Trial by Click Rank\n(each line = one click transition)")
ax.set_xlabel("Trial Number"); ax.set_ylabel("ICI (ms)"); ax.legend(title="Click Pair")

plt.tight_layout()
save("fig_ici_04_trial_trend.png")


# ── FIG ICI-05  Cumulative click time profile (mean ± SE ribbon) ─────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Cumulative Click Time Profile — Mean ± SE\n"
             "(How long it takes to reach each target within a trial)",
             fontsize=12, fontweight="bold")

# Collect abs time at each click rank per trial
abs_by_rank = {r: [] for r in range(1, 6)}
for _, row in ici_df.iterrows():
    abs_by_rank[row["to_click"]].append(row["abs_t2_ms"])

# Add rank 1 (first click = abs_t1 of first ICI)
first_clicks = ici_df[ici_df["from_click"]==1]["abs_t1_ms"].values
abs_by_rank[1] = list(first_clicks)

ranks_with_data = sorted([r for r in abs_by_rank if len(abs_by_rank[r]) >= 2])
means_cum = [np.mean(abs_by_rank[r]) for r in ranks_with_data]
sems_cum  = [np.std(abs_by_rank[r], ddof=1)/np.sqrt(len(abs_by_rank[r]))
             for r in ranks_with_data]

ax.plot(ranks_with_data, means_cum, "D-", color="#2D6A9F",
        lw=2.5, markersize=8, zorder=5, label="Mean cumulative RT")
ax.fill_between(ranks_with_data,
                np.array(means_cum) - np.array(sems_cum),
                np.array(means_cum) + np.array(sems_cum),
                alpha=0.2, color="#2D6A9F")

# Annotate each point
for r, m, s in zip(ranks_with_data, means_cum, sems_cum):
    ax.annotate(f"{m:.0f} ms", (r, m), textcoords="offset points",
                xytext=(8, 5), fontsize=9, fontweight="bold")

ax.set_xticks(ranks_with_data)
ax.set_xticklabels([f"Click {r}" for r in ranks_with_data])
ax.set_xlabel("Click Rank (1 = first target hit)")
ax.set_ylabel("Time from Stimulus Onset (ms)")
ax.legend()

plt.tight_layout()
save("fig_ici_05_cumulative_click_profile.png")


# ── FIG ICI-06  Heatmap: ICI per trial × click rank ─────────────────────────
pivot = ici_df.pivot_table(index="trial_n", columns="click_pair",
                            values="ICI_ms", aggfunc="mean")
pivot = pivot[pairs]  # enforce column order

fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(pivot)*0.45 + 2)))
fig.suptitle("ICI Heatmap: Trial × Click Rank",
             fontsize=14, fontweight="bold")

# Left: raw values
ax = axes[0]
sns.heatmap(pivot, ax=ax, cmap="YlOrRd",
            annot=len(pivot) <= 20,
            fmt=".0f" if len(pivot) <= 20 else "",
            linewidths=0.4, linecolor="white",
            cbar_kws={"label":"ICI (ms)", "shrink":0.7})
ax.set_title("Raw ICI (ms)")
ax.set_xlabel("Click Pair"); ax.set_ylabel("Trial Number")

# Right: z-scored within columns (highlight which trials are fast/slow per rank)
pivot_z = (pivot - pivot.mean()) / (pivot.std() + 1e-9)
ax = axes[1]
sns.heatmap(pivot_z, ax=ax, cmap="RdBu_r", center=0,
            annot=len(pivot) <= 20,
            fmt=".1f" if len(pivot) <= 20 else "",
            linewidths=0.4, linecolor="white",
            cbar_kws={"label":"z-score", "shrink":0.7})
ax.set_title("z-scored ICI (highlights fast/slow relative to rank mean)")
ax.set_xlabel("Click Pair"); ax.set_ylabel("Trial Number")

plt.tight_layout()
save("fig_ici_06_heatmap_trial_rank.png")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  SUMMARY")
print("="*65)
print(f"""
  Dataset   : Multiple Target Lab  ({n_ptpt} participants, {n_trials} trials)
  ICI obs   : {n_ici}  ({len(pairs)} transitions × {n_trials} trials × {n_ptpt} participant(s))

  Overall   : M={d_all['M']:.1f} ms  SD={d_all['SD']:.1f}  Mdn={d_all['Mdn']:.1f}
              Skew={d_all['Skew']:.2f}  SW-p={d_all['SW_p']:.4f}
              ({"non-normal" if d_all['SW_p']<.05 else "normal"} — use non-parametric tests ✓)

  By Rank   : 1→2={ici_df[ici_df['click_pair']=='1→2']['ICI_ms'].mean():.0f}  "
              2→3={ici_df[ici_df['click_pair']=='2→3']['ICI_ms'].mean():.0f}  "
              3→4={ici_df[ici_df['click_pair']=='3→4']['ICI_ms'].mean():.0f}  "
              4→5={ici_df[ici_df['click_pair']=='4→5']['ICI_ms'].mean():.0f} ms

  KW test   : H({len(pairs)-1})={H:.3f}  p={p_kw:.4f} {stars(p_kw)}
  MW colour : U={U2:.0f}  p={p_mw:.4f} {stars(p_mw)}  r={r_mw:.3f}
  Trend/trial: ρ={rho:+.3f}  p={p_sp:.4f} {stars(p_sp)}  slope={slope:+.2f} ms/trial
  Trend/rank : ρ={rho2:+.3f}  p={p2:.4f} {stars(p2)}

  Figures saved (6):
    fig_ici_01_distributions_by_rank.png
    fig_ici_02_boxplot_by_rank.png
    fig_ici_03_by_target_type.png
    fig_ici_04_trial_trend.png
    fig_ici_05_cumulative_click_profile.png
    fig_ici_06_heatmap_trial_rank.png

  CSVs saved:
    ici_descriptives.csv
    ici_inferential_stats.csv
""")

"""
╔══════════════════════════════════════════════════════════════════════╗
║  SELECTIVE ATTENTION STUDY — COMPLETE ANALYSIS                       ║
║  PART 5: Master Runner + Full Results Summary Report                 ║
╚══════════════════════════════════════════════════════════════════════╝

Runs all 4 parts in sequence and produces a single printed summary
of every statistical result, plus a consolidated CSV.

Usage:
    python analysis_part5_master.py
    (must be run from the folder containing preprocessed_data/)
"""

import subprocess, sys, time
from pathlib import Path

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

PARTS = [
    ("PART 1 — Data Loading",         "analysis_part1_loading.py"),
    ("PART 2 — Descriptive Statistics","analysis_part2_descriptives.py"),
    ("PART 3 — Visualizations",        "analysis_part3_visualization.py"),
    ("PART 4 — Inferential Statistics","analysis_part4_inferential.py"),
]

print("╔" + "═"*66 + "╗")
print("║  SELECTIVE ATTENTION STUDY — FULL ANALYSIS PIPELINE           ║")
print("╚" + "═"*66 + "╝\n")

total_start = time.time()
all_ok = True

for label, script in PARTS:
    print(f"{'─'*66}")
    print(f"  ▶  {label}")
    print(f"{'─'*66}")
    start = time.time()
    result = subprocess.run([sys.executable, script],
                            capture_output=False, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"  ✗  {script} FAILED (exit code {result.returncode})")
        all_ok = False
    else:
        print(f"  ⏱  Completed in {elapsed:.1f}s\n")

total_elapsed = time.time() - total_start

print("╔" + "═"*66 + "╗")
print(f"║  {'PIPELINE COMPLETE' if all_ok else 'PIPELINE FINISHED WITH ERRORS':<64}  ║")
print(f"║  Total time: {total_elapsed:.1f}s{' '*(52-len(str(round(total_elapsed,1))))}  ║")
print("╚" + "═"*66 + "╝\n")

# ── Collect and print all saved CSVs ─────────────────────────────────────────
import pandas as pd, warnings
warnings.filterwarnings("ignore")

result_files = [
    ("Load Log",              "load_log.csv"),
    ("Descriptives (4-cell)", "descriptives_4cell.csv"),
    ("Ptpt Lab Aggregated",   "participant_lab_aggregated.csv"),
    ("Ptpt Game Aggregated",  "participant_game_aggregated.csv"),
    ("RQ1 Validity",          "results_rq1_validity.csv"),
    ("ANOVA",                 "results_anova.csv"),
    ("RQ2 Indep t-test",      "results_rq2_ttest_indep.csv"),
    ("RQ3 Paired t-test",     "results_rq3_ttest_paired.csv"),
    ("RQ4 Level Trends",      "results_rq4_level_trends.csv"),
    ("Reliability",           "results_reliability.csv"),
]

print("─"*66)
print("  SAVED OUTPUT FILES")
print("─"*66)

figures = sorted(OUT.glob("fig_*.png"))
csvs    = sorted(OUT.glob("*.csv"))

print(f"\n  Figures ({len(figures)}):")
for f in figures:
    print(f"    📊  {f.name}")

print(f"\n  Data/Results CSVs ({len(csvs)}):")
for f in csvs:
    print(f"    📄  {f.name}")

# ── Consolidated results printout ────────────────────────────────────────────
print("\n" + "="*66)
print("  FINAL RESULTS SUMMARY")
print("="*66)

for label, fname in result_files:
    path = OUT / fname
    if not path.exists():
        print(f"\n  {label}: file not found — run preceding parts first")
        continue
    df = pd.read_csv(path)
    print(f"\n  ── {label} ──")
    print(df.to_string(index=False, max_colwidth=40))

print("\n" + "="*66)
print("""
  ANALYSIS STRUCTURE REFERENCE
  ─────────────────────────────
  Part 1 → Data loading, parsing, variable semantics
  Part 2 → Descriptive stats for all 4 dataset types
             (n, M, SD, SE, Mdn, IQR, Skew, Kurt, Shapiro-Wilk)
  Part 3 → 12 visualization figures:
             RT distributions, Q-Q, target type, trial duration,
             click timing, game metrics, score, level effects,
             inter-target time, interaction plot, validity scatter,
             participant heatmap
  Part 4 → Inferential stats:
             RQ1 Pearson/Spearman validity
             RQ2 2×2 Mixed ANOVA + Welch t-tests
             RQ3 Paired t-tests (Game vs Lab)
             RQ4 Spearman level trends
             Reliability: split-half, Spearman-Brown, Cronbach's α
""")

