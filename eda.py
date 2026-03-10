"""
=============================================================================
EDA ANALYSIS — Selective Attention Study (Full Dataset)
Directory structure expected:
  preprocessed_data/
    multiple/
      lab/    → {pid}_visual_search_*.csv
      phone/  → {pid}_attentional_spotter_results.csv
    single/
      lab/    → {pid}_visual_search_*.csv
      phone/  → {pid}_attentional_spotter_results.csv
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
from scipy.stats import pearsonr, skew as scipy_skew

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"Single": "#4C72B0", "Multiple": "#DD8452",
          "Lab": "#55A868", "Game": "#C44E52"}

# ── CONFIGURE THESE PATHS ─────────────────────────────────────────────────────
DATA_ROOT = Path("preprocessed_data")   # <-- change if needed
OUT       = Path("outputs")
OUT.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_pid(path: Path) -> str:
    """Extract leading numeric participant ID from filename."""
    m = re.match(r"^(\d+)", path.stem)
    return m.group(1) if m else path.stem


def load_lab(path: Path, participant_id: str, group: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["target_col"])
    df = df[df["target_col"].isin(["red", "white"])].copy()
    df["participant"] = participant_id
    df["group"]       = group
    df["trial_n"]     = range(1, len(df) + 1)

    def first_click_ms(val):
        try:
            lst = ast.literal_eval(str(val))
            return float(lst[0]) * 1000
        except Exception:
            return np.nan

    df["RT_ms"]     = df["mouse.time"].apply(first_click_ms)
    df["is_target"] = (df["target_col"] == "red").astype(int)
    df["hit"]       = 1
    return df[["participant", "group", "trial_n", "is_target", "RT_ms", "hit"]]


def load_game(path: Path, participant_id: str, group: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["participant"] = participant_id
    df["group"]       = group
    df = df.rename(columns={
        "Level":                    "level",
        "Completed":                "completed",
        "SuccessRate(%)":           "success_rate",
        "HitRate(%)":               "hit_rate",
        "FalseAlarms":              "false_alarms",
        "InitialResponseTime(ms)":  "RT_ms",
        "AvgInterTargetTime(ms)":   "avg_inter_target_ms",
        "FinalScore":               "final_score",
    })
    df["completed"] = df["completed"].astype(str).str.lower().isin(["true","1","yes"])
    return df[["participant","group","level","completed","success_rate",
               "hit_rate","false_alarms","RT_ms","avg_inter_target_ms","final_score"]]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD ALL PARTICIPANTS FROM DIRECTORY TREE
# ─────────────────────────────────────────────────────────────────────────────

all_lab_dfs  = []
all_game_dfs = []
load_log     = []

for group_dir, group_label in [("multiple", "Multiple"), ("single", "Single")]:
    lab_dir   = DATA_ROOT / group_dir / "lab"
    phone_dir = DATA_ROOT / group_dir / "phone"

    if not lab_dir.exists() or not phone_dir.exists():
        print(f"  ⚠  Missing directory: {lab_dir} or {phone_dir} — skipping {group_label}")
        continue

    lab_map   = {extract_pid(f): f for f in sorted(lab_dir.glob("*.csv"))}
    phone_map = {extract_pid(f): f for f in sorted(phone_dir.glob("*.csv"))}
    all_pids  = sorted(set(lab_map) | set(phone_map),
                       key=lambda x: int(x) if x.isdigit() else x)

    for pid in all_pids:
        has_lab  = pid in lab_map
        has_game = pid in phone_map
        status   = "complete" if (has_lab and has_game) else "partial"

        if has_lab:
            try:
                df = load_lab(lab_map[pid], pid, group_label)
                all_lab_dfs.append(df)
            except Exception as e:
                status  = f"lab_error: {e}"
                has_lab = False

        if has_game:
            try:
                df = load_game(phone_map[pid], pid, group_label)
                all_game_dfs.append(df)
            except Exception as e:
                status   = f"game_error: {e}"
                has_game = False

        load_log.append({"participant": pid, "group": group_label,
                          "has_lab": has_lab, "has_game": has_game, "status": status})

lab_all  = pd.concat(all_lab_dfs,  ignore_index=True) if all_lab_dfs  else pd.DataFrame()
game_all = pd.concat(all_game_dfs, ignore_index=True) if all_game_dfs else pd.DataFrame()
log_df   = pd.DataFrame(load_log)

print("=" * 65)
print("DATA LOADING SUMMARY")
print("=" * 65)
for grp in ["Single", "Multiple"]:
    sub = log_df[log_df["group"] == grp]
    print(f"\n  {grp} group")
    print(f"    Participants found   : {len(sub)}")
    print(f"    Complete (lab+game)  : {(sub['status']=='complete').sum()}")
    print(f"    Lab only             : {(sub['has_lab'] & ~sub['has_game']).sum()}")
    print(f"    Game only            : {(~sub['has_lab'] & sub['has_game']).sum()}")

print(f"\n  Total lab  trials : {len(lab_all)}")
print(f"  Total game levels : {len(game_all)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ATTRITION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("ATTRITION ANALYSIS")
print("=" * 65)

EXPECTED = {"Single": 21, "Multiple": 16}
for grp in ["Single", "Multiple"]:
    sub      = log_df[log_df["group"] == grp]
    enrolled = len(sub)
    complete = (sub["status"] == "complete").sum()
    expected = EXPECTED.get(grp, "?")
    dropout  = expected - complete if isinstance(expected, int) else "?"
    pct      = 100 * complete / enrolled if enrolled else 0
    print(f"\n  {grp} group  (expected n={expected})")
    print(f"    Files found         : {enrolled}")
    print(f"    Complete records    : {complete}  ({pct:.1f}% of found)")
    print(f"    Estimated dropout   : {dropout}")

print("\n  Trial-level RT Missingness (Lab):")
for grp in ["Single", "Multiple"]:
    if lab_all.empty: break
    sub  = lab_all[lab_all["group"] == grp]
    miss = sub["RT_ms"].isna().sum()
    tot  = len(sub)
    print(f"    {grp:<12}  total={tot}  missing={miss}  ({100*miss/tot:.1f}%)" if tot else f"    {grp}: no data")

print("\n  Game Level Completion:")
for grp in ["Single", "Multiple"]:
    if game_all.empty: break
    sub  = game_all[game_all["group"] == grp]
    comp = sub["completed"].sum()
    tot  = len(sub)
    print(f"    {grp:<12}  levels={tot}  completed={comp}  ({100*comp/tot:.1f}%)" if tot else f"    {grp}: no data")

# Attrition figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Attrition Overview", fontsize=14, fontweight="bold")

ax = axes[0]
status_counts = log_df.groupby(["group","status"]).size().unstack(fill_value=0)
status_counts.plot(kind="bar", ax=ax, edgecolor="white", colormap="Set2")
ax.set_title("Participant Status by Group")
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)

ax = axes[1]
if not lab_all.empty:
    data = lab_all.groupby("group")["RT_ms"].apply(
        lambda x: pd.Series({"Valid": x.notna().sum(), "Missing": x.isna().sum()})
    ).unstack()
    data.plot(kind="bar", ax=ax, color=["#4CAF50","#F44336"], edgecolor="white")
ax.set_title("Lab RT: Valid vs Missing")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)

ax = axes[2]
if not game_all.empty:
    data = game_all.groupby("group")["completed"].apply(
        lambda x: pd.Series({"Completed": x.sum(), "Incomplete": (~x).sum()})
    ).unstack()
    data.plot(kind="bar", ax=ax, color=["#2196F3","#FF9800"], edgecolor="white")
ax.set_title("Game Level Completion")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(OUT / "fig_00_attrition.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✔  Saved fig_00_attrition.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DESCRIPTIVE STATISTICS — 4 CELLS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("DESCRIPTIVE STATISTICS — 4 CELLS (Group × Modality)")
print("=" * 65)

def desc(arr):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return dict(n=0, M=np.nan, SD=np.nan, Mdn=np.nan,
                    Min=np.nan, Max=np.nan, SE=np.nan, skew=np.nan)
    return dict(n=len(arr), M=arr.mean(), SD=arr.std(ddof=1),
                Mdn=np.median(arr), Min=arr.min(), Max=arr.max(),
                SE=arr.std(ddof=1)/np.sqrt(len(arr)),
                skew=scipy_skew(arr))

print(f"\n{'Group':<12}{'Modality':<10}{'DV':<20}"
      f"{'n':>5}{'M':>10}{'SD':>10}{'Mdn':>10}{'Min':>10}{'Max':>10}{'Skew':>8}")
print("-" * 95)

cell_rows = []
for grp in ["Single", "Multiple"]:
    for mod, df, dvs in [
        ("Lab",  lab_all,  [("RT_ms","RT (ms)"), ("hit","Hit Rate")]),
        ("Game", game_all, [("RT_ms","RT (ms)"), ("success_rate","Success %"),
                             ("hit_rate","Hit Rate %"), ("false_alarms","False Alarms")]),
    ]:
        if df.empty: continue
        sub = df[df["group"] == grp]
        for col, label in dvs:
            if col not in sub.columns: continue
            d = desc(sub[col].dropna().values)
            print(f"{grp:<12}{mod:<10}{label:<20}"
                  f"{d['n']:>5}{d['M']:>10.2f}{d['SD']:>10.2f}"
                  f"{d['Mdn']:>10.2f}{d['Min']:>10.2f}{d['Max']:>10.2f}{d['skew']:>8.2f}")
            cell_rows.append({"Group":grp,"Modality":mod,"DV":label,**d})

pd.DataFrame(cell_rows).to_csv(OUT / "summary_descriptives.csv", index=False)
print("\n  ✔  Saved summary_descriptives.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PARTICIPANT-LEVEL AGGREGATION (Slide 15)
# ─────────────────────────────────────────────────────────────────────────────

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

ptpt_lab.to_csv(OUT  / "participant_lab_aggregated.csv",  index=False)
ptpt_game.to_csv(OUT / "participant_game_aggregated.csv", index=False)
print("\n  ✔  Saved participant_lab_aggregated.csv")
print("  ✔  Saved participant_game_aggregated.csv")

print("\n  Participant-Level Mean RT by Group:")
for df, label in [(ptpt_lab,"Lab"), (ptpt_game,"Game")]:
    if df.empty: continue
    gm = df.groupby("group")["RT_mean"].agg(["mean","std","count"])
    print(f"\n  {label}:"); print(gm.round(2).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RT DISTRIBUTION — 4 PANELS
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Reaction Time Distributions — All Participants",
             fontsize=15, fontweight="bold")

for ax, (df, grp, title) in zip(axes.flat, [
    (lab_all,  "Single",   "Single — Lab Task"),
    (lab_all,  "Multiple", "Multiple — Lab Task"),
    (game_all, "Single",   "Single — Game"),
    (game_all, "Multiple", "Multiple — Game"),
]):
    if df.empty: ax.set_title(title + " (no data)"); continue
    rt = df[df["group"] == grp]["RT_ms"].dropna()
    if rt.empty: ax.set_title(title + " (no data)"); continue
    ax.hist(rt, bins=min(30, max(5, len(rt)//3)),
            color=COLORS[grp], edgecolor="white", alpha=0.85)
    ax.axvline(rt.mean(),   color="black", linestyle="--", linewidth=1.5,
               label=f"M={rt.mean():.0f}")
    ax.axvline(rt.median(), color="gray",  linestyle=":",  linewidth=1.5,
               label=f"Mdn={rt.median():.0f}")
    ax.set_title(f"{title}  (n={len(rt)})")
    ax.set_xlabel("RT (ms)"); ax.set_ylabel("Frequency"); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "fig_01_rt_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✔  Saved fig_01_rt_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  BOX + STRIP — PARTICIPANT MEANS
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("RT by Group — Participant Means", fontsize=14, fontweight="bold")

for ax, (df, title) in zip(axes, [
    (ptpt_lab,  "Lab Task — Mean RT per Participant"),
    (ptpt_game, "Game Task — Mean RT per Participant"),
]):
    if df.empty: continue
    sns.boxplot(data=df, x="group", y="RT_mean", order=["Single","Multiple"],
                palette=COLORS, width=0.45, linewidth=1.3, ax=ax)
    sns.stripplot(data=df, x="group", y="RT_mean", order=["Single","Multiple"],
                  palette=COLORS, alpha=0.6, size=5, jitter=True, ax=ax)
    ax.set_title(title); ax.set_xlabel("Target Load"); ax.set_ylabel("Mean RT (ms)")

plt.tight_layout()
plt.savefig(OUT / "fig_02_boxplot_group.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔  Saved fig_02_boxplot_group.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  INTERACTION PLOT
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Interaction Plot: Mean RT by Group × Modality",
             fontsize=13, fontweight="bold")

for mod, df in [("Lab", ptpt_lab), ("Game", ptpt_game)]:
    if df.empty: continue
    means, sems, ns = [], [], []
    for grp in ["Single","Multiple"]:
        v = df[df["group"]==grp]["RT_mean"].dropna()
        means.append(v.mean()); sems.append(v.sem()); ns.append(len(v))
    ls = "--" if mod == "Lab" else "-"
    mk = "o"  if mod == "Lab" else "s"
    ax.errorbar(["Single","Multiple"], means, yerr=sems, label=mod,
                linestyle=ls, marker=mk, color=COLORS[mod],
                linewidth=2.2, markersize=9, capsize=6)

ax.set_xlabel("Target Load", fontsize=12)
ax.set_ylabel("Mean RT (ms)", fontsize=12)
ax.legend(title="Modality", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "fig_03_interaction_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔  Saved fig_03_interaction_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  GAME ACCURACY METRICS
# ─────────────────────────────────────────────────────────────────────────────

if not ptpt_game.empty:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Game Accuracy Metrics by Group — Participant Means",
                 fontsize=14, fontweight="bold")
    for ax, (col, ylabel) in zip(axes, [
        ("success_rate","Success Rate (%)"),
        ("hit_rate","Hit Rate (%)"),
        ("false_alarms","False Alarms"),
    ]):
        sns.boxplot(data=ptpt_game, x="group", y=col, order=["Single","Multiple"],
                    palette=COLORS, width=0.5, linewidth=1.3, ax=ax)
        sns.stripplot(data=ptpt_game, x="group", y=col, order=["Single","Multiple"],
                      palette=COLORS, alpha=0.55, size=5, jitter=True, ax=ax)
        ax.set_title(ylabel); ax.set_xlabel("Group"); ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUT / "fig_04_game_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved fig_04_game_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  LEVEL EFFECTS — GROUP MEAN ± SE
# ─────────────────────────────────────────────────────────────────────────────

if not game_all.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Game Performance by Level — Group Mean ± SE",
                 fontsize=14, fontweight="bold")
    for ax, (col, ylabel) in zip(axes, [
        ("RT_ms","Initial RT (ms)"), ("success_rate","Success Rate (%)")
    ]):
        for grp, color in [("Single",COLORS["Single"]),("Multiple",COLORS["Multiple"])]:
            sub = game_all[game_all["group"]==grp].groupby("level")[col].agg(["mean","sem"])
            ax.plot(sub.index, sub["mean"], "o-", color=color,
                    label=grp, linewidth=2, markersize=5, alpha=0.85)
            ax.fill_between(sub.index,
                            sub["mean"]-sub["sem"], sub["mean"]+sub["sem"],
                            alpha=0.15, color=color)
        ax.set_title(ylabel); ax.set_xlabel("Level"); ax.set_ylabel(ylabel); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "fig_05_level_effects.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved fig_05_level_effects.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  LAB — TARGET vs NON-TARGET RT
# ─────────────────────────────────────────────────────────────────────────────

if not lab_all.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Lab Task: RT by Target Type per Group",
                 fontsize=14, fontweight="bold")
    for ax, grp in zip(axes, ["Single","Multiple"]):
        sub = lab_all[lab_all["group"]==grp].dropna(subset=["RT_ms"]).copy()
        sub["Target Type"] = sub["is_target"].map({1:"Target (red)",0:"Non-Target (white)"})
        sns.boxplot(data=sub, x="Target Type", y="RT_ms",
                    palette={"Target (red)":"#DD8452","Non-Target (white)":"#4C72B0"},
                    width=0.5, ax=ax)
        ax.set_title(f"{grp} (n={len(sub)} trials)"); ax.set_ylabel("RT (ms)")
    plt.tight_layout()
    plt.savefig(OUT / "fig_06_target_vs_nontarget.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved fig_06_target_vs_nontarget.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  CONCURRENT VALIDITY SCATTER
# ─────────────────────────────────────────────────────────────────────────────

if not ptpt_lab.empty and not ptpt_game.empty:
    merged = pd.merge(
        ptpt_lab[["participant","group","RT_mean"]].rename(columns={"RT_mean":"lab_RT"}),
        ptpt_game[["participant","group","RT_mean"]].rename(columns={"RT_mean":"game_RT"}),
        on=["participant","group"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Concurrent Validity: Game vs Lab RT (Participant Means)",
                 fontsize=14, fontweight="bold")
    for ax, grp in zip(axes, ["Single","Multiple"]):
        sub = merged[merged["group"]==grp]
        if len(sub) < 2:
            ax.set_title(f"{grp} (n={len(sub)} — need ≥2)"); continue
        ax.scatter(sub["game_RT"], sub["lab_RT"],
                   color=COLORS[grp], s=70, alpha=0.8, edgecolors="white")
        for _, row in sub.iterrows():
            ax.annotate(row["participant"], (row["game_RT"], row["lab_RT"]),
                        textcoords="offset points", xytext=(5,3), fontsize=7, alpha=0.7)
        z  = np.polyfit(sub["game_RT"], sub["lab_RT"], 1)
        xs = np.linspace(sub["game_RT"].min(), sub["game_RT"].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), "--", color="black", linewidth=1.5)
        r, p = pearsonr(sub["game_RT"], sub["lab_RT"])
        sig  = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
        ax.set_title(f"{grp}  (n={len(sub)})   r = {r:.3f} {sig}")
        ax.set_xlabel("Game Mean RT (ms)"); ax.set_ylabel("Lab Mean RT (ms)")
    plt.tight_layout()
    plt.savefig(OUT / "fig_07_validity_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved fig_07_validity_scatter.png")


print("\n" + "=" * 65)
print("  ✔  EDA COMPLETE — all outputs saved to:", OUT.resolve())
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 13.  GAME ATTRITION — LEVEL PROGRESSION (OUT OF 15)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("GAME ATTRITION — LEVEL PROGRESSION (Phone Task)")
print("=" * 65)

TOTAL_LEVELS = 15

if not ptpt_game.empty:

    # Compute attrition
    ptpt_game["attrition_levels"] = TOTAL_LEVELS - ptpt_game["max_level"]

    print(f"\n{'Group':<12}{'n':>5}{'Mean MaxLvl':>15}{'SD':>10}"
          f"{'% Reached 15':>18}{'Mean Attrition':>18}")
    print("-" * 80)

    for grp in ["Single", "Multiple"]:
        sub = ptpt_game[ptpt_game["group"] == grp]
        if len(sub) == 0:
            continue

        mean_lvl = sub["max_level"].mean()
        sd_lvl   = sub["max_level"].std(ddof=1)
        pct_full = 100 * (sub["max_level"] == TOTAL_LEVELS).mean()
        mean_att = sub["attrition_levels"].mean()

        print(f"{grp:<12}{len(sub):>5}"
              f"{mean_lvl:>15.2f}{sd_lvl:>10.2f}"
              f"{pct_full:>18.1f}%{mean_att:>18.2f}")

    # Save CSV
    ptpt_game.to_csv(OUT / "participant_game_attrition.csv", index=False)
    print("\n  ✔  Saved participant_game_attrition.csv")

    # ─────────────────────────────────────────────────────────
    # HISTOGRAM — MAX LEVEL REACHED
    # ─────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Game Attrition — Max Level Reached (out of 15)",
                 fontsize=14, fontweight="bold")

    for ax, grp in zip(axes, ["Single", "Multiple"]):
        sub = ptpt_game[ptpt_game["group"] == grp]
        if len(sub) == 0:
            ax.set_title(f"{grp} (no data)")
            continue

        ax.hist(sub["max_level"],
                bins=np.arange(0.5, TOTAL_LEVELS + 1.5, 1),
                color=COLORS[grp],
                edgecolor="white",
                alpha=0.85)

        ax.set_xticks(range(1, TOTAL_LEVELS + 1))
        ax.set_xlabel("Max Level Reached")
        ax.set_ylabel("Number of Participants")
        ax.set_title(f"{grp}  (n={len(sub)})")

    plt.tight_layout()
    plt.savefig(OUT / "fig_08_attrition_histogram.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved fig_08_attrition_histogram.png")

    # ─────────────────────────────────────────────────────────
    # SURVIVAL-STYLE CURVE — % REMAINING BY LEVEL
    # ─────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Game Progression Curve — % Participants Reaching Each Level",
                 fontsize=13, fontweight="bold")

    for grp in ["Single", "Multiple"]:
        sub = ptpt_game[ptpt_game["group"] == grp]
        if len(sub) == 0:
            continue

        survival_pct = []
        for level in range(1, TOTAL_LEVELS + 1):
            pct = 100 * (sub["max_level"] >= level).mean()
            survival_pct.append(pct)

        ax.plot(range(1, TOTAL_LEVELS + 1),
                survival_pct,
                marker="o",
                linewidth=2.2,
                label=grp,
                color=COLORS[grp])

    ax.set_xlabel("Level")
    ax.set_ylabel("% Participants Remaining")
    ax.set_ylim(0, 105)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "fig_09_attrition_survival.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved fig_09_attrition_survival.png")

else:
    print("\n  No game data available — cannot compute attrition.")


# ─────────────────────────────────────────────────────────────────────────────
# 14.  PROGRESSION EFFECTS — LAB vs PHONE (Group Mean ± SE)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("PROGRESSION EFFECTS — LAB vs PHONE")
print("=" * 65)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Performance Progression: Lab vs Phone",
             fontsize=14, fontweight="bold")

# ─────────────────────────────────────────────────────────
# RT PROGRESSION
# ─────────────────────────────────────────────────────────

ax = axes[0]

# Phone progression
if not game_all.empty:
    for grp in ["Single", "Multiple"]:
        sub = (game_all[game_all["group"] == grp]
               .groupby("level")["RT_ms"]
               .agg(["mean", "sem"]))
        ax.plot(sub.index,
                sub["mean"],
                marker="o",
                linewidth=2,
                linestyle="-",
                label=f"{grp} — Phone",
                color=COLORS[grp])

        ax.fill_between(sub.index,
                        sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"],
                        alpha=0.15,
                        color=COLORS[grp])

# Lab progression
if not lab_all.empty:
    for grp in ["Single", "Multiple"]:
        sub = (lab_all[lab_all["group"] == grp]
               .groupby("trial_n")["RT_ms"]
               .agg(["mean", "sem"]))

        ax.plot(sub.index,
                sub["mean"],
                marker="s",
                linewidth=2,
                linestyle="--",
                label=f"{grp} — Lab",
                color=COLORS[grp])

ax.set_title("Reaction Time Progression")
ax.set_xlabel("Level / Trial")
ax.set_ylabel("Mean RT (ms)")
ax.legend(fontsize=9)

# ─────────────────────────────────────────────────────────
# ACCURACY PROGRESSION (if available)
# ─────────────────────────────────────────────────────────

ax = axes[1]

# Phone success progression
if not game_all.empty:
    for grp in ["Single", "Multiple"]:
        sub = (game_all[game_all["group"] == grp]
               .groupby("level")["success_rate"]
               .mean())

        ax.plot(sub.index,
                sub.values,
                marker="o",
                linewidth=2,
                linestyle="-",
                label=f"{grp} — Phone",
                color=COLORS[grp])

# Lab hit progression (currently always 1 unless changed)
if not lab_all.empty:
    for grp in ["Single", "Multiple"]:
        sub = (lab_all[lab_all["group"] == grp]
               .groupby("trial_n")["hit"]
               .mean())

        ax.plot(sub.index,
                sub.values * 100,
                marker="s",
                linewidth=2,
                linestyle="--",
                label=f"{grp} — Lab",
                color=COLORS[grp])

ax.set_title("Accuracy Progression")
ax.set_xlabel("Level / Trial")
ax.set_ylabel("Accuracy (%)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "fig_10_progression_lab_vs_phone.png",
            dpi=150, bbox_inches="tight")
plt.close()

print("  ✔  Saved fig_10_progression_lab_vs_phone.png")