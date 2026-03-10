# """
# EDA – Selective Attention Game Study
# 4 Conditions:
#   1. Multiple Target  | Lab  (visual_search CSVs, e.g. 22_visual_search_*.csv)
#   2. Multiple Target  | Phone (attentional_spotter_results CSVs, e.g. 22_attentional_spotter_results.csv)
#   3. Single Target    | Lab  (visual_search CSVs, e.g. 1_visual_search_*.csv)
#   4. Single Target    | Phone (attentional_spotter_results CSVs, e.g. 1_attentional_spotter_results.csv)

# Usage:
#     python eda_selective_attention.py \
#         --multi_lab   "data/multiple/lab/*.csv" \
#         --multi_phone "data/multiple/phone/*.csv" \
#         --single_lab  "data/single/lab/*.csv" \
#         --single_phone "data/single/phone/*.csv" \
#         --out_dir      results/eda
# """

# import argparse, ast, glob, os, warnings
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import seaborn as sns
# from scipy import stats

# warnings.filterwarnings("ignore")
# sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
# PALETTE = {"Single": "#4C72B0", "Multiple": "#DD8452"}
# MOD_PALETTE = {"Lab": "#55A868", "Phone": "#C44E52"}

# # ─────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────

# def safe_parse_list(val):
#     """Parse a stringified Python list safely."""
#     try:
#         result = ast.literal_eval(str(val))
#         if isinstance(result, (list, tuple)):
#             return list(result)
#         return [result]
#     except Exception:
#         return []


# def first_element(val):
#     lst = safe_parse_list(val)
#     return lst[0] if lst else np.nan


# def list_len(val):
#     lst = safe_parse_list(val)
#     return len(lst)


# # ─────────────────────────────────────────────
# # Loaders
# # ─────────────────────────────────────────────

# def load_lab_files(pattern: str, condition: str) -> pd.DataFrame:
#     """
#     Load PsychoPy visual_search CSVs.
#     Works for both Single (mouse.time = scalar list) and
#     Multiple (click_times = list of click times within trial).
#     """
#     files = sorted(glob.glob(pattern))
#     if not files:
#         print(f"  [WARN] No files matched: {pattern}")
#         return pd.DataFrame()

#     records = []
#     for fp in files:
#         try:
#             raw = pd.read_csv(fp, dtype=str)
#         except Exception as e:
#             print(f"  [WARN] Could not read {fp}: {e}")
#             continue

#         # Drop instruction / blank rows – keep only trial rows
#         trial_rows = raw[
#             raw["target_col"].notna() &
#             raw["target_col"].isin(["white", "red"])
#         ].copy()

#         if trial_rows.empty:
#             continue

#         pid = str(trial_rows["participant"].iloc[0]).strip()

#         for _, row in trial_rows.iterrows():
#             target_col = row["target_col"]
#             is_target = (target_col == "red")

#             # ── RT ──────────────────────────────────────────────────────────
#             # Single-target files have  mouse.time  (list with 1 element)
#             # Multiple-target files have  click_times  (list of click times)
#             rt_ms = np.nan
#             if condition == "Single":
#                 rt_raw = row.get("mouse.time", np.nan)
#                 rt_ms = first_element(rt_raw) * 1000  # seconds → ms
#             else:  # Multiple
#                 ct_raw = row.get("click_times", np.nan)
#                 rt_ms = first_element(ct_raw) * 1000  # first click latency

#             # ── Accuracy / hits ─────────────────────────────────────────────
#             clicked = safe_parse_list(row.get("mouse.clicked_name", "[]"))
#             n_clicks = len(clicked)

#             if condition == "Single":
#                 # Target-present trial: hit if 'target' in clicked names
#                 hit = int(any("target" in str(c) for c in clicked)) if is_target else np.nan
#                 false_alarm = int(n_clicks > 0) if not is_target else np.nan
#             else:
#                 # Multiple-target: count target hits vs expected (5 per trial based on data)
#                 n_targets_expected = n_clicks  # all boxes were targets in sample
#                 n_hits = sum(1 for c in clicked if "target" in str(c).lower())
#                 hit = n_hits / n_targets_expected if n_targets_expected > 0 else np.nan
#                 false_alarm = 0  # not tracked per-trial in multi lab

#             trial_duration = np.nan
#             try:
#                 trial_duration = (float(row["trial.stopped"]) - float(row["trial.started"])) * 1000
#             except Exception:
#                 pass

#             records.append({
#                 "participant_id": pid,
#                 "condition": condition,
#                 "modality": "Lab",
#                 "trial_type": target_col,
#                 "is_target_trial": is_target,
#                 "RT_ms": rt_ms,
#                 "n_clicks": n_clicks,
#                 "hit": hit,
#                 "false_alarm": false_alarm,
#                 "trial_duration_ms": trial_duration,
#                 "source_file": os.path.basename(fp),
#             })

#     df = pd.DataFrame(records)
#     print(f"  Loaded Lab/{condition}: {df['participant_id'].nunique()} participants, "
#           f"{len(df)} trial rows from {len(files)} files")
#     return df


# def load_phone_files(pattern: str, condition: str) -> pd.DataFrame:
#     """Load attentional_spotter_results CSVs (game / phone modality)."""
#     files = sorted(glob.glob(pattern))
#     if not files:
#         print(f"  [WARN] No files matched: {pattern}")
#         return pd.DataFrame()

#     dfs = []
#     for fp in files:
#         try:
#             df = pd.read_csv(fp)
#         except Exception as e:
#             print(f"  [WARN] Could not read {fp}: {e}")
#             continue

#         # Derive participant ID from filename (e.g. 1_attentional… → "1")
#         pid = os.path.basename(fp).split("_")[0]
#         df["participant_id"] = pid
#         df["source_file"] = os.path.basename(fp)
#         dfs.append(df)

#     if not dfs:
#         return pd.DataFrame()

#     raw = pd.concat(dfs, ignore_index=True)

#     rename = {
#         "InitialResponseTime(ms)": "RT_ms",
#         "AvgInterTargetTime(ms)":   "avg_inter_target_ms",
#         "SuccessRate(%)":           "success_rate",
#         "HitRate(%)":               "hit_rate",
#         "FalseAlarms":              "false_alarms",
#         "FinalScore":               "final_score",
#     }
#     raw.rename(columns=rename, inplace=True)

#     raw["condition"] = condition
#     raw["modality"]  = "Phone"
#     raw["hit"]       = raw["hit_rate"] / 100.0
#     raw["false_alarm"] = raw["false_alarms"]

#     # Level as integer
#     raw["Level"] = pd.to_numeric(raw["Level"], errors="coerce")

#     print(f"  Loaded Phone/{condition}: {raw['participant_id'].nunique()} participants, "
#           f"{len(raw)} level-rows from {len(files)} files")
#     return raw


# # ─────────────────────────────────────────────
# # EDA Functions
# # ─────────────────────────────────────────────

# def descriptive_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
#     metrics = ["RT_ms", "hit"]
#     extra_lab   = ["trial_duration_ms", "n_clicks", "false_alarm"]
#     extra_phone = ["success_rate", "false_alarms", "avg_inter_target_ms", "final_score"]

#     cols = metrics + [c for c in extra_lab + extra_phone if c in df.columns]
#     cols = [c for c in cols if c in df.columns]

#     desc = df[cols].describe().T
#     desc["label"] = label
#     return desc


# def compare_two(a: pd.Series, b: pd.Series, name_a: str, name_b: str):
#     """Paired or independent t-test depending on equal length."""
#     a_clean = a.dropna()
#     b_clean = b.dropna()
#     if len(a_clean) == 0 or len(b_clean) == 0:
#         return None
#     if len(a_clean) == len(b_clean):
#         t, p = stats.ttest_rel(a_clean, b_clean)
#         kind = "paired"
#     else:
#         t, p = stats.ttest_ind(a_clean, b_clean, equal_var=False)
#         kind = "independent (Welch)"
#     d_num = a_clean.mean() - b_clean.mean()
#     pooled = np.sqrt((a_clean.std()**2 + b_clean.std()**2) / 2)
#     cohen_d = d_num / pooled if pooled > 0 else np.nan
#     return {"test": kind, "t": round(t, 3), "p": round(p, 4),
#             "cohen_d": round(cohen_d, 3),
#             f"mean_{name_a}": round(a_clean.mean(), 2),
#             f"mean_{name_b}": round(b_clean.mean(), 2)}


# # ─────────────────────────────────────────────
# # Plotting
# # ─────────────────────────────────────────────

# def fig_overview(all_df: pd.DataFrame, out_dir: str):
#     """RT and Hit distributions across all 4 cells."""
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#     fig.suptitle("Overview: RT and Hit Rate across Conditions & Modalities", fontsize=14, fontweight="bold")

#     cells = [
#         ("Single",   "Lab",   axes[0, 0]),
#         ("Single",   "Phone", axes[0, 1]),
#         ("Multiple", "Lab",   axes[1, 0]),
#         ("Multiple", "Phone", axes[1, 1]),
#     ]

#     for cond, mod, ax in cells:
#         sub = all_df[(all_df["condition"] == cond) & (all_df["modality"] == mod)]
#         if sub.empty:
#             ax.set_title(f"{cond} / {mod}\n(no data)")
#             continue
#         rt = sub["RT_ms"].dropna()
#         if len(rt) > 1:
#             ax.hist(rt, bins=20, color=MOD_PALETTE[mod], alpha=0.75, edgecolor="white")
#             ax.axvline(rt.mean(), color="black", linestyle="--", linewidth=1.5,
#                        label=f"M={rt.mean():.0f}ms")
#             ax.legend(fontsize=9)
#         ax.set_title(f"{cond} Target | {mod}", fontweight="bold")
#         ax.set_xlabel("RT (ms)")
#         ax.set_ylabel("Frequency")

#     plt.tight_layout()
#     path = os.path.join(out_dir, "01_rt_distributions.png")
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  Saved: {path}")


# def fig_boxplots(all_df: pd.DataFrame, out_dir: str):
#     """Side-by-side boxplots for RT and Hit."""
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#     fig.suptitle("RT and Hit Rate: Condition × Modality", fontsize=13, fontweight="bold")

#     all_df["Cell"] = all_df["condition"] + "\n" + all_df["modality"]

#     for ax, metric, label in [
#         (axes[0], "RT_ms",  "Reaction Time (ms)"),
#         (axes[1], "hit",    "Hit Rate (proportion)"),
#     ]:
#         sub = all_df[all_df[metric].notna()]
#         if sub.empty:
#             continue
#         sns.boxplot(data=sub, x="Cell", y=metric, hue="modality",
#                     palette=MOD_PALETTE, ax=ax, width=0.5, dodge=False)
#         sns.stripplot(data=sub, x="Cell", y=metric, hue="modality",
#                       palette=MOD_PALETTE, ax=ax, alpha=0.4, dodge=False, legend=False)
#         ax.set_xlabel("")
#         ax.set_ylabel(label)
#         ax.set_title(label)
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles[:2], labels[:2], title="Modality")

#     plt.tight_layout()
#     path = os.path.join(out_dir, "02_boxplots_rt_hit.png")
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  Saved: {path}")


# def fig_level_trends(phone_df: pd.DataFrame, out_dir: str):
#     """RT and Hit across levels, by condition (phone data only)."""
#     if phone_df.empty or "Level" not in phone_df.columns:
#         return

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     fig.suptitle("Phone Game: Performance Across Levels", fontsize=13, fontweight="bold")

#     for ax, metric, label in [
#         (axes[0], "RT_ms",   "Initial Response Time (ms)"),
#         (axes[1], "hit_rate" if "hit_rate" in phone_df.columns else "hit", "Hit Rate (%)"),
#     ]:
#         if metric not in phone_df.columns:
#             continue
#         for cond, color in PALETTE.items():
#             sub = phone_df[phone_df["condition"] == cond].dropna(subset=["Level", metric])
#             if sub.empty:
#                 continue
#             grp = sub.groupby("Level")[metric].agg(["mean", "sem"]).reset_index()
#             ax.errorbar(grp["Level"], grp["mean"], yerr=grp["sem"],
#                         label=f"{cond}", color=color, marker="o", linewidth=2, capsize=4)
#         ax.set_xlabel("Game Level")
#         ax.set_ylabel(label)
#         ax.set_title(label)
#         ax.legend(title="Condition")

#     plt.tight_layout()
#     path = os.path.join(out_dir, "03_level_trends_phone.png")
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  Saved: {path}")


# def fig_validity_scatter(lab_df: pd.DataFrame, phone_df: pd.DataFrame, out_dir: str):
#     """Scatter: Lab RT vs Phone RT per participant (concurrent validity preview)."""
#     # Aggregate to participant level
#     def agg_pid(df, mod):
#         return df.groupby(["participant_id", "condition"])["RT_ms"].mean().reset_index().rename(
#             columns={"RT_ms": f"RT_{mod}"})

#     lab_agg   = agg_pid(lab_df,   "Lab")
#     phone_agg = agg_pid(phone_df, "Phone")

#     merged = pd.merge(lab_agg, phone_agg, on=["participant_id", "condition"])
#     if merged.empty:
#         print("  [WARN] No overlapping participants for validity scatter.")
#         return

#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     fig.suptitle("Concurrent Validity Preview: Lab RT vs Phone RT (per participant)",
#                  fontsize=12, fontweight="bold")

#     for ax, cond in zip(axes, ["Single", "Multiple"]):
#         sub = merged[merged["condition"] == cond].dropna()
#         if sub.empty:
#             ax.set_title(f"{cond} (no data)")
#             continue
#         ax.scatter(sub["RT_Lab"], sub["RT_Phone"], color=PALETTE[cond], s=80, alpha=0.8)
#         if len(sub) >= 3:
#             r, p = stats.pearsonr(sub["RT_Lab"], sub["RT_Phone"])
#             ax.set_title(f"{cond} Target\nr = {r:.3f}, p = {p:.3f}", fontweight="bold")
#         else:
#             ax.set_title(f"{cond} Target (n={len(sub)})")
#         ax.set_xlabel("Lab RT (ms)")
#         ax.set_ylabel("Phone RT (ms)")
#         # regression line
#         if len(sub) >= 2:
#             m, b = np.polyfit(sub["RT_Lab"], sub["RT_Phone"], 1)
#             x_line = np.linspace(sub["RT_Lab"].min(), sub["RT_Lab"].max(), 100)
#             ax.plot(x_line, m * x_line + b, "--", color="grey", linewidth=1.5)

#     plt.tight_layout()
#     path = os.path.join(out_dir, "04_validity_scatter.png")
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  Saved: {path}")


# def fig_participant_profiles(all_df: pd.DataFrame, out_dir: str):
#     """Per-participant mean RT heatmap (Lab vs Phone × condition)."""
#     pivot = all_df.groupby(["participant_id", "condition", "modality"])["RT_ms"].mean().reset_index()
#     pivot["cell"] = pivot["condition"] + "_" + pivot["modality"]
#     heat = pivot.pivot(index="participant_id", columns="cell", values="RT_ms")

#     if heat.empty:
#         return

#     fig, ax = plt.subplots(figsize=(10, max(4, len(heat) * 0.35 + 2)))
#     sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
#                 linewidths=0.5, cbar_kws={"label": "Mean RT (ms)"})
#     ax.set_title("Participant-Level Mean RT (ms) across Cells", fontweight="bold")
#     ax.set_xlabel("Condition_Modality")
#     ax.set_ylabel("Participant ID")
#     plt.tight_layout()
#     path = os.path.join(out_dir, "05_participant_heatmap.png")
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  Saved: {path}")


# def fig_false_alarms(all_df: pd.DataFrame, out_dir: str):
#     """False alarm comparison."""
#     fa_col = "false_alarm" if "false_alarm" in all_df.columns else None
#     if fa_col is None:
#         return

#     sub = all_df.dropna(subset=[fa_col])
#     if sub.empty:
#         return

#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.barplot(data=sub, x="condition", y=fa_col, hue="modality",
#                 palette=MOD_PALETTE, ax=ax, capsize=0.1, errwidth=1.5)
#     ax.set_title("False Alarms by Condition and Modality", fontweight="bold")
#     ax.set_xlabel("Condition")
#     ax.set_ylabel("False Alarm Rate")
#     ax.legend(title="Modality")
#     plt.tight_layout()
#     path = os.path.join(out_dir, "06_false_alarms.png")
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  Saved: {path}")


# def fig_score_progression(phone_df: pd.DataFrame, out_dir: str):
#     """Final score and success rate across levels."""
#     if phone_df.empty:
#         return

#     cols_needed = [c for c in ["final_score", "success_rate"] if c in phone_df.columns]
#     if not cols_needed:
#         return

#     fig, axes = plt.subplots(1, len(cols_needed), figsize=(7 * len(cols_needed), 5))
#     if len(cols_needed) == 1:
#         axes = [axes]
#     fig.suptitle("Phone Game: Score & Success Rate Progression", fontsize=12, fontweight="bold")

#     for ax, col in zip(axes, cols_needed):
#         for cond, color in PALETTE.items():
#             sub = phone_df[phone_df["condition"] == cond].dropna(subset=["Level", col])
#             if sub.empty:
#                 continue
#             grp = sub.groupby("Level")[col].agg(["mean", "sem"]).reset_index()
#             ax.errorbar(grp["Level"], grp["mean"], yerr=grp["sem"],
#                         label=cond, color=color, marker="s", linewidth=2, capsize=4)
#         ax.set_xlabel("Level")
#         ax.set_ylabel(col.replace("_", " ").title())
#         ax.set_title(col.replace("_", " ").title())
#         ax.legend(title="Condition")

#     plt.tight_layout()
#     path = os.path.join(out_dir, "07_score_progression.png")
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  Saved: {path}")


# # ─────────────────────────────────────────────
# # Report
# # ─────────────────────────────────────────────

# def save_report(all_df, lab_df, phone_df, out_dir):
#     lines = ["# EDA Report – Selective Attention Study\n"]

#     lines.append("## 1. Sample Sizes\n")
#     for (cond, mod), grp in all_df.groupby(["condition", "modality"]):
#         n_p = grp["participant_id"].nunique()
#         n_t = len(grp)
#         lines.append(f"- **{cond} / {mod}**: {n_p} participants, {n_t} observations\n")

#     lines.append("\n## 2. Descriptive Statistics\n")
#     for (cond, mod), grp in all_df.groupby(["condition", "modality"]):
#         lines.append(f"\n### {cond} | {mod}\n")
#         desc = descriptive_stats(grp, f"{cond}/{mod}")
#         lines.append(desc[["count", "mean", "std", "min", "50%", "max"]].to_string())
#         lines.append("\n")

#     lines.append("\n## 3. Inferential Tests\n")

#     # RQ3: Lab vs Phone within each condition
#     lines.append("### RQ3 – Modality Effect (Lab vs Phone)\n")
#     for cond in ["Single", "Multiple"]:
#         lab_rt   = all_df[(all_df["condition"] == cond) & (all_df["modality"] == "Lab")]["RT_ms"]
#         phone_rt = all_df[(all_df["condition"] == cond) & (all_df["modality"] == "Phone")]["RT_ms"]
#         result   = compare_two(lab_rt, phone_rt, "Lab", "Phone")
#         if result:
#             lines.append(f"  **{cond}**: {result}\n")

#     # RQ2: Single vs Multiple within each modality
#     lines.append("\n### RQ2 – Target Load Effect (Single vs Multiple)\n")
#     for mod in ["Lab", "Phone"]:
#         single_rt = all_df[(all_df["condition"] == "Single")   & (all_df["modality"] == mod)]["RT_ms"]
#         multi_rt  = all_df[(all_df["condition"] == "Multiple") & (all_df["modality"] == mod)]["RT_ms"]
#         result    = compare_two(single_rt, multi_rt, "Single", "Multiple")
#         if result:
#             lines.append(f"  **{mod}**: {result}\n")

#     # RQ1: Concurrent validity (per participant, per condition)
#     lines.append("\n### RQ1 – Concurrent Validity (Lab–Phone correlation)\n")
#     lab_pid   = lab_df.groupby(["participant_id", "condition"])["RT_ms"].mean().reset_index().rename(columns={"RT_ms": "RT_Lab"})
#     phone_pid = phone_df.groupby(["participant_id", "condition"])["RT_ms"].mean().reset_index().rename(columns={"RT_ms": "RT_Phone"})
#     merged    = pd.merge(lab_pid, phone_pid, on=["participant_id", "condition"])
#     for cond in ["Single", "Multiple"]:
#         sub = merged[merged["condition"] == cond].dropna()
#         if len(sub) >= 3:
#             r, p = stats.pearsonr(sub["RT_Lab"], sub["RT_Phone"])
#             lines.append(f"  **{cond}**: Pearson r = {r:.3f}, p = {p:.4f} (n={len(sub)})\n")
#         else:
#             lines.append(f"  **{cond}**: insufficient overlapping participants (n={len(sub)})\n")

#     report_path = os.path.join(out_dir, "eda_report.md")
#     with open(report_path, "w") as f:
#         f.writelines(lines)
#     print(f"  Saved: {report_path}")

#     # Also save tidy CSV
#     tidy_path = os.path.join(out_dir, "tidy_data.csv")
#     all_df.to_csv(tidy_path, index=False)
#     print(f"  Saved: {tidy_path}")


# # ─────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(description="EDA – Selective Attention Study")
#     parser.add_argument("--multi_lab",    required=True, help='Glob pattern for Multiple/Lab CSVs')
#     parser.add_argument("--multi_phone",  required=True, help='Glob pattern for Multiple/Phone CSVs')
#     parser.add_argument("--single_lab",   required=True, help='Glob pattern for Single/Lab CSVs')
#     parser.add_argument("--single_phone", required=True, help='Glob pattern for Single/Phone CSVs')
#     parser.add_argument("--out_dir",      default="results/eda", help='Output directory')
#     args = parser.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     print("\n── Loading data ──────────────────────────────────")
#     multi_lab_df   = load_lab_files(args.multi_lab,   "Multiple")
#     multi_phone_df = load_phone_files(args.multi_phone, "Multiple")
#     single_lab_df  = load_lab_files(args.single_lab,  "Single")
#     single_phone_df = load_phone_files(args.single_phone, "Single")

#     lab_df   = pd.concat([multi_lab_df,   single_lab_df],   ignore_index=True)
#     phone_df = pd.concat([multi_phone_df, single_phone_df], ignore_index=True)
#     all_df   = pd.concat([lab_df, phone_df], ignore_index=True)

#     if all_df.empty:
#         print("ERROR: No data loaded. Check your glob patterns.")
#         return

#     print(f"\nTotal observations: {len(all_df)}")
#     print(all_df.groupby(["condition", "modality"])[["RT_ms", "hit"]].describe().to_string())

#     print("\n── Generating plots ──────────────────────────────")
#     fig_overview(all_df, args.out_dir)
#     fig_boxplots(all_df, args.out_dir)
#     fig_level_trends(phone_df, args.out_dir)
#     fig_validity_scatter(lab_df, phone_df, args.out_dir)
#     fig_participant_profiles(all_df, args.out_dir)
#     fig_false_alarms(all_df, args.out_dir)
#     fig_score_progression(phone_df, args.out_dir)

#     print("\n── Saving report ─────────────────────────────────")
#     save_report(all_df, lab_df, phone_df, args.out_dir)

#     print("\n✓ EDA complete. All outputs saved to:", args.out_dir)


# if __name__ == "__main__":
#     main()

"""
eda_selective_attention.py
==========================
EDA for the Selective Attention Game Study.
Just run it — no command-line arguments needed.

    python eda_selective_attention.py

──────────────────────────────────────────────────────
SET YOUR DATA ROOT BELOW
──────────────────────────────────────────────────────
Expected folder layout:
  DATA_ROOT/
    multiple/lab/    ← visual_search_*.csv   (participants 22–37)
    multiple/phone/  ← attentional_spotter_results.csv
    single/lab/      ← visual_search_*.csv   (participants 1–21)
    single/phone/    ← attentional_spotter_results.csv

Fallback: if the above subfolders are missing, files are
auto-detected from a flat DATA_ROOT folder by filename prefix.
"""

import ast, glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
#  ▶▶  CONFIGURE HERE  ◀◀
# ══════════════════════════════════════════════════════
# DATA_ROOT = "/mnt/user-data/uploads"
# OUT_DIR   = "/mnt/user-data/outputs/eda"

DATA_ROOT = "data"
OUT_DIR   = "results/eda"

# Structured-folder globs
MULTI_LAB_GLOB    = os.path.join(DATA_ROOT, "multiple", "lab",   "*.csv")
MULTI_PHONE_GLOB  = os.path.join(DATA_ROOT, "multiple", "phone", "*.csv")
SINGLE_LAB_GLOB   = os.path.join(DATA_ROOT, "single",   "lab",   "*.csv")
SINGLE_PHONE_GLOB = os.path.join(DATA_ROOT, "single",   "phone", "*.csv")

# Fallback flat-folder globs (files named <N>_visual_search_*.csv etc.)
FALLBACK_MULTI_LAB    = os.path.join(DATA_ROOT, "2*_visual_search_*.csv")
FALLBACK_MULTI_PHONE  = os.path.join(DATA_ROOT, "2*_attentional_spotter_results.csv")
FALLBACK_SINGLE_LAB   = os.path.join(DATA_ROOT, "1*_visual_search_*.csv")
FALLBACK_SINGLE_PHONE = os.path.join(DATA_ROOT, "1*_attentional_spotter_results.csv")
# ══════════════════════════════════════════════════════

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.15)
COND_PAL = {"Single": "#4C72B0", "Multiple": "#DD8452"}
MOD_PAL  = {"Lab": "#55A868",    "Phone": "#C44E52"}


# ─────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────

def find_files(structured, fallback, label):
    files = sorted(glob.glob(structured))
    if files:
        return files
    files = sorted(glob.glob(fallback))
    if files:
        print(f"  [auto-detected {len(files)} {label} files via flat folder]")
    return files


# ─────────────────────────────────────────────────────
# Parse helpers
# ─────────────────────────────────────────────────────

def safe_list(val):
    """Parse a stringified Python list; return [] on failure."""
    try:
        r = ast.literal_eval(str(val))
        return list(r) if isinstance(r, (list, tuple)) else [r]
    except Exception:
        return []

def first_val(val):
    lst = safe_list(val)
    return lst[0] if lst else np.nan


# ─────────────────────────────────────────────────────
# Data cleaning helpers
# ─────────────────────────────────────────────────────

def clean_phone_df(df: pd.DataFrame, pid: str) -> pd.DataFrame:
    """
    Apply all confirmed cleaning rules to a single phone participant file:

    1. PlayerID is unreliable (all show 'Player_2') — replace with filename-derived pid.
    2. Two-player contamination (e.g. 6_attentional_spotter_results.csv): if >1 unique
       PlayerID detected AND timestamps suggest two sessions, keep only the LAST player
       (second participant, per supervisor instruction).
    3. Repeated completed levels: keep only the FIRST Completed==True row per level.
       Rationale: if a level was failed then passed, the pass is the valid attempt.
       Alternating completed levels (e.g. 11,12,11,12) are an artefact — first pass wins.
    4. Levels where the participant failed twice and stopped: these have no Completed==True
       row — they are excluded from performance analysis but counted as failures.
    """
    df = df.copy()

    # Rule 1: replace PlayerID with filename-derived pid
    df["participant_id"] = pid

    # Rule 2: two-player contamination — keep rows from the LAST player block
    # Detect by looking for a timestamp reset (each player starts from scratch)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    # If timestamps go backwards at some point, that marks the boundary
    boundary_idx = None
    ts = df["Timestamp"].dropna().reset_index(drop=True)
    for i in range(1, len(ts)):
        if ts[i] < ts[i - 1]:
            boundary_idx = df[df["Timestamp"] == ts[i]].index[0]
            break
    if boundary_idx is not None:
        print(f"    [CLEAN] {pid}: two-player contamination detected — "
              f"keeping rows from index {boundary_idx} onwards (second participant).")
        df = df.loc[boundary_idx:].reset_index(drop=True)

    # Rule 3: keep only first Completed==True per level; drop failed-only levels
    completed = df[df["Completed"] == True].copy()
    cleaned   = completed.groupby("Level", sort=False).first().reset_index()

    # Record how many levels were failed-and-stopped (no completed row)
    all_levels      = df["Level"].unique()
    completed_levels = cleaned["Level"].unique()
    failed_stopped  = [l for l in all_levels if l not in completed_levels]
    if failed_stopped:
        print(f"    [CLEAN] {pid}: levels failed twice and excluded: {sorted(failed_stopped)}")

    cleaned["failed_levels"] = str(sorted(failed_stopped))
    return cleaned


# ─────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────

def load_lab_files(files: list, condition: str) -> pd.DataFrame:
    """
    Load PsychoPy lab CSVs for one condition (Single or Multiple).

    RT extraction:
      Single   → mouse.time[0]  (seconds relative to trial onset)
                 True RT = mouse.time[0] - target.started  (target appears after delay)
      Multiple → click_times[0]  (custom-logged, relevant clicks only)
                 mouse.time is the PsychoPy hardware log — use click_times for Multiple.

    Accuracy:
      Any element containing 'target' in mouse.clicked_name = a hit.
      Non-target names = false alarm.  Multiple clicks on same trial are fine
      (PsychoPy sometimes combines them) — using `any()` covers all cases.
    """
    records = []
    for fp in files:
        try:
            raw = pd.read_csv(fp, dtype=str)
        except Exception as e:
            print(f"  [WARN] Cannot read {fp}: {e}")
            continue

        trials = raw[raw["target_col"].isin(["white", "red"])].copy()
        if trials.empty:
            continue

        pid = str(trials["participant"].iloc[0]).strip()

        for _, row in trials.iterrows():
            is_target = (row["target_col"] == "red")

            # ── Reaction Time ────────────────────────────────────────────────
            if condition == "Single":
                # Use mouse.time (PsychoPy hardware-level click log)
                # mouse.time[0] is relative to trial.started (seconds since trial onset)
                # target.started is ABSOLUTE (seconds since experiment start)
                # True RT = mouse.time[0] - (target.started - trial.started)
                raw_time = row.get("mouse.time", np.nan)
                click_sec = first_val(raw_time)  # relative to trial.started

                target_onset_relative = np.nan
                if "target.started" in row.index:
                    try:
                        trial_start = float(row["trial.started"])
                        target_abs  = float(row["target.started"])
                        target_onset_relative = target_abs - trial_start
                    except (ValueError, TypeError):
                        pass

                if np.isfinite(click_sec) and np.isfinite(target_onset_relative):
                    rt_ms = (click_sec - target_onset_relative) * 1000
                else:
                    rt_ms = click_sec * 1000 if np.isfinite(click_sec) else np.nan

            else:  # Multiple
                # Use click_times (custom software var — relevant clicks only)
                raw_ct = row.get("click_times", np.nan)
                click_sec = first_val(raw_ct)
                rt_ms = click_sec * 1000 if np.isfinite(click_sec) else np.nan

            # ── Accuracy ────────────────────────────────────────────────────
            clicked = safe_list(row.get("mouse.clicked_name", "[]"))
            # Any element containing 'target' (handles both 'target' and 'target_0' etc.)
            hit_clicks = [c for c in clicked if "target" in str(c).lower()]
            non_hit_clicks = [c for c in clicked if "target" not in str(c).lower()]

            if condition == "Single":
                hit         = int(len(hit_clicks) > 0) if is_target else np.nan
                false_alarm = int(len(non_hit_clicks) > 0) if not is_target else np.nan
            else:
                # 5 targets per trial; hit = proportion found
                n_expected = 5
                hit         = len(hit_clicks) / n_expected if is_target else np.nan
                false_alarm = len(non_hit_clicks)  # count of non-target clicks

            # ── Inter-click times (Multiple only) ───────────────────────────
            click_list = safe_list(row.get("click_times", "[]") if condition == "Multiple"
                                   else row.get("mouse.time", "[]"))
            inter_click_ms = np.nan
            if len(click_list) > 1:
                diffs = [click_list[i] - click_list[i-1] for i in range(1, len(click_list))]
                inter_click_ms = np.mean(diffs) * 1000

            trial_dur_ms = np.nan
            try:
                trial_dur_ms = (float(row["trial.stopped"]) - float(row["trial.started"])) * 1000
            except Exception:
                pass

            records.append({
                "participant_id":    pid,
                "condition":         condition,
                "modality":          "Lab",
                "trial_type":        row["target_col"],
                "is_target_trial":   is_target,
                "RT_ms":             rt_ms,
                "hit":               hit,
                "false_alarm":       false_alarm,
                "n_clicks":          len(clicked),
                "inter_click_ms":    inter_click_ms,
                "trial_duration_ms": trial_dur_ms,
                "source_file":       os.path.basename(fp),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        print(f"  Loaded Lab/{condition}: {df['participant_id'].nunique()} participants, "
              f"{len(df)} trials from {len(files)} files")
    else:
        print(f"  [WARN] Lab/{condition}: no data loaded from {len(files)} files")
    return df


def load_phone_files(files: list, condition: str) -> pd.DataFrame:
    """
    Load attentional_spotter_results CSVs for one condition.

    Cleaning applied per file (see clean_phone_df):
      - PlayerID replaced with filename-derived ID
      - Two-player contamination removed (keep second player)
      - Repeated levels: keep first Completed==True per level
      - Failed-twice levels excluded from analysis

    Formulas (confirmed):
      HitRate(%)    = (Hits / Total_Targets) × 100
      SuccessRate(%) = (Total_Targets / (Total_Targets + FalseAlarms)) × 100
      Completed      = (Hits == Total_Targets)
      FalseAlarms    = count of incorrect taps
    """
    frames = []
    for fp in files:
        try:
            raw = pd.read_csv(fp)
        except Exception as e:
            print(f"  [WARN] Cannot read {fp}: {e}")
            continue

        pid = os.path.basename(fp).split("_")[0]
        cleaned = clean_phone_df(raw, pid)
        cleaned["source_file"] = os.path.basename(fp)
        frames.append(cleaned)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Standardise column names
    df.rename(columns={
        "InitialResponseTime(ms)": "RT_ms",
        "AvgInterTargetTime(ms)":  "avg_inter_target_ms",
        "SuccessRate(%)":          "success_rate",
        "HitRate(%)":              "hit_rate",
        "FalseAlarms":             "false_alarms",
        "FinalScore":              "final_score",
    }, inplace=True)

    df["condition"]   = condition
    df["modality"]    = "Phone"
    df["hit"]         = df["hit_rate"] / 100.0
    df["false_alarm"] = df["false_alarms"]
    df["Level"]       = pd.to_numeric(df["Level"], errors="coerce")

    print(f"  Loaded Phone/{condition}: {df['participant_id'].nunique()} participants, "
          f"{len(df)} level-rows (after cleaning) from {len(files)} files")
    return df


# ─────────────────────────────────────────────────────
# Descriptive statistics
# ─────────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in ["RT_ms", "hit", "false_alarm", "trial_duration_ms",
                            "inter_click_ms", "success_rate", "hit_rate",
                            "false_alarms", "avg_inter_target_ms", "final_score"]
               if c in df.columns]
    return df[metrics].describe().T.round(3)


def print_descriptives(all_df: pd.DataFrame):
    print("\n── Descriptive Statistics ───────────────────────────────────────────")
    for (cond, mod), grp in all_df.groupby(["condition", "modality"]):
        print(f"\n  [{cond} / {mod}]  n_participants={grp['participant_id'].nunique()}  "
              f"n_rows={len(grp)}")
        desc = descriptive_stats(grp)
        print(desc[["count", "mean", "std", "min", "50%", "max"]].to_string())


# ─────────────────────────────────────────────────────
# Inferential tests
# ─────────────────────────────────────────────────────

def ttest_report(a: pd.Series, b: pd.Series, label_a: str, label_b: str):
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return f"  insufficient data (n_a={len(a)}, n_b={len(b)})"
    if len(a) == len(b):
        t, p = stats.ttest_rel(a, b)
        kind = "paired t"
    else:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        kind = "Welch t"
    d = (a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2 + 1e-9)
    sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
    return (f"  {kind}({len(a)+len(b)-2}) = {t:.3f}, p = {p:.4f} {sig}  "
            f"d = {d:.3f}  |  M_{label_a}={a.mean():.1f}  M_{label_b}={b.mean():.1f}")


def run_tests(all_df: pd.DataFrame, lab_df: pd.DataFrame, phone_df: pd.DataFrame):
    print("\n── Inferential Tests (preliminary) ─────────────────────────────────")

    print("\n  RQ2 – Target Load: Single vs Multiple RT")
    for mod in ["Lab", "Phone"]:
        s = all_df[(all_df["condition"] == "Single")   & (all_df["modality"] == mod)]["RT_ms"]
        m = all_df[(all_df["condition"] == "Multiple") & (all_df["modality"] == mod)]["RT_ms"]
        print(f"    {mod}: {ttest_report(s, m, 'Single', 'Multi')}")

    print("\n  RQ3 – Modality: Lab vs Phone RT")
    for cond in ["Single", "Multiple"]:
        l = all_df[(all_df["condition"] == cond) & (all_df["modality"] == "Lab")]["RT_ms"]
        p = all_df[(all_df["condition"] == cond) & (all_df["modality"] == "Phone")]["RT_ms"]
        print(f"    {cond}: {ttest_report(l, p, 'Lab', 'Phone')}")

    print("\n  RQ1 – Concurrent Validity: Lab RT vs Phone RT (per-participant means)")
    lab_agg   = lab_df.groupby(["participant_id","condition"])["RT_ms"].mean().reset_index()
    phone_agg = phone_df.groupby(["participant_id","condition"])["RT_ms"].mean().reset_index()
    merged    = pd.merge(lab_agg, phone_agg, on=["participant_id","condition"],
                         suffixes=("_Lab","_Phone"))
    for cond in ["Single", "Multiple"]:
        sub = merged[merged["condition"] == cond].dropna()
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub["RT_ms_Lab"], sub["RT_ms_Phone"])
            sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
            print(f"    {cond}: Pearson r = {r:.3f}, p = {p:.4f} {sig}  (n={len(sub)})")
        else:
            print(f"    {cond}: n={len(sub)} — need ≥ 3 overlapping participants for correlation")


# ─────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_rt_distributions(all_df):
    """Fig 1 — RT histograms for all 4 cells."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("RT Distributions across Conditions & Modalities", fontsize=14, fontweight="bold")
    cells = [("Single","Lab",axes[0,0]), ("Single","Phone",axes[0,1]),
             ("Multiple","Lab",axes[1,0]), ("Multiple","Phone",axes[1,1])]
    for cond, mod, ax in cells:
        sub = all_df[(all_df["condition"]==cond) & (all_df["modality"]==mod)]["RT_ms"].dropna()
        color = MOD_PAL[mod]
        if len(sub) > 1:
            ax.hist(sub, bins=20, color=color, alpha=0.75, edgecolor="white")
            ax.axvline(sub.mean(), color="black", ls="--", lw=1.8,
                       label=f"M={sub.mean():.0f}ms\nSD={sub.std():.0f}ms\nn={len(sub)}")
            ax.legend(fontsize=9)
        elif len(sub) == 1:
            ax.axvline(sub.iloc[0], color=color, lw=2)
        ax.set_title(f"{cond} | {mod}", fontweight="bold")
        ax.set_xlabel("RT (ms)")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    savefig(fig, "01_rt_distributions.png")


def plot_boxplots(all_df):
    """Fig 2 — Boxplots for RT and Hit across all 4 cells."""
    all_df = all_df.copy()
    all_df["Cell"] = all_df["condition"] + " / " + all_df["modality"]
    order = ["Single / Lab", "Single / Phone", "Multiple / Lab", "Multiple / Phone"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("RT and Hit Rate: Condition × Modality", fontsize=13, fontweight="bold")

    for ax, metric, ylabel in [
        (axes[0], "RT_ms", "Reaction Time (ms)"),
        (axes[1], "hit",   "Hit Rate (proportion)"),
    ]:
        sub = all_df[all_df[metric].notna()]
        if sub.empty:
            continue
        palette = {c: MOD_PAL["Lab"] if "Lab" in c else MOD_PAL["Phone"] for c in order}
        sns.boxplot(data=sub, x="Cell", y=metric, order=order,
                    palette=palette, ax=ax, width=0.5, fliersize=3)
        sns.stripplot(data=sub, x="Cell", y=metric, order=order,
                      palette=palette, ax=ax, alpha=0.5, jitter=True, size=4)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    savefig(fig, "02_boxplots_rt_hit.png")


def plot_level_trends(phone_df):
    """Fig 3 — RT, HitRate, SuccessRate, FalseAlarms across levels (phone only)."""
    if phone_df.empty:
        return

    metrics = [
        ("RT_ms",          "Initial RT (ms)"),
        ("hit_rate",       "Hit Rate (%)"),
        ("success_rate",   "Success Rate (%)"),
        ("false_alarms",   "False Alarms (count)"),
    ]
    metrics = [(m, l) for m, l in metrics if m in phone_df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phone Game: Performance Across Levels", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics):
        for cond, color in COND_PAL.items():
            sub = phone_df[phone_df["condition"]==cond].dropna(subset=["Level", metric])
            if sub.empty:
                continue
            grp = sub.groupby("Level")[metric].agg(["mean","sem"]).reset_index()
            ax.errorbar(grp["Level"], grp["mean"], yerr=grp["sem"],
                        label=cond, color=color, marker="o", lw=2, capsize=4)
        ax.set_xlabel("Game Level")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(title="Condition")

    # Hide unused axes if fewer than 4 metrics
    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    plt.tight_layout()
    savefig(fig, "03_level_trends_phone.png")


def plot_validity_scatter(lab_df, phone_df):
    """Fig 4 — Lab RT vs Phone RT per participant (concurrent validity)."""
    lab_agg   = lab_df.groupby(["participant_id","condition"])["RT_ms"].mean().reset_index()
    phone_agg = phone_df.groupby(["participant_id","condition"])["RT_ms"].mean().reset_index()
    merged    = pd.merge(lab_agg, phone_agg, on=["participant_id","condition"],
                         suffixes=("_Lab","_Phone"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Concurrent Validity: Lab RT vs Phone RT (per participant means)",
                 fontsize=12, fontweight="bold")

    for ax, cond in zip(axes, ["Single","Multiple"]):
        sub = merged[merged["condition"]==cond].dropna()
        color = COND_PAL[cond]
        if sub.empty:
            ax.set_title(f"{cond} — no data")
            continue
        ax.scatter(sub["RT_ms_Lab"], sub["RT_ms_Phone"], color=color, s=80, alpha=0.85,
                   edgecolors="white", linewidths=0.5)
        for _, r in sub.iterrows():
            ax.annotate(r["participant_id"], (r["RT_ms_Lab"], r["RT_ms_Phone"]),
                        fontsize=7, alpha=0.7, ha="left")
        if len(sub) >= 2:
            m_coef, b = np.polyfit(sub["RT_ms_Lab"], sub["RT_ms_Phone"], 1)
            xs = np.linspace(sub["RT_ms_Lab"].min(), sub["RT_ms_Lab"].max(), 100)
            ax.plot(xs, m_coef*xs + b, "--", color="grey", lw=1.5)
        title = f"{cond} Target\n"
        if len(sub) >= 3:
            r_val, p_val = stats.pearsonr(sub["RT_ms_Lab"], sub["RT_ms_Phone"])
            title += f"r = {r_val:.3f}, p = {p_val:.3f}  (n={len(sub)})"
        else:
            title += f"n={len(sub)} (need ≥3 for correlation)"
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Lab Mean RT (ms)")
        ax.set_ylabel("Phone Mean RT (ms)")

    plt.tight_layout()
    savefig(fig, "04_validity_scatter.png")


def plot_participant_heatmap(all_df):
    """Fig 5 — Heatmap of mean RT per participant per cell."""
    pivot = (all_df
             .groupby(["participant_id","condition","modality"])["RT_ms"]
             .mean()
             .reset_index())
    pivot["cell"] = pivot["condition"] + "_" + pivot["modality"]
    heat = pivot.pivot(index="participant_id", columns="cell", values="RT_ms")

    if heat.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(heat)*0.4 + 2)))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Mean RT (ms)"})
    ax.set_title("Per-Participant Mean RT (ms) across Cells", fontweight="bold")
    ax.set_xlabel("Condition × Modality")
    ax.set_ylabel("Participant ID")
    plt.tight_layout()
    savefig(fig, "05_participant_heatmap.png")


def plot_accuracy_comparison(all_df):
    """Fig 6 — Hit rate and false alarms side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Accuracy Metrics: Condition × Modality", fontsize=13, fontweight="bold")

    all_df = all_df.copy()
    all_df["Cell"] = all_df["condition"] + " / " + all_df["modality"]
    order = ["Single / Lab", "Single / Phone", "Multiple / Lab", "Multiple / Phone"]
    palette = {c: MOD_PAL["Lab"] if "Lab" in c else MOD_PAL["Phone"] for c in order}

    for ax, metric, ylabel in [
        (axes[0], "hit",         "Hit Rate (proportion)"),
        (axes[1], "false_alarm", "False Alarms"),
    ]:
        sub = all_df[all_df[metric].notna()]
        if sub.empty:
            ax.set_title(f"{ylabel}\n(no data)")
            continue
        sns.barplot(data=sub, x="Cell", y=metric, order=order,
                    palette=palette, ax=ax, capsize=0.1, errwidth=1.5,
                    estimator=np.mean, errorbar="se")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    savefig(fig, "06_accuracy_comparison.png")


def plot_score_progression(phone_df):
    """Fig 7 — Score and success rate progression by level (phone only)."""
    if phone_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phone Game: Score & Success Rate by Level", fontsize=13, fontweight="bold")

    for ax, metric, ylabel in [
        (axes[0], "final_score",  "Final Score"),
        (axes[1], "success_rate", "Success Rate (%)"),
    ]:
        if metric not in phone_df.columns:
            continue
        for cond, color in COND_PAL.items():
            sub = phone_df[phone_df["condition"]==cond].dropna(subset=["Level",metric])
            if sub.empty:
                continue
            grp = sub.groupby("Level")[metric].agg(["mean","sem"]).reset_index()
            ax.errorbar(grp["Level"], grp["mean"], yerr=grp["sem"],
                        label=cond, color=color, marker="s", lw=2, capsize=4)
        ax.set_xlabel("Level")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(title="Condition")

    plt.tight_layout()
    savefig(fig, "07_score_progression.png")


def plot_inter_click_times(all_df):
    """Fig 8 — Inter-click / inter-target times (Multiple condition only)."""
    multi = all_df[all_df["condition"] == "Multiple"].copy()
    if multi.empty:
        return

    # Lab: inter_click_ms from click_times diffs
    # Phone: avg_inter_target_ms column
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Inter-Target Time (Multiple Condition Only)", fontsize=13, fontweight="bold")

    for ax, mod, col in [
        (axes[0], "Lab",   "inter_click_ms"),
        (axes[1], "Phone", "avg_inter_target_ms"),
    ]:
        sub = multi[multi["modality"]==mod]
        if sub.empty or col not in sub.columns:
            ax.set_title(f"{mod} (no data)")
            continue
        data = sub[col].dropna()
        ax.hist(data, bins=15, color=MOD_PAL[mod], alpha=0.75, edgecolor="white")
        ax.axvline(data.mean(), color="black", ls="--", lw=1.8,
                   label=f"M={data.mean():.0f}ms")
        ax.set_xlabel("Inter-Target Time (ms)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Multiple | {mod}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    savefig(fig, "08_inter_target_times.png")


# ─────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────

def save_tidy_csv(all_df):
    path = os.path.join(OUT_DIR, "tidy_data.csv")
    all_df.to_csv(path, index=False)
    print(f"  Saved tidy data: {path}")


def save_report(all_df, lab_df, phone_df):
    lines = [
        "# EDA Report – Selective Attention Study\n\n",
        "## Cleaning Rules Applied\n",
        "- **Phone PlayerID**: Replaced with filename-derived participant ID (PlayerID column is unreliable — all show 'Player_2').\n",
        "- **Two-player contamination** (e.g. `6_attentional_spotter_results.csv`): Detected timestamp reset; kept only rows from second player.\n",
        "- **Repeated completed levels**: Kept first `Completed==True` row per level. Rationale: alternating level pattern (11,12,11,12) is a game artefact — only the first pass is the valid performance measure.\n",
        "- **Failed-twice levels**: Excluded from performance analysis (no `Completed==True` row exists).\n",
        "- **RT in Single/Lab**: Subtracted `target.started` from `mouse.time[0]` to get true RT (target appears after a delay within the trial).\n",
        "- **RT in Multiple/Lab**: Used `click_times[0]` (software-logged relevant clicks) not `mouse.time` (hardware log).\n\n",
        "## Confirmed Formulas (Phone)\n",
        "```\n",
        "HitRate(%)     = (Hits / Total_Targets) × 100\n",
        "SuccessRate(%) = (Total_Targets / (Total_Targets + FalseAlarms)) × 100\n",
        "Completed      = (Hits == Total_Targets)\n",
        "FalseAlarms    = count of incorrect taps per level\n",
        "```\n\n",
        "## 1. Sample Sizes\n",
    ]

    for (cond, mod), grp in all_df.groupby(["condition", "modality"]):
        n_p = grp["participant_id"].nunique()
        n_r = len(grp)
        lines.append(f"- **{cond} / {mod}**: {n_p} participants, {n_r} observations\n")

    lines.append("\n## 2. Descriptive Statistics\n")
    for (cond, mod), grp in all_df.groupby(["condition", "modality"]):
        lines.append(f"\n### {cond} | {mod}\n```\n")
        lines.append(descriptive_stats(grp)[["count","mean","std","min","50%","max"]].to_string())
        lines.append("\n```\n")

    lines.append("\n## 3. Preliminary Inferential Tests\n")
    lines.append("*(These are observation-level tests for EDA. Final ANOVA should use participant-aggregated data.)*\n\n")

    lines.append("### RQ2 – Target Load (Single vs Multiple)\n")
    for mod in ["Lab","Phone"]:
        s = all_df[(all_df["condition"]=="Single")   & (all_df["modality"]==mod)]["RT_ms"]
        m = all_df[(all_df["condition"]=="Multiple") & (all_df["modality"]==mod)]["RT_ms"]
        lines.append(f"- **{mod}**: {ttest_report(s, m, 'Single', 'Multiple')}\n")

    lines.append("\n### RQ3 – Modality (Lab vs Phone)\n")
    for cond in ["Single","Multiple"]:
        l = all_df[(all_df["condition"]==cond) & (all_df["modality"]=="Lab")]["RT_ms"]
        p = all_df[(all_df["condition"]==cond) & (all_df["modality"]=="Phone")]["RT_ms"]
        lines.append(f"- **{cond}**: {ttest_report(l, p, 'Lab', 'Phone')}\n")

    lines.append("\n### RQ1 – Concurrent Validity (per-participant Lab vs Phone correlation)\n")
    lab_agg   = lab_df.groupby(["participant_id","condition"])["RT_ms"].mean().reset_index()
    phone_agg = phone_df.groupby(["participant_id","condition"])["RT_ms"].mean().reset_index()
    merged    = pd.merge(lab_agg, phone_agg, on=["participant_id","condition"],
                         suffixes=("_Lab","_Phone"))
    for cond in ["Single","Multiple"]:
        sub = merged[merged["condition"]==cond].dropna()
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub["RT_ms_Lab"], sub["RT_ms_Phone"])
            sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."
            lines.append(f"- **{cond}**: r = {r:.3f}, p = {p:.4f} {sig}  (n={len(sub)})\n")
        else:
            lines.append(f"- **{cond}**: n={len(sub)} — insufficient for correlation\n")

    path = os.path.join(OUT_DIR, "eda_report.md")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  Saved report: {path}")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n── Discovering files ────────────────────────────────────────────────")
    multi_lab_files    = find_files(MULTI_LAB_GLOB,    FALLBACK_MULTI_LAB,    "Multiple/Lab")
    multi_phone_files  = find_files(MULTI_PHONE_GLOB,  FALLBACK_MULTI_PHONE,  "Multiple/Phone")
    single_lab_files   = find_files(SINGLE_LAB_GLOB,   FALLBACK_SINGLE_LAB,   "Single/Lab")
    single_phone_files = find_files(SINGLE_PHONE_GLOB, FALLBACK_SINGLE_PHONE, "Single/Phone")

    print(f"  Multiple / Lab   : {len(multi_lab_files)} files")
    print(f"  Multiple / Phone : {len(multi_phone_files)} files")
    print(f"  Single  / Lab    : {len(single_lab_files)} files")
    print(f"  Single  / Phone  : {len(single_phone_files)} files")

    print("\n── Loading & cleaning data ──────────────────────────────────────────")
    multi_lab_df    = load_lab_files(multi_lab_files,    "Multiple")
    multi_phone_df  = load_phone_files(multi_phone_files, "Multiple")
    single_lab_df   = load_lab_files(single_lab_files,   "Single")
    single_phone_df = load_phone_files(single_phone_files,"Single")

    lab_df   = pd.concat([multi_lab_df,   single_lab_df],   ignore_index=True)
    phone_df = pd.concat([multi_phone_df, single_phone_df], ignore_index=True)
    all_df   = pd.concat([lab_df, phone_df], ignore_index=True)

    if all_df.empty:
        print("\nERROR: No data loaded. Check DATA_ROOT path.")
        return

    print(f"\nTotal rows loaded (after cleaning): {len(all_df)}")

    print_descriptives(all_df)
    run_tests(all_df, lab_df, phone_df)

    print("\n── Generating plots ─────────────────────────────────────────────────")
    plot_rt_distributions(all_df)
    plot_boxplots(all_df)
    plot_level_trends(phone_df)
    plot_validity_scatter(lab_df, phone_df)
    plot_participant_heatmap(all_df)
    plot_accuracy_comparison(all_df)
    plot_score_progression(phone_df)
    plot_inter_click_times(all_df)

    print("\n── Saving outputs ───────────────────────────────────────────────────")
    save_tidy_csv(all_df)
    save_report(all_df, lab_df, phone_df)

    print(f"\n✓ EDA complete. All outputs saved to: {OUT_DIR}\n")


if __name__ == "__main__":
    main()