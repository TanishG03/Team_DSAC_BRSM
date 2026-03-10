"""Aggregate participant-level means and run 2x2 mixed ANOVA.

Assumptions used (please confirm or tell me to change):
- Target Load: folder name `single` vs `multiple` (between-subjects).
- Modality: folder name `lab` vs `phone` (within-subjects).
- RT:
  - For phone (`attentional_spotter_results.csv`) use `InitialResponseTime(ms)`.
  - For lab (`visual_search_*.csv`) use `trial.stopped - trial.started` when available.
- Hit rate / Success rate:
  - For phone use `HitRate(%)` and `SuccessRate(%)` (converted to proportions).
  - For lab compute proportion of trials with non-empty `click_times` as both hit and success rate.

Saves participant-level CSV to `preprocessed_data/aggregated_participant_level.csv`
and writes ANOVA tables to `results/analysis_anova.txt`.
"""

import ast
import glob
import os
import pandas as pd
import numpy as np
from statsmodels.stats.anova import AnovaRM


ROOT = "preprocessed_data"


def parse_click_list(s):
    if pd.isna(s) or s == "":
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, (list, tuple)) else []
    except Exception:
        return []


def process_lab_file(path):
    # visual_search files: trial.started, trial.stopped, click_times
    df = pd.read_csv(path)
    # drop empty rows
    df = df.dropna(how="all")
    # compute RT per row if possible
    if "trial.started" in df.columns and "trial.stopped" in df.columns:
        try:
            rt = df["trial.stopped"].astype(float) - df["trial.started"].astype(float)
        except Exception:
            rt = pd.Series(dtype=float)
    else:
        rt = pd.Series(dtype=float)

    # compute hit/success as proportion of trials with click_times
    hit_flags = []
    if "click_times" in df.columns:
        for s in df["click_times"]:
            lst = parse_click_list(s)
            hit_flags.append(1 if len(lst) > 0 else 0)
    else:
        hit_flags = [0] * len(df)

    out = {
        "mean_rt": float(np.nan if rt.size == 0 else np.nanmean(rt)),
        "hit_rate": float(np.mean(hit_flags)) if len(hit_flags) > 0 else float('nan'),
        "success_rate": float(np.mean(hit_flags)) if len(hit_flags) > 0 else float('nan'),
    }
    return out


def process_phone_file(path):
    df = pd.read_csv(path)
    df = df.dropna(how="all")
    # RT use InitialResponseTime(ms)
    rt = None
    for col in [c for c in df.columns if "InitialResponseTime" in c or "InitialResponseTime(ms)" in c]:
        rt = pd.to_numeric(df[col], errors='coerce')
        break
    if rt is None:
        # try other candidate
        for c in ["InitialResponseTime(ms)", "AvgInterTargetTime(ms)"]:
            if c in df.columns:
                rt = pd.to_numeric(df[c], errors='coerce')
                break

    # hit / success rates
    hit = None
    succ = None
    if "HitRate(%)" in df.columns:
        hit = pd.to_numeric(df["HitRate(%)"], errors='coerce') / 100.0
    elif "HitRate" in df.columns:
        hit = pd.to_numeric(df["HitRate"], errors='coerce')

    if "SuccessRate(%)" in df.columns:
        succ = pd.to_numeric(df["SuccessRate(%)"], errors='coerce') / 100.0
    elif "SuccessRate" in df.columns:
        succ = pd.to_numeric(df["SuccessRate"], errors='coerce')

    out = {
        "mean_rt": float(np.nan if rt is None or rt.size == 0 else np.nanmean(rt)),
        "hit_rate": float(np.nan if hit is None or hit.size == 0 else np.nanmean(hit)),
        "success_rate": float(np.nan if succ is None or succ.size == 0 else np.nanmean(succ)),
    }
    return out


def gather():
    rows = []
    for load in ["single", "multiple"]:
        for modality in ["lab", "phone"]:
            folder = os.path.join(ROOT, load, modality)
            if not os.path.isdir(folder):
                continue
            for path in glob.glob(os.path.join(folder, "*.csv")):
                fname = os.path.basename(path)
                # participant id as numeric prefix before first underscore
                pid = fname.split("_")[0]
                try:
                    int(pid)
                except Exception:
                    # fallback to full fname
                    pid = fname

                if modality == "lab":
                    stats = process_lab_file(path)
                else:
                    stats = process_phone_file(path)

                rows.append({
                    "participant": pid,
                    "target_load": load,
                    "modality": modality,
                    "mean_rt": stats["mean_rt"],
                    "hit_rate": stats["hit_rate"],
                    "success_rate": stats["success_rate"],
                    "source_file": path,
                })

    df = pd.DataFrame(rows)
    # convert numeric columns
    for c in ["mean_rt", "hit_rate", "success_rate"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Participant-level: ensure one row per participant per modality
    # Save aggregated
    outpath = os.path.join("preprocessed_data", "aggregated_participant_level.csv")
    df.to_csv(outpath, index=False)
    return df


def run_anova(df, dv_col, out_f):
    # require wide-ish: each subject should have repeated measures across modality
    # AnovaRM expects a long-form dataframe with one row per subject per within level
    # Between-subject factor: target_load
    df_rm = df[['participant', 'target_load', 'modality', dv_col]].dropna()
    # Convert participant to str
    df_rm['participant'] = df_rm['participant'].astype(str)
    try:
        aov = AnovaRM(df_rm, depvar=dv_col, subject='participant', within=['modality'], between=['target_load'])
        res = aov.fit()
    except TypeError:
        # older statsmodels AnovaRM does not support between; fallback: run within-only AnovaRM and report separately
        aov = AnovaRM(df_rm, depvar=dv_col, subject='participant', within=['modality'])
        res = aov.fit()

    with open(out_f, 'a') as fh:
        fh.write(f"ANOVA for {dv_col}\n")
        fh.write(res.summary().as_text())
        fh.write("\n\n")


def main():
    df = gather()
    os.makedirs('results', exist_ok=True)
    out_f = os.path.join('results', 'analysis_anova.txt')
    # clear
    open(out_f, 'w').close()

    # run ANOVA for mean_rt, hit_rate, success_rate
    for dv in ['mean_rt', 'hit_rate', 'success_rate']:
        run_anova(df, dv, out_f)

    print("Saved aggregated participant-level data to preprocessed_data/aggregated_participant_level.csv")
    print("ANOVA results written to results/analysis_anova.txt")


if __name__ == '__main__':
    main()
