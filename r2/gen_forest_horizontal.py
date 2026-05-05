"""
Regenerate fig_s3_02_forest_plot in a wide horizontal (landscape) layout
suitable for fitting across a poster column.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind, mannwhitneyu

OUT = Path("outputs_r2")

with open(OUT / "_r2_cache.pkl", "rb") as f:
    cache = pickle.load(f)

ptpt_lab       = cache["ptpt_lab"]
ptpt_game_all  = cache["ptpt_game_all"]

PAL = {"Single": "#2D6A9F", "Multiple": "#C05621"}

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / sp if sp else np.nan

def stars(p):
    if p is None or np.isnan(p): return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

# ── Compute effect sizes ─────────────────────────────────────────────────────
dvs = [
    (ptpt_lab,      "logRT_mean",  "Lab Log RT"),
    (ptpt_game_all, "logRT_mean",  "Game Log RT"),
    (ptpt_game_all, "false_alarms","False Alarms"),
    (ptpt_game_all, "max_level",   "Max Level"),
    (ptpt_game_all, "success_rate","Success Rate (%)"),
]

results = []
for df, col, label in dvs:
    if col not in df.columns:
        continue
    s = df[df["group"] == "Single"][col].dropna().values
    m = df[df["group"] == "Multiple"][col].dropna().values
    if len(s) < 2 or len(m) < 2:
        continue
    _, p = ttest_ind(s, m, equal_var=False)
    d = cohens_d(s, m)
    results.append({"label": label, "d": d, "p": p, "sig": stars(p),
                    "s_m": s.mean(), "m_m": m.mean()})

# ── Plot: HORIZONTAL bar chart (wide, short) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 3.2))  # Wide & short for poster

x_pos  = range(len(results))
d_vals = [r["d"] for r in results]
labels = [r["label"] for r in results]
sigs   = [r["sig"] for r in results]
colors = [PAL["Single"] if d >= 0 else PAL["Multiple"] for d in d_vals]

bars = ax.bar(list(x_pos), d_vals, color=colors, alpha=0.82,
              edgecolor="white", width=0.55)

# Annotate each bar with d value + significance
for i, (d, sig) in enumerate(zip(d_vals, sigs)):
    y_offset = 0.08 if d >= 0 else -0.08
    va = "bottom" if d >= 0 else "top"
    ax.text(i, d + y_offset, f"d={d:.2f}\n{sig}",
            ha="center", va=va, fontsize=8.5, fontweight="bold", color="#1a1a2e")

# Reference lines
ax.axhline(0, color="black", lw=1.2)
for v, ls, lbl in [(0.2, ":", "small"), (0.5, "--", "med"), (0.8, "-.", "large")]:
    ax.axhline(v,  color="#718096", ls=ls, lw=1, alpha=0.55)
    ax.axhline(-v, color="#718096", ls=ls, lw=1, alpha=0.55)
    ax.text(len(results) - 0.5, v + 0.03, lbl, fontsize=7, color="#718096", ha="right")

ax.set_xticks(list(x_pos))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Cohen's d\n(+ve = Single > Multiple)", fontsize=9)
ax.set_title("H2 — Target Load Effect: Single vs. Multiple (Cohen's d)",
             fontsize=11, fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_els = [Patch(color=PAL["Single"], label="Single > Multiple"),
              Patch(color=PAL["Multiple"], label="Multiple > Single")]
ax.legend(handles=legend_els, fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(OUT / "fig_s3_02_forest_plot_horizontal.png", dpi=300, bbox_inches="tight")
plt.close()
print("✔  Saved: outputs_r2/fig_s3_02_forest_plot_horizontal.png")
