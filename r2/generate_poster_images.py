import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs_r2")
POSTER_OUT = Path("outputs_poster")
POSTER_OUT.mkdir(exist_ok=True)

with open(OUT / "_r2_cache.pkl", "rb") as f:
    cache = pickle.load(f)

PAL = {"Single": "#2D6A9F", "Multiple": "#C05621"}
wide = cache["wide"]
wide_comp = cache["wide_comp"]
anova_logRT = cache["anova_logRT"]

# 1. Simple Interaction Plot (1 panel)
fig, ax = plt.subplots(figsize=(6, 5))
for grp, color, ls, mk in [("Single", PAL["Single"], "--", "o"),
                           ("Multiple", PAL["Multiple"], "-", "s")]:
    sub = wide[wide["group"]==grp].dropna(subset=["logRT_Lab", "logRT_Game"])
    lab_m = sub["logRT_Lab"].mean()
    game_m = sub["logRT_Game"].mean()
    lab_se = sub["logRT_Lab"].std(ddof=1) / np.sqrt(len(sub))
    game_se = sub["logRT_Game"].std(ddof=1) / np.sqrt(len(sub))
    ax.errorbar(["Lab", "Game"], [lab_m, game_m], yerr=[lab_se, game_se],
                color=color, ls=ls, marker=mk, linewidth=2.5, markersize=10, capsize=6,
                label=f"{grp}")

ax.set_title("Modality × Target Load Interaction", fontsize=14, fontweight="bold")
ax.set_ylabel("Log Reaction Time", fontsize=12)
ax.set_xlabel("Modality", fontsize=12)
ax.tick_params(labelsize=11)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(POSTER_OUT / "poster_fig1_interaction.png", dpi=300)
plt.close()

# 2. Simple Validity Scatter (2 panels: Single vs Multiple for Levels 1-10)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for i, grp in enumerate(["Single", "Multiple"]):
    ax = axes[i]
    sub = wide_comp[wide_comp["group"]==grp]
    x = sub["logRT_Lab"]
    y = sub["logRT_Game"]
    ax.scatter(x, y, color=PAL[grp], alpha=0.7, s=50)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, color="black", lw=2)
    ax.set_title(f"{grp} (Levels 1-10)", fontsize=13)
    ax.set_xlabel("Lab Log RT", fontsize=11)
    if i == 0:
        ax.set_ylabel("Game Log RT", fontsize=11)
plt.tight_layout()
plt.savefig(POSTER_OUT / "poster_fig2_validity.png", dpi=300)
plt.close()

print("Poster images generated in 'outputs_poster' directory!")
