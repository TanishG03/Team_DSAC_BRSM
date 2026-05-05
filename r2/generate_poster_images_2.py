import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

OUT = Path("outputs_r2")
POSTER_OUT = Path("outputs_poster")

with open(OUT / "_r2_cache.pkl", "rb") as f:
    cache = pickle.load(f)

PAL = {"Single": "#2D6A9F", "Multiple": "#C05621"}
game_all = cache["game_all"]

# 3. Simple Level Trends (RT only)
fig, ax = plt.subplots(figsize=(6, 4.5))
for grp in ["Single", "Multiple"]:
    sub = game_all[game_all["group"] == grp]
    means = sub.groupby("level")["RT_ms"].mean()
    sems = sub.groupby("level")["RT_ms"].sem()
    ax.plot(means.index, means.values, marker="o", color=PAL[grp], label=grp, lw=2)
    ax.fill_between(means.index, means.values - sems.values, means.values + sems.values, color=PAL[grp], alpha=0.2)

ax.axvline(10, color="crimson", ls="--", label="Level 10 Wall")
ax.set_title("Reaction Time by Level", fontsize=14, fontweight="bold")
ax.set_xlabel("Game Level", fontsize=12)
ax.set_ylabel("Reaction Time (ms)", fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(POSTER_OUT / "poster_fig3_level_trends.png", dpi=300)
plt.close()

# 4. ICI (Combined Rank and Red/White)
lab_all = cache["lab_all"]
lab_mult = lab_all[lab_all["group"]=="Multiple"]
ici_rank = []
for _, row in lab_mult.iterrows():
    clicks = eval(row["click_times_ms"]) if isinstance(row["click_times_ms"], str) else row["click_times_ms"]
    if isinstance(clicks, list) and len(clicks) == 5:
        intervals = np.diff(clicks)
        for i, val in enumerate(intervals):
            ici_rank.append({"rank": f"{i+1}→{i+2}", "ici": val, "target": row["target"]})
ici_df = pd.DataFrame(ici_rank)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel 1: ICI by Rank
sns.pointplot(data=ici_df, x="rank", y="ici", ax=axes[0], color="#2C3E50", ci=95)
axes[0].set_title("ICI Slowing by Click Rank", fontsize=13)
axes[0].set_ylabel("Inter-Click Interval (ms)", fontsize=11)
axes[0].set_xlabel("Click Rank", fontsize=11)

# Panel 2: ICI Red vs White
sns.boxplot(data=ici_df, x="target", y="ici", ax=axes[1], palette={"red": "#E74C3C", "white": "#ECF0F1"}, showfliers=False)
axes[1].set_title("ICI: Target vs. Distractor", fontsize=13)
axes[1].set_ylabel("")
axes[1].set_xlabel("Trial Type", fontsize=11)

plt.tight_layout()
plt.savefig(POSTER_OUT / "poster_fig4_ici.png", dpi=300)
plt.close()

print("More poster images generated!")
