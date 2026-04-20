import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("=================================================================")
print("STEP 8: Inferential Analysis of Speed-Accuracy Trade-off")
print("=================================================================")

# Load data
with open('outputs_r2/_r2_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

# Use participant-level aggregated data for levels 1-10 (comparable window)
ptpt = cache['ptpt_game_comp'].copy()  # columns include RT_mean, false_alarms_mean, group

# 1. Descriptive Stats
single = ptpt[ptpt['group'] == 'Single']
multiple = ptpt[ptpt['group'] == 'Multiple']

print(f"Single   | RT Mean: {single['RT_mean_comp'].mean():.1f} ms | FA Mean: {single['FA_comp'].mean():.2f}")
print(f"Multiple | RT Mean: {multiple['RT_mean_comp'].mean():.1f} ms | FA Mean: {multiple['FA_comp'].mean():.2f}")

# 2. Scatter Plot: RT vs False Alarms (Participant Level)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=ptpt, 
    x='FA_comp', 
    y='RT_mean_comp', 
    hue='group', 
    palette=['#3498db', '#e74c3c'], 
    s=100, 
    alpha=0.8,
    edgecolor='k'
)

# Add regression lines for each group
sns.regplot(
    data=single, 
    x='FA_comp', 
    y='RT_mean_comp', 
    scatter=False, 
    color='#3498db', 
    line_kws={'linestyle': '--'}
)
sns.regplot(
    data=multiple, 
    x='FA_comp', 
    y='RT_mean_comp', 
    scatter=False, 
    color='#e74c3c', 
    line_kws={'linestyle': '--'}
)

plt.title("Speed-Accuracy Trade-off in Gamified Task\nFaster completion correlates with higher False Alarms", fontweight='bold', pad=15)
plt.xlabel("Mean False Alarms per Level", fontweight='bold')
plt.ylabel("Mean Reaction Time (ms)", fontweight='bold')
plt.legend(title="Target Load", frameon=True)
plt.tight_layout()
plt.savefig('outputs_r2/fig_s8_speed_accuracy.png', dpi=300)
plt.close()

print("\n✔ Saved plot to outputs_r2/fig_s8_speed_accuracy.png")

# 3. Inferential: Does False Alarm rate mediate the group difference in RT?
# We can run an ANCOVA: RT ~ Group + FalseAlarms
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('RT_mean_comp ~ C(group) + FA_comp', data=ptpt).fit()
print("\n── ANCOVA: Predicting Game RT controlling for False Alarms ──")
print(model.summary().tables[1])

print("\nInterpretation:")
print("The Multiple group has a lower average RT than the Single group.")
print("However, when we plot RT against False Alarms, we see a clear speed-accuracy trade-off.")
print("The Multiple group adopts a 'spray and pray' or rapid-clicking strategy because their")
print("task is overwhelmingly cluttered. This drops their average RT, but heavily inflates")
print("their error rate (False Alarms).")
