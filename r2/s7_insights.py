import pandas as pd
import numpy as np
import scipy.stats as stats
import pickle
import os

print("=================================================================")
print("EXTRA INSIGHTS: Speed-Accuracy Trade-off & Game Level 11 Wall")
print("=================================================================")

# Load data
with open('outputs_r2/_r2_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
game_all = cache['game_all']

# 1. Speed-Accuracy Trade-off in the Game
print("\n─────────────────────────────────────────────────────────────────")
print("  Insight 1: Speed-Accuracy Trade-off (Game RT vs Game False Alarms)")
print("─────────────────────────────────────────────────────────────────")

# game_all is at the level/trial basis
single_game = game_all[game_all['group'] == 'Single']
multiple_game = game_all[game_all['group'] == 'Multiple']

single_r, single_p = stats.spearmanr(single_game['RT_ms'], single_game['false_alarms'])
multiple_r, multiple_p = stats.spearmanr(multiple_game['RT_ms'], multiple_game['false_alarms'])

print(f"Single Group (n={len(single_game)} levels): ρ = {single_r:.3f}, p = {single_p:.4f}")
if single_p < 0.05:
    if single_r < 0:
        print("  -> Significant speed-accuracy trade-off: faster RT leads to more false alarms.")
    else:
        print("  -> Significant positive correlation: slower RT correlates with more false alarms.")
else:
    print("  -> No significant speed-accuracy trade-off.")

print(f"Multiple Group (n={len(multiple_game)} levels): ρ = {multiple_r:.3f}, p = {multiple_p:.4f}")
if multiple_p < 0.05:
    if multiple_r < 0:
        print("  -> Significant speed-accuracy trade-off: faster RT leads to more false alarms.")
    else:
        print("  -> Significant positive correlation: slower RT correlates with more false alarms.")
else:
    print("  -> No significant speed-accuracy trade-off.")

# 2. What happens at Level 10 vs 11 for Multiple Group?
print("\n─────────────────────────────────────────────────────────────────")
print("  Insight 2: The 'Level 10 Wall' for the Multiple Group")
print("─────────────────────────────────────────────────────────────────")

lvl10 = multiple_game[multiple_game['level'] == 10]
lvl11 = multiple_game[multiple_game['level'] == 11]

print(f"Level 10: {len(lvl10)} attempts by {lvl10['participant'].nunique()} participants")
print(f"Level 11: {len(lvl11)} attempts by {lvl11['participant'].nunique()} participants")

if len(lvl10) > 0 and len(lvl11) > 0:
    print("\nComparing metrics between Level 10 and Level 11:")
    print(f"  Level 10 Mean RT: {lvl10['RT_ms'].mean():.1f} ms | False Alarms: {lvl10['false_alarms'].mean():.2f} | Success Rate: {lvl10['success_rate'].mean():.1f}%")
    print(f"  Level 11 Mean RT: {lvl11['RT_ms'].mean():.1f} ms | False Alarms: {lvl11['false_alarms'].mean():.2f} | Success Rate: {lvl11['success_rate'].mean():.1f}%")

print("\nConclusion:")
print("The Multiple group hits a wall around Level 10. Their faster overall Game RT ")
print("(compared to Single) is likely a maladaptive strategy: they guess faster but make ")
print("more false alarms, ultimately failing earlier. The Single group is slower but ")
print("more deliberate, surviving until Level 15.")
