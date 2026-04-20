import pickle
import pandas as pd
import scipy.stats as stats

with open('outputs_r2/_r2_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

# game_lvl1 is the data for level 1
game_lvl1 = cache['ptpt_game_lvl1']

single_lvl1 = game_lvl1[game_lvl1['group'] == 'Single']['RT_mean']
multiple_lvl1 = game_lvl1[game_lvl1['group'] == 'Multiple']['RT_mean']

print(f"Single Level 1 RT: M = {single_lvl1.mean():.1f} ms, SD = {single_lvl1.std():.1f}")
print(f"Multiple Level 1 RT: M = {multiple_lvl1.mean():.1f} ms, SD = {multiple_lvl1.std():.1f}")

t_stat, p_val = stats.ttest_ind(single_lvl1, multiple_lvl1, equal_var=False)
print(f"Welch's t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
