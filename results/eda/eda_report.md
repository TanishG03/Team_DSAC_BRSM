# EDA Report – Selective Attention Study

## Cleaning Rules Applied
- **Phone PlayerID**: Replaced with filename-derived participant ID (PlayerID column is unreliable — all show 'Player_2').
- **Two-player contamination** (e.g. `6_attentional_spotter_results.csv`): Detected timestamp reset; kept only rows from second player.
- **Repeated completed levels**: Kept first `Completed==True` row per level. Rationale: alternating level pattern (11,12,11,12) is a game artefact — only the first pass is the valid performance measure.
- **Failed-twice levels**: Excluded from performance analysis (no `Completed==True` row exists).
- **RT in Single/Lab**: Subtracted `target.started` from `mouse.time[0]` to get true RT (target appears after a delay within the trial).
- **RT in Multiple/Lab**: Used `click_times[0]` (software-logged relevant clicks) not `mouse.time` (hardware log).

## Confirmed Formulas (Phone)
```
HitRate(%)     = (Hits / Total_Targets) × 100
SuccessRate(%) = (Total_Targets / (Total_Targets + FalseAlarms)) × 100
Completed      = (Hits == Total_Targets)
FalseAlarms    = count of incorrect taps per level
```

## 1. Sample Sizes
- **Multiple / Lab**: 16 participants, 240 observations
- **Multiple / Phone**: 16 participants, 165 observations
- **Single / Lab**: 21 participants, 315 observations
- **Single / Phone**: 21 participants, 315 observations

## 2. Descriptive Statistics

### Multiple | Lab
```
                     count      mean       std       min       50%        max
RT_ms                240.0  1669.900   745.264   889.780  1480.804   9368.018
hit                   80.0     0.950     0.170     0.200     1.000      1.400
false_alarm          240.0     0.000     0.000     0.000     0.000      0.000
trial_duration_ms    240.0  6658.945  2244.395  3319.226  6138.414  22932.329
inter_click_ms       240.0  1247.425   488.182   474.891  1137.803   5098.223
success_rate           0.0       NaN       NaN       NaN       NaN        NaN
hit_rate               0.0       NaN       NaN       NaN       NaN        NaN
false_alarms           0.0       NaN       NaN       NaN       NaN        NaN
avg_inter_target_ms    0.0       NaN       NaN       NaN       NaN        NaN
final_score            0.0       NaN       NaN       NaN       NaN        NaN
```

### Multiple | Phone
```
                     count      mean       std     min     50%      max
RT_ms                165.0  1754.406  1435.106  927.00  1388.0  12246.0
hit                  165.0     1.000     0.000    1.00     1.0      1.0
false_alarm          165.0     0.485     1.413    0.00     0.0     13.0
trial_duration_ms      0.0       NaN       NaN     NaN     NaN      NaN
inter_click_ms         0.0       NaN       NaN     NaN     NaN      NaN
success_rate         165.0    96.843     7.326   51.85   100.0    100.0
hit_rate             165.0   100.000     0.000  100.00   100.0    100.0
false_alarms         165.0     0.485     1.413    0.00     0.0     13.0
avg_inter_target_ms  165.0   997.339   487.235  237.00   958.0   2056.0
final_score          165.0   433.212   294.617   50.00   405.0   1310.0
```

### Single | Lab
```
                     count      mean      std     min       50%       max
RT_ms                315.0  1495.434  542.630  718.47  1367.848  5817.438
hit                  105.0     1.000    0.000    1.00     1.000     1.000
false_alarm          210.0     0.000    0.000    0.00     0.000     0.000
trial_duration_ms    315.0  1612.710  554.931  897.16  1475.560  6012.414
inter_click_ms         0.0       NaN      NaN     NaN       NaN       NaN
success_rate           0.0       NaN      NaN     NaN       NaN       NaN
hit_rate               0.0       NaN      NaN     NaN       NaN       NaN
false_alarms           0.0       NaN      NaN     NaN       NaN       NaN
avg_inter_target_ms    0.0       NaN      NaN     NaN       NaN       NaN
final_score            0.0       NaN      NaN     NaN       NaN       NaN
```

### Single | Phone
```
                     count      mean       std     min     50%      max
RT_ms                315.0  3143.883  2511.162  864.00  2120.0  14711.0
hit                  315.0     1.000     0.000    1.00     1.0      1.0
false_alarm          315.0     0.035     0.216    0.00     0.0      2.0
trial_duration_ms      0.0       NaN       NaN     NaN     NaN      NaN
inter_click_ms         0.0       NaN       NaN     NaN     NaN      NaN
success_rate         315.0    98.466     9.038   33.33   100.0    100.0
hit_rate             315.0   100.000     0.000  100.00   100.0    100.0
false_alarms         315.0     0.035     0.216    0.00     0.0      2.0
avg_inter_target_ms  315.0     0.000     0.000    0.00     0.0      0.0
final_score          315.0    66.651    42.488    5.00    60.0    150.0
```

## 3. Preliminary Inferential Tests
*(These are observation-level tests for EDA. Final ANOVA should use participant-aggregated data.)*

### RQ2 – Target Load (Single vs Multiple)
- **Lab**:   Welch t(553) = -3.061, p = 0.0023 **  d = -0.268  |  M_Single=1495.4  M_Multiple=1669.9
- **Phone**:   Welch t(478) = 7.707, p = 0.0000 ***  d = 0.679  |  M_Single=3143.9  M_Multiple=1754.4

### RQ3 – Modality (Lab vs Phone)
- **Single**:   paired t(628) = -11.189, p = 0.0000 ***  d = -0.907  |  M_Lab=1495.4  M_Phone=3143.9
- **Multiple**:   Welch t(403) = -0.695, p = 0.4879 n.s.  d = -0.074  |  M_Lab=1669.9  M_Phone=1754.4

### RQ1 – Concurrent Validity (per-participant Lab vs Phone correlation)
- **Single**: r = 0.144, p = 0.5341 n.s.  (n=21)
- **Multiple**: r = 0.066, p = 0.8154 n.s.  (n=15)
