"""
REPORT 2 — STEP 4: Mixed Effects Models (LME on log-RT + Gamma GLMM)
Requires outputs_r2/_r2_cache.pkl from Steps 1–3
"""

import pandas as pd
import numpy as np
import pickle, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, pearsonr

warnings.filterwarnings("ignore")
OUT = Path("outputs_r2"); OUT.mkdir(exist_ok=True)

# ── Install dependencies if needed ───────────────────────────────────────────
import subprocess, sys
for pkg in ["statsmodels"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"--quiet"])

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

with open(OUT / "_r2_cache.pkl","rb") as f:
    cache = pickle.load(f)

lab_all         = cache["lab_all"]
game_all        = cache["game_all"]
game_comparable = cache["game_comparable"]
ptpt_lab        = cache["ptpt_lab"]
ptpt_game_all   = cache["ptpt_game_all"]
max_comp_level  = cache["max_comparable_level"]

PAL = {"Single":"#2D6A9F","Multiple":"#C05621"}

def stars(p):
    if p is None or (isinstance(p,float) and np.isnan(p)): return ""
    return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

def save(name):
    plt.savefig(OUT/name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  {name}")

def section(title):
    print("\n"+"─"*65)
    print(f"  {title}")
    print("─"*65)

print("="*65)
print("STEP 4: MIXED EFFECTS MODELS")
print("="*65)


# =============================================================================
# DATA PREPARATION
# Build trial-level datasets:
#   (A) Lab trials only — predictor = Modality (dummy), group as between var
#   (B) Combined lab + game trial-level — long format
#   (C) Game-only by level — comparable window
# =============================================================================

section("DATA PREPARATION — Trial-Level Long Format")

# ── Lab trial-level ──────────────────────────────────────────────────────────
lab_tl = lab_all[["participant","group","trial_n","RT_ms","log_RT",
                   "target_col","is_target"]].copy()
lab_tl["modality"]    = "Lab"
lab_tl["modality_01"] = 0   # 0=Lab, 1=Game
lab_tl["group_01"]    = (lab_tl["group"]=="Multiple").astype(int)  # 0=Single, 1=Multiple
lab_tl = lab_tl.dropna(subset=["RT_ms","log_RT"])

# ── Game trial-level (comparable window) ─────────────────────────────────────
game_tl = game_comparable[["participant","group","level","RT_ms","log_RT",
                            "success_rate","hit_rate","false_alarms"]].copy()
game_tl["modality"]    = "Game"
game_tl["modality_01"] = 1
game_tl["group_01"]    = (game_tl["group"]=="Multiple").astype(int)
game_tl["trial_n"]     = game_tl["level"]  # level as trial proxy
game_tl = game_tl.dropna(subset=["RT_ms","log_RT"])

# ── Combined long format ──────────────────────────────────────────────────────
combined = pd.concat([
    lab_tl[["participant","group","group_01","modality","modality_01",
            "trial_n","RT_ms","log_RT"]],
    game_tl[["participant","group","group_01","modality","modality_01",
             "trial_n","RT_ms","log_RT"]],
], ignore_index=True)

combined["participant"] = combined["participant"].astype(str)
combined["group_fac"]   = combined["group"].astype("category")
combined["modality_fac"]= combined["modality"].astype("category")

# Separate by group for group-specific models
comb_s = combined[combined["group"]=="Single"].copy()
comb_m = combined[combined["group"]=="Multiple"].copy()

print(f"\n  Lab trial-level rows    : {len(lab_tl)}")
print(f"  Game trial-level rows   : {len(game_tl)}  (levels 1–{max_comp_level})")
print(f"  Combined (all) rows     : {len(combined)}")
print(f"  Combined Single rows    : {len(comb_s)}")
print(f"  Combined Multiple rows  : {len(comb_m)}")
print(f"\n  Modality coding: 0=Lab, 1=Game")
print(f"  Group coding:    0=Single, 1=Multiple")
print(f"\n  Participants in combined:")
for grp in ["Single","Multiple"]:
    n = combined[combined["group"]==grp]["participant"].nunique()
    print(f"    {grp}: {n} participants")


# =============================================================================
# MODEL 1: LINEAR MIXED EFFECTS — LOG RT
# Equation (as per professor's suggestion):
#   log(RT_ij) = β0 + β1*Modality_ij + u0i + ε_ij
#   where i=participant, j=trial/level
#   Random intercept per participant
#   Run separately for Single and Multiple groups
# =============================================================================

section("MODEL 1: LME — Log RT ~ Modality (random intercept per participant)")

print("\n  Model equation:")
print("  log(RT_ij) = β0 + β1*Modality_ij + u0i + ε_ij")
print("  Modality: 0=Lab, 1=Game")
print("  u0i ~ N(0, σ²_u0)  [random intercept per participant]")
print("  ε_ij ~ N(0, σ²_ε)  [residual error]")

lme_results = {}
residuals_for_diagnostics = {}

for grp, df in [("Single", comb_s), ("Multiple", comb_m)]:
    print(f"\n  ── {grp} group  "
          f"(n_participants={df['participant'].nunique()}, "
          f"n_trials={len(df)}) ──")

    df = df.copy()
    df["participant"] = df["participant"].astype("category")

    try:
        # Random intercept model
        model = smf.mixedlm(
            "log_RT ~ modality_01",
            data=df,
            groups=df["participant"],
        )
        result = model.fit(reml=True, method="lbfgs")

        print(result.summary().tables[1].to_string())

        # Extract key values
        b0       = result.fe_params["Intercept"]
        b1       = result.fe_params["modality_01"]
        b0_se    = result.bse["Intercept"]
        b1_se    = result.bse["modality_01"]
        b0_p     = result.pvalues["Intercept"]
        b1_p     = result.pvalues["modality_01"]
        b0_ci    = result.conf_int().loc["Intercept"].values
        b1_ci    = result.conf_int().loc["modality_01"].values
        rand_var = float(result.cov_re.iloc[0,0])
        resid_var= result.scale
        icc      = rand_var / (rand_var + resid_var)

        print(f"\n  Fixed effects:")
        print(f"    β0 (Intercept/Lab logRT): {b0:.4f} ± {b0_se:.4f}  "
              f"p={b0_p:.4f}{stars(b0_p)}  "
              f"95%CI[{b0_ci[0]:.4f},{b0_ci[1]:.4f}]")
        print(f"    β1 (Modality: Game−Lab):  {b1:.4f} ± {b1_se:.4f}  "
              f"p={b1_p:.4f}{stars(b1_p)}  "
              f"95%CI[{b1_ci[0]:.4f},{b1_ci[1]:.4f}]")
        print(f"\n  Back-transformed (multiplicative effect on RT):")
        print(f"    exp(β0) = {np.exp(b0):.1f} ms  [baseline Lab RT]")
        print(f"    exp(β1) = {np.exp(b1):.4f}  "
              f"[Game RT is {(np.exp(b1)-1)*100:+.1f}% of Lab RT]")
        print(f"\n  Random effects:")
        print(f"    Participant variance (σ²_u0) : {rand_var:.6f}")
        print(f"    Residual variance    (σ²_ε)  : {resid_var:.6f}")
        print(f"    ICC (intraclass corr)        : {icc:.4f}  "
              f"[{icc*100:.1f}% of variance due to participant]")
        print(f"\n  Model fit:")
        print(f"    Log-likelihood : {result.llf:.3f}")
        print(f"    AIC            : {result.aic:.3f}")
        print(f"    BIC            : {result.bic:.3f}")

        # Residuals for diagnostic plots
        resids = result.resid.values
        fitted = result.fittedvalues.values
        residuals_for_diagnostics[grp] = {
            "resids": resids, "fitted": fitted, "log_RT": df["log_RT"].values
        }

        lme_results[grp] = {
            "group":grp, "n_ptpt":df["participant"].nunique(),
            "n_obs":len(df),
            "b0":b0,"b0_se":b0_se,"b0_p":b0_p,
            "b1":b1,"b1_se":b1_se,"b1_p":b1_p,
            "b1_ci_lo":b1_ci[0],"b1_ci_hi":b1_ci[1],
            "exp_b0":np.exp(b0),"exp_b1":np.exp(b1),
            "pct_change":(np.exp(b1)-1)*100,
            "rand_var":rand_var,"resid_var":resid_var,"icc":icc,
            "aic":result.aic,"bic":result.bic,"llf":result.llf,
        }

    except Exception as e:
        print(f"  ⚠  Model failed: {e}")
        lme_results[grp] = {"group":grp,"error":str(e)}


# =============================================================================
# MODEL 2: LME WITH RANDOM SLOPE (Modality | Participant)
# Tests whether the Lab→Game shift varies across participants
# =============================================================================

section("MODEL 2: LME — Random Slope for Modality per Participant")

print("\n  Model equation:")
print("  log(RT_ij) = β0 + β1*Modality_ij + u0i + u1i*Modality_ij + ε_ij")
print("  Adds random slope for Modality — does the Game inflation")
print("  differ meaningfully across participants?")

lme_rs_results = {}

for grp, df in [("Single", comb_s), ("Multiple", comb_m)]:
    print(f"\n  ── {grp} group ──")
    df = df.copy()
    df["participant"] = df["participant"].astype("category")

    try:
        # Random slope model
        model_rs = smf.mixedlm(
            "log_RT ~ modality_01",
            data=df,
            groups=df["participant"],
            re_formula="~modality_01",
        )
        result_rs = model_rs.fit(reml=True, method="lbfgs")

        b1_rs    = result_rs.fe_params["modality_01"]
        b1_se_rs = result_rs.bse["modality_01"]
        b1_p_rs  = result_rs.pvalues["modality_01"]

        print(f"    β1 (Modality) = {b1_rs:.4f} ± {b1_se_rs:.4f}  "
              f"p={b1_p_rs:.4f}{stars(b1_p_rs)}")
        print(f"    AIC={result_rs.aic:.3f}  BIC={result_rs.bic:.3f}  "
              f"LLF={result_rs.llf:.3f}")

        # LRT: random slope vs random intercept only
        if grp in lme_results and "aic" in lme_results[grp]:
            aic_ri = lme_results[grp]["aic"]
            aic_rs = result_rs.aic
            print(f"    AIC comparison: RI={aic_ri:.3f}  RS={aic_rs:.3f}  "
                  f"ΔAIC={aic_ri-aic_rs:.3f}  "
                  f"({'RS preferred' if aic_rs<aic_ri else 'RI preferred (simpler)'})")

        lme_rs_results[grp] = {
            "group":grp,"b1":b1_rs,"b1_se":b1_se_rs,"b1_p":b1_p_rs,
            "aic":result_rs.aic,"bic":result_rs.bic,
        }

    except Exception as e:
        print(f"  ⚠  Random slope model failed: {e}")
        lme_rs_results[grp] = {"group":grp,"error":str(e)}


# =============================================================================
# MODEL 3: GAMMA GLMM (RT ~ Modality, Gamma family, log link)
# RT is strictly positive + right-skewed → Gamma is theoretically correct
# Use statsmodels GEE as approximation (true GLMM requires R/lme4)
# =============================================================================

section("MODEL 3: Gamma GLM — Raw RT ~ Modality (log link, per group)")

print("\n  Note: True Gamma GLMM with random effects requires R/lme4.")
print("  Running Gamma GLM per group (ignoring participant clustering)")
print("  + GEE with exchangeable correlation as approximation.")
print("  Compare AIC/BIC with LME models to assess fit.")

gamma_results = {}

for grp, df in [("Single", comb_s), ("Multiple", comb_m)]:
    print(f"\n  ── {grp} group ──")
    df = df.dropna(subset=["RT_ms","modality_01"]).copy()
    df["participant"] = df["participant"].astype("category")

    # Model 3a: Gamma GLM (no random effects — baseline)
    try:
        glm_gamma = smf.glm(
            "RT_ms ~ modality_01",
            data=df,
            family=sm.families.Gamma(link=sm.families.links.Log()),
        ).fit()

        b0_g  = glm_gamma.params["Intercept"]
        b1_g  = glm_gamma.params["modality_01"]
        b0_p  = glm_gamma.pvalues["Intercept"]
        b1_p  = glm_gamma.pvalues["modality_01"]
        b0_ci = glm_gamma.conf_int().loc["Intercept"].values
        b1_ci = glm_gamma.conf_int().loc["modality_01"].values

        print(f"\n  Gamma GLM (log link):")
        print(f"    β0 = {b0_g:.4f} ± {glm_gamma.bse['Intercept']:.4f}  "
              f"p={b0_p:.4f}{stars(b0_p)}")
        print(f"    β1 = {b1_g:.4f} ± {glm_gamma.bse['modality_01']:.4f}  "
              f"p={b1_p:.4f}{stars(b1_p)}")
        print(f"    exp(β0) = {np.exp(b0_g):.1f} ms  [Lab RT estimate]")
        print(f"    exp(β1) = {np.exp(b1_g):.4f}  "
              f"[Game/Lab ratio: {(np.exp(b1_g)-1)*100:+.1f}%]")
        print(f"    AIC={glm_gamma.aic:.3f}  Deviance={glm_gamma.deviance:.3f}")
        print(f"    95%CI β1: [{b1_ci[0]:.4f},{b1_ci[1]:.4f}]")

        gamma_results[grp] = {
            "group":grp, "b0":b0_g, "b1":b1_g,
            "b1_p":b1_p, "exp_b1":np.exp(b1_g),
            "pct_change":(np.exp(b1_g)-1)*100,
            "aic":glm_gamma.aic,
        }

    except Exception as e:
        print(f"  ⚠  Gamma GLM failed: {e}")

    # Model 3b: GEE Gamma (accounts for participant clustering)
    try:
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.cov_struct import Exchangeable

        gee_gamma = GEE(
            endog=df["RT_ms"],
            exog=sm.add_constant(df["modality_01"]),
            groups=df["participant"],
            family=sm.families.Gamma(link=sm.families.links.Log()),
            cov_struct=Exchangeable(),
        ).fit()

        b0_gee = gee_gamma.params[0]
        b1_gee = gee_gamma.params[1]
        b1_p_gee = gee_gamma.pvalues[1]
        b1_ci_gee = gee_gamma.conf_int()[1]

        print(f"\n  GEE Gamma (exchangeable correlation — accounts for clustering):")
        print(f"    β0 = {b0_gee:.4f}  β1 = {b1_gee:.4f}  "
              f"p={b1_p_gee:.4f}{stars(b1_p_gee)}")
        print(f"    exp(β1) = {np.exp(b1_gee):.4f}  "
              f"[Game/Lab: {(np.exp(b1_gee)-1)*100:+.1f}%]")
        print(f"    95%CI β1: [{b1_ci_gee[0]:.4f},{b1_ci_gee[1]:.4f}]")

        if grp in gamma_results:
            gamma_results[grp].update({
                "gee_b1":b1_gee,"gee_b1_p":b1_p_gee,
                "gee_exp_b1":np.exp(b1_gee),
            })

    except Exception as e:
        print(f"  ⚠  GEE failed: {e}")


# =============================================================================
# MODEL 4: LME WITH LEVEL AS COVARIATE (H4 — does RT change with level?)
# Game-only, comparable window, random intercept per participant
# =============================================================================

section("MODEL 4: LME — Log RT ~ Level (Game only, random intercept)")

print("\n  Model equation:")
print("  log(RT_ij) = β0 + β1*Level_ij + u0i + ε_ij")
print("  Tests H4: Does RT increase with game difficulty level?")
print("  Run separately per group on levels 1–10")

lme_level_results = {}

for grp in ["Single","Multiple"]:
    df = game_comparable[game_comparable["group"]==grp].copy()
    df = df.dropna(subset=["log_RT","level"])
    df["participant"] = df["participant"].astype("category")
    # Centre level for interpretability
    df["level_c"] = df["level"] - df["level"].mean()

    print(f"\n  ── {grp} group  "
          f"(n_ptpt={df['participant'].nunique()}, n_obs={len(df)}) ──")

    try:
        model = smf.mixedlm(
            "log_RT ~ level_c",
            data=df,
            groups=df["participant"],
        )
        result = model.fit(reml=True, method="lbfgs")

        b0     = result.fe_params["Intercept"]
        b1     = result.fe_params["level_c"]
        b1_se  = result.bse["level_c"]
        b1_p   = result.pvalues["level_c"]
        b1_ci  = result.conf_int().loc["level_c"].values

        print(f"    β0 (Intercept, at mean level): {b0:.4f}")
        print(f"    β1 (Level_c):  {b1:.6f} ± {b1_se:.6f}  "
              f"p={b1_p:.4f}{stars(b1_p)}  "
              f"95%CI[{b1_ci[0]:.6f},{b1_ci[1]:.6f}]")
        print(f"    Multiplicative per level: exp(β1)={np.exp(b1):.6f}  "
              f"[{(np.exp(b1)-1)*100:+.4f}% per level]")
        print(f"    AIC={result.aic:.3f}  BIC={result.bic:.3f}")

        lme_level_results[grp] = {
            "group":grp,"b0":b0,"b1":b1,"b1_se":b1_se,"b1_p":b1_p,
            "b1_ci_lo":b1_ci[0],"b1_ci_hi":b1_ci[1],
            "exp_b1":np.exp(b1),"pct_per_level":(np.exp(b1)-1)*100,
            "aic":result.aic,"bic":result.bic,
        }

    except Exception as e:
        print(f"  ⚠  Model failed: {e}")


# =============================================================================
# MODEL COMPARISON TABLE
# =============================================================================

section("MODEL COMPARISON SUMMARY")

print("\n  ── LME (log RT ~ Modality) per group ──────────────────────────")
print(f"  {'Group':<12} {'β1':>8} {'SE':>7} {'p':>10} {'sig':>5}"
      f" {'exp(β1)':>9} {'%change':>9} {'ICC':>7} {'AIC':>9}")
print("  " + "─"*78)
for grp in ["Single","Multiple"]:
    r = lme_results.get(grp, {})
    if "error" in r:
        print(f"  {grp:<12} ERROR: {r['error']}")
        continue
    print(f"  {grp:<12} {r['b1']:>8.4f} {r['b1_se']:>7.4f} "
          f"{r['b1_p']:>10.4f} {stars(r['b1_p']):>5} "
          f"{r['exp_b1']:>9.4f} {r['pct_change']:>+9.1f}% "
          f"{r['icc']:>7.4f} {r['aic']:>9.1f}")

print("\n  ── Gamma GLM (RT ~ Modality) per group ─────────────────────────")
print(f"  {'Group':<12} {'β1':>8} {'p':>10} {'sig':>5}"
      f" {'exp(β1)':>9} {'%change':>9} {'AIC':>9}")
print("  " + "─"*60)
for grp in ["Single","Multiple"]:
    r = gamma_results.get(grp, {})
    if not r:
        print(f"  {grp:<12} No results")
        continue
    print(f"  {grp:<12} {r['b1']:>8.4f} {r['b1_p']:>10.4f} "
          f"{stars(r['b1_p']):>5} "
          f"{r['exp_b1']:>9.4f} {r['pct_change']:>+9.1f}% {r['aic']:>9.1f}")


# =============================================================================
# RESIDUAL DIAGNOSTICS FOR LME MODELS
# =============================================================================

section("LME RESIDUAL DIAGNOSTICS")

if residuals_for_diagnostics:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("LME Residual Diagnostics (log RT ~ Modality)\n"
                 "Top: Single group | Bottom: Multiple group",
                 fontsize=12, fontweight="bold")

    for row_i, grp in enumerate(["Single","Multiple"]):
        if grp not in residuals_for_diagnostics:
            continue
        rd     = residuals_for_diagnostics[grp]
        resids = rd["resids"]
        fitted = rd["fitted"]

        # 1. Residuals vs fitted
        ax = axes[row_i, 0]
        ax.scatter(fitted, resids, color=PAL[grp], s=20, alpha=0.5)
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_title(f"{grp}: Resid vs Fitted")
        ax.set_xlabel("Fitted"); ax.set_ylabel("Residual")

        # 2. Q-Q plot of residuals
        ax = axes[row_i, 1]
        (osm, osr), (slope, intercept, _) = stats.probplot(resids, dist="norm")
        ax.scatter(osm, osr, color=PAL[grp], s=20, alpha=0.7)
        lx = np.array([osm.min(), osm.max()])
        ax.plot(lx, slope*lx+intercept, color="black", lw=1.5, ls="--")
        sw, p_sw = shapiro(resids) if len(resids)<=5000 else (np.nan, np.nan)
        ax.set_title(f"{grp}: Q-Q Residuals\nSW p={p_sw:.4f}")
        ax.set_xlabel("Theoretical"); ax.set_ylabel("Sample")

        # 3. Histogram of residuals
        ax = axes[row_i, 2]
        ax.hist(resids, bins=20, color=PAL[grp], alpha=0.7, edgecolor="white",
                density=True)
        x = np.linspace(resids.min(), resids.max(), 200)
        ax.plot(x, stats.norm.pdf(x, resids.mean(), resids.std()), "k--", lw=1.5)
        ax.set_title(f"{grp}: Residual Distribution")
        ax.set_xlabel("Residual"); ax.set_ylabel("Density")

        # 4. Scale-location (sqrt |resid| vs fitted)
        ax = axes[row_i, 3]
        ax.scatter(fitted, np.sqrt(np.abs(resids)), color=PAL[grp], s=20, alpha=0.5)
        ax.set_title(f"{grp}: Scale-Location")
        ax.set_xlabel("Fitted"); ax.set_ylabel("√|Residual|")

    plt.tight_layout()
    save("fig_s4_01_lme_residuals.png")


# =============================================================================
# VISUALIZATIONS — Observed vs Fitted, Random Effects
# =============================================================================

section("MIXED EFFECTS VISUALIZATIONS")

# ── Fig: Observed vs model-predicted RT by modality ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("LME Model Predictions vs Observed: Log RT ~ Modality\n"
             "(Points = participant means; lines = model fixed effects)",
             fontsize=12, fontweight="bold")

for ax, grp in zip(axes, ["Single","Multiple"]):
    r = lme_results.get(grp, {})
    if "error" in r or not r: ax.set_title(f"{grp} (model failed)"); continue

    sub = (combined[combined["group"]==grp]
           .groupby(["participant","modality"])["log_RT"]
           .mean().reset_index())
    sub_w = sub.pivot(index="participant", columns="modality", values="log_RT").reset_index()

    color = PAL[grp]
    for _, row in sub_w.dropna().iterrows():
        ax.plot(["Lab","Game"],[row["Lab"],row["Game"]],
                "o-", color=color, alpha=0.25, lw=1.2, ms=5)

    # Model prediction lines
    pred_lab  = r["b0"]
    pred_game = r["b0"] + r["b1"]
    ax.plot(["Lab","Game"],[pred_lab,pred_game],
            "D-", color=color, lw=3, ms=12, zorder=6,
            label=f"Model: β1={r['b1']:.3f}{stars(r['b1_p'])}\n"
                  f"exp(β1)={r['exp_b1']:.3f} ({r['pct_change']:+.1f}%)\n"
                  f"ICC={r['icc']:.3f}")
    ax.set_title(f"{grp} group (n={r['n_ptpt']})")
    ax.set_xlabel("Modality"); ax.set_ylabel("Log RT")
    ax.legend(fontsize=9)

plt.tight_layout()
save("fig_s4_02_lme_predictions.png")


# ── Fig: Random effects — participant intercepts ─────────────────────────────
# Re-fit to extract random effects
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("LME Random Effects: Participant-Level Intercepts\n"
             "(Deviation from group mean; shows individual differences captured by model)",
             fontsize=12, fontweight="bold")

for ax, (grp, df) in zip(axes, [("Single",comb_s),("Multiple",comb_m)]):
    df = df.copy()
    df["participant"] = df["participant"].astype("category")
    try:
        model = smf.mixedlm("log_RT ~ modality_01", data=df,
                             groups=df["participant"])
        result = model.fit(reml=True, method="lbfgs")

        # Extract random effects
        re = result.random_effects
        re_vals = pd.DataFrame([
            {"participant": pid, "random_intercept": float(v.iloc[0])}
            for pid, v in re.items()
        ]).sort_values("random_intercept")

        color = PAL[grp]
        y_pos = range(len(re_vals))
        ax.barh(list(y_pos), re_vals["random_intercept"].values,
                color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", lw=1.2, ls="--")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(re_vals["participant"].astype(str).values, fontsize=8)
        ax.set_title(f"{grp} group — Random Intercepts (u0i)\n"
                     f"σ²_u0={float(result.cov_re.iloc[0,0]):.5f}")
        ax.set_xlabel("Random Intercept (deviation from fixed intercept)")

    except Exception as e:
        ax.set_title(f"{grp}: failed ({e})")

plt.tight_layout()
save("fig_s4_03_random_effects.png")


# =============================================================================
# SAVE RESULTS
# =============================================================================

all_lme = pd.DataFrame([v for v in lme_results.values() if "error" not in v])
all_gamma = pd.DataFrame([v for v in gamma_results.values() if v])
all_lme_level = pd.DataFrame([v for v in lme_level_results.values() if v])

all_lme.to_csv(OUT/"results_r2_lme.csv", index=False)
all_gamma.to_csv(OUT/"results_r2_gamma.csv", index=False)
all_lme_level.to_csv(OUT/"results_r2_lme_level.csv", index=False)

print(f"\n  ✔  Saved results_r2_lme.csv")
print(f"  ✔  Saved results_r2_gamma.csv")
print(f"  ✔  Saved results_r2_lme_level.csv")

cache.update({
    "lme_results":       lme_results,
    "lme_rs_results":    lme_rs_results,
    "gamma_results":     gamma_results,
    "lme_level_results": lme_level_results,
    "combined":          combined,
    "comb_s":            comb_s,
    "comb_m":            comb_m,
})
with open(OUT/"_r2_cache.pkl","wb") as f:
    pickle.dump(cache, f)
print("  ✔  Updated _r2_cache.pkl")

print("\n  STEP 4 COMPLETE — paste full output before Step 5.\n")