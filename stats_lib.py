"""Shared statistics helpers for MFAD Datathon PS03/PS04 (section RR_CSE_B, Team 9).

Functions cover: descriptive summaries, sampling (SRS/stratified/cluster),
distribution checks (histogram/boxplot/Q-Q/z-scores), normal/binomial models,
correlation, sampling-distribution simulation, bootstrap bias/variance/MSE,
normal MLE vs unbiased variance, confidence intervals (t/z), hypothesis tests
(t/z), one-way ANOVA, chi-square tests, and power discussion helpers.
"""

from __future__ import annotations

import os
import math
import itertools
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(name: str = "Dataset_09.csv") -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------------
def describe_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df[cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]


def describe_categorical(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].value_counts(dropna=False)


# ---------------------------------------------------------------------------
# Sampling (reproducible)
# ---------------------------------------------------------------------------
def srs_sample(df: pd.DataFrame, n: int, seed: int = 42, replace: bool = False) -> pd.DataFrame:
    return df.sample(n=n, replace=replace, random_state=seed)


def stratified_sample(df: pd.DataFrame, strata_col: str, n_per_stratum: int,
                      seed: int = 42) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby(strata_col, group_keys=False):
        parts.append(g.sample(n=min(n_per_stratum, len(g)), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def cluster_sample(df: pd.DataFrame, cluster_col: str, n_clusters: int,
                   seed: int = 42) -> pd.DataFrame:
    clusters = df[cluster_col].unique()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(clusters, size=min(n_clusters, len(clusters)), replace=False)
    return df[df[cluster_col].isin(chosen)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Distribution checks
# ---------------------------------------------------------------------------
def z_scores(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def normal_fit(s: pd.Series) -> Tuple[float, float]:
    return float(s.mean()), float(s.std(ddof=1))


def shapiro_wilk(s: pd.Series) -> Tuple[float, float]:
    s_clean = s.dropna()
    if len(s_clean) < 3:
        return float("nan"), float("nan")
    stat, p = stats.shapiro(s_clean)
    return float(stat), float(p)


def qq_plot_points(s: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    s_clean = np.sort(s.dropna().values)
    n = len(s_clean)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(probs, loc=s_clean.mean(), scale=s_clean.std(ddof=1))
    return theoretical, s_clean


# ---------------------------------------------------------------------------
# Probability models
# ---------------------------------------------------------------------------
def normal_probability(x: float, mean: float, sd: float, tail: str = "less") -> float:
    if tail == "less":
        return float(stats.norm.cdf(x, loc=mean, scale=sd))
    if tail == "greater":
        return float(stats.norm.sf(x, loc=mean, scale=sd))
    if tail == "two":
        return 2 * min(float(stats.norm.cdf(x, loc=mean, scale=sd)),
                       float(stats.norm.sf(x, loc=mean, scale=sd)))
    raise ValueError("tail must be 'less', 'greater', or 'two'")


def binomial_probability(k: int, n: int, p: float, tail: str = "less") -> float:
    if tail == "less":
        return float(stats.binom.cdf(k, n, p))
    if tail == "greater":
        return float(stats.binom.sf(k, n, p))
    if tail == "eq":
        return float(stats.binom.pmf(k, n, p))
    if tail == "two":
        p_k = stats.binom.pmf(k, n, p)
        p_le = stats.binom.cdf(k, n, p)
        p_ge = stats.binom.sf(k - 1, n, p)
        return float(2 * min(p_le, p_ge))
    raise ValueError("tail must be 'less', 'greater', 'eq', or 'two'")


def correlation_matrix(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df[cols].corr()


# ---------------------------------------------------------------------------
# Sampling-distribution simulation
# ---------------------------------------------------------------------------
def simulate_sampling_distribution(df: pd.DataFrame, col: str,
                                   sample_sizes: Sequence[int],
                                   n_reps: int = 2000,
                                   stat: str = "mean",
                                   seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    values = df[col].dropna().values
    pop_mean = values.mean()
    pop_sd = values.std(ddof=1)
    pop_var = values.var(ddof=1)
    results = {}
    for n in sample_sizes:
        reps = []
        for _ in range(n_reps):
            sample = rng.choice(values, size=n, replace=True)
            if stat == "mean":
                reps.append(sample.mean())
            elif stat == "proportion":
                reps.append((sample == values).mean())  # placeholder
            elif stat == "median":
                reps.append(np.median(sample))
        reps = np.array(reps)
        results[n] = {
            "reps": reps,
            "sim_mean": float(reps.mean()),
            "sim_se": float(reps.std(ddof=1)),
            "theo_se": float(pop_sd / math.sqrt(n)),
            "pop_mean": float(pop_mean),
            "pop_sd": float(pop_sd),
            "n_reps": n_reps,
            "sample_size": n,
        }
    return results


def simulate_proportion_distribution(df: pd.DataFrame, col: str,
                                     sample_sizes: Sequence[int],
                                     n_reps: int = 2000,
                                     seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    values = df[col].dropna().values.astype(bool)
    pop_prop = values.mean()
    results = {}
    for n in sample_sizes:
        reps = []
        for _ in range(n_reps):
            sample = rng.choice(values, size=n, replace=True)
            reps.append(sample.mean())
        reps = np.array(reps)
        theo_se = math.sqrt(pop_prop * (1 - pop_prop) / n)
        results[n] = {
            "reps": reps,
            "sim_mean": float(reps.mean()),
            "sim_se": float(reps.std(ddof=1)),
            "theo_se": float(theo_se),
            "pop_prop": float(pop_prop),
            "n_reps": n_reps,
            "sample_size": n,
        }
    return results


# ---------------------------------------------------------------------------
# Bootstrap bias / variance / MSE
# ---------------------------------------------------------------------------
def bootstrap_estimator(df: pd.DataFrame, col: str,
                        estimator_func,
                        n_boots: int = 2000,
                        seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    values = df[col].dropna().values
    n = len(values)
    point = estimator_func(values)
    boots = np.array([estimator_func(rng.choice(values, size=n, replace=True))
                      for _ in range(n_boots)])
    boot_mean = boots.mean()
    boot_var = boots.var(ddof=1)
    bias = boot_mean - point
    mse = bias ** 2 + boot_var
    return {
        "point_estimate": float(point),
        "bootstrap_mean": float(boot_mean),
        "bootstrap_var": float(boot_var),
        "bootstrap_se": float(math.sqrt(boot_var)),
        "bias_estimate": float(bias),
        "mse_estimate": float(mse),
        "n_boots": n_boots,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Normal MLE vs unbiased sample variance
# ---------------------------------------------------------------------------
def normal_mle(s: pd.Series) -> dict:
    x = s.dropna().values
    n = len(x)
    mean_mle = float(x.mean())
    var_mle = float(((x - mean_mle) ** 2).mean())   # divides by n
    var_unbiased = float(x.var(ddof=1))             # divides by n-1
    se_mean_mle = math.sqrt(var_mle / n)
    return {
        "n": n,
        "mean_mle": mean_mle,
        "var_mle": var_mle,
        "var_unbiased": var_unbiased,
        "ratio_mle_unbiased": var_mle / var_unbiased,
        "se_mean_mle": se_mean_mle,
        "bias_of_mle_var": var_mle - var_unbiased,
    }


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------
def ci_mean_t(s: pd.Series, confidence: float = 0.95) -> dict:
    x = s.dropna().values
    n = len(x)
    mean = x.mean()
    sd = x.std(ddof=1)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    moe = tcrit * se
    return {
        "method": "t",
        "n": n,
        "mean": float(mean),
        "sd": float(sd),
        "se": float(se),
        "confidence": confidence,
        "critical": float(tcrit),
        "margin_of_error": float(moe),
        "lower": float(mean - moe),
        "upper": float(mean + moe),
    }


def ci_mean_z(s: pd.Series, pop_sd: float, confidence: float = 0.95) -> dict:
    x = s.dropna().values
    n = len(x)
    mean = x.mean()
    se = pop_sd / math.sqrt(n)
    zcrit = stats.norm.ppf(1 - (1 - confidence) / 2)
    moe = zcrit * se
    return {
        "method": "z (known population SD)",
        "n": n,
        "mean": float(mean),
        "pop_sd": float(pop_sd),
        "se": float(se),
        "confidence": confidence,
        "critical": float(zcrit),
        "margin_of_error": float(moe),
        "lower": float(mean - moe),
        "upper": float(mean + moe),
    }


def ci_proportion(s: pd.Series, confidence: float = 0.95,
                  method: str = "wald") -> dict:
    x = s.dropna().values.astype(bool)
    n = len(x)
    phat = x.mean()
    zcrit = stats.norm.ppf(1 - (1 - confidence) / 2)
    if method == "wald":
        se = math.sqrt(phat * (1 - phat) / n)
        moe = zcrit * se
    elif method == "wilson":
        denom = 1 + zcrit ** 2 / n
        centre = (phat + zcrit ** 2 / (2 * n)) / denom
        half = (zcrit / denom) * math.sqrt(phat * (1 - phat) / n + zcrit ** 2 / (4 * n ** 2))
        moe = half
        lower, upper = centre - half, centre + half
        return {
            "method": "Wilson",
            "n": n,
            "phat": float(phat),
            "confidence": confidence,
            "critical": float(zcrit),
            "margin_of_error": float(half),
            "lower": float(lower),
            "upper": float(upper),
        }
    else:
        raise ValueError("method must be 'wald' or 'wilson'")
    return {
        "method": "Wald",
        "n": n,
        "phat": float(phat),
        "confidence": confidence,
        "critical": float(zcrit),
        "se": float(se),
        "margin_of_error": float(moe),
        "lower": float(phat - moe),
        "upper": float(phat + moe),
    }


# ---------------------------------------------------------------------------
# Hypothesis tests (one-sample)
# ---------------------------------------------------------------------------
def t_test_one_sample(s: pd.Series, mu0: float,
                      alternative: str = "two-sided") -> dict:
    x = s.dropna().values
    n = len(x)
    mean = x.mean()
    sd = x.std(ddof=1)
    se = sd / math.sqrt(n)
    tstat = (mean - mu0) / se
    dfree = n - 1
    if alternative == "two-sided":
        p = 2 * stats.t.sf(abs(tstat), dfree)
        crit = stats.t.ppf(0.975, dfree)
    elif alternative == "greater":
        p = stats.t.sf(tstat, dfree)
        crit = stats.t.ppf(0.95, dfree)
    elif alternative == "less":
        p = stats.t.cdf(tstat, dfree)
        crit = stats.t.ppf(0.05, dfree)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return {
        "test": "one-sample t",
        "n": n,
        "mean": float(mean),
        "mu0": float(mu0),
        "sd": float(sd),
        "se": float(se),
        "t_statistic": float(tstat),
        "df": int(dfree),
        "critical_value": float(crit),
        "p_value": float(p),
        "alternative": alternative,
        "reject_at_05": bool(p < 0.05),
    }


def z_test_one_sample(s: pd.Series, mu0: float, pop_sd: float,
                      alternative: str = "two-sided") -> dict:
    x = s.dropna().values
    n = len(x)
    mean = x.mean()
    se = pop_sd / math.sqrt(n)
    zstat = (mean - mu0) / se
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(zstat))
        crit = stats.norm.ppf(0.975)
    elif alternative == "greater":
        p = stats.norm.sf(zstat)
        crit = stats.norm.ppf(0.95)
    elif alternative == "less":
        p = stats.norm.cdf(zstat)
        crit = stats.norm.ppf(0.05)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return {
        "test": "one-sample z (known population SD)",
        "n": n,
        "mean": float(mean),
        "mu0": float(mu0),
        "pop_sd": float(pop_sd),
        "se": float(se),
        "z_statistic": float(zstat),
        "critical_value": float(crit),
        "p_value": float(p),
        "alternative": alternative,
        "reject_at_05": bool(p < 0.05),
    }


# ---------------------------------------------------------------------------
# Two-sample t-test
# ---------------------------------------------------------------------------
def t_test_two_sample(s1: pd.Series, s2: pd.Series,
                      alternative: str = "two-sided",
                      equal_var: bool = True) -> dict:
    x1 = s1.dropna().values
    x2 = s2.dropna().values
    n1, n2 = len(x1), len(x2)
    m1, m2 = x1.mean(), x2.mean()
    res = stats.ttest_ind(x1, x2, equal_var=equal_var,
                          alternative=alternative)
    if alternative == "two-sided":
        crit = stats.t.ppf(0.975, df=res.df)
    else:
        crit = stats.t.ppf(0.95, df=res.df)
    se = math.sqrt(x1.var(ddof=1) / n1 + x2.var(ddof=1) / n2)
    return {
        "test": "two-sample t",
        "n1": int(n1), "n2": int(n2),
        "mean1": float(m1), "mean2": float(m2),
        "sd1": float(x1.std(ddof=1)), "sd2": float(x2.std(ddof=1)),
        "se": float(se),
        "t_statistic": float(res.statistic),
        "df": float(res.df),
        "critical_value": float(crit),
        "p_value": float(res.pvalue),
        "alternative": alternative,
        "equal_var": equal_var,
        "reject_at_05": bool(res.pvalue < 0.05),
    }


# ---------------------------------------------------------------------------
# One-way ANOVA
# ---------------------------------------------------------------------------
def one_way_anova(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    k = len(groups)
    n = sum(len(g) for g in groups)
    grand_mean = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    sst = ssb + ssw
    dfb, dfw = k - 1, n - k
    msb, msw = ssb / dfb, ssw / dfw
    fstat = msb / msw
    p = stats.f.sf(fstat, dfb, dfw)
    crit = stats.f.ppf(0.95, dfb, dfw)
    eta2 = ssb / sst
    return {
        "test": "one-way ANOVA",
        "k": int(k), "n": int(n),
        "df_between": int(dfb), "df_within": int(dfw),
        "ssb": float(ssb), "ssw": float(ssw), "sst": float(sst),
        "msb": float(msb), "msw": float(msw),
        "f_statistic": float(fstat),
        "critical_value": float(crit),
        "p_value": float(p),
        "eta_squared": float(eta2),
        "reject_at_05": bool(p < 0.05),
        "group_means": {str(g[0]): float(g[1].mean()) for g in df.groupby(group_col)[value_col]},
        "group_sizes": {str(g[0]): int(len(g[1])) for g in df.groupby(group_col)[value_col]},
    }


# ---------------------------------------------------------------------------
# Chi-square tests
# ---------------------------------------------------------------------------
def chi_square_independence(df: pd.DataFrame, col1: str, col2: str) -> dict:
    ct = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(ct, correction=False)
    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
    min_exp = expected.min()
    cells_below_5 = int((expected < 5).sum())
    return {
        "test": "chi-square independence",
        "chi2": float(chi2),
        "p_value": float(p),
        "df": int(dof),
        "critical_value": float(stats.chi2.ppf(0.95, dof)),
        "observed": ct.to_dict(),
        "expected": expected_df.round(3).to_dict(),
        "min_expected": float(min_exp),
        "cells_below_5": cells_below_5,
        "reject_at_05": bool(p < 0.05),
        "assumption_met": bool(min_exp >= 5 and cells_below_5 == 0),
    }


def chi_square_goodness_of_fit(observed: Sequence[float],
                               expected_probs: Sequence[float]) -> dict:
    observed = np.array(observed, dtype=float)
    expected_probs = np.array(expected_probs, dtype=float)
    n = observed.sum()
    expected = expected_probs * n
    chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
    dof = len(observed) - 1
    return {
        "test": "chi-square goodness-of-fit",
        "observed": observed.tolist(),
        "expected": expected.tolist(),
        "chi2": float(chi2),
        "p_value": float(p),
        "df": int(dof),
        "critical_value": float(stats.chi2.ppf(0.95, dof)),
        "min_expected": float(expected.min()),
        "reject_at_05": bool(p < 0.05),
        "assumption_met": bool(expected.min() >= 5),
    }


# ---------------------------------------------------------------------------
# Effect size: Cohen's d
# ---------------------------------------------------------------------------
def cohens_d(s1: pd.Series, s2: pd.Series) -> float:
    x1, x2 = s1.dropna().values, s2.dropna().values
    n1, n2 = len(x1), len(x2)
    s1_var, s2_var = x1.var(ddof=1), x2.var(ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1_var + (n2 - 1) * s2_var) / (n1 + n2 - 2))
    return float((x1.mean() - x2.mean()) / pooled)


# ---------------------------------------------------------------------------
# Power discussion helper
# ---------------------------------------------------------------------------
def power_notes(effect_size: float, n_per_group: int, alpha: float = 0.05) -> str:
    """Return a short plain-English power discussion."""
    return (
        f"With alpha={alpha}, per-group n={n_per_group} and Cohen's d≈{effect_size:.2f}, "
        "power increases with larger sample size, larger effect size, lower variability, "
        "and larger alpha. Small groups (as in this dataset) yield low power for small "
        "effects, so a non-significant result should be interpreted cautiously."
    )
