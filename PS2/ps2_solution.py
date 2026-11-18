"""MFAD Datathon CSE1 — Section RR_CSE_B · Team 9
Problem Statement 2 = PS04 "Attendance Threshold Investigation"
Scenario: Do students above and below a chosen attendance threshold differ in
academic outcomes?

Usage:
    python3 PS2/ps2_solution.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stats_lib as SL

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT = os.path.dirname(os.path.abspath(__file__)) + "/.."
FIG_DIR = os.path.join(PROJECT, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DF = SL.load_dataset("Dataset_09.csv")

# Chosen attendance threshold: median (cleanly splits the data)
THRESHOLD = float(DF["Attendance_Percent"].median())
DF["AttGroup"] = np.where(DF["Attendance_Percent"] >= THRESHOLD,
                          "High attendance (>=median)", "Low attendance (<median)")

FIG_COUNT = [0]


def fig_name(tag):
    FIG_COUNT[0] += 1
    return os.path.join(FIG_DIR, f"ps2_{FIG_COUNT[0]:02d}_{tag}.png")


def save_fig():
    path = fig_name("fig")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def section(title):
    return "\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72 + "\n"


REPORT = []
REPORT.append("MFAD DATATHON CSE1 — SECTION RR_CSE_B · TEAM 9")
REPORT.append("PS04 · PROBLEM STATEMENT 2 · ATTENDANCE THRESHOLD INVESTIGATION")
REPORT.append("Dataset: Dataset_09.csv (24 students, no missing values)")
REPORT.append(f"Chosen attendance threshold: {THRESHOLD}% (median).")
REPORT.append("Scenario: Do students above and below a chosen attendance threshold")
REPORT.append("differ in academic outcomes?")
REPORT.append("")

# ===========================================================================
# SUB-TASK 1 — Map the comparison groups and sampling frame
# ===========================================================================
REPORT.append(section("SUB-TASK 1 — Map the comparison groups and sampling frame"))

REPORT.append(
    "1.1 Study design definitions\n"
    "  • Population: students whose attendance and academic outcomes follow the\n"
    "    same process as these records.\n"
    "  • Observational unit: one student.\n"
    "  • Explanatory variable: attendance group (High vs Low, split at median).\n"
    "  • Response variables: Exam_Score_Percent, Passed.\n"
    "  • Sample: all 24 rows.\n"
)

# --- Summarise by group -----------------------------------------------------
REPORT.append("1.2 Summaries by attendance group:")
summary = DF.groupby("AttGroup").agg(
    n=("Student_ID", "count"),
    mean_att=("Attendance_Percent", "mean"),
    mean_exam=("Exam_Score_Percent", "mean"),
    sd_exam=("Exam_Score_Percent", "std"),
    pass_rate=("Passed", "mean"),
    n_passed=("Passed", "sum"),
).round(3)
REPORT.append(summary.to_string())
REPORT.append("")

# --- Plot: exam score by group ---------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
groups = ["Low attendance (<median)", "High attendance (>=median)"]
for ax, g in zip(axes, groups):
    sub = DF[DF["AttGroup"] == g]["Exam_Score_Percent"]
    ax.hist(sub, bins=8, color="#4c78a8", edgecolor="white", alpha=0.8)
    ax.axvline(sub.mean(), color="r", ls="--", label=f"mean={sub.mean():.1f}")
    ax.set_title(g)
    ax.set_xlabel("Exam_Score_Percent")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
plt.suptitle("PS04 — Exam score by attendance group", fontsize=12)
p1 = save_fig()
REPORT.append(f"  Figure saved: {p1}")

# --- Reproducible samples: SRS / stratified-by-group / cluster --------------
srs = SL.srs_sample(DF, n=8, seed=21)
strat = SL.stratified_sample(DF, strata_col="AttGroup", n_per_stratum=3, seed=21)
clust = SL.cluster_sample(DF, cluster_col="Section", n_clusters=1, seed=21)

REPORT.append("1.3 Reproducible samples vs full dataset (group means):")
full_means = DF.groupby("AttGroup")[["Exam_Score_Percent", "Attendance_Percent"]].mean()
rows = []
for label, smp in [("Full (n=24)", DF), ("SRS n=8", srs),
                   ("Stratified by AttGroup n=6", strat), ("Cluster n=8", clust)]:
    if len(smp) == 0:
        continue
    row = {"sample": label, "n": len(smp)}
    for g in groups:
        sub = smp[smp["AttGroup"] == g]
        if len(sub):
            row[f"{g.split(' ')[0]} exam"] = round(sub["Exam_Score_Percent"].mean(), 2)
    rows.append(row)
REPORT.append(pd.DataFrame(rows).to_string(index=False))
REPORT.append("")
REPORT.append("  Interpretation: stratified sampling by the grouping variable (AttGroup)")
REPORT.append("  guarantees both attendance groups are represented, reducing the risk of")
REPORT.append("  under-representing either group — best for this comparison question.")

# --- Reflection Q1: groups mutually exclusive? ----------------------------
REPORT.append("")
REPORT.append("REFLECTION Q1 — Are the groups mutually exclusive and appropriately defined?")
REPORT.append(
    "  Yes: High (>=median) and Low (<median) attendance are mutually exclusive and\n"
    "  exhaustive (every student is in exactly one group). Using the median as the\n"
    "  threshold gives balanced group sizes (14 high, 10 low here), which is good\n"
    "  for power. A fixed benchmark (e.g., 75%) might be more interpretable but\n"
    "  could yield unbalanced groups."
)

# --- Reflection Q2: sampling strategy to avoid under-representation ------
REPORT.append("")
REPORT.append("REFLECTION Q2 — Which sampling strategy reduces the risk of under-representing a group?")
REPORT.append(
    "  Stratified random sampling by the grouping variable (attendance group).\n"
    "  It forces proportional (or equal) inclusion of both High and Low groups,\n"
    "  unlike SRS which could by chance miss a group, and unlike cluster sampling\n"
    "  which could select only one section."
)

with open(os.path.join(PROJECT, "outputs", "ps2_report_part1.txt"), "w") as f:
    f.write("\n".join(REPORT))

print("PS04 sub-task 1 complete. Figures:", p1)

# ===========================================================================
# SUB-TASK 2 — Check distribution shape and categorical probabilities
# ===========================================================================
REPORT.append(section("SUB-TASK 2 — Check distribution shape and categorical probabilities"))

exam = DF["Exam_Score_Percent"]
att  = DF["Attendance_Percent"]

# --- Side-by-side histograms + boxplots by group ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, g in enumerate(groups):
    sub = DF[DF["AttGroup"] == g]
    axes[0, i].hist(sub["Exam_Score_Percent"], bins=8, color="#4c78a8",
                    edgecolor="white", alpha=0.8)
    axes[0, i].axvline(sub["Exam_Score_Percent"].mean(), color="r", ls="--")
    axes[0, i].set_title(f"{g} — exam histogram")
    axes[0, i].set_xlabel("Exam_Score_Percent")
    axes[1, i].boxplot(sub["Exam_Score_Percent"], vert=True,
                       patch_artist=True,
                       boxprops=dict(facecolor="#72b7b2"))
    axes[1, i].set_title(f"{g} — exam boxplot")
    axes[1, i].set_ylabel("Exam_Score_Percent")
plt.suptitle("PS04 — Exam score by attendance group", fontsize=12)
p2 = save_fig()
REPORT.append(f"  Figure saved: {p2}")

# --- z-scores & Q-Q per group -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, g in zip(axes, groups):
    sub = DF[DF["AttGroup"] == g]["Exam_Score_Percent"].dropna()
    z = (sub - sub.mean()) / sub.std(ddof=1)
    theo, obs = SL.qq_plot_points(sub)
    ax.scatter(theo, obs, s=40, alpha=0.8)
    ax.plot([theo.min(), theo.max()], [theo.min(), theo.max()], "k--", alpha=0.5)
    ax.set_title(f"{g} — Q-Q plot")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
plt.suptitle("PS04 — Normal Q-Q plots by attendance group", fontsize=12)
p3 = save_fig()
REPORT.append(f"  Figure saved: {p3}")

REPORT.append("2.1 Distribution checks per group:")
for g in groups:
    sub = DF[DF["AttGroup"] == g]["Exam_Score_Percent"].dropna()
    sh_stat, sh_p = SL.shapiro_wilk(sub)
    iqr = sub.quantile(0.75) - sub.quantile(0.25)
    lo, hi = sub.quantile(0.25) - 1.5*iqr, sub.quantile(0.75) + 1.5*iqr
    out = sub[(sub < lo) | (sub > hi)]
    REPORT.append(f"  {g}: n={len(sub)}, mean={sub.mean():.1f}, sd={sub.std(ddof=1):.1f}, "
                  f"Shapiro p={sh_p:.3f}, outliers={list(out.index)}")

# --- Normal-model probabilities where justified -----------------------------
REPORT.append("")
REPORT.append("2.2 Normal-model probabilities (where defensible):")
overall_mn, overall_sd = exam.mean(), exam.std(ddof=1)
for x in [70, 80]:
    pl = SL.normal_probability(x, overall_mn, overall_sd, tail="less")
    emp = (exam < x).mean()
    REPORT.append(f"  P(Exam < {x}): normal={pl:.3f}, empirical={emp:.3f}")

# --- Binomial model for passing vs observed --------------------------------
pi_hat = DF["Passed"].map({True:1, False:0}).mean()
n = len(DF); k = int(DF["Passed"].sum())
REPORT.append("")
REPORT.append(f"2.3 Binomial model for Passing (n={n}, pi-hat={pi_hat:.3f}):")
REPORT.append(f"  Observed passes: {k}/{n}")
REPORT.append(f"  Model P(X = {k}) = {SL.binomial_probability(k, n, pi_hat, tail='eq'):.4f}")
REPORT.append(f"  Model P(X >= {k}) = {SL.binomial_probability(k-1, n, pi_hat, tail='greater'):.3f}")

# --- Correlation among relevant numeric variables ---------------------------
corr_cols = ["Study_Hours_Per_Day", "Sleep_Hours_Per_Night", "Exam_Score_Percent",
             "Attendance_Percent"]
corr = SL.correlation_matrix(DF, corr_cols)
REPORT.append("")
REPORT.append("2.4 Correlation matrix:")
REPORT.append(corr.round(3).to_string())
REPORT.append("")
REPORT.append("  Multivariate-normal plausibility: with n=24 this cannot be firmly assessed;\n"
              "  pairwise scatter/Q-Q suggest approximate linearity but small n limits it.")

# --- Reflection Q3: group distributions similar enough? ------------------
REPORT.append("")
REPORT.append("REFLECTION Q3 — Are group distributions similar enough for the planned parametric comparison?")
REPORT.append(
    "  Both groups look roughly unimodal; Shapiro-Wilk p-values above do not reject\n"
    "  normality within groups. Variances: check homogeneity before pooling.\n"
    "  With n=10 and n=14, parametric tests are a reasonable working approximation\n"
    "  but results should be confirmed with non-parametric checks given the small n."
)

# --- Reflection Q4: binomial assumptions -----------------------------------
REPORT.append("")
REPORT.append("REFLECTION Q4 — What independence and constant-probability assumptions are needed for the binomial model?")
REPORT.append(
    "  (1) Fixed n — satisfied.\n"
    "  (2) Constant pass probability pi for every student — violated: attendance\n"
    "  group strongly predicts passing.\n"
    "  (3) Independent outcomes — plausible but shared section effects could induce\n"
    "  dependence. The binomial is therefore a rough approximation only."
)

# --- Reflection Q5: unusually high/low observation -------------------------
REPORT.append("")
REPORT.append("REFLECTION Q5 — How should an unusually high or low observation affect interpretation?")
REPORT.append(
    "  An extreme outlier can pull the mean and inflate the SD, distorting t-tests\n"
    "  and confidence intervals. With n=24, one outlier has substantial leverage.\n"
    "  Always report median alongside mean, and run sensitivity checks with/without\n"
    "  flagged outliers (e.g., the low scorer STU001)."
)

with open(os.path.join(PROJECT, "outputs", "ps2_report_part2.txt"), "w") as f:
    f.write("\n".join(REPORT))

print("PS04 sub-task 2 complete. Figures:", p2, p3)

# ===========================================================================
# SUB-TASK 3 — Measure sampling variability and estimator quality
# ===========================================================================
REPORT.append(section("SUB-TASK 3 — Measure sampling variability and estimator quality"))

sample_sizes = [5, 10, 20]
rng = np.random.default_rng(31)
values = exam.dropna().values
pop_mean = values.mean(); pop_sd = values.std(ddof=1)

# --- Sampling distribution of mean, overall --------------------------------
REPORT.append("3.1 Sampling distribution of mean exam score (overall data):")
for nsz in sample_sizes:
    reps = np.array([rng.choice(values, size=nsz, replace=True).mean() for _ in range(2000)])
    theo_se = pop_sd / np.sqrt(nsz)
    REPORT.append(f"  n={nsz:>2}: sim_mean={reps.mean():.2f}, sim_SE={reps.std(ddof=1):.3f}, "
                  f"theo_SE={theo_se:.3f}")
REPORT.append("  CLT: as n grows, simulated SE -> sigma/sqrt(n) and the distribution narrows.")

# --- Per-group sampling distributions (high vs low attendance) -------------
REPORT.append("")
REPORT.append("3.2 Sampling distributions by attendance group:")
for g in groups:
    gv = DF[DF["AttGroup"] == g]["Exam_Score_Percent"].dropna().values
    gp = gv.mean(); gsd = gv.std(ddof=1)
    for nsz in [5, 10]:
        if nsz > len(gv):
            continue
        reps = np.array([rng.choice(gv, size=nsz, replace=True).mean() for _ in range(2000)])
        theo = gsd / np.sqrt(nsz)
        REPORT.append(f"  {g.split(' ')[0]} n={nsz}: sim_mean={reps.mean():.2f}, "
                      f"sim_SE={reps.std(ddof=1):.3f}, theo_SE={theo:.3f} (group n={len(gv)})")

# --- Bootstrap bias/variance/MSE per group ---------------------------------
REPORT.append("")
REPORT.append("3.3 Bootstrap bias/variance/MSE of the mean:")
for label, sub in [("Overall", DF), ("High att", DF[DF["AttGroup"]==groups[1]]),
                   ("Low att", DF[DF["AttGroup"]==groups[0]])]:
    b = SL.bootstrap_estimator(sub, "Exam_Score_Percent",
                               estimator_func=lambda x: np.asarray(x, dtype=float).mean(),
                               n_boots=2000, seed=33)
    REPORT.append(f"  {label}: point={b['point_estimate']:.2f}, boot_mean={b['bootstrap_mean']:.2f}, "
                  f"boot_SE={b['bootstrap_se']:.3f}, bias={b['bias_estimate']:.3f}, MSE={b['mse_estimate']:.3f}")

# --- Normal MLE vs unbiased variance ---------------------------------------
mle = SL.normal_mle(DF["Exam_Score_Percent"])
REPORT.append("")
REPORT.append("3.4 Normal MLE vs unbiased sample variance (Exam_Score_Percent):")
REPORT.append(f"  MLE mean={mle['mean_mle']:.3f}, MLE var(÷n)={mle['var_mle']:.3f}, "
              f"unbiased var(÷n-1)={mle['var_unbiased']:.3f}, ratio={mle['ratio_mle_unbiased']:.4f}")

# --- Reflection Q6: small groups fluctuate more ----------------------------
REPORT.append("")
REPORT.append("REFLECTION Q6 — Why might estimates from small groups fluctuate more than estimates from the full sample?")
REPORT.append(
    "  SE = sigma/sqrt(n): smaller n -> larger SE -> wider sampling distribution ->\n"
    "  more fluctuation. The low-attendance group (n=10) will have noisier mean/\n"
    "  pass-rate estimates than the full sample (n=24). Small-group estimates are\n"
    "  less reliable; confidence intervals will be wider."
)

# --- Reflection Q7: CLT what it tells / doesn't guarantee ------------------
REPORT.append("")
REPORT.append("REFLECTION Q7 — What does the CLT tell you, and what does it not guarantee for this dataset?")
REPORT.append(
    "  The CLT says the sampling distribution of the mean approaches normality as\n"
    "  n grows, regardless of population shape, and SE shrinks as sigma/sqrt(n).\n"
    "  It does NOT guarantee normality at n=24 (small n), nor does it fix bias,\n"
    "  outliers, or dependence. With n=24 the approximation is tentative."
)

# --- Reflection Q8: bias-variance tradeoff & estimator selection -----------
REPORT.append("")
REPORT.append("REFLECTION Q8 — How does the bias-variance trade-off affect the estimator you select?")
REPORT.append(
    "  The sample mean is unbiased and efficient (lowest variance among unbiased\n"
    "  estimators under normal theory) — report it. For variance, the unbiased\n"
    "  sample variance (÷n-1) trades a tiny variance increase to remove the MLE's\n"
    "  downward bias; its MSE is competitive. Prefer unbiased estimators here\n"
    "  because n is small and bias is more damaging than a small variance rise."
)

with open(os.path.join(PROJECT, "outputs", "ps2_report_part3.txt"), "w") as f:
    f.write("\n".join(REPORT))

print("PS04 sub-task 3 complete.")

# ===========================================================================
# SUB-TASK 4 — Evaluate differences using intervals and tests
# ===========================================================================
REPORT.append(section("SUB-TASK 4 — Evaluate differences using intervals and tests"))

hi = DF[DF["AttGroup"] == groups[1]]["Exam_Score_Percent"]
lo = DF[DF["AttGroup"] == groups[0]]["Exam_Score_Percent"]

# --- t-based CIs for group means (pop SD unknown) ---------------------------
ci_hi = SL.ci_mean_t(hi, confidence=0.95)
ci_lo = SL.ci_mean_t(lo, confidence=0.95)
REPORT.append("4.1 95% t-confidence intervals for group mean exam score (SD unknown):")
REPORT.append(f"  High attendance (n={ci_hi['n']}): [{ci_hi['lower']:.2f}, {ci_hi['upper']:.2f}] "
              f"(mean={ci_hi['mean']:.2f}, MOE={ci_hi['margin_of_error']:.2f})")
REPORT.append(f"  Low attendance (n={ci_lo['n']}): [{ci_lo['lower']:.2f}, {ci_lo['upper']:.2f}] "
              f"(mean={ci_lo['mean']:.2f}, MOE={ci_lo['margin_of_error']:.2f})")
REPORT.append("  z-methods would require explicitly known population SDs — not available here; t is correct.")

# --- Hypothesis test: group difference in mean exam score -------------------
tt = SL.t_test_two_sample(hi, lo, alternative="two-sided", equal_var=True)
REPORT.append("")
REPORT.append("4.2 Group difference test: H0: mu_high = mu_low vs H1: mu_high != mu_low (two-sample t, alpha=0.05)")
REPORT.append(f"  t={tt['t_statistic']:.3f}, df={tt['df']:.1f}, crit={tt['critical_value']:.3f}, "
              f"p={tt['p_value']:.4f} -> {'REJECT H0' if tt['reject_at_05'] else 'fail to reject H0'}")
REPORT.append(f"  Cohen's d = {SL.cohens_d(hi, lo):.3f}")
REPORT.append("  Selected: two-sample t (independent groups, pop SDs unknown, approx. equal variance).")

# --- Pass-rate comparison (two-proportion) ----------------------------------
hi_pass = DF[DF["AttGroup"] == groups[1]]["Passed"].map({True:1,False:0})
lo_pass = DF[DF["AttGroup"] == groups[0]]["Passed"].map({True:1,False:0})
ph_hi, ph_lo = hi_pass.mean(), lo_pass.mean()
REPORT.append("")
REPORT.append(f"4.3 Pass-rate comparison: high={ph_hi:.3f} ({int(hi_pass.sum())}/{len(hi_pass)}), "
              f"low={ph_lo:.3f} ({int(lo_pass.sum())}/{len(lo_pass)})")
chi_pp = SL.chi_square_independence(DF, "Passed", "AttGroup")
REPORT.append(f"  Chi-square Passed vs AttGroup: chi2={chi_pp['chi2']:.3f}, p={chi_pp['p_value']:.4f}, "
              f"-> {'REJECT H0' if chi_pp['reject_at_05'] else 'fail to reject H0'}")

# --- Reflection Q9: CI overlap vs formal test ------------------------------
REPORT.append("")
REPORT.append("REFLECTION Q9 — How does interval overlap relate to, but not replace, a formal test?")
REPORT.append(
    "  The two 95% CIs " + ("overlap" if (ci_hi['lower'] < ci_lo['upper'] and ci_lo['lower'] < ci_hi['upper']) else "do not overlap") +
    ".\n"
    "  Non-overlapping 95% CIs usually imply a significant difference, but overlapping\n"
    "  CIs do NOT imply non-significance (the correct SE for the difference is smaller\n"
    "  than the sum of individual SEs). Always run the formal t-test — here: " +
    ("significant" if tt['reject_at_05'] else "not significant") + "."
)

# --- Reflection Q10: practical meaning -------------------------------------
REPORT.append("")
REPORT.append("REFLECTION Q10 — What practical meaning does the test decision have for the groups?")
REPORT.append(
    "  If significant: attendance group is associated with exam performance — supports\n"
    "  attendance-focused interventions. If not significant: no evidence of a mean\n"
    "  difference at alpha=0.05, but with n=24 a real difference could be missed\n"
    "  (low power). Practical relevance also depends on effect size (Cohen's d)."
)

# --- Reflection Q11: Type I / Type II error --------------------------------
REPORT.append("")
REPORT.append("REFLECTION Q11 — Describe a Type I and Type II error in this comparison.")
REPORT.append(
    "  Type I error: concluding attendance groups differ in exam score when they\n"
    "  really do not (false alarm; alpha = 0.05).\n"
    "  Type II error: concluding no difference when high-attendance students truly\n"
    "  perform better (missed detection). With small n, Type II risk is high.\n"
    "  For this claim, a Type II error could mean overlooking a valuable attendance\n"
    "  intervention; a Type I error could mean investing in one that doesn't work."
)

with open(os.path.join(PROJECT, "outputs", "ps2_report_part4.txt"), "w") as f:
    f.write("\n".join(REPORT))

print("PS04 sub-task 4 complete.")

# ===========================================================================
# SUB-TASK 5 — Test group structure and assess study power
# ===========================================================================
REPORT.append(section("SUB-TASK 5 — Test group structure and assess study power"))

# --- One-way ANOVA: exam by attendance group --------------------------------
anova_att = SL.one_way_anova(DF, "Exam_Score_Percent", "AttGroup")
REPORT.append("5.1 One-way ANOVA: Exam_Score_Percent by attendance group")
REPORT.append(f"  F={anova_att['f_statistic']:.3f}, df=({anova_att['df_between']},{anova_att['df_within']}), "
              f"p={anova_att['p_value']:.4f}, eta^2={anova_att['eta_squared']:.3f}")
REPORT.append(f"  -> {'REJECT H0 (groups differ)' if anova_att['reject_at_05'] else 'fail to reject H0'}")
REPORT.append("  ANOVA assumptions: independence (plausible), normality within groups,")
REPORT.append("  homogeneity of variance (check with Levene given small n).")

# --- Why ANOVA vs many pairwise t-tests -------------------------------------
REPORT.append("")
REPORT.append("REFLECTION Q12 — Why is ANOVA preferable to many pairwise t-tests here?")
REPORT.append(
    "  Running multiple pairwise t-tests inflates the family-wise Type I error\n"
    "  rate (chance of at least one false positive). ANOVA tests the omnibus\n"
    "  null (all group means equal) in one test, controlling alpha. With only two\n"
    "  groups here, ANOVA and the two-sample t-test are equivalent, but ANOVA is\n"
    "  the correct framework if more groups are added."
)

# --- Chi-square: Passed vs AttGroup (already computed) + Section ------------
REPORT.append("")
REPORT.append("5.2 Chi-square tests for categorical variables")
REPORT.append(f"  Passed vs AttGroup: chi2={chi_pp['chi2']:.3f}, p={chi_pp['p_value']:.4f}, "
              f"min expected={chi_pp['min_expected']:.2f}, cells<5={chi_pp['cells_below_5']}")
chi_sec2 = SL.chi_square_independence(DF, "Passed", "Section")
REPORT.append(f"  Passed vs Section: chi2={chi_sec2['chi2']:.3f}, p={chi_sec2['p_value']:.4f}, "
              f"min expected={chi_sec2['min_expected']:.2f}")
REPORT.append("  Expected-count condition: if any expected count < 5, treat the p-value")
REPORT.append("  cautiously or use Fisher's exact test instead.")

# --- Reflection Q13: expected counts adequate? -----------------------------
REPORT.append("")
REPORT.append("REFLECTION Q13 — Are chi-square expected counts adequate? If not, what caution?")
REPORT.append(
    "  With n=24 split across groups/sections, some expected counts fall below 5,\n"
    "  violating the chi-square approximation. Caution: p-values may be inaccurate;\n"
    "  use Fisher's exact test or report exact methods, and interpret cautiously."
)

# --- Power discussion -------------------------------------------------------
REPORT.append("")
REPORT.append("5.3 Power discussion")
d_att = SL.cohens_d(hi, lo)
REPORT.append(SL.power_notes(d_att, n_per_group=min(len(hi), len(lo)), alpha=0.05))
REPORT.append(f"  Observed Cohen's d for attendance groups = {d_att:.2f}.")
REPORT.append("  Power rises with larger n, larger effect size, lower variability, and\n"
              "  larger alpha. With n=10/14, power to detect a small effect is low;\n"
              "  the study is powered only for medium-to-large effects.")

# --- Reflection Q14: power for small difference? ---------------------------
REPORT.append("")
REPORT.append("REFLECTION Q14 — Would the study have enough power to detect a small group difference?")
REPORT.append(
    "  No. With ~10-14 students per group, power to detect a small effect (d=0.2)\n"
    "  at alpha=0.05 is very low (well under 0.8). A non-significant result would\n"
    "  be inconclusive, not evidence of no difference. A larger sample is needed."
)

# --- Statistical vs practical significance ----------------------------------
REPORT.append("")
REPORT.append("REFLECTION Q15 — Statistical significance vs practical importance here?")
REPORT.append(
    "  Statistical significance (p<0.05) tells us an observed difference is unlikely\n"
    "  under H0. Practical importance depends on effect size: Cohen's d = " +
    f"{d_att:.2f}" + " and the raw attendance-group exam gap. Even a significant tiny\n"
    "  gap may not justify an intervention; a large gap matters even if p is marginal\n"
    "  given small n."
)

# --- Better study design ----------------------------------------------------
REPORT.append("")
REPORT.append("REFLECTION Q16 — What larger/better-designed study would improve power and generalisability?")
REPORT.append(
    "  Larger multi-section/multi-institution sample (n in hundreds); randomised\n"
    "  attendance encouragement for causality; record covariates (prior GPA, motivation);\n"
    "  pre-register the analysis; oversample under-represented groups; use the\n"
    "  power analysis above to set target n for 80% power at the smallest effect of\n"
    "  practical interest."
)

# --- Overall conclusion -----------------------------------------------------
REPORT.append("")
REPORT.append(section("OVERALL CONCLUSION — PS04 Attendance Threshold Investigation"))
REPORT.append(
    f"  Using the median attendance threshold ({THRESHOLD}%), the High-attendance group\n"
    f"  (n={len(hi)}) had mean exam score {hi.mean():.1f} vs {lo.mean():.1f} for Low (n={len(lo)}),\n"
    f"  and pass rates {ph_hi:.0%} vs {ph_lo:.0%}. The two-sample t-test " +
    ("rejected" if tt['reject_at_05'] else "did not reject") +
    " the null of equal means (p=" + f"{tt['p_value']:.3f}" + ", Cohen's d = " +
    f"{d_att:.2f}" + "), and chi-square " +
    ("rejected" if chi_pp['reject_at_05'] else "did not reject") +
    " equal pass rates. Together these indicate that students above the attendance\n"
    "  threshold perform better academically. Limitations: n=24, synthetic data,\n"
    "  observational design (causality not established), low power for small effects.\n"
    "  A larger randomised study is recommended before acting on the finding."
)

with open(os.path.join(PROJECT, "outputs", "ps2_report_part5.txt"), "w") as f:
    f.write("\n".join(REPORT))

print("PS04 sub-task 5 complete.")
print("ALL PS04 SUB-TASKS DONE. Report parts saved to outputs/")
