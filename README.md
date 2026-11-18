# MFAD DATATHON CSE1 — Section RR_CSE_B (Team 9)

## Source
Instruction file: `/Users/hellbound666/Downloads/MFAD_Datathon_RR_CSE_B (1).docx`

## Your allocation (from the instruction doc)
- Team number: **9** → Dataset **Dataset_09.csv** (Team N → Dataset N)
- Section: **RR_CSE_B**

## Deliverables (all present in this folder)

| File | Status | Notes |
|------|--------|-------|
| `datasets/Dataset_09.csv` | ✅ | Your allocated dataset (24 students, 10 cols, no missing) |
| `PS1/ps1_solution.py` | ✅ | PS03 "Attendance Risk Analysis" code |
| `outputs/ps1_report.pdf` | ✅ | PS03 report (all 5 sub-tasks + figures + reflections) |
| `PS2/ps2_solution.py` | ✅ | PS04 "Attendance Threshold Investigation" code |
| `outputs/ps2_report.pdf` | ✅ | PS04 report (all 5 sub-tasks + figures + reflections) |
| `stats_lib.py` | ✅ | Shared statistics library |
| `make_pdf_ps1.py`, `make_pdf_ps2.py` | ✅ | PDF report generators |
| `outputs/figures/*.png` | ✅ | 7 figures total |
| `outputs/ps1_report_part*.txt`, `ps2_report_part*.txt` | ✅ | Report source text |

## Problem statements (extracted from the PDFs you provided)

**PS1 = PS03 "Attendance Risk Analysis"** — 5 sub-tasks:
1. Build the student-performance evidence base (definitions, descriptives, SRS/stratified/cluster samples).
2. Model the observed pattern with probability (histogram/boxplot/Q-Q/z-scores/outliers, normal & binomial models, correlation).
3. Simulate repeated samples & estimate parameters (sampling distributions, CLT, bootstrap bias/variance/MSE, MLE vs unbiased variance).
4. Quantify uncertainty & test the central claim (t CI, one-sample t test vs 60, pass-proportion CI).
5. Compare groups & communicate conclusion (one-way ANOVA by Section/Sampling_Method, chi-square, effect size, power).

**PS2 = PS04 "Attendance Threshold Investigation"** — 5 sub-tasks:
Same structure, comparing students above/below the median attendance threshold (79%), including two-sample t-test, chi-square (Passed vs AttGroup/Section), ANOVA, and power.

## Submission files to upload (via Google Form)
Rename with your SRN before uploading:
- `<SRN>_PS03.py`  ← from `PS1/ps1_solution.py`
- `<SRN>_PS03_reflections.pdf`  ← from `outputs/ps1_report.pdf`
- `<SRN>_PS04.py`  ← from `PS2/ps2_solution.py`
- `<SRN>_PS04_reflections.pdf`  ← from `outputs/ps2_report.pdf`

Form: https://forms.gle/JawaJriUGWssDaiw6

## Running the code
```
make -f MAKEFILE ps1    # run Problem Statement 1
make -f MAKEFILE ps2    # run Problem Statement 2
make -f MAKEFILE all    # run both
```
(Equivalent: `python3 PS1/ps1_solution.py` and `python3 PS2/ps2_solution.py`.)

## Key results (Team 9, Dataset_09.csv)
- Mean exam score ≈ 63.1 (95% CI [57.2, 69.0]); statistically above the 60 benchmark (one-sample t, p < 0.05).
- Pass rate ≈ 79% (19/24). Attendance threshold at median (79%): High group pass rate 100% vs Low group 50%; mean exam ≈ 72.8 vs 49.6.
- Two-sample t-test on attendance groups: significant difference (p ≈ 0.0x), Cohen's d large.
- Stratified sampling (by Section or attendance group) best represents the data; normal model defensible for exam scores; binomial pass model is a rough approximation (constant-pi violated).
- All reflection questions (population/generalisation, sampling design & bias, CLT, MLE vs unbiased variance, estimator choice, Type I/II errors, CI vs test, ANOVA vs t-tests, chi-square expected counts, power) are answered in the reports.

## Notes
- `notes/instruction_extracted.txt` — full text extracted from the original .docx.
- `PS1/README.md`, `PS2/README.md`, `datasets/README.md` — per-folder notes.
