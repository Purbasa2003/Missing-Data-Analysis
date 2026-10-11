# Missing Data Analysis

### Comparing missing-data handling methods for regression inference under MCAR, MAR and MNAR

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Purbasa2003/Missing-Data-Analysis/blob/main/Notebooks/Missing_Data_Part1_Replication.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Phase%201%20complete-brightgreen)

An independent research project in applied statistics, built for learning and for a data science portfolio. It asks one practical question:

> **When the response variable of a regression has missing values, how do different handling methods affect the estimated coefficients and their uncertainty, and how does the answer depend on *why* the data are missing?**

The project is organised in phases. **Phase 1 (this README)** is a controlled single-dataset comparison of seven methods under three missingness mechanisms. **Phase 2** will extend it into a Monte Carlo *severity analysis* (10%–50% missingness). See the [Roadmap](#roadmap).

![Beta1 estimates by method and mechanism](figures/beta1_estimates_by_mechanism.png)

---

## Table of contents
1. [Phase 1 at a glance](#phase-1-at-a-glance)
2. [Background: the three mechanisms](#background-the-three-missingness-mechanisms)
3. [Simulation design](#simulation-design)
4. [Methods compared](#methods-compared)
5. [Evaluation and inference](#evaluation-and-inference)
6. [Results](#results)
7. [Interpretation](#interpretation)
8. [Validation of the replication](#validation-of-the-replication)
9. [Limitations](#limitations)
10. [Repository structure](#repository-structure)
11. [How to reproduce](#how-to-reproduce)
12. [Roadmap](#roadmap)
13. [Acknowledgements and references](#acknowledgements-and-references)

---

## Phase 1 at a glance

| Item | Detail |
|---|---|
| Sample size | n = 100 |
| Data-generating model | $Y = 5 + 2X_1 - 1.5X_2 + X_3 + \varepsilon,\ \varepsilon \sim N(0,1)$ |
| True coefficients | $\beta_0 = 5,\ \beta_1 = 2,\ \beta_2 = -1.5,\ \beta_3 = 1$ |
| Incomplete variable | $Y$ only; $X_1, X_2, X_3$ always observed |
| Mechanisms | MCAR, MAR, MNAR (applied to the same complete dataset) |
| Methods | 7 (Complete Case, Mean, Hot-Deck, Regression, Stochastic Regression, MICE/PMM, EM) |
| Target parameter | $\beta_1$ (true value 2) |
| Uncertainty | 500-replicate bootstrap; Rubin's rules for MICE |
| Language / tools | Python, Google Colab (NumPy, pandas, SciPy, statsmodels, Matplotlib) |

---

## Background: the three missingness mechanisms

Let $R_i = 1$ if $Y_i$ is missing. The mechanisms differ in what $P(R_i = 1)$ is allowed to depend on.

| Mechanism | Meaning | Rule used in this project |
|---|---|---|
| **MCAR** (Missing Completely At Random) | Missingness is unrelated to any data, observed or not. | Every row has a flat 20% chance of losing $Y$. |
| **MAR** (Missing At Random) | Missingness depends only on *observed* data. | 30% chance if $X_1 < 0$, 10% chance if $X_1 \ge 0$. |
| **MNAR** (Missing Not At Random) | Missingness depends on the *unobserved* value itself. | $P(R=1) = \text{expit}(-1.3863 + 0.5\,Y^{*})$, where $Y^{*}$ is standardised $Y$. |

The focus is on **comparing methods within each mechanism**, not on ranking the mechanisms themselves.

---

## Simulation design

1. Generate a complete dataset from the model above (seeded, so it is exactly reproducible).
2. Fit OLS to the complete data as a **gold-standard baseline**.
3. Create three incomplete datasets by deleting $Y$ under MCAR, MAR and MNAR, each starting from the same complete data.
4. Give the *same* incomplete dataset to all seven methods within a mechanism, so methods are compared under identical conditions.
5. Record $\hat\beta_1$, bias, bootstrap SE and 95% interval for every method–mechanism pair.

---

## Methods compared

| # | Method | Idea | Implementation notes |
|---|---|---|---|
| 1 | **Complete Case** | Drop rows with missing $Y$; run OLS on the rest. | Bootstrap resamples all 100 rows, then deletes inside each resample. |
| 2 | **Mean Imputation** | Replace every missing $Y$ with the observed mean. | Deterministic; flattens the $X$–$Y$ relationship. |
| 3 | **Hot-Deck** | Copy the actual $Y$ of the most similar observed row. | Nearest donor by squared Euclidean distance on $(X_1, X_2, X_3)$; deterministic. |
| 4 | **Regression Imputation** | Fit $Y \sim X$ on observed rows, fill holes with predictions. | Imputed points lie on the fitted plane, removing residual scatter. |
| 5 | **Stochastic Regression** | Regression imputation plus a resampled residual. | Residuals drawn with replacement from the observed fit. |
| 6 | **MICE with PMM** | Several completed datasets via predictive mean matching, pooled by Rubin's rules. | m = 5 imputations, k = 5 donors; see [Limitations](#limitations). |
| 7 | **EM** | Iterate expected sufficient statistics (E-step) and parameter updates (M-step). | Starts from $\beta^{(0)} = 0$; stops when changes are below $10^{-8}$. |

**Rubin's rules (MICE).** With $m$ completed datasets giving estimates $\hat\beta_1^{(j)}$ and variances $U^{(j)}$:

$$\bar Q = \frac{1}{m}\sum_j \hat\beta_1^{(j)},\qquad W = \frac{1}{m}\sum_j U^{(j)},\qquad B = \frac{1}{m-1}\sum_j \big(\hat\beta_1^{(j)} - \bar Q\big)^2,\qquad T = W + \Big(1 + \frac{1}{m}\Big)B.$$

The pooled standard error is $\sqrt{T}$, and the 95% interval is a $t$-interval with Rubin's degrees of freedom.

---

## Evaluation and inference

| Quantity | Definition |
|---|---|
| **Estimate** | $\hat\beta_1$ from the final regression |
| **Bias** | $\hat\beta_1 - \beta_1$, with $\beta_1 = 2$ |
| **Bootstrap SE** | Standard deviation of $\hat\beta_1$ over 500 bootstrap replicates, with the *entire method* (imputation + regression) re-run inside each resample |
| **95% interval** | 2.5th and 97.5th percentiles of the bootstrap distribution (MICE: Rubin's $t$-interval) |

Bias, standard error and interval answer different questions, so each is reported separately.

---

## Results

Results below come from the dataset generated with `DATA_SEED = 2026`. In this realisation, **17 / 15 / 17** values of $Y$ were missing under MCAR / MAR / MNAR.

### Estimates of $\beta_1$ (bootstrap SE in parentheses; true value = 2)

| Method | MCAR | MAR | MNAR |
|---|---|---|---|
| Complete Case | 2.005 (0.122) | 1.985 (0.113) | 2.026 (0.121) |
| Mean Imputation | 1.590 (0.208) | 1.904 (0.125) | 1.587 (0.200) |
| Hot-Deck | 1.851 (0.149) | 1.983 (0.111) | 1.958 (0.131) |
| Regression Imputation | 2.005 (0.122) | 1.985 (0.113) | 2.026 (0.121) |
| Stochastic Regression | 2.022 (0.133) | 2.012 (0.113) | 2.072 (0.130) |
| MICE (PMM) | 1.954 (0.129) | 1.985 (0.095) | 1.907 (0.117) |
| EM | 2.005 (0.122) | 1.985 (0.113) | 2.026 (0.121) |

### Bias of $\hat\beta_1$

| Method | MCAR | MAR | MNAR |
|---|---|---|---|
| Complete Case | +0.005 | −0.015 | +0.026 |
| Mean Imputation | **−0.410** | −0.096 | **−0.413** |
| Hot-Deck | −0.149 | −0.017 | −0.042 |
| Regression Imputation | +0.005 | −0.015 | +0.026 |
| Stochastic Regression | +0.022 | +0.012 | +0.072 |
| MICE (PMM) | −0.046 | −0.015 | −0.093 |
| EM | +0.005 | −0.015 | +0.026 |

Full tables, including 95% intervals, are in [`results/master_results.csv`](results/master_results.csv). MICE's five individual imputation estimates and Rubin quantities are in [`results/mice_details.csv`](results/mice_details.csv).

### Figures

| | MCAR | MAR | MNAR |
|---|---|---|---|
| 95% intervals | [view](figures/beta1_ci_mcar.png) | [view](figures/beta1_ci_mar.png) | [view](figures/beta1_ci_mnar.png) |
| Absolute bias | [view](figures/beta1_absolute_bias_mcar.png) | [view](figures/beta1_absolute_bias_mar.png) | [view](figures/beta1_absolute_bias_mnar.png) |

Also: [overall comparison](figures/beta1_estimates_by_mechanism.png) · [EM convergence](figures/em_convergence_beta1.png)

---

## Interpretation

* **Mean imputation distorts the most.** Filling every hole with one constant weakens the $X_1$–$Y$ relationship and pulls $\hat\beta_1$ towards zero (bias around −0.41 under MCAR and MNAR).
* **Complete Case, Regression Imputation and EM coincide.** This is a consequence of the setup (only $Y$ is incomplete, predictors fully observed, and EM treats missingness as ignorable), not a general theorem. Regression imputation places imputed points exactly on the observed-data regression plane, so it cannot change the coefficients; EM converges to the same normal-model maximum-likelihood solution.
* **Hot-Deck and Stochastic Regression restore scatter** that regression imputation removes, at the price of extra variability in the estimate.
* **MICE propagates imputation uncertainty** through Rubin's rules, which is why its SE reflects both within- and between-imputation variation.
* **MNAR was not dramatic in this one draw.** Most methods showed small bias under MNAR here. That should *not* be read as "MNAR is harmless": MAR-based methods cannot correct a mechanism that depends on the hidden value, and a single realisation at moderate missingness is too weak to expose it. This is precisely the motivation for Phase 2.

---

## Validation of the replication

Phase 1 reproduces the design of an earlier project (see [Acknowledgements](#acknowledgements-and-references)). Because the source notebook was incomplete, the six non-complete-case methods were **re-implemented from the documented procedure** rather than copied, and then checked against the original published results using the original seed (`DATA_SEED = 42`):

* Dataset, missing counts (25 / 25 / 16), Complete Case, Mean, Hot-Deck, Regression, Stochastic Regression and EM: reproduced **exactly**, including bootstrap SEs, intervals and EM iteration counts (20 / 20 / 14).
* MICE under MCAR: reproduced exactly. MICE under MAR and MNAR: same method, but point estimates differ by roughly 0.01–0.02 because the original random seeds for those runs were not recorded.

Setting `DATA_SEED = 42` in the notebook triggers a built-in comparison table against the original numbers (Section 9 of the notebook).

---

## Limitations

These are deliberate to state, and they define the agenda for later phases.

1. **One dataset per mechanism.** Every number is a single draw. A single interval that misses the truth says nothing about long-run coverage; no coverage probability is estimated.
2. **Missingness is not controlled.** The MCAR/MAR/MNAR rules produce somewhat different overall missing rates (15–25%), so "mechanism" and "amount of missing data" are partly confounded.
3. **MICE is a simplified PMM demonstration.** Only $Y$ is incomplete, so the model never changes between cycles and the generator is re-seeded each cycle; the five cycles add nothing here. $m = 5$ is also small for applied work.
4. **EM does not model the missingness process**, so it is not an MNAR-specific method.
5. **Small sample (n = 100)** and a single data-generating model.
6. **Bootstrap SEs for single-imputation methods** describe sampling variability of the whole procedure but do not by themselves fix bias from a wrong missingness assumption.

---

## Repository structure

```text
Missing-Data-Analysis/
├── README.md
├── data/
│   ├── complete_data.csv        # full dataset, no missing values
│   ├── data_mcar.csv            # Y deleted under MCAR
│   ├── data_mar.csv             # Y deleted under MAR
│   └── data_mnar.csv            # Y deleted under MNAR
├── notebooks/
│   └── Missing_Data_Part1_Replication.ipynb
├── figures/
│   ├── beta1_estimates_by_mechanism.png
│   ├── beta1_ci_{mcar,mar,mnar}.png
│   ├── beta1_absolute_bias_{mcar,mar,mnar}.png
│   └── em_convergence_beta1.png
└── results/
    ├── master_results.csv
    ├── mice_details.csv
    └── point_estimates_all_coefficients.csv
```

---

## How to reproduce

**Option A: Google Colab (recommended)**
1. Click the *Open in Colab* badge at the top.
2. Run `Runtime → Run all` (about 1–3 minutes).
3. Outputs are written to `figures/` and `results/` inside the Colab session.

**Option B: locally**
```bash
git clone https://github.com/Purbasa2003/Missing-Data-Analysis.git
cd Missing-Data-Analysis
pip install numpy pandas scipy statsmodels matplotlib jupyter
jupyter notebook notebooks/Missing_Data_Part1_Replication.ipynb
```

**Reproducibility settings** (first code cell of the notebook):

| Setting | Default | Purpose |
|---|---|---|
| `DATA_SEED` | `2026` | Seed for the generated dataset. Use `42` to reproduce the original project's data. |
| `B` | `500` | Bootstrap replicates |
| `BOOT_SEED` | `123` | Seed for bootstrap resampling |

---

## Roadmap

- [x] **Phase 1:** single-dataset comparison of seven methods under MCAR, MAR, MNAR
- [ ] **Phase 2:** severity analysis at 10%, 20%, 30%, 40% and 50% missingness
  - calibrate each mechanism to hit the exact target missing rate
  - Monte Carlo simulation with many replications per condition
  - proper multi-cycle MICE
  - metrics: bias, RMSE, empirical coverage and interval width, with Monte Carlo error
- [ ] **Phase 3:** write-up, MNAR sensitivity analysis, and extensions

---

## Acknowledgements and references

Phase 1 follows the design of an earlier open-source project on missing data and statistical inference: **[add link to the original repository here]**. Phase 1 here is an independent re-implementation validated against its published results; later phases are original work.

* Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.
* Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.
* Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1–38.
* Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.
* van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). CRC Press.

---

**Author:** Purbasa Bhaduri · M.Sc. Statistics, St. Xavier's University, Kolkata
