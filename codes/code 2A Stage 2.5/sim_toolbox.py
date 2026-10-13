"""
sim_toolbox.py  --  Part 2A simulation toolbox
Missing Data Analysis project  |  Author: Purbasa Bhaduri

Everything the Part 2A severity simulation needs, in one importable module:
data generator, missingness generators, seven missing-data methods + a full-data
reference, bootstrap / Rubin inference, a one-replicate engine and a metrics summary.

VERSION 2 (Stage 2.5): identical simulation protocol and random streams as v1, but every row now
also stores the standard error and 95% interval of ALL FOUR coefficients (columns se_b0..se_b3,
lo_b0..lo_b3, hi_b0..hi_b3). The v1 columns se_b1, ci_lo, ci_hi are kept (= the b1 values).
Speed-up: EM and Regression Imputation reuse the Complete Case bootstrap (they are mathematically
identical when only Y is missing; v1 verified this to 1.6e-07). Set REUSE_TRIO = False to run all three.

Design in one line: every replicate draws ONE complete dataset, deletes Y under every
mechanism and severity level, runs every method, and records the results in a tidy table.
"""
import numpy as np
import pandas as pd
from scipy import stats

# =====================================================================
# PROTOCOL CONSTANTS (the fixed design of Part 2A)
# =====================================================================
MASTER_SEED = 42                                   # same seed family as Part 1
N = 100                                            # sample size
BETA_TRUE = np.array([5.0, 2.0, -1.5, 1.0])        # beta0, beta1, beta2, beta3
TRUE = {"b0": 5.0, "b1": 2.0, "b2": -1.5, "b3": 1.0}
SIGMA = 1.0                                        # error SD
SEVERITY_LEVELS = [10, 20, 30, 40, 50]             # % of Y deleted

# Missingness mechanisms. "slope" = how strongly deletion depends on the driver
# (standardised X1 for MAR, standardised Y for MNAR). Every extra SD of the driver
# multiplies a row's deletion weight by exp(slope).
MECH_SPECS = {
    "MCAR":      {"kind": "mcar", "slope": 0.0},
    "MAR":       {"kind": "mar",  "slope": 1.0},    # low X1  -> more likely to lose Y
    "MNAR":      {"kind": "mnar", "slope": 1.0},    # high Y  -> more likely to lose Y  (main run)
    "MNAR_mild": {"kind": "mnar", "slope": 0.5},    # Part 1's strength (sensitivity run)
}
MAIN_MECHANISMS = ["MCAR", "MAR", "MNAR"]

METHODS = ["Complete Case", "Mean Imputation", "Hot-Deck", "Regression Imputation",
           "Stochastic Regression", "MICE", "EM"]
REFERENCE = "Full Data"                            # analysis of the data BEFORE deletion

MICE_M = 20                                        # number of imputations
PMM_K = 5                                          # PMM donor pool size
EM_TOL = 1e-8                                      # EM stopping rule (as in Part 1)
EM_MAXIT = 1000
TOOLBOX_VERSION = 2
REUSE_TRIO = True                                  # EM / Regression Imputation reuse the Complete Case bootstrap
COEFS = ["b0", "b1", "b2", "b3"]

# =====================================================================
# SECTION 2 -- DATA GENERATOR
# =====================================================================
def make_rng(*keys):
    """Independent, reproducible random stream identified by a tuple of integers."""
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in keys]))


def generate_complete_data(rng, n=N):
    """One complete dataset from  Y = 5 + 2*X1 - 1.5*X2 + X3 + e,  e ~ N(0, 1)."""
    X = rng.standard_normal((n, 3))
    eps = rng.normal(0.0, SIGMA, size=n)
    y = BETA_TRUE[0] + X @ BETA_TRUE[1:] + eps
    return X, y


def design(X):
    """Add the intercept column:  [1, X1, X2, X3]."""
    return np.column_stack([np.ones(len(X)), X])


# ---------- ordinary least squares helpers (used throughout) ----------
def ols_beta(Xd, y):
    """OLS coefficients (fast normal-equations solve, lstsq fallback)."""
    try:
        return np.linalg.solve(Xd.T @ Xd, Xd.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(Xd, y, rcond=None)[0]


def ols(Xd, y):
    """OLS with covariance matrix. Returns beta, cov(beta), sigma2_hat, residuals."""
    beta = ols_beta(Xd, y)
    resid = y - Xd @ beta
    dfree = len(y) - Xd.shape[1]
    s2 = float(resid @ resid) / dfree
    cov = s2 * np.linalg.inv(Xd.T @ Xd)
    return beta, cov, s2, resid

# =====================================================================
# SECTION 3 -- MISSINGNESS GENERATORS  (exact severity, three mechanisms)
# =====================================================================
def n_missing_for(level, n=N):
    """Number of Y values to delete for a severity level given in percent."""
    return int(round(n * level / 100.0))


def zscore(v):
    return (v - v.mean()) / v.std(ddof=1)


def draw_deletion_noise(rng, n=N):
    """Gumbel noise, one value per row. Shared by every mechanism and severity level
    inside a replicate (common random numbers), which makes the deleted sets NESTED:
    the 10% set sits inside the 20% set, and so on."""
    return rng.gumbel(size=n)


def selection_scores(X, y, mechanism, noise):
    """Score = driver term + Gumbel noise. The rows with the HIGHEST scores are deleted.
    Taking the top-k of (log-weight + Gumbel noise) is exactly sampling k rows without
    replacement with probabilities proportional to exp(log-weight)."""
    spec = MECH_SPECS[mechanism]
    if spec["kind"] == "mcar":
        eta = 0.0                                   # no driver: pure luck
    elif spec["kind"] == "mar":
        eta = -spec["slope"] * zscore(X[:, 0])      # depends on OBSERVED X1 (low X1 -> deleted)
    elif spec["kind"] == "mnar":
        eta = spec["slope"] * zscore(y)             # depends on the HIDDEN Y (high Y -> deleted)
    else:
        raise ValueError(mechanism)
    return eta + noise


def make_missing_mask(X, y, mechanism, n_missing, noise):
    """Boolean mask, True where Y is deleted. Exactly `n_missing` Trues."""
    mask = np.zeros(len(y), dtype=bool)
    if n_missing > 0:
        order = np.argsort(-selection_scores(X, y, mechanism, noise), kind="stable")
        mask[order[:n_missing]] = True
    return mask

# =====================================================================
# SECTION 4 -- METHOD TOOLBOX
# Every method takes (Xd, y, rng): Xd = design matrix WITH intercept, y = outcome with NaN
# for missing values. Single-imputation methods and EM return (beta_vector, extra).
# =====================================================================
# ---------- imputers: return a COMPLETED copy of y ----------
def impute_mean(Xd, y, rng=None):
    out = y.copy()
    miss = np.isnan(out)
    if miss.any():
        out[miss] = out[~miss].mean()
    return out


def impute_hotdeck(Xd, y, rng=None):
    """Deterministic nearest-donor hot-deck: copy the actual Y of the observed row whose
    (X1, X2, X3) is closest in squared Euclidean distance (same rule as Part 1)."""
    out = y.copy()
    miss = np.isnan(out)
    if not miss.any():
        return out
    obs = ~miss
    d2 = ((Xd[miss, 1:][:, None, :] - Xd[obs, 1:][None, :, :]) ** 2).sum(axis=2)
    out[miss] = y[obs][d2.argmin(axis=1)]
    return out


def impute_regression(Xd, y, rng=None):
    """Fit Y~X on observed rows, fill holes with the predictions (points on the plane)."""
    out = y.copy()
    miss = np.isnan(out)
    if not miss.any():
        return out
    beta = ols_beta(Xd[~miss], y[~miss])
    out[miss] = Xd[miss] @ beta
    return out


def impute_stochastic(Xd, y, rng):
    """Regression imputation + a residual resampled (with replacement) from the observed fit."""
    out = y.copy()
    miss = np.isnan(out)
    if not miss.any():
        return out
    beta = ols_beta(Xd[~miss], y[~miss])
    resid = y[~miss] - Xd[~miss] @ beta
    out[miss] = Xd[miss] @ beta + rng.choice(resid, size=int(miss.sum()), replace=True)
    return out


def pmm_complete(Xd, y, rng, k=PMM_K):
    """ONE proper PMM imputation (Rubin / van Buuren 'type-1' matching):
       1. fit Y~X on observed rows;  2. DRAW sigma^2 and beta from their posterior
       (this is what makes the imputation 'proper');  3. for each missing row, find the k
       observed rows whose fitted value is closest to the row's DRAWN prediction;
       4. copy the actual Y of one of those k donors, chosen at random."""
    out = y.copy()
    miss = np.isnan(out)
    n_mis = int(miss.sum())
    if n_mis == 0:
        return out
    obs = ~miss
    Xo, yo, Xm = Xd[obs], y[obs], Xd[miss]
    p = Xd.shape[1]
    beta_hat = ols_beta(Xo, yo)
    resid = yo - Xo @ beta_hat
    rss = float(resid @ resid)
    dfree = len(yo) - p
    L = np.linalg.cholesky(np.linalg.inv(Xo.T @ Xo))
    sigma_star = np.sqrt(rss / rng.chisquare(dfree))
    beta_star = beta_hat + sigma_star * (L @ rng.standard_normal(p))
    k_eff = min(k, len(yo))
    dist = np.abs((Xm @ beta_star)[:, None] - (Xo @ beta_hat)[None, :])
    nearest = np.argpartition(dist, k_eff - 1, axis=1)[:, :k_eff]
    pick = nearest[np.arange(n_mis), rng.integers(0, k_eff, size=n_mis)]
    out[miss] = yo[pick]
    return out


# ---------- EM for the normal linear model with Y missing ----------
def em_regression(Xd, y, tol=EM_TOL, max_iter=EM_MAXIT, return_path=False):
    """EM for  Y | X ~ N(X beta, sigma^2)  with missing Y (ignorable missingness).
       E-step: E[y_i] = x_i'beta,  E[y_i^2] = (x_i'beta)^2 + sigma^2   for missing rows.
       M-step: beta = (X'X)^-1 X' E[y],   sigma^2 = mean of E[(y_i - x_i'beta_new)^2].
       Starts from beta = 0, sigma^2 = 1; stops when no parameter moves by more than tol."""
    n, p = Xd.shape
    miss = np.isnan(y)
    obs = ~miss
    n_mis = int(miss.sum())
    P = np.linalg.solve(Xd.T @ Xd, Xd.T)
    beta = np.zeros(p)
    sigma2 = 1.0
    path = []
    n_iter = 0
    for it in range(1, max_iter + 1):
        mu = Xd @ beta
        y_exp = np.where(miss, mu, y)                              # E-step
        beta_new = P @ y_exp                                       # M-step (beta)
        fit_new = Xd @ beta_new
        r_obs = y[obs] - fit_new[obs]
        sigma2_new = (float(r_obs @ r_obs)
                      + float(((mu[miss] - fit_new[miss]) ** 2).sum()) + n_mis * sigma2) / n
        delta = max(np.max(np.abs(beta_new - beta)), abs(sigma2_new - sigma2))
        beta, sigma2, n_iter = beta_new, sigma2_new, it
        if return_path:
            path.append(beta.copy())
        if delta < tol:
            break
    if return_path:
        return beta, sigma2, n_iter, np.array(path)
    return beta, sigma2, n_iter


# ---------- the seven methods in the common (Xd, y, rng) -> (beta, extra) form ----------
def m_complete_case(Xd, y, rng=None):
    obs = ~np.isnan(y)
    return ols_beta(Xd[obs], y[obs]), None


def m_mean(Xd, y, rng=None):
    return ols_beta(Xd, impute_mean(Xd, y)), None


def m_hotdeck(Xd, y, rng=None):
    return ols_beta(Xd, impute_hotdeck(Xd, y)), None


def m_regression(Xd, y, rng=None):
    return ols_beta(Xd, impute_regression(Xd, y)), None


def m_stochastic(Xd, y, rng):
    return ols_beta(Xd, impute_stochastic(Xd, y, rng)), None


def m_em(Xd, y, rng=None):
    beta, sigma2, n_iter = em_regression(Xd, y)
    return beta, n_iter


# MICE is handled separately (it is pooled by Rubin's rules, not bootstrapped)
METHOD_FUNCS = {
    "Complete Case": m_complete_case,
    "Mean Imputation": m_mean,
    "Hot-Deck": m_hotdeck,
    "Regression Imputation": m_regression,
    "Stochastic Regression": m_stochastic,
    "EM": m_em,
}

# =====================================================================
# SECTION 5 -- INFERENCE: Rubin's rules (MICE), bootstrap (everything else)
# =====================================================================
def rubin_pool(betas, variances, n_complete, n_params=4, df_method="barnard-rubin"):
    """Rubin's rules for every coefficient at once.
       betas, variances : arrays (m, p) of per-imputation estimates and their variances.
       df_method        : 'classical' (what Part 1 used) or 'barnard-rubin' (small-sample
                          adjusted, what R's mice uses; better with n = 100)."""
    betas = np.atleast_2d(betas)
    variances = np.atleast_2d(variances)
    m = betas.shape[0]
    qbar = betas.mean(axis=0)
    W = variances.mean(axis=0)
    B = betas.var(axis=0, ddof=1)
    T = W + (1.0 + 1.0 / m) * B
    lam = (1.0 + 1.0 / m) * B / T                       # fraction of variance due to missingness
    with np.errstate(divide="ignore", invalid="ignore"):
        df_old = (m - 1) / lam ** 2
        df_com = n_complete - n_params
        df_obs = (df_com + 1.0) / (df_com + 3.0) * df_com * (1.0 - lam)
        df_bar = df_old * df_obs / (df_old + df_obs)
    if df_method == "classical":
        df = df_old
    elif df_method == "barnard-rubin":
        df = np.where(lam > 0, df_bar, float(df_com))
    else:
        raise ValueError(df_method)
    return {"q": qbar, "se": np.sqrt(T), "df": df, "W": W, "B": B, "T": T, "lambda": lam}


def mice_pmm(Xd, y, rng, m=MICE_M, k=PMM_K, df_method="barnard-rubin"):
    """MICE with predictive mean matching: m proper PMM imputations, OLS on each completed
    dataset, pooled with Rubin's rules. (With only Y incomplete one cycle is enough -- there
    is nothing to iterate over -- so each imputation is a single proper PMM draw.)"""
    betas, variances = [], []
    for _ in range(m):
        yc = pmm_complete(Xd, y, rng, k=k)
        b, cov, _, _ = ols(Xd, yc)
        betas.append(b)
        variances.append(np.diag(cov))
    return rubin_pool(np.array(betas), np.array(variances), n_complete=len(y), df_method=df_method)


def bootstrap_all(method_fn, Xd, y, idx_matrix, rng, min_obs=10):
    """Part 1 bootstrap scheme (resample ALL rows, re-run the WHOLE method inside every resample),
    now keeping ALL FOUR coefficients. Returns arrays se (4,), lo (4,), hi (4,) and the number of
    usable resamples. Random-number consumption is identical to v1's bootstrap_beta1."""
    draws = []
    for idx in idx_matrix:
        yb = y[idx]
        if np.isfinite(yb).sum() < min_obs:
            continue
        beta, _ = method_fn(Xd[idx], yb, rng)
        draws.append(np.asarray(beta, float))
    draws = np.asarray(draws)
    return (draws.std(axis=0, ddof=1), np.percentile(draws, 2.5, axis=0),
            np.percentile(draws, 97.5, axis=0), len(draws))


def bootstrap_beta1(method_fn, Xd, y, idx_matrix, rng, min_obs=10):
    """v1 interface (beta1 only), kept for backward compatibility."""
    se, lo, hi, n = bootstrap_all(method_fn, Xd, y, idx_matrix, rng, min_obs)
    return float(se[1]), float(lo[1]), float(hi[1]), n


def _se_col(coef):
    return "se_b1" if coef == "b1" else f"se_{coef}"


def _ci_cols(coef):
    return ("ci_lo", "ci_hi") if coef == "b1" else (f"lo_{coef}", f"hi_{coef}")


def summarise_results(res, coef="b1"):
    """Performance measures per (mechanism, severity, method) for ONE coefficient, each with
    Monte Carlo error: bias, empirical SD, RMSE, mean estimated SE, SE ratio (calibration),
    95% coverage, mean interval width, and the share of intervals missing low / high."""
    truth = TRUE[coef]
    sec = _se_col(coef); loc, hic = _ci_cols(coef)
    rows = []
    for (mech, sev, meth), g in res.groupby(["mechanism", "severity", "method"], sort=False):
        est = g[coef].to_numpy()
        R = len(est)
        err = est - truth
        bias = err.mean()
        sd = est.std(ddof=1) if R > 1 else np.nan
        rmse = np.sqrt((err ** 2).mean())
        avg_se = np.sqrt((g[sec] ** 2).mean())
        lo, hi = g[loc].to_numpy(), g[hic].to_numpy()
        covered = (lo <= truth) & (truth <= hi)
        cov = covered.mean()
        mse_se = (err ** 2).std(ddof=1) / np.sqrt(R) if R > 1 else np.nan
        pl, ph = (hi < truth).mean(), (lo > truth).mean()
        rows.append({
            "mechanism": mech, "severity": sev, "method": meth, "coef": coef, "R": R,
            "bias": bias, "mc_se_bias": sd / np.sqrt(R) if R > 1 else np.nan,
            "emp_sd": sd, "rmse": rmse,
            "mc_se_rmse": mse_se / (2 * rmse) if rmse > 0 else np.nan,
            "avg_se": avg_se, "se_ratio": avg_se / sd if sd and sd > 0 else np.nan,
            "coverage": cov, "mc_se_cov": np.sqrt(cov * (1 - cov) / R),
            "width": float((hi - lo).mean()),
            "miss_low": pl, "miss_high": ph, "mc_se_miss": np.sqrt(max(pl, ph) * (1 - max(pl, ph)) / R),
        })
    return pd.DataFrame(rows)


def summarise_all(res):
    """summarise_results for all four coefficients, stacked (column 'coef')."""
    return pd.concat([summarise_results(res, c) for c in COEFS], ignore_index=True)

# =====================================================================
# SECTION 6 -- ONE-REPLICATE ENGINE
# =====================================================================
def _row(rep, mech, sev, nmiss, method, beta, se, lo, hi, em_iters=np.nan, mice_df=np.nan):
    """se, lo, hi: arrays of length 4 (one entry per coefficient)."""
    r = {"rep": rep, "mechanism": mech, "severity": sev, "n_missing": nmiss, "method": method,
         "b0": beta[0], "b1": beta[1], "b2": beta[2], "b3": beta[3],
         "se_b1": se[1], "ci_lo": lo[1], "ci_hi": hi[1]}               # v1 columns (beta1)
    for k, c in enumerate(COEFS):
        r[f"se_{c}"] = se[k]; r[f"lo_{c}"] = lo[k]; r[f"hi_{c}"] = hi[k]
    r["em_iters"] = em_iters; r["mice_df"] = mice_df
    return r


def run_replicate(rep, mechanisms=MAIN_MECHANISMS, levels=SEVERITY_LEVELS, boot_B=200,
                  mice_m=MICE_M, master_seed=MASTER_SEED, n=N, reuse_trio=None):
    """One full replicate: draw a complete dataset, then for every mechanism x severity level
    delete Y, run every method and record estimates, SEs and intervals for all four coefficients.
    Returns a list of row-dicts.

    Reproducibility: every random stream is identified by (seed, replicate, purpose, ...), so a
    replicate gives identical output however many cores are used or in which order they run."""
    reuse = REUSE_TRIO if reuse_trio is None else reuse_trio
    X, y_full = generate_complete_data(make_rng(master_seed, rep, 0), n)
    Xd = design(X)
    noise = draw_deletion_noise(make_rng(master_seed, rep, 1), n)
    p = Xd.shape[1]
    rows = []

    # reference: analysis of the complete data (analytic t-intervals, all four coefficients)
    beta, cov, _, _ = ols(Xd, y_full)
    se = np.sqrt(np.diag(cov))
    tcrit = stats.t.ppf(0.975, n - p)
    rows.append(_row(rep, "NONE", 0, 0, REFERENCE, beta, se, beta - tcrit * se, beta + tcrit * se))

    spec_names = list(MECH_SPECS)
    for mech in mechanisms:
        for li, level in enumerate(levels):
            k = n_missing_for(level, n)
            mask = make_missing_mask(X, y_full, mech, k, noise)
            y = y_full.copy()
            y[mask] = np.nan
            cid = spec_names.index(mech) * 100 + li               # condition id
            idx_matrix = make_rng(master_seed, rep, 2, cid).integers(0, n, size=(boot_B, n))
            cc_boot = None                                         # Complete Case bootstrap (reused by the trio)
            for mi, meth in enumerate(METHODS):
                if meth == "MICE":
                    out = mice_pmm(Xd, y, make_rng(master_seed, rep, 3, cid, mi), m=mice_m)
                    q, s_, d = out["q"], out["se"], out["df"]
                    tc = stats.t.ppf(0.975, d)                     # one critical value per coefficient
                    rows.append(_row(rep, mech, level, k, meth, q, s_, q - tc * s_, q + tc * s_,
                                     mice_df=d[1]))
                    continue
                fn = METHOD_FUNCS[meth]
                beta, extra = fn(Xd, y, make_rng(master_seed, rep, 3, cid, mi))
                if reuse and meth in ("Regression Imputation", "EM") and cc_boot is not None:
                    se_, lo_, hi_ = cc_boot                        # identical to Complete Case (see docstring)
                else:
                    se_, lo_, hi_, _ = bootstrap_all(fn, Xd, y, idx_matrix,
                                                     make_rng(master_seed, rep, 4, cid, mi))
                    if meth == "Complete Case":
                        cc_boot = (se_, lo_, hi_)
                rows.append(_row(rep, mech, level, k, meth, beta, se_, lo_, hi_,
                                 em_iters=(extra if meth == "EM" else np.nan)))
    return rows
