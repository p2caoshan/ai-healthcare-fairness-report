import numpy as np


def _with_intercept(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.column_stack([np.ones(len(x)), x])


def _least_squares(x, y):
    return np.linalg.lstsq(_with_intercept(x), np.asarray(y, dtype=float), rcond=None)[0]


def _predict(coef, x):
    return _with_intercept(x) @ coef


def _propensity(x, treatment):
    design = _with_intercept(x)
    beta = np.zeros(design.shape[1])
    y = np.asarray(treatment, dtype=float)
    for _ in range(30):
        eta = np.clip(design @ beta, -20, 20)
        p = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(p * (1.0 - p), 1e-6, None)
        working = eta + (y - p) / weight
        scale = np.sqrt(weight)
        beta = np.linalg.lstsq(design * scale[:, None], working * scale, rcond=None)[0]
    eta = np.clip(design @ beta, -20, 20)
    return np.clip(1.0 / (1.0 + np.exp(-eta)), 0.01, 0.99)


def iptw_ate(x, treatment, outcome):
    treatment = np.asarray(treatment, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    ps = _propensity(x, treatment)
    treated = treatment / ps
    control = (1.0 - treatment) / (1.0 - ps)
    return float(np.sum(treated * outcome) / np.sum(treated) - np.sum(control * outcome) / np.sum(control))


def aipw_ate(x, treatment, outcome):
    treatment = np.asarray(treatment, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    x = np.asarray(x, dtype=float)
    ps = _propensity(x, treatment)
    treated = treatment == 1
    mu1 = _predict(_least_squares(x[treated], outcome[treated]), x)
    mu0 = _predict(_least_squares(x[~treated], outcome[~treated]), x)
    score = mu1 - mu0
    score += treatment * (outcome - mu1) / ps
    score -= (1.0 - treatment) * (outcome - mu0) / (1.0 - ps)
    return float(np.mean(score))


def g_formula_ate(x, treatment, outcome):
    treatment = np.asarray(treatment, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    x = np.asarray(x, dtype=float)
    coef = _least_squares(np.column_stack([treatment, x]), outcome)
    ones = np.ones(len(treatment))
    zeros = np.zeros(len(treatment))
    y1 = _predict(coef, np.column_stack([ones, x]))
    y0 = _predict(coef, np.column_stack([zeros, x]))
    return float(np.mean(y1 - y0))


def two_period_g_formula(baseline, a0, later, a1, outcome):
    baseline = np.asarray(baseline, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    later = np.asarray(later, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    later_coef = _least_squares(np.column_stack([a0, baseline]), later)
    outcome_coef = _least_squares(np.column_stack([a0, a1, later]), outcome)

    def mean_under(value):
        assigned = np.full(len(baseline), value)
        later_cf = _predict(later_coef, np.column_stack([assigned, baseline]))
        y_cf = _predict(outcome_coef, np.column_stack([assigned, assigned, later_cf]))
        return float(np.mean(y_cf))

    return mean_under(1.0) - mean_under(0.0)


def two_stage_least_squares(instrument, treatment, outcome, covariates=None):
    instrument = np.asarray(instrument, dtype=float).reshape(-1, 1)
    treatment = np.asarray(treatment, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if covariates is None:
        first_design = instrument
        second_tail = None
    else:
        covariates = np.asarray(covariates, dtype=float)
        first_design = np.column_stack([instrument, covariates])
        second_tail = covariates
    fitted = _predict(_least_squares(first_design, treatment), first_design)
    if second_tail is None:
        second_design = fitted.reshape(-1, 1)
    else:
        second_design = np.column_stack([fitted, second_tail])
    return float(_least_squares(second_design, outcome)[1])


def wald_ratio(genotype, exposure, outcome):
    genotype = np.asarray(genotype, dtype=float).reshape(-1, 1)
    exposure_coef = _least_squares(genotype, exposure)[1]
    outcome_coef = _least_squares(genotype, outcome)[1]
    return float(outcome_coef / exposure_coef)


def segmented_regression(time, outcome, interruption):
    time = np.asarray(time, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    interruption = np.asarray(interruption, dtype=float)
    start = time[interruption == 1][0]
    since = (time - start) * interruption
    design = np.column_stack([np.ones(len(time)), time, interruption, since])
    coef, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    return {"level_change": float(coef[2]), "slope_change": float(coef[3])}


def demo():
    rng = np.random.default_rng(7)
    n = 6000
    covariates = rng.normal(size=(n, 2))
    logits = -0.1 + 0.9 * covariates[:, 0] - 0.5 * covariates[:, 1]
    propensity = 1.0 / (1.0 + np.exp(-logits))
    treatment = rng.binomial(1, propensity)
    outcome = 1.5 * treatment + 1.1 * covariates[:, 0] + rng.normal(size=n)
    print("iptw", round(iptw_ate(covariates, treatment, outcome), 3))
    print("aipw", round(aipw_ate(covariates, treatment, outcome), 3))
    print("g_formula", round(g_formula_ate(covariates, treatment, outcome), 3))

    baseline = rng.normal(size=n)
    a0 = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.4 * baseline)))
    later = 0.6 * baseline - 0.8 * a0 + rng.normal(size=n)
    a1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.3 * later + 0.5 * a0))))
    y_time = 1.2 * a0 + 0.8 * a1 + 0.7 * later + rng.normal(size=n)
    print("two_period_g", round(two_period_g_formula(baseline, a0, later, a1, y_time), 3))

    confounder = rng.normal(size=n)
    instrument = rng.binomial(1, 0.5, size=n)
    treated = ( -0.2 + 1.1 * instrument + 0.35 * confounder + rng.normal(scale=0.25, size=n) > 0.5).astype(float)
    y_iv = 2.0 * treated + 1.2 * confounder + rng.normal(size=n)
    print("iv", round(two_stage_least_squares(instrument, treated, y_iv), 3))

    genotype = rng.binomial(2, 0.3, size=n)
    exposure = 0.4 * genotype + rng.normal(size=n)
    y_mr = 1.7 * exposure + rng.normal(size=n)
    print("wald", round(wald_ratio(genotype, exposure, y_mr), 3))

    time = np.arange(80, dtype=float)
    post = (time >= 40).astype(float)
    series = 10 + 0.05 * time + 3.0 * post - 0.08 * (time - 40) * post
    series += rng.normal(scale=0.4, size=80)
    fit = segmented_regression(time, series, post)
    print("its_level", round(fit["level_change"], 3))
    print("its_slope", round(fit["slope_change"], 3))


if __name__ == "__main__":
    demo()
