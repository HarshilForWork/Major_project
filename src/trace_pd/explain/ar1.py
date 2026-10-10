"""AR(1)-SHAP, part 1 -- a model of how ONE patient moves from one visit to the next.

The briefing's transition model is  X_t | X_{t-1} ~ N(A X_{t-1} + b, Sigma).  Fitted
literally on a real clinical table it breaks in four places, so this module makes
four changes (design note: docs/ar1_shap.md section 2). Built on PPMI, now GENERIC --
every dataset-specific choice is detected from the data, not hard-coded, so the same
code runs on a different cohort (e.g. ADNI) with no changes:

  1. Feature roles.  STATIC features (e.g. sex, handedness) never change within a
     patient, and CLOCK features (e.g. age, years since diagnosis) advance in lock-step
     with elapsed time. Given the previous visit and the visit date, both are KNOWN --
     there is nothing to sample. Only the DYNAMIC features get a transition model. The
     current values of the static and clock features enter that model as known inputs.
     Which features are which is DETECTED statistically (detect_feature_roles, below),
     not hard-coded to any one dataset's column names.
  2. Discrete features.  Many clinical features are 0-4 items, counts or a yes/no flag.
     Each is mapped to a normal score by its rank (a Gaussian copula), modelled there,
     and mapped back through the empirical quantile function, so every sampled value is
     a value a real patient has had.
  3. Irregular gaps.  The carry-over A and the drift depend on the gap dt (interaction
     terms), and the residual spread is estimated separately for ~3, ~6 and >=9 month
     gaps: Sigma(dt) = D(dt) R D(dt), with one shared correlation matrix R.
  4. Missing data.  The previous value of a feature is carried forward from the last
     visit that recorded it (up to 12 months) with a "stale" flag; older or absent
     values fall back to the population mean. Missing targets are left out of the
     likelihood exactly (Gaussian marginalisation), never imputed.

Parameters are shrunk: ridge on the mean model (alpha chosen by patient-grouped CV) and
shrinkage of Sigma toward its diagonal (lambda chosen by held-out likelihood).
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

# Reference values for PPMI, kept only as a known-answer check (see
# tests/test_ar1_shap.py::test_detect_feature_roles_matches_ppmi_reference) and as an
# explicit override if a caller wants PPMI's exact roles without re-detecting them.
# Nothing in this module uses these by default any more -- TransitionModel calls
# detect_feature_roles() on whatever table it's given.
PPMI_STATIC = ["SEX", "HANDED"]
PPMI_CLOCK = ["AGE_AT_VISIT", "YRS_SINCE_DIAGNOSIS", "YRS_SINCE_SYMPTOM_ONSET"]

LOCF_MAX_MONTHS = 12
GAP_EDGES = (4.5, 7.5)                       # buckets: ~3 mo | ~6 mo | >= 9 mo
ALPHAS = (0.3, 3.0, 30.0, 300.0)
LAMBDAS = (0.0, 0.05, 0.15, 0.3, 0.5)

# Thresholds for detect_feature_roles(). Conservative on purpose: when a feature's
# behaviour is ambiguous, the safe default is DYNAMIC (it gets modelled and sampled),
# not STATIC or CLOCK (which would wrongly treat it as fixed/known and never sample
# it at all). Getting a dynamic feature wrong costs a little model quality; getting a
# static/clock feature wrong corrupts every explanation that touches it.
STATIC_MATCH_FRAC = 0.98     # a feature is static if >=98% of repeat-visit patients never change
CLOCK_R2 = 0.97              # within-patient linear-in-time fit must explain this much variance
MIN_PATIENTS_FOR_DETECTION = 10
MIN_POINTS_PER_PATIENT = 3   # for the clock check -- 2 points always fit a line exactly, so is uninformative


def fit_category_maps(df, cols):
    """Build a {column: {value: code}} mapping for every non-numeric column in `cols`,
    from `df` (meant to be called ONCE, on the full training data). Codes are assigned
    by sorted value, so they're reproducible given the same training data."""
    maps = {}
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            uniq = sorted(df[c].dropna().unique())
            maps[c] = {v: i for i, v in enumerate(uniq)}
    return maps


def numeric_matrix(df, cols, cat_maps=None):
    """Like df[cols].to_numpy(float), but a text/categorical column (e.g. a STATIC
    feature recorded as 'Male'/'Female' or 'Left'/'Right', rather than PPMI's numeric
    codes) is mapped to a numeric code instead of crashing.

    `cat_maps`, from fit_category_maps(), MUST be supplied and must have been built
    from the full training data, not from `df` itself: encoding a text column
    independently on every call (e.g. via a fresh pd.factorize) assigns codes relative
    to whatever categories happen to appear in THAT call's rows, which silently
    mis-aligns with a copula fitted on the full training distribution -- a single row
    containing only 'Right' would wrongly code it 0 even if training saw 'Left'=0,
    'Right'=1. A category absent from the training map becomes NaN, not an error.
    """
    cat_maps = cat_maps or {}
    out = np.empty((len(df), len(cols)), float)
    for j, c in enumerate(cols):
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            out[:, j] = pd.to_numeric(col, errors="coerce").to_numpy(float)
        elif c in cat_maps:
            out[:, j] = col.map(cat_maps[c]).astype(float).to_numpy()
        else:
            raise ValueError(
                f"column '{c}' is non-numeric and has no fitted category map -- "
                f"pass cat_maps from fit_category_maps(TRAINING_data, cols)")
    return out


def feature_roles_from_lists(features, static, clock):
    """Explicit override: split `features` using caller-supplied static/clock lists,
    instead of detecting them. Use this when the roles are already known (e.g. PPMI's
    PPMI_STATIC/PPMI_CLOCK above) or need to be forced for a specific experiment."""
    static, clock = set(static), set(clock)
    s = [f for f in features if f in static]
    c = [f for f in features if f in clock]
    d = [f for f in features if f not in static | clock]
    return s, c, d


def detect_feature_roles(visits, features, id_col="PATNO", time_col="VISIT_MONTH",
                          static_thresh=STATIC_MATCH_FRAC, clock_r2=CLOCK_R2,
                          min_patients=MIN_PATIENTS_FOR_DETECTION,
                          min_points=MIN_POINTS_PER_PATIENT):
    """Detect which features are STATIC, CLOCK, or DYNAMIC from the long, one-row-
    per-patient-visit table (not a pair table -- see the note on why, below). No
    dataset-specific knowledge -- this is what makes the method a framework rather
    than a PPMI script.

      STATIC   among patients with >= 2 recorded values of the feature, it is the SAME
               value every time for >= static_thresh of them. Works for numbers, text
               and categories alike (exact equality), so it catches both a numeric sex
               code and a genotype string.
      CLOCK    among patients with >= min_points recorded values, fit value ~ time_col
               separately WITHIN each patient (ordinary least squares); pool the
               leftover variance across all of them. A clock feature's value is
               explained almost perfectly (R2 >= clock_r2) by each patient's own
               straight line in time, because it's deterministic given the calendar
               (age, years-since-X).
      DYNAMIC  everything else -- gets the full AR(1) treatment.

    Why the FULL trajectory, not just consecutive-visit pairs: many real clinical
    tables round a clock feature to a coarse precision (e.g. age to one decimal
    year). Over a short gap, that rounding noise can be as large as the true change
    (3 months of age is 0.25 years -- the same size as the rounding step), so a
    consecutive-pair test wrongly calls a textbook clock feature "dynamic". Over a
    patient's full multi-year span the same rounding noise is negligible next to the
    true drift, so the within-patient linear fit used here is robust to it. Verified
    against PPMI's known clock features (see the test suite).

    Returns (static, clock, dynamic, report); report[feature] is a one-line string
    explaining the decision, meant to be printed so a new dataset's classification can
    be sanity-checked by eye before it's trusted. Ties are broken toward DYNAMIC: a
    feature wrongly left dynamic costs a little model quality; a feature wrongly
    declared static/clock is never sampled at all, which corrupts every explanation
    that touches it.
    """
    static, clock, dynamic, report = [], [], [], {}

    for f in features:
        sub = visits[[id_col, time_col, f]].dropna(subset=[f])
        sizes = sub.groupby(id_col)[f].transform("size")
        elig = sub[sizes >= 2]
        n_pat = elig[id_col].nunique()
        if n_pat < min_patients:
            dynamic.append(f)
            report[f] = f"dynamic (fewer than {min_patients} patients have a repeat value, n_patients={n_pat})"
            continue

        same_frac = float((elig.groupby(id_col)[f].nunique() == 1).mean())
        if same_frac >= static_thresh:
            static.append(f)
            report[f] = f"static ({same_frac * 100:.1f}% of {n_pat} repeat-visit patients never change)"
            continue

        try:
            pd.to_numeric(sub[f])
        except (ValueError, TypeError):
            dynamic.append(f)
            report[f] = f"dynamic (non-numeric, so not a clock; {same_frac * 100:.1f}% unchanged)"
            continue

        ss_res = ss_tot = 0.0
        n_used = 0
        for pid, g in sub.groupby(id_col):
            if len(g) < min_points:
                continue
            t = g[time_col].to_numpy(float)
            y = pd.to_numeric(g[f]).to_numpy(float)
            if np.ptp(t) == 0:
                continue
            slope, intercept = np.polyfit(t, y, 1)
            resid = y - (slope * t + intercept)
            ss_res += float(np.sum(resid ** 2))
            ss_tot += float(np.sum((y - y.mean()) ** 2))
            n_used += 1

        if n_used < min_patients or ss_tot <= 0:
            dynamic.append(f)
            report[f] = (f"dynamic (fewer than {min_patients} patients with >= {min_points} points "
                         f"for a clock check, n={n_used}; {same_frac * 100:.1f}% unchanged)")
            continue

        r2 = 1 - ss_res / ss_tot
        if r2 >= clock_r2:
            clock.append(f)
            report[f] = f"clock (within-patient linear-in-time R2={r2:.4f}, n_patients={n_used})"
        else:
            dynamic.append(f)
            report[f] = f"dynamic (within-patient linear-in-time R2={r2:.4f} < {clock_r2}, n_patients={n_used})"

    return static, clock, dynamic, report


# ------------------------------------------------------------------ copula transform

class CopulaTransform:
    """Rank -> normal score, and back through the empirical quantile function.

    to_normal uses the mid-rank, so tied values (a 0-4 item) map to one score;
    from_normal maps any real number back to a value present in the fitting data.
    Round trip is exact on observed values.
    """

    def fit(self, X):
        X = np.asarray(X, float)
        self.sorted_ = [np.sort(c[~np.isnan(c)]) for c in X.T]
        return self

    def to_normal(self, X):
        X = np.asarray(X, float)
        U = np.full(X.shape, np.nan)
        for j, s in enumerate(self.sorted_):
            v = X[:, j]; ok = ~np.isnan(v); n = len(s)
            lo = np.searchsorted(s, v[ok], "left"); hi = np.searchsorted(s, v[ok], "right")
            p = np.clip((lo + hi) / (2.0 * n), 0.5 / n, 1 - 0.5 / n)
            U[ok, j] = ndtri(p)
        return U

    def from_normal_col(self, u, j):
        s = self.sorted_[j]
        idx = np.clip(np.floor(ndtr(u) * len(s)).astype(int), 0, len(s) - 1)
        return s[idx]


# ------------------------------------------------------------------------ pair table

def build_pairs(df, features, id_col="PATNO", time_col="VISIT_MONTH", passthrough=("LABEL",)):
    """One row per visit that has a previous SCHEDULED visit of the same patient.

    `df` is the long table (any subset of patients, any dataset -- `id_col`/`time_col`
    name the patient-id and months-since-baseline columns; PPMI's names are the
    default). Returns a frame with `id_col`, `time_col`, DT (months since the previous
    scheduled visit), the current features, and PREV_<f> / STALE_<f> for every
    feature: PREV is the last recorded value at or before the previous visit (carried
    forward up to LOCF_MAX_MONTHS), STALE = 1 when it was carried forward or is
    missing. `passthrough` columns (e.g. the label) are copied through unchanged if
    present, for convenience in evaluation code -- they play no role in the model.
    """
    d = df.sort_values([id_col, time_col]).reset_index(drop=True)
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(d[f]) or d[f].dropna().empty]
    non_numeric_features = [f for f in features if f not in numeric_features]
    X = d[features].copy()
    if numeric_features:
        X[numeric_features] = X[numeric_features].apply(pd.to_numeric, errors="coerce")
    g = d.groupby(id_col)
    prev_time = g[time_col].shift(1)
    out = d[[id_col, time_col]].copy()
    out["DT"] = d[time_col] - prev_time
    for f in features:
        obs_time = d[time_col].where(X[f].notna())
        last_val = X[f].groupby(d[id_col]).ffill().groupby(d[id_col]).shift(1)
        last_t = obs_time.groupby(d[id_col]).ffill().groupby(d[id_col]).shift(1)
        too_old = (d[time_col] - last_t) - out["DT"] > LOCF_MAX_MONTHS
        prev = last_val.where(~too_old)
        out["PREV_" + f] = prev
        out["STALE_" + f] = ((X[f].groupby(d[id_col]).shift(1).isna()) | prev.isna()).astype(float)
        out[f] = X[f]
    for c in passthrough:
        if c in d:
            out[c] = d[c]
    return out[out["DT"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------- Gaussian utils

def nearest_psd(S, floor_frac=1e-3):
    S = (S + S.T) / 2
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, floor_frac * max(np.mean(np.diag(S)), 1e-8))
    return (V * w) @ V.T


def pairwise_cov(R, min_periods=30):
    C = pd.DataFrame(R).cov(min_periods=min_periods).to_numpy()
    d = np.nanvar(R, axis=0)
    C[np.isnan(C)] = 0.0
    np.fill_diagonal(C, d)
    return C


def shrink(S, lam):
    return (1 - lam) * S + lam * np.diag(np.diag(S))


def gauss_loglik_rows(resid, Sigmas):
    """Mean log-density per OBSERVED value. resid (n, p) with NaN = unobserved;
    Sigmas is one (p, p) matrix or a list of n matrices."""
    tot, cnt = 0.0, 0
    for i in range(len(resid)):
        o = ~np.isnan(resid[i])
        if not o.any():
            continue
        S = (Sigmas if isinstance(Sigmas, np.ndarray) and Sigmas.ndim == 2 else Sigmas[i])[np.ix_(o, o)]
        r = resid[i, o]
        sign, logdet = np.linalg.slogdet(S)
        tot += -0.5 * (r @ np.linalg.solve(S, r) + logdet + o.sum() * np.log(2 * np.pi))
        cnt += o.sum()
    return tot / max(cnt, 1)


def gap_bucket(dt):
    dt = np.asarray(dt, float)
    return np.digitize(dt, GAP_EDGES)                 # 0, 1, 2


# ------------------------------------------------------------------ transition model

class TransitionModel:
    """Next-visit model for the dynamic features, in normal-score space.

    variant:
      "population"  no history at all: mean and covariance of all visits
      "scalar"      each feature from its OWN previous value only, diagonal Sigma
      "full"        all previous values + current static/clock, shared Sigma
      "full_dt"     "full" + gap-dependent carry-over/drift + gap-dependent spread (default)
    """

    def __init__(self, features, variant="full_dt", alpha=None, lam=None, seed=0, roles=True,
                 static=None, clock=None, id_col="PATNO", time_col="VISIT_MONTH"):
        """
        roles=False         no feature is declared static or clock -- all are modelled, and
                             the model must LEARN that e.g. age advances with the calendar
                             (robustness check).
        roles=True (default) and static/clock not given: roles are DETECTED from the pair
                             table at fit() time (detect_feature_roles) -- works on any
                             dataset, nothing PPMI-specific.
        static=[...], clock=[...]   explicit override: use exactly these lists instead of
                             detecting (e.g. PPMI_STATIC/PPMI_CLOCK, or to force a role for
                             an experiment). Only meaningful when roles=True.
        id_col, time_col     the patient-id and months-since-baseline column names in
                             whatever table fit() is given; default to PPMI's names.
        """
        self.features = list(features)
        self.roles = roles
        self._static_override = static
        self._clock_override = clock
        self.id_col, self.time_col = id_col, time_col
        if roles and static is not None and clock is not None:
            self.static, self.clock, self.dynamic = feature_roles_from_lists(self.features, static, clock)
            self.role_report_ = {f: "explicit override" for f in self.features}
        elif not roles:
            self.static, self.clock, self.dynamic = [], [], list(self.features)
            self.role_report_ = {f: "dynamic (roles=False)" for f in self.features}
        else:
            self.static = self.clock = self.dynamic = self.role_report_ = None  # set in fit()
        self.known = (self.static + self.clock) if self.static is not None else None
        self.variant = variant
        self.alpha, self.lam, self.seed = alpha, lam, seed

    # -- design matrices ------------------------------------------------------------
    def _inputs(self, P):
        """Normal-score inputs from a pair frame."""
        U_prev = self.cop_dyn.to_normal(numeric_matrix(P, ["PREV_" + f for f in self.dynamic]))
        stale = P[["STALE_" + f for f in self.dynamic]].to_numpy(float)
        U_prev = np.where(np.isnan(U_prev), self.mu_pop, U_prev)     # absent -> population mean
        Z = self.cop_known.to_normal(numeric_matrix(P, self.known, self._known_cat_maps)) if self.known else np.zeros((len(P), 0))
        Z = np.where(np.isnan(Z), 0.0, Z)
        dt = P["DT"].to_numpy(float)[:, None] / 6.0
        return U_prev, stale, Z, dt

    def _design(self, U_prev, stale, Z, dt, j=None):
        if self.variant == "scalar":
            return np.column_stack([U_prev[:, j], stale[:, j]])
        cols = [U_prev, stale, Z]
        if self.variant == "full_dt":
            cols += [U_prev * dt, dt]
        return np.column_stack(cols)

    def _targets(self, P):
        return self.cop_dyn.to_normal(numeric_matrix(P, self.dynamic))

    # -- mean model ------------------------------------------------------------------
    def _fit_mean(self, inputs, U, alpha):
        models = []
        for j in range(U.shape[1]):
            ok = ~np.isnan(U[:, j])
            Xd = self._design(*inputs, j=j)
            models.append(Ridge(alpha=alpha).fit(Xd[ok], U[ok, j]))
        return models

    def _predict_mean(self, models, inputs):
        if self.variant == "population":
            return np.tile(self.mu_t, (len(inputs[0]), 1))
        return np.column_stack([m.predict(self._design(*inputs, j=j)) for j, m in enumerate(models)])

    # -- covariance -------------------------------------------------------------------
    def _fit_cov(self, resid, dt, lam):
        S = pairwise_cov(resid)
        if self.variant == "scalar":
            S = np.diag(np.diag(S))
        S = nearest_psd(shrink(S, lam))
        cov = {"pooled": S}
        if self.variant == "full_dt":
            sd_pool = np.sqrt(np.diag(S)); Rm = S / np.outer(sd_pool, sd_pool)
            b = gap_bucket(dt)
            for k in range(len(GAP_EDGES) + 1):
                rk = resid[b == k]
                cnt = (~np.isnan(rk)).sum(0)
                sd = np.where(cnt >= 30, np.sqrt(np.nanmean(rk ** 2, axis=0)) if len(rk) else sd_pool, sd_pool)
                cov[k] = nearest_psd(Rm * np.outer(sd, sd))
        return cov

    def sigma_for(self, dt):
        if self.variant != "full_dt":
            return self.cov_["pooled"]
        return self.cov_[int(gap_bucket([dt])[0])]

    # -- fit ---------------------------------------------------------------------------
    def fit(self, P, visits):
        """P: pair frame (dev patients only, from build_pairs). visits: per-visit frame
        used for the copula and the population model (dev patients only)."""
        if self.static is None:  # roles=True, no explicit override -> detect from the full trajectory
            self.static, self.clock, self.dynamic, self.role_report_ = detect_feature_roles(
                visits, self.features, id_col=self.id_col, time_col=self.time_col)
            self.known = self.static + self.clock
        self.patients_seen_ = set(P[self.id_col].unique()) | set(visits[self.id_col].unique())
        self.cop_dyn = CopulaTransform().fit(numeric_matrix(visits, self.dynamic))
        self._known_cat_maps = fit_category_maps(visits, self.known) if self.known else {}
        self.cop_known = CopulaTransform().fit(numeric_matrix(visits, self.known, self._known_cat_maps)) if self.known else None
        U_all = self.cop_dyn.to_normal(numeric_matrix(visits, self.dynamic))
        self.mu_pop = np.nanmean(U_all, axis=0)
        U = self._targets(P)
        self.mu_t = np.nanmean(U, axis=0)
        inputs = self._inputs(P)
        groups = P[self.id_col].to_numpy()

        if self.variant == "population":
            self.alpha_ = None; self.models_ = None
            resid = U - self.mu_t
            self.lam_ = self.lam if self.lam is not None else 0.0
            self.cov_ = {"pooled": nearest_psd(shrink(pairwise_cov(resid), self.lam_))}
            return self

        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(U, groups=groups))

        # alpha: held-out squared error on observed targets
        if self.alpha is None:
            best = None
            for a in ALPHAS:
                err = 0.0
                for tr, te in folds:
                    sub = lambda X, ix: tuple(x[ix] for x in X)
                    ms = self._fit_mean(sub(inputs, tr), U[tr], a)
                    r = U[te] - self._predict_mean(ms, sub(inputs, te))
                    err += np.nansum(r ** 2)
                if best is None or err < best[0]:
                    best = (err, a)
            self.alpha_ = best[1]
        else:
            self.alpha_ = self.alpha

        # lambda: held-out log-likelihood
        if self.lam is None:
            scores = {lam: 0.0 for lam in LAMBDAS}
            for tr, te in folds:
                sub = lambda X, ix: tuple(x[ix] for x in X)
                ms = self._fit_mean(sub(inputs, tr), U[tr], self.alpha_)
                r_tr = U[tr] - self._predict_mean(ms, sub(inputs, tr))
                r_te = U[te] - self._predict_mean(ms, sub(inputs, te))
                dt_te = P.DT.to_numpy()[te]
                for lam in LAMBDAS:
                    cov = self._fit_cov(r_tr, P.DT.to_numpy()[tr], lam)
                    self.cov_ = cov
                    Sig = [self.sigma_for(x) for x in dt_te] if self.variant == "full_dt" else cov["pooled"]
                    scores[lam] += gauss_loglik_rows(r_te, Sig)
            self.lam_ = max(scores, key=scores.get)
        else:
            self.lam_ = self.lam

        self.models_ = self._fit_mean(inputs, U, self.alpha_)
        resid = U - self._predict_mean(self.models_, inputs)
        self.cov_ = self._fit_cov(resid, P.DT.to_numpy(), self.lam_)
        return self

    # -- use -----------------------------------------------------------------------------
    def predict_mean(self, P):
        return self._predict_mean(self.models_, self._inputs(P))

    def loglik(self, P):
        """Mean held-out log-density per observed dynamic value (normal-score space)."""
        r = self._targets(P) - self.predict_mean(P)
        Sig = [self.sigma_for(x) for x in P.DT.to_numpy()] if self.variant == "full_dt" else self.cov_["pooled"]
        return gauss_loglik_rows(r, Sig)

    def r2(self, P):
        """Per-feature share of next-visit variance explained (normal-score space)."""
        U = self._targets(P); r = U - self.predict_mean(P)
        return 1 - np.nanmean(r ** 2, 0) / np.nanvar(U, 0)
