import numpy as np
from scipy.stats import chi2
from scipy.linalg import solve_triangular


def olivehawkins_robustcov(
    X,
    *,
    outlier_fraction=0.5,
    num_trials=None,
    reweighting_method="rfch",
    num_concentration_steps=10,
    start_method=None,   # None, string, callable, or list/tuple of those
    random_state=None,
):
    """
    MATLAB-like robustcov(..., Method='olivehawkins').

    Parameters
    ----------
    X : array-like, shape (n, p)
        Rows are observations.
    outlier_fraction : float in [0, 0.5]
    num_trials : int or None
        MATLAB semantics:
          - default 2 if start_method is None
          - otherwise default len(start_method)
          - if elemental is present and num_trials > len(start_method),
            elemental starts are repeated to fill
    reweighting_method : {'rfch', 'rmvn'}
    num_concentration_steps : int
    start_method : None, str, callable, or sequence
        Allowed strings: 'classical', 'medianball', 'elemental'
    random_state : int or None

    Returns
    -------
    Sig : ndarray, shape (p, p)
    Mu : ndarray, shape (p,)
    Mahal : ndarray, shape (n_original,)
        sqrt robust squared Mahalanobis distances; NaN for dropped rows
    Outliers : ndarray, shape (n_original,), dtype=bool
    result : dict
    """
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n_original = X.shape[0]

    # Remove NaN rows, like MATLAB robustcov
    rows_with_nans = np.any(np.isnan(X), axis=1)
    X_clean = X[~rows_with_nans]
    n, p = X_clean.shape

    # MATLAB requires n >= 2p
    if n < 2 * p:
        raise ValueError("MATLAB robustcov requires at least n >= 2*p.")

    if not (0.0 <= outlier_fraction <= 0.5):
        raise ValueError("outlier_fraction must be between 0 and 0.5.")

    if not (isinstance(num_concentration_steps, int) and num_concentration_steps > 0):
        raise ValueError("num_concentration_steps must be a positive integer.")

    reweighting_method = reweighting_method.lower()
    if reweighting_method not in {"rfch", "rmvn"}:
        raise ValueError("reweighting_method must be 'rfch' or 'rmvn'.")

    # MATLAB:
    # Alpha = 1 - outlier_fraction
    # n2 = floor((n+p+1)/2)
    # h = floor(2*n2 - n + 2*(n-n2)*Alpha)
    alpha = 1.0 - outlier_fraction
    n2 = int(np.floor((n + p + 1) / 2))
    h = int(np.floor(2 * n2 - n + 2 * (n - n2) * alpha))

    starts = _expand_start_methods_like_matlab(start_method, num_trials)
    has_mb_attractor = any(_is_named_start(s, "medianball") for s in starts)
    is_full_subset = (h == n)

    # Median-ball quantities used for attractor screening
    med_X = None
    median_median_ball_dists = None
    if has_mb_attractor:
        med_X = np.median(X_clean, axis=0)
        median_ball_dists = np.sum((X_clean - med_X) ** 2, axis=1)
        median_median_ball_dists = np.median(median_ball_dists)

    has_singular_subset = False
    t_best = None
    V_best = None
    det_min = np.inf

    # MATLAB computes MB attractor up front if present
    if has_mb_attractor:
        t, V = _compute_start_estimate(X_clean, "medianball", rng)
        if not is_full_subset:
            for _ in range(num_concentration_steps):
                if not has_singular_subset:
                    t, V, _, has_singular_subset = _compute_c_step(X_clean, t, V, h)
        t_best = t
        V_best = V
        det_min = np.linalg.det(V)

    # Compute all other attractors
    for current_start in starts:
        if has_singular_subset:
            break

        if _is_named_start(current_start, "medianball"):
            continue

        t, V = _compute_start_estimate(X_clean, current_start, rng)

        if not is_full_subset:
            for _ in range(num_concentration_steps):
                if not has_singular_subset:
                    t, V, _, has_singular_subset = _compute_c_step(X_clean, t, V, h)

        det_V = np.linalg.det(V)

        if has_mb_attractor:
            # Accept non-MB attractor only if its location is inside median ball
            if np.sum((t - med_X) ** 2) <= median_median_ball_dists:
                if det_V < det_min:
                    t_best = t
                    V_best = V
                    det_min = det_V
        else:
            if det_V < det_min:
                t_best = t
                V_best = V
                det_min = det_V

    if t_best is None or V_best is None:
        raise RuntimeError("No valid attractor was selected.")

    if has_singular_subset:
        # MATLAB: no reweighting if singular subset encountered
        Mu = t_best
        Sig = V_best
    else:
        t_fch = t_best
        V_fch = (
            np.median(_local_squared_mahal(X_clean, t_best, V_best))
            / chi2.ppf(0.5, p)
        ) * V_best

        if reweighting_method == "rfch":
            # Step 1
            n1_mask = _local_squared_mahal(X_clean, t_fch, V_fch) <= chi2.ppf(0.975, p)
            Sig1, Mu1 = _local_mean_cov(X_clean[n1_mask, :])

            Sig1_bar = (
                np.median(_local_squared_mahal(X_clean, Mu1, Sig1))
                / chi2.ppf(0.5, p)
            ) * Sig1

            # Step 2
            n2_mask = _local_squared_mahal(X_clean, Mu1, Sig1_bar) <= chi2.ppf(0.975, p)
            Sig2_bar, t_rfch = _local_mean_cov(X_clean[n2_mask, :])

            V_rfch = (
                np.median(_local_squared_mahal(X_clean, t_rfch, Sig2_bar))
                / chi2.ppf(0.5, p)
            ) * Sig2_bar

            Mu = t_rfch
            Sig = V_rfch

        elif reweighting_method == "rmvn":
            # Step 1
            n1_mask = _local_squared_mahal(X_clean, t_fch, V_fch) <= chi2.ppf(0.975, p)
            Sig1, Mu1 = _local_mean_cov(X_clean[n1_mask, :])
            n1_in = int(np.sum(n1_mask))

            q1 = min(0.5 * 0.975 * n / n1_in, 0.995)
            Sig1_bar = (
                np.median(_local_squared_mahal(X_clean, Mu1, Sig1))
                / chi2.ppf(q1, p)
            ) * Sig1

            # Step 2
            n2_mask = _local_squared_mahal(X_clean, Mu1, Sig1_bar) <= chi2.ppf(0.975, p)
            Sig2, t_rmvn = _local_mean_cov(X_clean[n2_mask, :])
            n2_in = int(np.sum(n2_mask))

            q2 = min(0.5 * 0.975 * n / n2_in, 0.995)
            V_rmvn = (
                np.median(_local_squared_mahal(X_clean, t_rmvn, Sig2))
                / chi2.ppf(q2, p)
            ) * Sig2

            Mu = t_rmvn
            Sig = V_rmvn

    # Final robust distances / outliers on original indexing
    Mahal = np.zeros(n_original, dtype=float)
    Mahal[rows_with_nans] = np.nan

    if not _zero_cov_det(Sig)[0]:
        sq_mahal = _local_squared_mahal(X_clean, Mu, Sig)
        Mahal[~rows_with_nans] = np.sqrt(sq_mahal)
    else:
        Mahal[~rows_with_nans] = 0.0

    Outliers = np.zeros(n_original, dtype=bool)
    Outliers[~rows_with_nans] = Mahal[~rows_with_nans] > np.sqrt(chi2.ppf(0.975, p))

    result = {
        "Mu": Mu,
        "Sigma": Sig,
        "Method": "olivehawkins",
        "Distances": Mahal,
        "Outliers": Outliers,
        "OutlierFraction": outlier_fraction,
        "ReweightingMethod": reweighting_method,
        "NumTrials": len(starts),
        "NumConcentrationSteps": num_concentration_steps,
        "StartMethodExpanded": starts,
        "RowsWithNaNsRemoved": rows_with_nans,
        "SingularSubsetFound": has_singular_subset,
        "h": h,
    }

    return Sig, Mu, Mahal, Outliers, result


def _is_named_start(start, name):
    return isinstance(start, str) and start.lower() == name.lower()


def _expand_start_methods_like_matlab(start_method, num_trials):
    """
    Match MATLAB's OH StartMethod / NumTrials behavior closely.
    """
    if start_method is None:
        starts = ["classical", "medianball"]
        if num_trials is None:
            num_trials = 2
    else:
        if isinstance(start_method, (str,)) or callable(start_method):
            starts = [start_method]
        else:
            starts = list(start_method)

        # MATLAB removes duplicate classical / medianball if repeated
        deduped = []
        seen_named = set()
        for s in starts:
            if isinstance(s, str) and s.lower() in {"classical", "medianball"}:
                key = s.lower()
                if key in seen_named:
                    continue
                seen_named.add(key)
            deduped.append(s)
        starts = deduped

        if num_trials is None:
            num_trials = len(starts)

    if not (isinstance(num_trials, int) and num_trials > 0):
        raise ValueError("num_trials must be a positive integer.")

    has_elemental = any(_is_named_start(s, "elemental") for s in starts)
    num_starts = len(starts)

    if num_starts == 1:
        if num_trials > 1:
            if _is_named_start(starts[0], "elemental"):
                starts = starts * num_trials
            else:
                if num_trials != num_starts:
                    raise ValueError(
                        "If NumTrials > 1 with a single non-elemental start, MATLAB requires NumTrials == numStarts."
                    )
    else:
        if num_trials != num_starts:
            if has_elemental:
                num_elemental_to_add = num_trials - num_starts
                if num_elemental_to_add < 0:
                    raise ValueError("NumTrials must equal numStarts unless elemental starts are expanded.")
                starts = starts + ["elemental"] * num_elemental_to_add
            else:
                raise ValueError("NumTrials must equal the number of starts when no elemental start is present.")

    return starts


def _compute_start_estimate(X, current_start, rng):
    if isinstance(current_start, str):
        s = current_start.lower()

        if s == "medianball":
            med = np.median(X, axis=0)
            d2 = np.sum((X - med) ** 2, axis=1)
            keep = d2 <= np.median(d2)
            return _local_mean_cov(X[keep, :])[1], _local_mean_cov(X[keep, :])[0]

        if s == "classical":
            S, T = _local_mean_cov(X)
            return T, S

        if s == "elemental":
            return _elemental_subset_estimate(X, rng)

        raise ValueError(f"Unsupported start method: {current_start}")

    if callable(current_start):
        t, V = current_start(X)
        return np.asarray(t, dtype=float), np.asarray(V, dtype=float)

    raise ValueError(f"Invalid start method: {current_start}")


def _elemental_subset_estimate(X, rng):
    """
    MATLAB:
      - draw p+1 cases
      - if covariance singular, keep adding random cases until nonsingular
        or until n-p additions attempted
    """
    n, p = X.shape
    idx = rng.choice(n, size=p + 1, replace=False)
    chosen = list(idx)

    J = X[chosen, :]
    C = _local_mean_cov(J)[0]
    has_zero_cov_det, _ = _zero_cov_det(C)

    num_new_points = 0
    while has_zero_cov_det and num_new_points < (n - p):
        hnew = int(rng.integers(0, n))
        if hnew not in chosen:
            chosen.append(hnew)
            J = X[chosen, :]
            C = _local_mean_cov(J)[0]
            has_zero_cov_det, _ = _zero_cov_det(C)
        num_new_points += 1

    Tinit = np.sum(J, axis=0) / J.shape[0]
    Sinit = C
    return Tinit, Sinit


def _zero_cov_det(S):
    """
    MATLAB ZeroCovDet uses chol and returns true if chol fails.
    """
    try:
        L = np.linalg.cholesky(S)
        return False, L
    except np.linalg.LinAlgError:
        return True, None


def _compute_c_step(X, T, S, h):
    """
    MATLAB computeCStep behavior:
      - if S not PD, flag singular subset and return current estimate
      - else compute distances, sort, keep h smallest, refit mean/cov
    """
    has_singular_subset, L = _zero_cov_det(S)

    if has_singular_subset:
        sorted_inds = np.arange(h)
        return T, S, sorted_inds, True

    Dold = _local_squared_mahal_from_chol(X, T, L)
    sorted_inds = np.argsort(Dold)[:h]
    Xnew = X[sorted_inds, :]
    Snew, Tnew = _local_mean_cov(Xnew)
    return Tnew, Snew, sorted_inds, False


def _local_mean_cov(X):
    n = X.shape[0]
    T = np.sum(X, axis=0) / n
    Xc = X - T
    S = (Xc.T @ Xc) / (n - 1)
    return S, T


def _local_squared_mahal(X, T, S):
    has_zero_cov_det, L = _zero_cov_det(S)
    if has_zero_cov_det:
        raise np.linalg.LinAlgError("Covariance is singular or not positive definite.")
    return _local_squared_mahal_from_chol(X, T, L)


def _local_squared_mahal_from_chol(X, T, L):
    """
    MATLAB computes sum(((X - T) / C).^2, 2) where C is chol(S) upper-triangular.
    NumPy cholesky returns lower L with S = L @ L.T.
    Equivalent: solve L z^T = (X-T)^T, then sum(z^2).
    """
    XC = (X - T).T
    Z = solve_triangular(L, XC, lower=True, check_finite=False)
    return np.sum(Z * Z, axis=0)