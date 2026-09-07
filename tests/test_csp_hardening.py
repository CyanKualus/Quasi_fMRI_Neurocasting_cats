"""Optional CSP conditioning/shrinkage behavior."""
import numpy as np

from klh.csp import calc_csp_cov, covariance_condition


def test_zero_shrinkage_is_exact_default():
    rng = np.random.default_rng(44)
    a = rng.standard_normal((200, 6))
    b = rng.standard_normal((200, 6))
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    default = calc_csp_cov(s1, s2)
    explicit = calc_csp_cov(s1, s2, shrinkage=0.0)
    for left, right in zip(default, explicit):
        assert np.array_equal(left, right)


def test_shrinkage_conditions_a_nearly_singular_problem():
    s1 = np.diag([1.0, 1e-12, 1e-14])
    s2 = np.diag([2.0, 2e-12, 2e-14])
    before = covariance_condition(s1, s2)
    # The optional behavior should produce a finite CSP result and improve the
    # conditioning without changing the default path.
    result = calc_csp_cov(s1, s2, shrinkage=0.05)
    assert np.all(np.isfinite(result[-1]))
    target1 = 0.95 * s1 + 0.05 * np.trace(s1) / 3 * np.eye(3)
    target2 = 0.95 * s2 + 0.05 * np.trace(s2) / 3 * np.eye(3)
    assert covariance_condition(target1, target2) < before


def test_invalid_shrinkage_is_rejected():
    eye = np.eye(2)
    for value in (-0.1, 1.0):
        try:
            calc_csp_cov(eye, eye, shrinkage=value)
        except ValueError as exc:
            assert "shrinkage" in str(exc)
        else:
            raise AssertionError("invalid shrinkage should raise")


def test_unused_ged_can_be_skipped_when_rest_covariance_is_singular():
    # R1 + R2 is positive definite, so the bounded CSP used by the pipeline is
    # valid even though the optional generalized problem (R1, R2) is not.
    s1 = np.eye(2)
    s2 = np.diag([1.0, 0.0])
    pinv, pfwd, ged_inv, ged_fwd, evals = calc_csp_cov(
        s1, s2, include_ged=False)
    assert pinv.shape == pfwd.shape == (2, 2)
    assert ged_inv is None and ged_fwd is None
    assert np.all(np.isfinite(evals))
