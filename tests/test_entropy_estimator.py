"""SNIS entropy estimator against a closed-form target.

compute_entropy_via_importance_sampling treats log_weights as
log dPi^0/dPi^u - log Pi^0_1(X_1) + log|det J_f| and estimates the entropy of
pi* proportional to exp(Q/T) over squashed actions (uniform base measure after the
erf pushforward; see its docstring). The idealized zero-variance-bridge case
is sampling actions directly: a ~ Uniform(-1,1), log_weights = -log q(a)
= log 2. Then pi* proportional to exp(-c a^2 / T) on (-1,1) is a truncated normal
with s^2 = T/(2c); for s << 1 the truncation is negligible and
H(pi*) = 1/2 log(2 pi e s^2) up to < 1e-5.
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.jaxrl.lagrangian_utils import compute_entropy_via_importance_sampling


def _estimate(key, n, c, temperature=1.0):
    a = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    log_weights = jnp.full((n,), jnp.log(2.0))
    q_value = -c * a**2
    return float(
        compute_entropy_via_importance_sampling(log_weights, q_value, temperature)
    )


def test_converges_to_truncated_gaussian_entropy():
    c, temperature = 12.5, 1.0
    s_sq = temperature / (2 * c)  # 0.04 -> mass outside (-1,1) ~ 6e-7
    h_true = 0.5 * np.log(2 * np.pi * np.e * s_sq)

    h_small = _estimate(jax.random.PRNGKey(0), 2_000, c, temperature)
    h_large = _estimate(jax.random.PRNGKey(1), 200_000, c, temperature)

    # ESS ~ 0.35 N for this weight profile; 5-sigma-ish practical bounds.
    assert abs(h_small - h_true) < 0.3
    assert abs(h_large - h_true) < 0.03


def test_temperature_scaling():
    """Doubling T doubles s^2, raising the entropy by 1/2 log 2."""
    c = 12.5
    h1 = _estimate(jax.random.PRNGKey(2), 200_000, c, temperature=1.0)
    h2 = _estimate(jax.random.PRNGKey(3), 200_000, c, temperature=2.0)
    np.testing.assert_allclose(h2 - h1, 0.5 * np.log(2.0), atol=0.05)


def test_batched_axis_semantics():
    """A (B, N) batch returns one entropy per row (axis=-1 reduction)."""
    key = jax.random.PRNGKey(4)
    n, cs = 100_000, np.array([5.0, 12.5, 50.0])
    a = jax.random.uniform(key, (3, n), minval=-1.0, maxval=1.0)
    log_weights = jnp.full((3, n), jnp.log(2.0))
    q_value = -cs[:, None] * np.asarray(a) ** 2
    h = compute_entropy_via_importance_sampling(
        log_weights, jnp.asarray(q_value), 1.0
    )
    assert h.shape == (3,)
    h_true = 0.5 * np.log(2 * np.pi * np.e / (2 * cs))
    np.testing.assert_allclose(np.asarray(h), h_true, atol=0.05)
