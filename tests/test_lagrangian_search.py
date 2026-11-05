"""parallel_nary_search and find_optimum_kl_lagrangian.

find_optimum_kl_lagrangian returns (approximately) the root of the dual
gradient g(lam) = mean(w^2/(w+lam)^2 * sse) - eps: the internal unit-mean
weight normalization cancels in w^2/(w+lam)^2, so the returned lambda can be
checked directly against a brute-force root in the raw parameterization.
g is strictly decreasing on lam > 0 (w > 0) from mean(sse) to 0, so any
eps in (0, mean(sse)) has a unique root.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.jaxrl.lagrangian_utils import (
    find_optimum_kl_lagrangian,
    parallel_nary_search,
)


@pytest.mark.parametrize(
    "lam_star",
    [3.7, 5.0e4, 1.0e-5],  # in range, above high (expansion), below low (expansion)
)
def test_nary_search_finds_convex_minimum(lam_star):
    f = lambda lam: (lam - lam_star) ** 2
    result = float(parallel_nary_search(f, low=1e-3, high=1e3))
    np.testing.assert_allclose(result, lam_star, rtol=1e-2)


def test_nary_search_matches_dense_grid():
    lam_star = 41.3
    f = lambda lam: (jnp.log(lam) - np.log(lam_star)) ** 2  # convex in log-space too
    result = float(parallel_nary_search(f, low=1e-3, high=1e3))
    grid = np.geomspace(1e-3, 1e3, 400_001)
    brute = grid[np.argmin((np.log(grid) - np.log(lam_star)) ** 2)]
    np.testing.assert_allclose(result, brute, rtol=1e-2)


def _dual_grad_terms(key, n=256, dim=4):
    k1, k2, k3 = jax.random.split(key, 3)
    w_t = jnp.abs(jax.random.normal(k1, (n,))) + 0.1
    old_ctrl = jax.random.normal(k2, (n, dim))
    ctrl_target = old_ctrl + 0.5 * jax.random.normal(k3, (n, dim))
    sse = 0.5 * jnp.sum(jnp.square(old_ctrl - ctrl_target), axis=-1)
    return w_t, old_ctrl, ctrl_target, sse


@pytest.mark.parametrize("norm_weights", [True, False])
def test_kl_lagrangian_solves_the_dual_root(norm_weights):
    w_t, old_ctrl, ctrl_target, sse = _dual_grad_terms(jax.random.PRNGKey(42))
    eps = 0.3 * float(sse.mean())

    lam = float(
        find_optimum_kl_lagrangian(
            w_t, old_ctrl, ctrl_target, eps, norm_weights=norm_weights
        )
    )

    w = np.asarray(w_t, dtype=np.float64)
    s = np.asarray(sse, dtype=np.float64)
    grid = np.geomspace(1e-6, 1e6, 2_000_001)
    g = np.mean(
        (w[None, :] ** 2 / (w[None, :] + grid[:, None]) ** 2) * s[None, :], axis=1
    ) - eps
    brute_root = grid[np.argmin(np.abs(g))]

    np.testing.assert_allclose(lam, brute_root, rtol=2e-2)


def test_kl_lagrangian_jit_parity():
    w_t, old_ctrl, ctrl_target, sse = _dual_grad_terms(jax.random.PRNGKey(7))
    eps = 0.3 * float(sse.mean())

    eager = find_optimum_kl_lagrangian(w_t, old_ctrl, ctrl_target, eps)
    jitted = jax.jit(
        lambda w, o, c: find_optimum_kl_lagrangian(w, o, c, eps)
    )(w_t, old_ctrl, ctrl_target)

    np.testing.assert_allclose(float(eager), float(jitted), rtol=1e-5)
