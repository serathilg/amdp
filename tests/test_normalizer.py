"""NormalizeVec's streaming (Chan et al.) mean/variance against numpy.

The parallel-Welford merge with biased batch variances is exact: after any
number of batch folds the state must equal the pooled biased statistics of
every row seen so far (reset batch included). update_stats=False must freeze
the statistics while still normalizing with them.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from src.env_utils.jax_wrappers import NormalizeVec

B, D, DC, K = 8, 4, 6, 10  # envs, obs dim, critic-obs dim, steps


@struct.dataclass
class DummyState:
    idx: jnp.ndarray
    truncated: jnp.ndarray
    info: dict


class DummyVecEnv:
    """Plays back pre-generated observation streams."""

    def __init__(self, obs_seq, critic_seq):
        self.obs_seq = obs_seq
        self.critic_seq = critic_seq

    def reset(self, key, params=None):
        state = DummyState(
            idx=jnp.array(0), truncated=jnp.zeros(B), info={}
        )
        return self.obs_seq[0], self.critic_seq[0], state

    def step(self, key, state, action):
        idx = state.idx + 1
        new_state = DummyState(idx=idx, truncated=jnp.zeros(B), info={})
        return (
            self.obs_seq[idx],
            self.critic_seq[idx],
            new_state,
            jnp.zeros(B),
            jnp.zeros(B),
            {},
        )


@pytest.fixture
def rollout():
    rng = np.random.default_rng(0)
    obs_seq = jnp.asarray(rng.normal(2.0, 3.0, size=(K + 1, B, D)), dtype=jnp.float32)
    critic_seq = jnp.asarray(
        rng.normal(-1.0, 0.5, size=(K + 1, B, DC)), dtype=jnp.float32
    )
    env = NormalizeVec(DummyVecEnv(obs_seq, critic_seq))
    return env, obs_seq, critic_seq


def _run_steps(env, state, n, update_stats=True):
    outs = None
    for _ in range(n):
        outs = env.step(None, state, None, update_stats=update_stats)
        state = outs[2]
    return state, outs


def test_stats_match_pooled_numpy(rollout):
    env, obs_seq, critic_seq = rollout
    _, _, state = env.reset(None)
    state, _ = _run_steps(env, state, K)

    rows = np.asarray(obs_seq, dtype=np.float64).reshape(-1, D)
    critic_rows = np.asarray(critic_seq, dtype=np.float64).reshape(-1, DC)

    assert int(state.count) == (K + 1) * B
    np.testing.assert_allclose(np.asarray(state.mean), rows.mean(0), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(state.var), rows.var(0), rtol=1e-4)
    np.testing.assert_allclose(
        np.asarray(state.critic_mean), critic_rows.mean(0), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(state.critic_var), critic_rows.var(0), rtol=1e-4
    )


def test_step_normalizes_with_updated_stats(rollout):
    env, obs_seq, _ = rollout
    _, _, state = env.reset(None)
    obs_n, _, state, _, _, _ = env.step(None, state, None)
    expected = (np.asarray(obs_seq[1]) - np.asarray(state.mean)) / np.sqrt(
        np.asarray(state.var) + 1e-2
    )
    np.testing.assert_allclose(np.asarray(obs_n), expected, rtol=1e-5, atol=1e-6)


def test_update_stats_false_freezes(rollout):
    env, obs_seq, _ = rollout
    _, _, state = env.reset(None)
    state, _ = _run_steps(env, state, 3)
    frozen = state

    obs_n, _, state2, _, _, _ = env.step(None, state, None, update_stats=False)

    np.testing.assert_array_equal(np.asarray(state2.mean), np.asarray(frozen.mean))
    np.testing.assert_array_equal(np.asarray(state2.var), np.asarray(frozen.var))
    assert float(state2.count) == float(frozen.count)
    expected = (np.asarray(obs_seq[4]) - np.asarray(frozen.mean)) / np.sqrt(
        np.asarray(frozen.var) + 1e-2
    )
    np.testing.assert_allclose(np.asarray(obs_n), expected, rtol=1e-5, atol=1e-6)


def test_reset_with_params_restores_stats(rollout):
    env, obs_seq, _ = rollout
    _, _, state = env.reset(None)
    state, _ = _run_steps(env, state, K)

    obs_n, _, restored = env.reset(None, params=state)
    np.testing.assert_array_equal(np.asarray(restored.mean), np.asarray(state.mean))
    np.testing.assert_array_equal(np.asarray(restored.var), np.asarray(state.var))
    assert float(restored.count) == float(state.count)
    expected = (np.asarray(obs_seq[0]) - np.asarray(state.mean)) / np.sqrt(
        np.asarray(state.var) + 1e-2
    )
    np.testing.assert_allclose(np.asarray(obs_n), expected, rtol=1e-5, atol=1e-6)
