from typing import NamedTuple

from jaxtyping import Array, Float


class SDESample(NamedTuple):
    """Per-sample statistics of one controlled-SDE action draw.

    Shapes document the batched output of DiffusionActor.sde_sample
    ("batch act" etc.); they are documentation, not runtime-checked. The batch
    axis is whatever the caller feeds in: envs during a rollout, a minibatch
    during updates.
    log_weight is the single-path estimate of -log pi(a|s):
      log dPi^0/dPi^u - log Pi^0_1(X_1) + log|det J_f(X_1)|
    with a = f(X_1) the terminal sample (log_p_T_ref stores log Pi^0_1); the
    path weights are its accumulated Girsanov terms (they keep the
    integrator's per-sample (1,) axis, hence "batch 1").
    """

    action: Float[Array, "batch act"]
    action_unsquashed: Float[Array, "batch act"]
    prior_action: Float[Array, "batch act"]
    squashing_correction_grad: Float[Array, "batch act"]
    log_weight: Float[Array, "batch 1"]
    log_path_weight_deterministic: Float[Array, "batch 1"]
    log_path_weight_stochastic: Float[Array, "batch 1"]
    log_p_T_ref: Float[Array, "batch 1"]
    squashing_correction_val: Float[Array, "batch"]
