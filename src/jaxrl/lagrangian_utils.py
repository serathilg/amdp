import jax
import jax.numpy as jnp
from jax._src.scipy.special import logsumexp
from jaxtyping import Array, Float


def compute_entropy_via_importance_sampling(
    log_weights: Float[Array, "*batch n"],
    Q_value: Float[Array, "*batch n"],
    lm_entr: Float[Array, ""] | float,
) -> Float[Array, "*batch"]:
    r"""SNIS estimate of the entropy of the optimal policy pi*(a|s).

    Target. pi*(a|s) \propto exp(Q(a,s)/T) over squashed actions; in latent
    space pi~*(x|s) = pi*(f(x)|s) |det J_f(x)| for any bijective squashing f —
    the estimator below only needs this change of variables. (For the scaled-erf
    squashing specifically, the pushforward of the reference terminal marginal
    Pi^0_1 through f is uniform, i.e. |det J_f| \propto Pi^0_1, and the target
    becomes the exp(Q/T)-tilting of the uncontrolled reference Gaussian.)

    Weights. Estimation is in path space w.r.t. the uncontrolled reference
    Pi^0. The target path measure Pi* is Pi^0 tilted at the terminal time,
        log dPi*/dPi^0(X) = log pi~*(X_1) - log Pi^0_1(X_1)
                          = Q(f(X_1))/T + log|det J_f(X_1)| - log Pi^0_1(X_1) - log Z,
    and sde_sample provides
        log_weights = log dPi^0/dPi^u(X) - log Pi^0_1(X_1) + log|det J_f(X_1)|,
    i.e. the path RND w.r.t. Pi^0 (Girsanov terms) plus the terminal-marginal
    terms. Together:
        log w := Q(f(X_1))/T + log_weights = log dPi*/dPi^u(X) + log Z.
    The tilting acts on X_1 only, so the path weights marginalize:
    E_{Pi^u}[w g(f(X_1))] = Z E_{pi*}[g(a)], and SNIS over sampled paths
    estimates expectations under the action marginal pi* (the intractable
    marginal ratio pi~*(X_1)/pi^u_1(X_1) is replaced by the tractable path
    RND at the price of extra variance from the bridges).

    Args:
        log_weights: per-path log dPi^0/dPi^u - log Pi^0_1(X_1) + log|det J_f(X_1)|
            from sde_sample.
        Q_value: Q(a, s) at the sampled actions a = f(X_1).
        lm_entr: the temperature T.
    """
    T = lm_entr
    # Unnormalized target log-density over squashed actions: log q~*(a) = Q(a)/T
    log_q_tilde = Q_value / T
    # log w = log q~*(a) + log_weights = log dPi*/dPi^u + log Z (see docstring)
    log_importance_weights = log_q_tilde + log_weights
    # log Z = log E_{Pi^u}[exp(log w)], batch estimate via logsumexp
    N = log_q_tilde.shape[-1]
    log_Z = logsumexp(log_importance_weights, axis=-1) - jnp.log(N)
    # self-normalized weights = softmax(log w)
    norm_weights = jax.nn.softmax(log_importance_weights, axis=-1)
    # H(pi*) = -E_{pi*}[log pi*(a)] = -E_{pi*}[log q~*(a)] + log Z
    entropy = -jnp.sum(norm_weights * log_q_tilde, axis=-1) + log_Z
    return entropy

def find_optimum_kl_lagrangian(
    w_t: Float[Array, "n"],
    old_ctrl: Float[Array, "n act"],
    ctrl_target: Float[Array, "n act"],
    eps: float,
    min=1e-3,
    max=1e3,
    norm_weights=True,
) -> Float[Array, ""]:
    if norm_weights:
        # lagrangian is linear in constant scaling of weights but unit mean weights improve numerics
        w_t_factor = w_t.mean()
    else:
        w_t_factor = 1.0
    w_t_norm = w_t / w_t_factor
    sse = 0.5 * jnp.sum(jnp.square(old_ctrl - ctrl_target), axis=-1)
    w2 = w_t_norm**2
    def eval_single_lambda(lam):
        w_lam_2 = (w_t_norm + lam)**2
        dual_grad = jnp.mean(w2 / w_lam_2 * sse) - eps
        return dual_grad**2
    min_norm_lambda = parallel_nary_search(eval_single_lambda, low=min, high=max)
    return min_norm_lambda * w_t_factor

def parallel_nary_search(f, low, high, n_points=64, rtol=1e-4, atol=1e-6, max_iter=50):
    """
    Parallel N-ary search of convex 1d function of positive input with expansion of range.
    """
    f_batched = jax.vmap(f)

    assert low > 0 and high > 0, f"low and high of range must be positive, got {low=} and {high=}"

    # Track (low, high, iteration number) as state

    def cond_fun(state):
        a, b, i = state
        # Continue refinement until max_iter or precise range
        return (i < max_iter) & ((b / a) > (1.0 + rtol)) & (jnp.abs(b - a) > atol)

    def body_fun(state):
        low, high, i = state

        grid = jnp.geomspace(low, high, n_points)

        vals = f_batched(grid)
        idx = jnp.argmin(vals)

        # Initial range might be wrong, so expand with same high/low ratio
        ratio = high / low

        # Best is at lower bound, convexity implies that minimmum
        # either at smaller value or between two first grid points
        is_at_lower = (idx == 0)
        high_lower = grid[1]
        low_lower = high_lower / ratio

        # Best is at upper bound, convexity implies that minimmum
        # either at larger value or between two last grid points
        is_at_upper = (idx == n_points - 1)
        low_upper = grid[n_points - 2]
        high_upper = low_upper * ratio
        
        # Best is in range, convexity implies that minimmum within previous and next
        # grid points of current minimum point
        # We use jnp.maximum/minimum to handle illegal indices.
        low_bracketing = grid[jnp.maximum(0, idx - 1)]
        high_bracketing = grid[jnp.minimum(n_points - 1, idx + 1)]
        
        new_low = jnp.where(is_at_lower, low_lower, jnp.where(is_at_upper, low_upper, low_bracketing))
        new_high = jnp.where(is_at_lower, high_lower, jnp.where(is_at_upper, high_upper, high_bracketing))
        
        return (new_low, new_high, i + 1)

    init_state = (jnp.float32(low), jnp.float32(high), 0)

    final_a, final_b, total_iters = jax.lax.while_loop(cond_fun, body_fun, init_state)

    return (final_a + final_b) / 2
