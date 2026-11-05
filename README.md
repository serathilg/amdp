# Scalable Maximum Entropy Reinforcement Learning for Diffusion Policies via Adjoint Matching

Official implementation of AMDP (NeurIPS 2026;
[OpenReview](https://openreview.net/forum?id=h8DYJlF7Jt),
[arXiv:2606.22630](https://arxiv.org/abs/2606.22630)),
an actor-critic algorithm for online maximum-entropy reinforcement learning with
diffusion policies. The policy is a controlled SDE trained by reciprocal adjoint
matching against the soft Q-function; entropy regularization uses a path-space lower
bound on the policy entropy. The training loop builds on the REPPO codebase
([arXiv:2507.11019](https://arxiv.org/abs/2507.11019)). JAX and PyTorch
implementations are provided.

## Installation

Requires Python 3.12 and a GPU with CUDA 12 compatible drivers. With
[uv](https://docs.astral.sh/uv/):

```bash
git submodule update --init      # dependencies/humanoid-bench
uv sync                          # core: JAX + PyTorch, MuJoCo Playground, ManiSkill3
uv sync --extra humanoid-bench   # additionally: HumanoidBench
```

With other environment managers, create a Python 3.12 environment and run
`pip install -e .`.

## Running experiments

Each trainer is a [Hydra](https://hydra.cc) entry point with its own base config in
`config/`:

| Entry point | Base config | Description |
| --- | --- | --- |
| `src/jaxrl/amdp_reppo.py` | `amdp_reppo` | JAX, Brownian reference with Dirac prior (default) |
| `src/jaxrl/amdp_ou_reppo.py` | `amdp_ou_reppo` | JAX, Ornstein-Uhlenbeck reference (ablation) |
| `src/jaxrl/amdp_reppo_benchmark.py` | `amdp_reppo` | JAX, fully-jitted monolith of the default trainer with per-phase timing (MJX envs only) |
| `src/torchrl/amdp_reppo.py` | `amdp_reppo_torch` | PyTorch, Brownian reference with Dirac prior |

Examples:

```bash
uv run src/jaxrl/amdp_reppo.py env=mjx_dmc env.name=CartpoleSwingup
uv run src/jaxrl/amdp_ou_reppo.py env=mjx_humanoid env.name=G1JoystickFlatTerrain seed=1
uv run src/torchrl/amdp_reppo.py env=maniskill env.name=PickCube-v1
```

Common overrides: `env=<group>` selects an environment family from `config/env/`,
`env.name=<task>` the task, `hyperparameters.*` the algorithm settings, and
`experiment_overrides=<group>` a benchmark-specific hyperparameter set. Logging uses
Weights & Biases; adjust the `wandb:` block in the base configs or pass
`wandb.mode=disabled`.

`num_seeds=<n>` trains several seeds in parallel in a single process: every training
phase is vmapped over the seed axis, and non-JAX environments batch all seeds in one
vectorized environment. Metrics are logged as the seed mean plus per-seed
`<key>/seed_i` curves.

Each environment family sets `jax_native` in its config: MuJoCo Playground
environments step inside the jitted rollout, while ManiSkill and HumanoidBench are
stepped by the host loop between jitted phases.

On the first run you will see an alternation of compilation pauses and short GPU
bursts before throughput settles: each training phase (init, rollout, learn, eval,
and — on the host path — the small pre/post-step helpers) is a separate jitted
program that compiles on its first call, and evaluation necessarily has a different
shape than training. This is expected and happens once: compiled programs are stored
in a persistent cache (`~/.cache/reppo_jax_cache`), so subsequent runs start almost
immediately.

### Archived experiments and Slurm submission

`config/run/<name>/` archives the experiments as a Hydra override file
(`config.yaml`) plus a sweep definition (`sweep.yaml`: entry point, base override,
grid/list/ablation values, array layout). A single configuration can be reproduced
locally with

```bash
uv run <entry_point> -cn <base_config> +run=<name>/config
```

`scripts/submit.py` expands a sweep and submits it as a Slurm array job, assembling
the sbatch header from a cluster profile `config/slurm/<name>.yaml`. Copy
`config/slurm/example.yaml`, which documents the supported options, and adapt it to
your cluster:

```bash
scripts/submit.py --run 052_mjx_bench --dry-run                                # print expanded runs and script
scripts/submit.py --run 052_mjx_bench --cluster <name> --account <account>     # submit
```

Keep the account out of the profile: pass `--account` or export `SBATCH_ACCOUNT`.
sbatch options after `--` override the profile's `#SBATCH` directives.

## Repository structure

### `src/`

- `jaxrl/` — JAX trainers and shared training utilities.
  - `amdp_reppo.py`, `amdp_ou_reppo.py` — the main trainers; the default trainer
    supports Best-of-N evaluation (`hyperparameters.best_of_n_values`), the OU
    trainer probability-flow ODE evaluation (`hyperparameters.eval_ode`).
  - `amdp_reppo_benchmark.py` — the default trainer with per-phase wall-clock
    timing for benchmarking.
  - `types.py` — `Transition`, `SACTrainState`, and the `ReppoConfigBase`
    hyperparameter dataclass.
  - `eval.py` — evaluation rollouts (traced for JAX-native environments,
    host-stepped otherwise).
  - `lagrangian_utils.py` — trust-region dual solvers and the SNIS estimator of the
    optimal-policy entropy.
  - `utils.py` — metric-dict helpers, HL-Gauss targets, TD(λ) targets, effective
    sample size.
- `torchrl/` — PyTorch trainer.
  - `amdp_reppo.py` — trainer and losses.
  - `envs.py`, `env_hyperparams.py` — environment construction and per-task value
    ranges.
  - `reppo_util.py` — observation normalization and HL-Gauss targets.
  - `trajectory_utils.py` — trajectory export and plotting.
- `networks/` — model definitions.
  - `jax_models.py`, `torch_models.py` — critic architectures (categorical/HL-Gauss
    critics, MLP backbones).
  - `amdp/` — the diffusion actor.
    - `jax_policy.py`, `jax_policy_ou.py`, `torch_policy.py` — actor modules: SDE
      sampling with path-space log-weights (Girsanov terms, reference terminal
      density, squashing correction), Erf/Tanh squashing bijectors, temperature and
      trust-region Lagrange parameters.
    - `jax_integrators.py`, `jax_integrators_ou.py`, `torch_integrators.py` —
      Euler–Maruyama integrators accumulating the Girsanov increments of
      log dΠ⁰/dΠᵘ; the OU file also contains the probability-flow ODE integrator.
    - `jax_control_net.py`, `torch_control_net.py` — control networks (time
      embedding, optional residual/AdaLN conditioning).
    - `jax_scheduler.py`, `torch_scheduler.py` — geometric noise schedules and
      bridge coefficients.
    - `types.py` — the `SDESample` result record.
- `env_utils/` — environment wrappers.
  - `jax_wrappers.py` — MuJoCo Playground, ManiSkill, and HumanoidBench wrappers for
    the JAX trainers (normalization, episode statistics).
  - `mixed_vec_env.py` — vectorized environment (subprocess/in-process mix) used by
    the JAX HumanoidBench wrapper.
  - `torch_wrappers/` — MuJoCo Playground, ManiSkill, and HumanoidBench wrappers for
    the PyTorch trainer.

### `config/`

- `amdp_reppo.yaml`, `amdp_ou_reppo.yaml`, `amdp_reppo_torch.yaml` — base configs of
  the entry points; `amdp_reppo_torch.yaml` also holds the `platform:` device and
  precision settings of the PyTorch trainer.
- `env/` — environment families (`mjx_dmc`, `mjx_humanoid`, `mjx_manipulation`,
  `maniskill`, `humanoid_bench`).
- `experiment_overrides/` — benchmark-specific hyperparameter sets.
- `run/` — archived experiment configurations and sweep definitions.
- `slurm/` — example cluster profile for `scripts/submit.py`.

### Other

- `scripts/submit.py` — sweep expansion and Slurm submission.
- `dependencies/humanoid-bench` — HumanoidBench git submodule.

## Tests

```bash
uv run pytest
```

The suite runs on CPU and covers the diffusion math (integrators, path-space
log-weights, schedulers, terminal marginals, the probability-flow ODE, squashing
bijectors), the TD(λ) recursion, the trust-region and entropy estimators, and the
observation normalizer.

## Citation

```bibtex
@inproceedings{thilges2026scalable,
  title     = {Scalable Maximum Entropy Reinforcement Learning for Diffusion Policies via Adjoint Matching},
  author    = {Thilges, Serge and Celik, Onur and Blessing, Denis and Gospodinov, Emiliyan and Neumann, Gerhard},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=h8DYJlF7Jt}
}
```

## License

MIT, see `LICENCE`. The repository builds on the
[REPPO](https://arxiv.org/abs/2507.11019) code release, which in turn builds on
[PureJaxRL](https://github.com/luchris429/purejaxrl) and
[FastTD3](https://github.com/younggyoseo/FastTD3).
