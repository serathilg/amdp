#!/usr/bin/env python3
"""Submit an archived experiment sweep to Slurm.

Expands config/run/<run>/sweep.yaml into hydra override strings, writes a
self-contained sbatch script to outputs/<run>/ using the profile
config/slurm/<cluster>.yaml (see config/slurm/example.yaml), and submits it.

Usage:
    scripts/submit.py --run 020_mjx_eval --dry-run
    scripts/submit.py --run 020_mjx_eval --cluster <name> --account <account>
    scripts/submit.py --run 020_mjx_eval --cluster <name> -- --time=04:00:00

Options after "--" go to sbatch and override the profile's #SBATCH directives
(sbatch precedence: command line > SBATCH_* environment > script). The account
can also come from SBATCH_ACCOUNT.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def build_runs(spec: dict) -> list[str]:
    """Expand grid (cartesian), list (zipped), and ablations into override strings.

    Reproduces the enumeration order of the original bash sweep scripts exactly.
    """
    grid_combos = [""]
    for key, vals in (spec.get("grid") or {}).items():
        grid_combos = [f"{c} {key}={v}" for c in grid_combos for v in vals]

    lists = spec.get("list") or {}
    if lists:
        keys = list(lists)
        lengths = {len(lists[k]) for k in keys}
        assert len(lengths) == 1, f"list values must have equal length, got {lengths}"
        list_combos = [
            "".join(f" {k}={lists[k][i]}" for k in keys) for i in range(lengths.pop())
        ]
    else:
        list_combos = [""]

    base = spec["base_override"]
    runs = []
    for g in grid_combos:
        for l in list_combos:
            exp = f"{g} {l}"
            runs.append(f"{base} {exp}")
            for ovr in spec.get("ablations") or []:
                runs.append(f"{base} {exp} {ovr}")
    return runs


def make_sbatch_script(spec, profile, runs, run_name, out_dir):
    opts = dict(profile.get("sbatch_options") or {})
    env_lines = "".join(
        f"export {k}={str(v).replace('${REPO}', str(REPO))}\n"
        for k, v in (profile.get("env") or {}).items()
    )
    wandb_dir = profile.get("wandb_dir", "tmpdir")
    if wandb_dir == "outputs":
        wandb_export = f'export WANDB_DIR="{out_dir}"'
    elif wandb_dir == "tmpdir":
        wandb_export = 'export WANDB_DIR="${TMPDIR:-/tmp}/wandb"'
    else:
        raise ValueError(f"unknown wandb_dir: {wandb_dir}")
    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name={spec['job_name']}",
        f"#SBATCH --array={spec['array']}",
        f"#SBATCH --time={spec['time']}",
        *(f"#SBATCH --{k}={v}" for k, v in opts.items()),
        f"#SBATCH --output={out_dir}/%x_%A_%a.out",
        f"#SBATCH --error={out_dir}/%x_%A_%a.err",
        f"#SBATCH --chdir={REPO}",
    ]

    run_mode = profile.get("run_mode", "sequential")
    max_parallel = int(profile.get("max_parallel", 4))
    if run_mode == "sequential":
        launch = 'HYDRA_FULL_ERROR=1 uv run "$ENTRY_POINT" $CURRENT_PARAMS'
        epilogue = ""
    elif run_mode == "parallel":
        launch = (
            f"while (( $(jobs -p | wc -l) >= {max_parallel} )); do wait -n; done\n"
            '    HYDRA_FULL_ERROR=1 srun --ntasks=1 --exclusive uv run "$ENTRY_POINT" $CURRENT_PARAMS &'
        )
        epilogue = "wait\n"
    else:
        raise ValueError(f"unknown run_mode: {run_mode}")

    quoted_runs = "\n".join(f"    {shlex.quote(r)}" for r in runs)
    body = f"""
ENTRY_POINT={shlex.quote(spec["entry_point"])}
JOBS_PER_TASK={spec["jobs_per_task"]}

FINAL_RUNS=(
{quoted_runs}
)
TOTAL_COUNT=${{#FINAL_RUNS[@]}}

{wandb_export}
mkdir -p "$WANDB_DIR"
export WANDB_CONSOLE=off
export MUJOCO_EGL_DEVICE_ID=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
{env_lines}
: "${{SLURM_ARRAY_TASK_ID:=0}}"
echo "Job ID: ${{SLURM_ARRAY_JOB_ID:-local}}_${{SLURM_ARRAY_TASK_ID}} on ${{SLURMD_NODENAME:-$(hostname)}}"

# ensure the venv is importable before burning array slots
uv run python -c "import torch, jax"

START_IDX=$(( SLURM_ARRAY_TASK_ID * JOBS_PER_TASK ))
END_IDX=$(( START_IDX + JOBS_PER_TASK ))
for (( i=START_IDX; i<END_IDX && i<TOTAL_COUNT; i++ )); do
    CURRENT_PARAMS=${{FINAL_RUNS[$i]}}
    echo "------------------------------------------------"
    echo "Run $i / $TOTAL_COUNT: $CURRENT_PARAMS"
    echo "------------------------------------------------"
    {launch}
done
{epilogue}"""
    return "\n".join(header) + "\n" + body


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run name under config/run/")
    parser.add_argument("--cluster", help="profile name under config/slurm/")
    parser.add_argument("--account", help="slurm account (forwarded to sbatch)")
    parser.add_argument("--dry-run", action="store_true", help="print the expanded run list and script; do not submit")
    parser.add_argument("--run-dir", type=Path, help="override the run directory (default config/run/<run>)")
    parser.add_argument("sbatch_args", nargs="*", help="extra sbatch options after --")
    args = parser.parse_args()

    run_dir = args.run_dir or (REPO / "config" / "run" / args.run)
    spec = yaml.safe_load((run_dir / "sweep.yaml").read_text())

    if args.cluster:
        profile = yaml.safe_load((REPO / "config" / "slurm" / f"{args.cluster}.yaml").read_text())
    elif args.dry_run:
        profile = {}
    else:
        parser.error("--cluster is required unless --dry-run")

    runs = build_runs(spec)
    out_dir = REPO / "outputs" / args.run
    script = make_sbatch_script(spec, profile, runs, args.run, out_dir)

    if args.dry_run:
        print(f"ENTRY={spec['entry_point']}")
        for r in runs:
            print(r)
        print(f"# {len(runs)} runs; generated script follows", file=sys.stderr)
        print(script, file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / "sbatch_generated.sh"
    script_path.write_text(script)
    cmd = ["sbatch"]
    if args.account:
        cmd.append(f"--account={args.account}")
    cmd += args.sbatch_args
    cmd.append(str(script_path))
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
