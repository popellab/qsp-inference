#!/bin/bash
# QSP-scale population fit (docs ch. 4) on a Slurm cluster. Written against JHU
# Rockfish; the only cluster-specific line is the partition.
#
# Submit from the repo root, passing your allocation:
#
#     mkdir -p logs
#     sbatch -A <your_account> examples/run_toy_population_full.sbatch
#
# The account is deliberately not baked in, and neither is any path: the job runs
# from SLURM_SUBMIT_DIR and builds its environment beside the checkout.
#
# examples/toy_population_fit.py and examples/qsp_scale_mechanism.py import only
# jax, numpyro, diffrax, numpy and scipy. They do NOT import qsp_inference or
# maple, so this job does not install the package. That matters here: the package
# depends on maple over git+ssh, which a compute node generally cannot reach.

#SBATCH --job-name=qsp-vpop-full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CPUS="${SLURM_CPUS_PER_TASK:-8}"
CHAIN_DEVICES=4  # matches numpyro.set_host_device_count(4) in the script

# Thread budget. XLA gives every host device its own pool, so leaving
# OMP_NUM_THREADS at the core count would put CHAIN_DEVICES x CPUS threads on
# CPUS cores and thrash. Divide instead.
export OMP_NUM_THREADS=$(( CPUS / CHAIN_DEVICES ))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export XLA_FLAGS="--xla_force_host_platform_device_count=${CHAIN_DEVICES}"
export JAX_PLATFORMS=cpu

# uv brings its own Python, so no module load and no dependence on whatever the
# cluster's default interpreter happens to be this month.
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

VENV=".venv-toy"
if [ ! -d "$VENV" ]; then
    uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install --quiet "jax>=0.4" "jaxlib>=0.4" "numpyro>=0.15" "diffrax>=0.6" \
    "numpy>=1.24" "scipy>=1.11"

echo "host $(hostname), ${CPUS} cpus, OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "started $(date -Is)"

# The progress bar is kept, not silenced: tqdm writes to stderr, and stderr now
# goes to its own file. So the .out file is clean prose and the .err file carries
# a live ETA. Watch .out for phase markers, .err for the bar.
# Setup is roughly 18 minutes before NUTS starts: emulator design and training,
# then V_c built twice with E_c dominating. The Laplace metric then needs a
# forward-mode Jacobian over 572 coordinates, which is the most likely place for
# this to stall; if it does, rerun with --no-laplace-mass --max-tree-depth 7.
srun python -u examples/toy_population_fit.py \
    --size full \
    --warmup 500 \
    --samples 500 \
    --chains 2

echo "finished $(date -Is)"
