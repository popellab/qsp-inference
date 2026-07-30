#!/bin/bash
# QSP-scale population fit (docs ch. 4) on a Slurm cluster. Written against JHU
# Rockfish, whose default partition is "parallel" (there is no defq); that is the
# only cluster-specific line.
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
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CPUS="${SLURM_CPUS_PER_TASK:-8}"

# One host device per core. Devices are virtual, but they are the only axis JAX
# parallelises well on CPU: the true-ODE solves are pmapped across them and each
# NUTS chain runs on one. Threads are therefore set to 1 per device rather than
# many -- with a device per core, extra threads only oversubscribe.
#
# This replaces a much worse setting. Giving 4 devices and CPUS/4 threads each
# measured 4.4 cores of 24 during the emulator build, because a vmapped diffrax
# solve is not automatically threaded by the XLA CPU backend. That build took
# 15.5 minutes.
export TOY_HOST_DEVICES="$CPUS"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
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

echo "host $(hostname), ${CPUS} cpus, ${TOY_HOST_DEVICES} host devices, OMP_NUM_THREADS=${OMP_NUM_THREADS}"
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
    --chains 4

echo "finished $(date -Is)"
