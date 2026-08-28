"""cProfile artifact for one process-cold GP.train call (n=128, num_restarts=3).

The profiled call is wrapped in jax.block_until_ready so the profile
covers the same synchronized work the bench timings measure.

cProfile is used instead of jax.profiler to avoid the TensorBoard UI
dependency stack; the interesting hotspots here are Python-level anyway
(restart loop, scipy L-BFGS, jit dispatch and compilation).

Run:  .venv/Scripts/python.exe benchmarks/profile_train.py
Writes results/2026-07-28-train-cprofile.prof and a top-40 text dump.
"""

import cProfile
import io
import logging
import pstats
from pathlib import Path

import jax
import jax.random as jax_random

from conftest import NUM_RESTARTS, SEED, make_problem

LOGGER = logging.getLogger(__name__)
OUT = Path(__file__).parent / "results"
"""Directory receiving profile artifacts."""


def main():
    """Run the main benchmark."""
    OUT.mkdir(exist_ok=True)
    gp, batch, _, _ = make_problem(128)
    seed = SEED
    key = jax_random.PRNGKey(seed)

    prof = cProfile.Profile()
    prof.enable()
    jax.block_until_ready(gp.train(batch, key, num_restarts=NUM_RESTARTS))
    prof.disable()

    prof.dump_stats(str(OUT / "2026-07-28-train-cprofile.prof"))
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(40)
    (OUT / "2026-07-28-train-cprofile.txt").write_text(s.getvalue())
    LOGGER.info("%s", s.getvalue()[:3000])


if __name__ == "__main__":
    main()
