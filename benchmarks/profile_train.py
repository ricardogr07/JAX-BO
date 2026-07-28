"""cProfile artifact for one cold GP.train call (n=128, num_restarts=3).

cProfile is used instead of jax.profiler to avoid the TensorBoard UI
dependency stack; the interesting hotspots here are Python-level anyway
(restart loop, scipy L-BFGS, jit dispatch and compilation).

Run:  .venv/Scripts/python.exe benchmarks/profile_train.py
Writes results/2026-07-28-train-cprofile.prof and a top-40 text dump.
"""

import cProfile
import io
import pstats
from pathlib import Path

from jax import random

from conftest import NUM_RESTARTS, SEED, make_problem

OUT = Path(__file__).parent / "results"


def main():
    OUT.mkdir(exist_ok=True)
    gp, batch, _, _ = make_problem(128)
    key = random.PRNGKey(SEED)

    prof = cProfile.Profile()
    prof.enable()
    gp.train(batch, key, num_restarts=NUM_RESTARTS)
    prof.disable()

    prof.dump_stats(str(OUT / "2026-07-28-train-cprofile.prof"))
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(40)
    (OUT / "2026-07-28-train-cprofile.txt").write_text(s.getvalue())
    print(s.getvalue()[:3000])


if __name__ == "__main__":
    main()
