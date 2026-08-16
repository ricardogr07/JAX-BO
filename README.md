# jaxbo: Bayesian optimization in JAX

[![CI](https://github.com/ricardogr07/JAX-BO/actions/workflows/ci.yml/badge.svg)](https://github.com/ricardogr07/JAX-BO/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/jaxbo)](https://pypi.org/project/jaxbo/)
[![License](https://img.shields.io/github/license/ricardogr07/JAX-BO)](LICENSE)

A maintained, modern-JAX Bayesian optimization library: a small stable core
(exact GP regression, 5 kernels, the classic acquisition functions plus a
batched `score_candidates` helper, input priors, L-BFGS training) with
optional research extras (MCMC inference, multifidelity and manifold GPs,
weighted sampling) that install on demand and are never paid for otherwise.

This is a maintained fork of
[PredictiveIntelligenceLab/JAX-BO](https://github.com/PredictiveIntelligenceLab/JAX-BO);
see the [acknowledgment](#acknowledgment) below.

## Install

```bash
pip install jaxbo

# The quickstart below needs 0.2.2 (score_candidates, extras). Until it is
# on PyPI, install from the repo:
# pip install "jaxbo @ git+https://github.com/ricardogr07/JAX-BO"
```

The core installs with exactly 4 dependencies: `jax`, `jaxlib`, `numpy`,
`scipy`. Everything else lives behind optional extras:

| Install | Adds | Extra dependencies |
|---|---|---|
| `pip install jaxbo` | core: `jaxbo.gp`, `jaxbo.kernels`, `jaxbo.acquisitions`, `jaxbo.optimizers`, `jaxbo.priors`, `jaxbo.test_functions` | none |
| `pip install jaxbo[mcmc]` | `jaxbo.mcmc`: NUTS-based GP models | numpyro |
| `pip install jaxbo[multifidelity]` | `jaxbo.multifidelity`: multifidelity, manifold, gradient, and multi-output GPs | none |
| `pip install jaxbo[weighted]` | `jaxbo.weights`: GMM/KDE weighted acquisition machinery | scikit-learn, KDEpy |
| `pip install jaxbo[all]` | all of the above | union |

`import jaxbo` never imports an extra's dependencies; importing an extra
without them raises an ImportError naming the `pip install jaxbo[extra]`
fix.

## Supported versions

The package pins `jax>=0.6,<0.11` (matching jaxlib) and requires Python
3.10 or newer. CI tests every lane below at the jax floor and the newest
jax the pin allows:

| Python | jax tested | Status |
|---|---|---|
| 3.10 | 0.6.0 to 0.6.2 | supported (0.6.2 is the last jax with 3.10 wheels) |
| 3.11 | 0.6.0 to 0.10.2 | supported |
| 3.12 | 0.6.0 to 0.10.2 | supported |
| 3.13 | 0.6.0 to 0.10.2 | tested, advisory until the CI lanes hold a green streak |
| 3.14 | 0.7.2 to 0.10.2 | tested, advisory until the CI lanes hold a green streak |

## Quickstart: 60 seconds to the next point

Fit a GP to observations of an objective, then score a batch of candidates
with expected improvement in one vmapped pass:

```python
import jax.numpy as jnp
import numpy as np
from jax import random

from jaxbo.acquisitions import score_candidates
from jaxbo.gp import GP
from jaxbo.priors import uniform_prior
from jaxbo.utils import normalize


def f(x):
    return ((x - 1.5) ** 2).ravel()


# Domain and observations (raw domain)
lb, ub = jnp.array([-2.0]), jnp.array([3.0])
bounds = {"lb": lb, "ub": ub}
X = jnp.linspace(-2.0, 3.0, 8)[:, None]
y = f(X)

# GP expects an already normalized training batch
batch, norm_const = normalize(X, y, bounds)

gp = GP({"kernel": "RBF", "input_prior": uniform_prior(lb, ub), "criterion": "EI"})
params = gp.train(batch, random.PRNGKey(0), num_restarts=5)

# Score 256 raw-domain candidates in one vmapped pass and pick the best
X_cand = np.linspace(-2.0, 3.0, 256)[:, None]
scores = score_candidates(
    gp, X_cand, params=params, batch=batch, bounds=bounds,
    best=float(np.min(batch["y"])),
)
x_next = X_cand[np.argmin(scores)]
print("next point to evaluate:", x_next)  # close to the true minimum at 1.5
```

Two contract points worth knowing before you swap in a real objective:

- **Normalization:** `train` consumes `batch["X"]` exactly as given, so
  pass the already normalized batch (`utils.normalize`). `predict` and
  `score_candidates` normalize raw-domain inputs internally against
  `bounds`. Normalized batch in, raw candidates in; mixing these up fails
  silently.
- **Scores:** lower is better for every acquisition in
  `jaxbo.acquisitions` (EI is returned negated), so the next point is
  `X_cand[np.argmin(scores)]`.

Prefer a notebook? Launch the interactive tutorial on Colab:
[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ricardogr07/JAX-BO/blob/main/jaxbo_colab.ipynb)

## Docker

A quickstart image with `jaxbo[all]`, JupyterLab, and the `examples/`
notebooks is published to GHCR with each release, starting with 0.2.0:

```bash
docker run -p 8888:8888 ghcr.io/ricardogr07/jaxbo
```

Then open the printed `http://127.0.0.1:8888/lab?token=...` URL.

## Project documentation

- [CONTRIBUTING.md](CONTRIBUTING.md): dev setup (uv + committed lockfile),
  tox envs, the CI gate map, PR rules, and the benchmark delta rule
- [SCOPE.md](SCOPE.md): the 0.2.0 revamp scope, architecture, and locked
  decisions
- [benchmarks/](benchmarks/README.md): the performance harness and the "no
  optimization without a delta" rule
- [docs/audits/2026-07-28-reposage-baseline.md](docs/audits/2026-07-28-reposage-baseline.md): RepoSage revamp baseline
- [docs/audits/2026-08-15-reposage-rc.md](docs/audits/2026-08-15-reposage-rc.md): RepoSage release-candidate audit
- [benchmarks/results/2026-08-15-rc-final.md](benchmarks/results/2026-08-15-rc-final.md): final benchmark summary and provenance

- [CHANGELOG.md](CHANGELOG.md): release notes, generated by release-please

## Final benchmark evidence

Four clean runs from the existing `.venv`, using `.venv\Scripts\python.exe -m pytest benchmarks -q`, produced raw JSON artifacts in `benchmarks/results/2026-08-15-rc-run{1,2,3,4}.json`. Values below are medians across the four run medians. IQRs and cross-session noise bands, including comparison ratios, are in `benchmarks/results/2026-08-15-rc-final.md`.

| Bench | RC median | vs 2026-07-28 | vs v0.2.0 |
|---|---:|---:|---:|
| Train warm, n=32 | 16.53 ms | 0.19x | 0.31x |
| Train warm, n=128 | 110.84 ms | 0.62x | 0.80x |
| Train warm, n=512 | 715.70 ms | 0.29x | 0.40x |
| Train fresh, n=128 | 101.37 ms | 0.16x | 0.20x |
| Predict, 256 points | 1.35 ms | 1.26x | 1.36x |
| EI consumer, 256 candidates | 116.83 ms | 0.85x | 0.96x |
| EI fused, 256 candidates | 65.63 ms | 0.86x | 0.92x |
| `score_candidates`, 256 candidates | 1.38 ms | N/A | 0.86x |

The cross-session noise bands are wide. Ratios within those bands are not treated as meaningful regressions. The 2026-07-28 baseline predates `score_candidates`, so its comparison is N/A.

## Acknowledgment

This project is a fork of
[JAX-BO](https://github.com/PredictiveIntelligenceLab/JAX-BO) by the
[Predictive Intelligence Lab](https://github.com/PredictiveIntelligenceLab)
at the University of Pennsylvania (Paris Perdikaris, Yibo Yang, Mohamed
Aziz Bhouri). The core model structure, kernels, and acquisition functions
originate there; this fork modernizes the library (current jax/python
support, core/extras packaging, tests, benchmarks, CI) and is maintained
independently by Ricardo García Ramírez. It is not affiliated with the
original authors.

If you use this library in your research, please cite the original work
(see also [CITATION.cff](CITATION.cff)):

```bibtex
@software{jaxbo2020github,
  author = {Paris Perdikaris, Yibo Yang, Mohamed Aziz Bhouri},
  title = {{JAX-BO}: A Bayesian optimization library in {JAX}},
  url = {https://github.com/PredictiveIntelligenceLab/JAX-BO},
  version = {0.2},
  year = {2020},
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
