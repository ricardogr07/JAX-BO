# jaxbo 0.2.0 revamp: scope v2

Status: DRAFT v2, supersedes the v1 scope previously on this branch. Pending Ricardo's approval; not pushed until he approves the text. Refs ricardogr07/forja#13; implementation epic parent is forja#17, sub-issues live in ricardogr07/JAX-BO via `orchestration/epics/jaxbo-revamp.json`. Session guardrails: `AGENTS.md` (fork-only rule, benchmark-delta rule, house rules). All workers in this repo read `AGENTS.md` first.

## 1. Why revamp

Two independent lines of evidence say the library is broken as shipped, and both become the revamp's before/after story.

**The audit baseline is 0/6.** reposage 0.4.1 (deterministic Six Standards audit, DS/ML profile) graded the repo 0/6 on 2026-07-28: `docs/audits/2026-07-28-reposage-baseline.md` (+ `.json`). Every standard fails: `import jaxbo` dies on a clean install (`ModuleNotFoundError: pyDOE`), pytest collects 0 tests, no lockfile, docstring coverage 52% (gate is 70%), no container, and the deploy workflow is not gated on tests. The audit re-runs at RC; the 0/6-baseline-to-6/6-target arc is the headline metric of the whole revamp. Its 10-item fix list is folded into the slices in section 7.

**Both real consumers ship workarounds.**

- **shelter-pulse** (`Dockerfile` lines 13 to 20) carries two shims: it writes a fake `pyDOE.py` module into site-packages (`from pydoe import *`) because jaxbo imports the dead `pyDOE` package name, and it runs `sed` over every installed `jaxbo/*.py` to rewrite `np.clip(x, a_min=...)` to positional form because the `a_min` keyword was removed in modern JAX (0.10+).
- **ecg-purkinje-npe** avoids PyPI entirely: it vendors a full editable copy at `packages/jax-bo` for its planned BO+ABC baseline.

**Defect inventory** (verified by grep, 2026-07-28): 6 model files + 2 examples import the dead `pyDOE` package; 23 `clip(a_min=)` call sites are broken on modern jax (17 in `models/`, 4 in `mcmc_models.py`, 1 in `utils.py`, 1 in `acquisitions.py` EIC; EI was already migrated, so the codebase is inconsistent today); jax is unpinned, the root cause of the clip breakage; not one of the 12 model classes has a test; `requires-python >= 3.6` is untested fiction; and numpyro, scikit-learn, KDEpy, and pyDOE are hard install-time and import-time dependencies even for a consumer that only wants GP + EI (`mcmc_models` is imported eagerly by `__init__.py` and defines a second class named `GP` that shadows `models.GP`).

**The fork is the project.** All work happens on github.com/ricardogr07/JAX-BO. Upstream (PredictiveIntelligenceLab/JAX-BO) is credited prominently (README acknowledgment section + CITATION.cff, section 6) but is never touched in any way: no PRs, no issues, no comments, no pushes. The rule and its enforcement mechanics live in `AGENTS.md`.

## 2. What revamped jaxbo IS

A maintained, modern-JAX Bayesian optimization library with a small stable core and optional research extras. The core: exact GP regression, 5 kernels, the 10 classic acquisition functions plus a new batched scoring helper, input priors, and L-BFGS training. The extras: MCMC inference, the multifidelity/manifold model family, and the weighted-sampling machinery, each installable on demand and never paid for otherwise. It installs clean on pinned, tested jax versions, carries tests, benchmarks, executable notebooks, a Docker quickstart, and full open source hygiene. It is the BO engine for shelter-pulse and the ecg-purkinje-npe baseline, and a credible small library for anyone else.

**Compatibility promise:** the shelter-pulse call pattern works unchanged in 0.2.0: `GP` dict options, `train(batch, rng_key, num_restarts)`, `predict(X_star, params=, batch=, bounds=)`, `EI(mu, std, best)`, `input_priors.uniform_prior`. **Normalization contract (as it exists today, callers must follow it):** `train` consumes `batch["X"]` exactly as given, so the caller passes an already normalized batch (unit cube via `utils.normalize`), while `predict` normalizes `X_star` internally against `bounds` and so takes raw-domain points (`jaxbo/models/gp_model.py:41-58, 85-94`). Pass normalized `batch`, raw `X_star`; mixing these up fails silently. Consumer ground truth (verified by grep, 2026-07-28): shelter-pulse (`shelterpulse/optimize/jaxbo_optimizer.py`) imports exactly `jaxbo.models.GP`, `jaxbo.acquisitions` (uses `EI`), and `jaxbo.input_priors` (uses `uniform_prior`), and hand-rolls a 256-candidate Python EI loop that `score_candidates` replaces. ecg-purkinje-npe has zero first-party jaxbo imports yet; it only declares the vendored copy as its optional `baseline` extra, and its planned baseline is the same GP + EI pattern the core serves.

## 3. Architecture: core + optional extras

```
jaxbo/                  core, target deps: jax, jaxlib, numpy, scipy
  gp.py                 GPmodel base + GP (Matern52 et al), train/predict
  kernels.py            5 kernels (jit): RBF, Matern52, Matern32, Matern12, RatQuad
  acquisitions.py       10 acquisitions + NEW score_candidates (vmap batched)
  optimizers.py         minimize_lbfgs / minimize_lbfgs_grad
  priors.py             uniform_prior, gaussian_prior et al
  initializers.py       trimmed; LHS via qmc.LatinHypercube seeded from the
                        caller's rng_key (PR #44 pattern, see slice 1a)
  utils.py              trimmed
  test_functions.py     kept (no deps beyond the jax core + input_priors,
                        CI smoke + quickstart)
jaxbo/mcmc/             extra [mcmc] (numpyro); MCMC GP class renamed, it no
                        longer shadows models.GP; lazy import, core never
                        imports it
jaxbo/multifidelity/    extra [multifidelity]: 6 MF models + 2 manifold models
                        + GradientGP + MultipleIndependentOutputsGP on a
                        shared base
jaxbo/weights.py        extra [weighted] (scikit-learn, KDEpy): fit_gmm /
                        fit_kernel_density plus the model-method weight paths
                        (section 4, decision 7); LW_* pure functions stay in
                        core taking precomputed weights as plain arguments;
                        gmm_vars-driven flows move behind [weighted]
[all] = mcmc + multifidelity + weighted
```

Rules of the architecture:

- **Core installs with exactly 4 dependencies** (`jax`, `jaxlib`, `numpy`, `scipy`) and `python -c "import jaxbo"` succeeds WITHOUT numpyro, scikit-learn, or KDEpy installed. Extras are lazy: importing `jaxbo` never triggers an extra's dependency; importing an extra without its dependency raises a clear error naming the `pip install jaxbo[extra]` fix. **This is the TARGET state that slices 2a/2b deliver, not the current one.** Today the core import graph reaches every extras dependency: `jaxbo/__init__.py:16-26` imports `mcmc_models` (numpyro) eagerly, `models/__init__.py:16-29` imports the full multifidelity/manifold/gradient/multiple-output family eagerly, `models/base_gpmodel.py:10-17` imports pyDOE and sklearn, and `utils.py:4-10` imports KDEpy and `jax.example_libraries.stax`. 2a/2b acceptance therefore includes: break the import graph so no core module transitively imports an extras dependency, specifically severing the `utils.py` sklearn/KDEpy/stax reach and splitting `models/__init__.py` so core exposes only `GPmodel` + `GP`.
- Every surviving file: pyDOE gone, clip sites swept to positional, consistent `jnp` usage, type hints and docstrings (reposage s1 gates docstring coverage at 70%).
- `serializable.py`'s MF param serialization moves with the multifidelity extra it serves.

## 4. Decisions locked by Ricardo

These were v1's open questions. They are closed. Do not reopen them in PRs; a PR that contradicts a locked decision is wrong by definition.

| # | Decision | Locked outcome |
|---|---|---|
| 1 | Package shape | **Core + optional extras, not the v1 drop list.** The research modules (mcmc, multifidelity, manifold, GradientGP, MultipleIndependentOutputsGP, weighted sampling) move into extras instead of being deleted. |
| 2 | Python floor | **3.10**, tested through 3.14. 3.13 and 3.14 start allowed-to-fail (jax wheel availability governs) and are promoted to required when green. |
| 3 | jax pin | **`jax>=0.6,<0.11`** with matching jaxlib. CI exercises floor and latest; the ceiling is raised only on green CI. |
| 4 | `score_candidates` | **Lands in 0.2.0**, in core `acquisitions`. It is the ergonomics friction that motivated the revamp; shipping it lets shelter-pulse delete its 256-candidate Python loop in the same bump. |
| 5 | Examples | **MORE, not fewer.** v1's delete-examples proposal is reversed. The 13 existing notebooks get fixed and mapped core-vs-extras (core tutorials against 0.2.0 core; MF/manifold tutorials become the extras' documentation), plus a new 5-minute quickstart. Execution-gated in CI via pytest-nbmake (quickstart + one per extra). |
| 6 | `test_functions` | **Kept in core.** No dependencies beyond the jax core (it imports jax and `jaxbo.input_priors`), tested, used by CI smoke tests and the quickstart. |
| 7 | Weighted machinery | **Moves to the `[weighted]` extra** (`jaxbo/weights.py`: `fit_gmm`, `fit_kernel_density`; deps scikit-learn + KDEpy). Today this machinery lives as model methods, not free functions, and the extraction covers that whole surface: `GPmodel.fit_gmm` (`base_gpmodel.py:139-210`, sklearn + pyDOE), the LW_* branches of `GPmodel.acquisition` that compute weights internally from `gmm_vars` via `utils.compute_w_gmm` (`base_gpmodel.py:245-278`, `utils.py:275-305`), and the constrained multifidelity variant `MultipleIndependentMFGP.fit_gmm` (`multiple_independent_mfgp.py:224-281`). The `LW_*` pure functions (`LW_LCB`, `LW_US`, `LW_CLSF`) stay in core taking precomputed weights as plain arguments; the `gmm_vars`-driven flows (criterion `LW_*` through `model.acquisition`) move behind `[weighted]` and raise a clear error naming `pip install jaxbo[weighted]` when used without it. |
| 8 | Docker | **One quickstart image:** `jaxbo[all]` + jupyterlab + `examples/`, default CMD jupyter lab, `docker run -p 8888:8888 ghcr.io/ricardogr07/jaxbo`. GHCR publish wired into `release.yml`. |
| 9 | Release | **0.2.0 via release-please**, changelog with an explicit Removed section covering every module that moved or was renamed. Conventional commits only; every future removal or signature change gets a changelog line before merge. |

## 5. Profiling and optimization

Optimization is benchmark-gated, never vibes-gated.

**Benchmark harness** (pytest-benchmark, committed baseline numbers, phase 0c):

- `GP.train` at n = 32, 128, and 512
- `predict` on a batch
- EI single-point loop vs `score_candidates` batched
- jit compile cost vs steady-state cost, split via `block_until_ready`

Plus one `jax.profiler` trace to locate the real hot spots before touching anything.

**The rule** (also in `AGENTS.md`): NO optimization lands without a before/after benchmark delta in the PR. A perf change without numbers is rejected regardless of how plausible it looks.

**Candidate optimizations** (each stands or falls on its measured delta):

| Candidate | Hypothesis |
|---|---|
| vmap the `num_restarts` training loop | restarts are embarrassingly parallel; the Python loop retraces |
| jit boundary audit | some hot paths cross the jit boundary per call instead of once |
| kill np/jnp host round-trips in hot paths | device-to-host copies inside loops dominate small-n cost |
| cholesky reuse between acquisition calls | the same factorization is recomputed per candidate today |
| dtype policy documented (x64 flag) | silent x64/x32 mixing costs both speed and reproducibility |

## 6. Open source hardening

Current hygiene surface: LICENSE only. 0.2.0 adds the full set:

- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and PR templates.
- `CITATION.cff` citing the original JAX-BO work by the PredictiveIntelligenceLab authors, with upstream credited prominently in the README acknowledgment section. Credit is loud; contact is zero (section 1 hard rule).
- README rebuilt: badges (PyPI, CI, license), 5-minute quickstart, extras table, Docker one-liner, acknowledgment section.
- CI matrix: {3.10, 3.11, 3.12, 3.13, 3.14} x {jax floor, jax latest}; 3.13/3.14 allowed-to-fail until green, then required; tox synced; the release job gated on the test job (`needs:`).

## 7. Slices: phases 0 to 5

| Phase | Slice | Size |
|---|---|---|
| 0 guardrails + baselines (W1) | 0a Upstream PRs scrubbed, upstream push URL neutered, `AGENTS.md` with the fork-only rule | S |
| | 0b reposage baseline committed (`docs/audits/`), fix list feeds the slices, re-run at RC | S |
| | 0c Benchmark harness + `jax.profiler` trace, baseline numbers committed | M |
| | 0d SCOPE v2 (this document) | S |
| 1 mechanical unbreak (W1 to W2) | 1a pyDOE to `scipy.stats.qmc`: call sites hold a JAX `rng_key`, not an int seed, so the shipped pattern (PR #44) is `qmc.LatinHypercube(d=dim, seed=int(rng_key[0])).random(n)`; dependency deleted | S |
| | 1b `clip(a_min=)` sweep to positional + grep guard in CI so it never returns | S |
| | 1c Packaging: `jax>=0.6,<0.11` pin, `requires-python >= 3.10`, prune and declare deps, lockfile committed, classifiers, metadata | S |
| 2 refactor to core + extras (W2) | 2a Core package restructure per section 3, lazy extras, exports, mcmc GP rename, docstrings + type hints to the 70% gate. Acceptance adds: resolve or document the train/predict normalization asymmetry in the public API (section 2); break the core import graph so no core module transitively imports an extras dependency, severing `utils.py`'s sklearn/KDEpy/stax reach and splitting `models/__init__.py` (section 3) | L |
| | 2b Extras subpackages moved + refactored on shared bases. Acceptance adds: extract the method-level weighted surface listed in decision 7 (`GPmodel.fit_gmm`, the `gmm_vars` LW_* acquisition branches, `MultipleIndependentMFGP.fit_gmm`); after the move, no core module imports sklearn or KDEpy | L |
| | 2c `score_candidates` batched helper with shape-convention docs | M |
| | 2d Model tests: GP train/predict round-trips (1D + 4D), extras import guards, EI sanity, `score_candidates` shapes; coverage on the 12 classes that today have none | M |
| 3 optimization (W2 to W3) | 3a The section 5 candidates, each landing only with its benchmark delta | L |
| 4 open source (W3) | 4a CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, templates, CITATION.cff | M |
| | 4b README rebuild | S |
| | 4c CI matrix {3.10 to 3.14} x {jax floor, latest}, release gated on tests, tox synced | M |
| | 4d Notebooks: fix the 13, map core-vs-extras, add quickstart, nbmake gate | L |
| | 4e Docker quickstart image on GHCR, wired into `release.yml` | M |
| 5 release + validation (W5 RC) | 5a Re-run reposage (target 6/6) + final benchmark table; 0.2.0 via release-please with Removed section | S |
| | 5b Consumer validation: shelter-pulse deletes both Dockerfile shims and builds green on 0.2.0; ecg-purkinje-npe swaps the vendored copy for the PyPI dep | M |

**reposage fix list to slice map** (all 10 baseline items land somewhere):

| # | Fix (standard) | Slice |
|---|---|---|
| 1 | Declare missing distributions in pyproject (s0.env_spec) | 1c |
| 2 | Generate and commit a lockfile (s0.lockfile) | 1c |
| 3 | Seed the unseeded random call in `base_gpmodel.py` (s0.determinism) | 1a (qmc LHS seeded from `rng_key`, PR #44) + 2a |
| 4 | Docstring coverage 52% to 70%+ (s1.docs) | 2a + 2b |
| 5 | Fix `import jaxbo` on clean install (s2.package) | 1a (pyDOE) + 2a (lazy numpyro) |
| 6 | Make pytest collect tests (s3.suite) | 2d |
| 7 | Behavioral assertions over smoke tests (s3.behavioral) | 2d |
| 8 | Container installing from a committed lockfile (s4.env_isolation) | 4e (+ 4c frozen CI install) |
| 9 | `needs: <test-job>` on the deploy job (s4.cicd) | 4c |
| 10 | Experiment metrics on the training surface (s5.metrics) | 0c (committed benchmark baselines; honest note: s5 may grade N/A for a library with no running system) |

## 8. Timeline

| Week | Milestone |
|---|---|
| W1 | Baselines + scope: guardrails (0a), reposage 0/6 baseline (0b), benchmark harness (0c), SCOPE v2 approved (0d), phase 1 unbreak started |
| W2 | Refactor core: phases 1 and 2 land, `pip install jaxbo` clean with 4 deps, `import jaxbo` works without numpyro |
| W2 to W3 | Optimization: phase 3, each change with its benchmark delta |
| W3 | OSS + notebooks + Docker: phase 4 |
| W5 | RC: re-audit targeting 6/6, final benchmark table, 0.2.0 released, both consumers validated (shims deleted, vendored copy replaced) |

## Adversarial review log

Reviewed 2026-07-28 by GitHub Codex (PR #14 review) plus a local Codex adversarial pass; findings verified against code before amending.

1. HIGH, amended section 2: documented the train/predict normalization asymmetry as the current contract; added a resolve-or-document item to slice 2a.
2. MED, amended sections 3 and 7: pyDOE replacement rewritten to the PR #44 pattern (`qmc.LatinHypercube(d=dim, seed=int(rng_key[0])).random(n)`), since call sites hold a JAX rng_key, not an int seed.
3. HIGH, amended section 3: "exactly 4 dependencies" reframed as the 2a/2b TARGET; current import graph (pyDOE, sklearn, KDEpy, stax) named with file refs; import-graph break added to 2a acceptance.
4. HIGH, amended section 3: lazy-extras claim reframed as target state; splitting the eager `models/__init__.py` added to 2a acceptance.
5. MED, amended decision 7: extraction now covers the model-method weight paths; gmm_vars-driven LW_* flows move behind `[weighted]` with a clear error.
6. MED, amended decision 7: method-level surface 2b must extract is listed explicitly, including the constrained multifidelity variant.
7. LOW, acknowledged in `docs/audits/2026-07-28-reposage-baseline.md`: erratum appended noting the PyPI publisher is `release.yml`, not `release-please.yml`; the generated table is untouched.
8. LOW, amended sections 3 and 4 (decision 6): "dependency-free" reworded to "no dependencies beyond the jax core" (it imports jax and `jaxbo.input_priors`).
