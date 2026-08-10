# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2](https://github.com/ricardogr07/JAX-BO/compare/v0.2.1...v0.2.2) (2026-08-07)


### Bug Fixes

* problem-scaled Cholesky jitter in the multifidelity and mcmc models ([#75](https://github.com/ricardogr07/JAX-BO/issues/75)) ([27fbcda](https://github.com/ricardogr07/JAX-BO/commit/27fbcda4f92555cede04f9f93c424cfd4e65ad6e)), closes [#71](https://github.com/ricardogr07/JAX-BO/issues/71)
* scale the Cholesky jitter to the problem, stop clipping negative variances ([#74](https://github.com/ricardogr07/JAX-BO/issues/74)) ([d63e424](https://github.com/ricardogr07/JAX-BO/commit/d63e4244e574078dd8de0f97b441e6f91125037b))
* use minimize_lbfgs_grad in extras models, add round-trip smoke tests ([#76](https://github.com/ricardogr07/JAX-BO/issues/76)) ([2adbfbe](https://github.com/ricardogr07/JAX-BO/commit/2adbfbee836d55578e470a1535d2db52fceef28e)), closes [#71](https://github.com/ricardogr07/JAX-BO/issues/71)

## [0.2.1](https://github.com/ricardogr07/JAX-BO/compare/v0.2.0...v0.2.1) (2026-08-05)


### Bug Fixes

* authenticate release-please with a PAT so tags trigger release.yml ([#73](https://github.com/ricardogr07/JAX-BO/issues/73)) ([7287ddd](https://github.com/ricardogr07/JAX-BO/commit/7287ddd3c5ebfe1ffb968a6f990a36baaee915d3)), closes [#66](https://github.com/ricardogr07/JAX-BO/issues/66)


### Performance Improvements

* batched-start acquisition multi-start via score_candidates (3c) ([#72](https://github.com/ricardogr07/JAX-BO/issues/72)) ([27f607f](https://github.com/ricardogr07/JAX-BO/commit/27f607fecd32a4cb26a51dc6d9511adf87e33ba8))
* GP.train multi-start fully on device, vmapped across restarts (3b) ([#70](https://github.com/ricardogr07/JAX-BO/issues/70)) ([469c0b6](https://github.com/ricardogr07/JAX-BO/commit/469c0b643edffc09b70b5873bc11e2a512d684fe))
* key the GP jit cache on config, not instance identity (3a) ([#68](https://github.com/ricardogr07/JAX-BO/issues/68)) ([2e2bb87](https://github.com/ricardogr07/JAX-BO/commit/2e2bb873282d91e7e790e413d4b0eef4ae85a8a9))

## [0.2.0](https://github.com/ricardogr07/JAX-BO/compare/v0.1.2...v0.2.0) (2026-08-01)


### ⚠ BREAKING CHANGES

* EI and EIC use the textbook closed form for mean > best ([#61](https://github.com/ricardogr07/JAX-BO/issues/61))

### Features

* acquisitions.score_candidates, vmap-batched candidate scoring (2c) ([#54](https://github.com/ricardogr07/JAX-BO/issues/54)) ([19649c9](https://github.com/ricardogr07/JAX-BO/commit/19649c9903cea8b3ec9584c4f7b201e1433d33aa))
* Docker quickstart image on GHCR, published from the release workflow (4g) ([#60](https://github.com/ricardogr07/JAX-BO/issues/60)) ([9cda188](https://github.com/ricardogr07/JAX-BO/commit/9cda188b2330c0dfdd2c1cc83911cc5982c78258))


### Bug Fixes

* EI and EIC use the textbook closed form for mean &gt; best ([#61](https://github.com/ricardogr07/JAX-BO/issues/61)) ([f6e5c47](https://github.com/ricardogr07/JAX-BO/commit/f6e5c477c250ee1d282262a08b87655650966d9f))
* LCBC and LW_LCBC use the shared NaN-free constraint feasibility ([#63](https://github.com/ricardogr07/JAX-BO/issues/63)) ([f3f9f63](https://github.com/ricardogr07/JAX-BO/commit/f3f9f633a6ea5b14b084472e1f963be11b41078a)), closes [#62](https://github.com/ricardogr07/JAX-BO/issues/62)
* migrate clip a_min/a_max kwargs to positional (JAX 0.10 compat) ([#43](https://github.com/ricardogr07/JAX-BO/issues/43)) ([6350647](https://github.com/ricardogr07/JAX-BO/commit/63506476a1fbf16b605d65bc7eb2e4e19216d49d))
* pin jax to &gt;=0.6,&lt;0.11, require python 3.10, commit uv lockfile ([#52](https://github.com/ricardogr07/JAX-BO/issues/52)) ([4b0ee23](https://github.com/ricardogr07/JAX-BO/commit/4b0ee2329d4077e95304de82ba22ea48a28444d6))
* replace pyDOE with scipy.stats.qmc.LatinHypercube ([#44](https://github.com/ricardogr07/JAX-BO/issues/44)) ([062ab8e](https://github.com/ricardogr07/JAX-BO/commit/062ab8edf998834ad1b77ce0023fca56d6fead79))


### Documentation

* add community files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, templates, CITATION.cff (4a) ([#58](https://github.com/ricardogr07/JAX-BO/issues/58)) ([dee1b70](https://github.com/ricardogr07/JAX-BO/commit/dee1b705241d23511e8b3ccf4314f01f98b9abad))
* rebuild README: badges, quickstart, extras table, support matrix, acknowledgment (4b) ([#59](https://github.com/ricardogr07/JAX-BO/issues/59)) ([3e30113](https://github.com/ricardogr07/JAX-BO/commit/3e30113e53a337217861912d10fb3e75e10aca35))
* SCOPE v2, expanded revamp scope + audit baseline + guardrails ([#14](https://github.com/ricardogr07/JAX-BO/issues/14)) ([1ce21ff](https://github.com/ricardogr07/JAX-BO/commit/1ce21ff11250c19cb4f98d80dc8922cd76029a7a))


### Miscellaneous Chores

* release-please manifest config, keep pre-1.0 breaking bumps on 0.x ([#64](https://github.com/ricardogr07/JAX-BO/issues/64)) ([3ebffc8](https://github.com/ricardogr07/JAX-BO/commit/3ebffc855850712afa474305733acae190f7327d))

## [0.1.2](https://github.com/ricardogr07/JAX-BO/compare/v0.1.1...v0.1.2) (2025-07-21)


### Bug Fixes

* fix release pipeline, changelog, and trigger logic ([ef0150c](https://github.com/ricardogr07/JAX-BO/commit/ef0150c5d857e81527e78b1f6780648f386838dd))

## [0.1.1](https://github.com/ricardogr07/JAX-BO/compare/v0.1.0...v0.1.1) (2025-07-21)


### Bug Fixes

* fix release pipeline ([#8](https://github.com/ricardogr07/JAX-BO/issues/8)) ([d20575e](https://github.com/ricardogr07/JAX-BO/commit/d20575e0e408c93afc12a16abe34c0b7de264bbb))

## 0.1.0 (2025-07-21)


### Bug Fixes

* release-please pipeline ([#6](https://github.com/ricardogr07/JAX-BO/issues/6)) ([d8a69af](https://github.com/ricardogr07/JAX-BO/commit/d8a69afb496f4ae82fd8fb97e7e5105307ebc31b))

### Added
- Compatibility with Python 3.12 and latest `jax` / `jaxlib` releases
- `minimize_lbfgs_grad` as separate gradient-based optimizer
- Descriptive comments and usage instructions for all optimizer functions

### Fixed
- Multiple runtime errors due to deprecated JAX APIs
- Bug in `compute_next_point_lbfgs` when missing `rng_key`
- Missing `kappa` causing `KeyError` for `LW-LCB` and similar criteria
- Installation issues due to old `jaxlib` version pinning

### Changed
- Refactored optimizer interface for better debuggability
- README updated to reflect fork, license, and credit to original authors
- Modularized model internals for improved extensibility

---

## [0.2.0] 2020 (Original Release)
Initial release by Predictive Intelligence Lab.
