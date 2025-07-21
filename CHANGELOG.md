# Changelog

All notable changes to this project will be documented in this file.

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

## [0.2.0] – 2020 (Original Release)
Initial release by Predictive Intelligence Lab.
