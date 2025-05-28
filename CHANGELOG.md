# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] – 2025-05-28
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
