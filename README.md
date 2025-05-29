# JAX-BO (Extended): Bayesian Optimization in JAX

This is a modified and extended version of the original [JAX-BO](https://github.com/PredictiveIntelligenceLab/JAX-BO) library for Bayesian optimization, with improved compatibility and enhancements for modern Python and JAX versions.

---

## Getting Started

### Installation

You can install the latest version from PyPI:

```bash
pip install jaxbo
```

Launch the interactive tutorial on Google Colab:  
[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ricardogr07/JAX-BO/blob/master/jaxbo_colab.ipynb)

---

## Maintainer and Fork Information

This fork is maintained by Ricardo García Ramírez, as of May 2025.

### Summary of Modifications

- Updated for compatibility with Python 3.12
- Migrated to recent versions of `jax` and `jaxlib`
- Fixed and tested all demo notebooks and example scripts
- Added detailed documentation to all public functions and modules
- Improved error handling and logging output
- Refactored and expanded optimizer functionality
- Clarified model design and acquisition strategy logic

> **Note:** This fork is **not affiliated** with the original authors. It is maintained independently to support downstream research applications.

---

## Original Project

This project is based on the original [JAX-BO](https://github.com/PredictiveIntelligenceLab/JAX-BO) library developed by the [Predictive Intelligence Lab](https://github.com/PredictiveIntelligenceLab) at the University of Pennsylvania.

---

## Citation (Original Work)

If you use this library in your research, please cite the original authors:

```bibtex
@software{jaxbo2020github,
  author = {Paris Perdikaris, Yibo Yang, Mohamed Aziz Bhouri},
  title = {{JAX-BO}: A Bayesian optimization library in {JAX}},
  url = {https://github.com/PredictiveIntelligenceLab/JAX-BO},
  version = {0.2},
  year = {2020},
}
```
---

## Changelog ##

All modifications and release notes are documented in the [CHANGELOG](CHANGELOG.md) file.

---
## License ##
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.