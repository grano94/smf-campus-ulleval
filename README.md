## Content

Supplementary information assoicated with the published journal article:
**Surrogate Modeling Framework for Cantilever Sheet Pile Wall Stability Assessment: Case Study Campus Ullevål**

## Article reference

- **Authors:** Andreas‑Nizar Granitzer · Hilde Aas Nøst · Egil Monsås · Georg Erharter · Johannes Leo
- **Title:** Surrogate Modeling Framework for Cantilever Sheet Pile Wall Stability Assessment: Case Study Campus Ullevål
- **Accepted:** 16 May 2026
- **DOI:** https://doi.org/10.1007/s10706-026-03740-3

## Usage

The main package is `deepxtwin`, organized into subpackages for different functionalities. *(Note: Code associated with parametric FE modelling has not been uploaded due to size limitations.)*

Project structure:
```
├───data                    # Parametric FE input data and FoS predictions (500 samples).
├───deepxtwin
│   ├───sampling            # Latin Hypercube Sampling for sampling plan.
│   ├───surrogate           # Construction and analysis.
config.toml                 # Configuration settings (TOML format)
main.py                     # Top-level script for parametric FE simulations
poetry.lock                 # Dependency lock file
pyproject.toml              # Project metadata and dependencies
test_ngilive.py             # Testing script
```

## Installation

This project uses `poetry`. Install by running:
```bash
poetry install  # Install all dependencies
poetry shell    # Activate virtual environment
```

