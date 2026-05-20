# 🚧 Work in Progress

Supplementary information for the journal article:

**Surrogate Modeling Framework for Cantilever Sheet Pile Wall Stability Assessment: Case Study Campus Ullevål**

## Navigation

- [Article reference](#article-reference)
- [Supplementary Python content](#supplementary-python-content)

## Article reference

- **Authors:** Andreas‑Nizar Granitzer · Hilde Aas Nøst · Egil Monsås · Georg Erharter · Johannes Leo
- **Title:** Surrogate Modeling Framework for Cantilever Sheet Pile Wall Stability Assessment: Case Study Campus Ullevål
- **Accepted:** 16 May 2026
- **DOI:** https://doi.org/10.1007/s10706-026-03740-3

## Supplementary Python content

The `campus_smf` package contains classes and functions for:

- **Surrogate model building**
  - `LinearSurrogateModel`
  - `fit_linear_surrogate(...)`
  - `SurrogateModelBuilder.from_pairs(...)`
- **Data source descriptions**
  - `DataSource`
  - `DataSourceRegistry`
  - `default_campus_ullevall_data_sources()`
