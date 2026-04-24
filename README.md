# SymPG — Synthetic Multivariate Data Generator

SymPG generates simulated datasets with arbitrary marginal distributions and a
user-specified correlation structure, using the **Gaussian copula** approach. This was originally developed for DNA-methylation (CpG) data simulation.

[![PyPI version](https://badge.fury.io/py/sympg.svg)](https://pypi.org/project/sympg/)
[![Python versions](https://img.shields.io/pypi/pyversions/sympg.svg)](https://pypi.org/project/sympg/)

**SymPG** is a lightweight, high-performance Python package for generating synthetic multivariate tabular data using a Gaussian Copula framework. 

It allows researchers and data scientists to simulate complex datasets where variables have drastically different distribution shapes (like highly skewed Beta distributions or bimodal mixtures) while enforcing precise, user-defined rank (Spearman) correlations between them.

## Key Features
* **Flexible Marginal Distributions**: Seamlessly map variables to standard SciPy distributions, custom functions, or complex bimodal mixture models.
* **Block-Diagonal Grid Simulation**: Easily generate massive "wide" datasets testing every combination of variables across multiple correlation levels in one vectorized shot.
* **Positive-Definite Repair**: Automatically repairs intended correlation matrices that lose positive-definiteness.
* **Automated Metadata**: Automatically tracks variables, generated parameters, and both intended vs. observed pairwise correlations.

---

## Installation

Install SymPG via `pip`:

```bash
pip install sympg
```

*Requires Python 3.8+ and depends on `numpy`, `pandas`, `scipy`, and `scikit-learn`.*

---

## Quickstart Tutorial

This tutorial demonstrates how to simulate a dataset with three drastically different variable shapes (a heavily skewed Beta, a standard Normal, and a bimodal Mixture) while forcing them to correlate with one another at $r=0.5$.

### 1. Define your distributions
You can define variables using simple strings, dictionaries with custom parameters, or pass SciPy distribution objects directly.

```python
import sympg
from scipy.stats import norm

distributions = [
    # 1. A highly skewed beta distribution (e.g., modeling DNA methylation)
    dict(name="skewed", type="beta", a=9.8, b=430.2),
    
    # 2. A standard normal control
    "normal",
    
    # 3. A custom bimodal mixture model
    dict(
        name="bimodal.mix", 
        type="mixture", 
        submodels=[norm(loc=0.2, scale=0.05), norm(loc=0.8, scale=0.05)], 
        weights=[0.5, 0.5]
    )
]
```

# 2. Define one or more correlation scenarios

```python
correlations=[0.01, 0.5, 0.9]  # The intended Spearman correlation for all pairs
```

### 3. Run the simulation grid
Use `simulate_grid` to generate the data. By passing `[0.5]` as the correlation, SymPG will force every pair of these variables to correlate at 0.5. You can also pass a list like `[0.1, 0.5, 0.9]` to instantly generate multiple distinct correlation blocks.

```python

# Generate 1,000 observations
dnam = sympg.simulate_grid(
    distributions=distributions,
    correlations=correlations,
    n_obs=1000,
    random_state=42
)

# Extract the generated Pandas DataFrame
data = dnam['data']
print(data.head())
```

### 4. Analyze the generated data
Even though the variables have completely different geometric shapes, SymPG's Gaussian copula ensures the underlying rank correlations remain robust.

```python
# Check the observed Spearman correlation between the skewed beta and the mixture
observed_corr = data['skewed0'].corr(data['bimodal.mix0'], method='spearman')
print(f"Observed Spearman Correlation: {observed_corr:.3f}")
# Output: Observed Spearman Correlation: 0.46
```

### 5. Access Metadata
SymPG automatically tracks the configuration of your simulation, including the exact parameters used and the discrepancy between intended and observed correlations.

```python
metadata_corr = dnam['meta']['corr']
print(metadata_corr.head())
```

---

## Why use SymPG?
When training Machine Learning models (like Lasso or Ridge regression), passing variables with vastly different non-linear shapes causes linear Pearson correlations to collapse, forcing the model to severely underestimate the true association coefficients. 

SymPG allows you to explicitly simulate these non-linear shape mismatches in a controlled environment, making it an invaluable tool for benchmarking feature engineering pipelines (like Quantile Normal Transformations) and testing the robustness of predictive models against complex real-world data.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License

## How it works

1. Draw correlated standard-normal samples using the target correlation matrix
   (Cholesky decomposition).
2. Map each column to \[0,1\] via the normal CDF (probability integral transform).
3. Apply the inverse CDF of the desired marginal distribution to each column.

This preserves the rank correlation structure while allowing any marginal shape.


### Multiple distribution sets

Pass a **list of lists** to sweep both distribution types and correlation levels:

### Single dataset

```python
single = sympg.simulate(
    distributions=["normal", dict(name="beta", a=2, b=5)],
    correlations=[0.7],   # one pair → one value
    n_obs=1000,
    var_names=["x", "y"],
    random_state=1,
)
```

---

## Built-in distributions

| Key | Scipy basis | Key parameters |
|---|---|---|
| `normal` | `scipy.stats.norm` | `loc`, `scale` |
| `beta` | `scipy.stats.beta` | `a`, `b` |
| `gamma` | `scipy.stats.gamma` | `a`, `scale` |
| `gamma_norm` | `scipy.stats.gamma` (min–max normalised) | `a`, `scale` |
| `lognormal` | `scipy.stats.lognorm` | `s`, `scale` |
| `uniform` | `scipy.stats.uniform` | `low`, `high` |
| `truncnorm` | `scipy.stats.truncnorm` | `a`, `b`, `loc`, `scale` |
| `mixture` | custom mixture of 3 normal distributions | |

Custom callables are also accepted: `f(uniform_samples: np.ndarray) -> np.ndarray`.

---

## Diagnostics helpers

```python
# Observed correlation summary across all scenarios
corr_summary = sympg.summarise_correlations(df, var_names=["phenotype", "CpG1", "CpG2"])
```

---

## Saving results

```python
df.to_csv("simulated_grid.csv", index=False)
```

---