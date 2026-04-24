"""
SymPG: Synthetic Multivariate Data Generator
Core simulation engine using the Gaussian copula approach.
"""

import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import block_diag
from typing import List, Dict, Optional, Union
import inspect

from .distributions import _build_distribution_registry
from .correlations import _is_pd, _nearest_pd

# ---------------------------------------------------------------------------
# Built-in distribution registry
# ---------------------------------------------------------------------------

DISTRIBUTIONS = _build_distribution_registry()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minmax_norm(cpg):
    """ Normalise values between 0 and 1. """
    norm_cpg = (cpg - cpg.min()) / (cpg.max() - cpg.min())
    return(norm_cpg)


def _apply_distribution(uniform_samples: np.ndarray, dist_spec) -> np.ndarray:

    if callable(dist_spec) and not isinstance(dist_spec, dict):
        return dist_spec(uniform_samples)
    
    if isinstance(dist_spec, str):
        dist_name, params = dist_spec, {}

    elif isinstance(dist_spec, dict):
        dist_spec = dict(dist_spec)  # don't mutate caller's dict
        
        # Resolve the actual distribution function name
        if "type" in dist_spec:
            dist_name = dist_spec.pop("type")
            # Pop 'name' so it isn't passed as a kwarg to the distribution function
            if "name" in dist_spec:
                dist_spec.pop("name")
        else:
            dist_name = dist_spec.pop("name")
            
        params = dist_spec
    else:
        raise TypeError(f"dist_spec must be str, dict, or callable; got {type(dist_spec)}")
        
    if dist_name not in DISTRIBUTIONS:
        raise ValueError(
            f"Unknown distribution '{dist_name}'. Available: {sorted(DISTRIBUTIONS.keys())}"
        )
    return DISTRIBUTIONS[dist_name](uniform_samples, **params)

def _dist_label(dist_spec) -> str:
    """Return a short human-readable name for a distribution spec."""
    if isinstance(dist_spec, str):
        return dist_spec

    if isinstance(dist_spec, dict):
        return dist_spec.get("name", dist_spec.get("type", "custom"))

    if callable(dist_spec):
        return getattr(dist_spec, "__name__", "custom")

    return str(dist_spec)


def _make_var_names(distributions: List) -> List[str]:
    """
    Extract variable names from distribution specifications and enforce uniqueness.
    If a user wants multiple of the same distribution (e.g., two normals), 
    they must explicitly provide unique names via dictionaries:
    e.g., [dict(name="norm1", type="normal"), dict(name="norm2", type="normal")]
    """
    names = []
    seen = set()
    
    for dist in distributions:
        if isinstance(dist, dict) and "name" in dist:
            name = dist["name"]
        elif isinstance(dist, dict) and "type" in dist:
            name = dist["type"]
        elif isinstance(dist, str):
            name = dist
        elif callable(dist):
            name = getattr(dist, "__name__", "custom")
        else:
            name = str(dist)
            
        if name in seen:
            raise ValueError(
                f"Duplicate distribution name '{name}'. Provide unique names using a "
                f"dictionary (e.g., dict(name='{name}1', type='{name}'))."
            )
        
        seen.add(name)
        names.append(name)
        
    return names


def _extract_dist_info(dist_spec):
    """
    Extracts the underlying distribution type and its full parameters, 
    merging any user-provided overrides with the function's default values.
    """
    # TODO: inspect custom callables too for params 

    # 1. Parse the user's input
    if isinstance(dist_spec, str):
        dist_type = dist_spec
        user_params = {}
    elif isinstance(dist_spec, dict):
        user_params = dist_spec.copy()
        dist_type = user_params.pop("type", user_params.pop("name", "custom"))
        user_params.pop("name", None)  # Remove name if it's still there
    elif callable(dist_spec):
        dist_type = getattr(dist_spec, "__name__", "custom")
        user_params = {}
    else:
        dist_type = "custom"
        user_params = {}

    # 2. Get defaults dynamically from the DISTRIBUTIONS registry
    final_params = {}
    if dist_type in DISTRIBUTIONS:
        func = DISTRIBUTIONS[dist_type]
        sig = inspect.signature(func)
        
        for param_name, param in sig.parameters.items():
            # Skip 'u' (the uniform array input)
            if param_name == 'u':
                continue
            # If the parameter has a default value, store it
            if param.default is not inspect.Parameter.empty:
                final_params[param_name] = param.default

    # 3. Override defaults with any parameters the user provided
    final_params.update(user_params)

    return dist_type, final_params

# ---------------------------------------------------------------------------
# simulate(): one dataset, arbitrary corr matrix
# ---------------------------------------------------------------------------

def simulate(
    distributions: List,
    correlations: Union[List[float], np.ndarray],
    n_obs: int = 1000,
    var_names: Optional[List[str]] = None,
    random_state: Optional[int] = None,
    repair_pd: bool = True,
    normalize: Union[bool, List[int], List[str]] = True,
) -> pd.DataFrame:
    """
    Simulate a single dataset from the Gaussian copula.

    Parameters
    ----------
    distributions : list of str | dict | callable
        One entry per variable.
        - str  → name of a built-in distribution (e.g. ``"normal"``)
        - dict → ``{"name": "beta", "a": 2, "b": 5}``
        - callable → ``f(uniform_samples: np.ndarray) -> np.ndarray``
    correlations : list or array, length = n*(n-1)/2
        Upper-triangle correlation values (row-major).
    n_obs : int
    var_names : list of str, optional
        If omitted, names are auto-generated as ``normal1``, ``beta1``, etc.
    random_state : int, optional
    repair_pd : bool

    Returns
    -------
    pd.DataFrame, shape (n_obs, n_vars)
    """
    rng = np.random.default_rng(random_state)
    n_vars = len(distributions)

    if var_names is None:
        var_names = _make_var_names(distributions)
    if len(var_names) != n_vars:
        raise ValueError("len(var_names) must equal len(distributions).")

    corr = np.eye(n_vars)
    idx = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars)]
    corr_vals = np.asarray(correlations, dtype=float)
    if len(corr_vals) != len(idx):
        raise ValueError(
            f"Expected {len(idx)} correlation value(s) for {n_vars} variables, "
            f"got {len(corr_vals)}."
        )
    for (i, j), v in zip(idx, corr_vals):
        corr[i, j] = v
        corr[j, i] = v

    if not _is_pd(corr):
        if repair_pd:
            corr = _nearest_pd(corr)
        else:
            raise ValueError("Correlation matrix is not positive-definite.")

    normal_data = rng.multivariate_normal(np.zeros(n_vars), corr, size=n_obs)
    uniform_data = norm.cdf(normal_data)

    df = pd.DataFrame({
        name: _apply_distribution(uniform_data[:, k], dist_spec)
        for k, (dist_spec, name) in enumerate(zip(distributions, var_names))
    })

    # Apply Normalization
    if normalize is True:
        cols_to_norm = df.columns

    elif isinstance(normalize, list):
        # Handle both integer indices and string names
        cols_to_norm = [df.columns[c] if isinstance(c, int) else c for c in normalize]
    else:
        cols_to_norm = []
        
    for col in cols_to_norm:
        df[col] = _minmax_norm(df[col])

    return df


# ---------------------------------------------------------------------------
# simulate_grid(): wide format, all (distribution × correlation) combos
# ---------------------------------------------------------------------------

import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import block_diag
from typing import List, Dict, Optional, Union

def simulate_grid(
    distributions: List,
    correlations: List[float],
    n_obs: int = 1000,
    var_names: Optional[List[str]] = None,
    random_state: Optional[int] = None,
    repair_pd: bool = True,
    normalize: Union[bool, List[int], List[str]] = True,
) -> Dict:
    
    n_dists = len(distributions)
    n_corrs = len(correlations)
    n_total = n_dists * n_corrs

    # 1. Validate and define base names
    if var_names is None:
        base_names = _make_var_names(distributions)
    elif len(var_names) != n_dists:
        raise ValueError("len(var_names) must equal len(distributions).")
    else:
        base_names = var_names

    rng = np.random.default_rng(random_state)

    # 2. Build the exact Intended Block-Diagonal Correlation Matrix
    blocks = []
    for r in correlations:
        corr_block = np.full((n_dists, n_dists), r)
        np.fill_diagonal(corr_block, 1.0)
        
        # Repair at the block level to perfectly preserve cross-block 0s!
        if not _is_pd(corr_block):
            if repair_pd:
                corr_block = _nearest_pd(corr_block)
            else:
                raise ValueError(f"Equicorrelation matrix with r={r} is not positive-definite.")
        blocks.append(corr_block)

    # Stitch blocks together instantaneously
    full_corr_matrix = block_diag(*blocks)

    # 3. Simulate all data in ONE vectorized shot
    normal_data = rng.multivariate_normal(np.zeros(n_total), full_corr_matrix, size=n_obs)
    uniform_data = norm.cdf(normal_data)

    # 4. Apply Distributions & Gather Meta
    df_data = {}
    variables_meta = []
    
    for idx in range(n_total):
        c_idx = idx // n_dists          # Identifies which correlation block (0, 1, 2...)
        d_idx = idx % n_dists           # Identifies which distribution (0, 1, 2...)
        
        r = correlations[c_idx]
        dist_spec = distributions[d_idx]
        base_name = base_names[d_idx]
        col_name = f"{base_name}{c_idx}"
        
        # Apply inverse CDF
        df_data[col_name] = _apply_distribution(uniform_data[:, idx], dist_spec)
        
        # Extract meta
        dist_type, params = _extract_dist_info(dist_spec)
        variables_meta.append({
            "column_name": col_name,
            "distribution_type": dist_type,
            "parameters": params,
            "block_corr": r,
        })

    data = pd.DataFrame(df_data)

    # 5. Apply Normalization
    if normalize is True:
        cols_to_norm = data.columns
    elif isinstance(normalize, list):
        cols_to_norm = [data.columns[c] if isinstance(c, int) else c for c in normalize]
    else:
        cols_to_norm = []
        
    for col in cols_to_norm:
        data[col] = _minmax_norm(data[col])

    # 6. Build the Correlation Pairwise Metadata
    observed_corr_df = data.corr(method="spearman").round(4)
    intended_corr_df = pd.DataFrame(full_corr_matrix, index=data.columns, columns=data.columns)
    
    pairwise_meta = []
    # Using combinations ensures distinct pairs (A-B) without reversing (B-A) or self (A-A)
    for var1, var2 in itertools.combinations(data.columns, 2):
        pairwise_meta.append({
            "var1": var1,
            "var2": var2,
            "intended_corr": intended_corr_df.loc[var1, var2],
            "observed_corr": observed_corr_df.loc[var1, var2]
        })

    return {
        "data": data,
        "meta": {
            "dist": pd.DataFrame(variables_meta),
            "corr": pd.DataFrame(pairwise_meta)
        }
    }