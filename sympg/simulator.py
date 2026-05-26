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

from .distributions import _apply_distribution, _extract_dist_info, _minmax_norm
from .correlations import _is_pd, _nearest_pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

    # Apply Normalization (if specified)
    df = _minmax_norm(df, normalize)

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
    mode: str = "equi",  # or "all_pairs"
    random_state: Optional[int] = None,
    repair_pd: bool = True,
    normalize: Union[bool, List[int], List[str]] = True,
) -> Dict:
    
    n_dists = len(distributions)

    # 1. Validate and define base names
    if var_names is None:
        base_names = _make_var_names(distributions)
    elif len(var_names) != n_dists:
        raise ValueError("len(var_names) must equal len(distributions).")
    else:
        base_names = var_names

    rng = np.random.default_rng(random_state)

    # 2. Build the Intended Correlation Blocks based on mode
    blocks = []
    
    if mode == "all_pairs":
        # Calculate number of unique pairs (upper triangle)
        n_pairs = n_dists * (n_dists - 1) // 2
        
        # Generate the Cartesian product of all possible correlation levels
        for combo in itertools.product(correlations, repeat=n_pairs):
            corr_block = np.eye(n_dists)
            
            # Fill the upper and lower triangles
            pair_idx = 0
            for i in range(n_dists):
                for j in range(i + 1, n_dists):
                    corr_block[i, j] = corr_block[j, i] = combo[pair_idx]
                    pair_idx += 1
            
            if not _is_pd(corr_block):
                if repair_pd:
                    corr_block = _nearest_pd(corr_block)
                else:
                    continue  # Skip impossible matrices if not repairing
                    
            blocks.append(corr_block)
            
    else:  # Original "equi" mode
        for r in correlations:
            corr_block = np.full((n_dists, n_dists), r)
            np.fill_diagonal(corr_block, 1.0)
            
            if not _is_pd(corr_block):
                if repair_pd:
                    corr_block = _nearest_pd(corr_block)
                else:
                    raise ValueError(f"Equicorrelation matrix with r={r} is not PD.")
            blocks.append(corr_block)

    n_blocks = len(blocks)
    n_total = n_dists * n_blocks

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
            "block_idx": c_idx,
        })

    data = pd.DataFrame(df_data)

    # 5. Apply Normalization
    data = _minmax_norm(data, normalize)

    # 6. Build the Correlation Pairwise Metadata
    observed_corr_df = data.corr(method="spearman").round(4)
    intended_corr_df = pd.DataFrame(full_corr_matrix, index=data.columns, columns=data.columns)
    
    pairwise_meta = []
    # Using combinations ensures distinct pairs (A-B) without reversing (B-A) or self (A-A)
    # for var1, var2 in itertools.combinations(data.columns, 2):
    #     pairwise_meta.append({
    #         "var1": var1,
    #         "var2": var2,
    #         "intended_corr": intended_corr_df.loc[var1, var2],
    #         "observed_corr": observed_corr_df.loc[var1, var2]
    #     })
    for var1, var2 in itertools.combinations(data.columns, 2):
        # We only care about saving pairwise metadata for variables inside the SAME block
        # Variables across different blocks have 0 correlation by definition of block_diag
        v1_block = int(var1.replace(next(b for b in base_names if var1.startswith(b)), ""))
        v2_block = int(var2.replace(next(b for b in base_names if var2.startswith(b)), ""))
        
        if v1_block == v2_block:
            pairwise_meta.append({
                "var1": var1,
                "var2": var2,
                "block_idx": v1_block,
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