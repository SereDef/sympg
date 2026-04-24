"""
SymPG diagnostics: per-scenario correlation summaries and optional plotting.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def summarise_correlations(
    data: pd.DataFrame,
    var_names: List[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute pairwise correlations for every (dist_set, corr_set) combination
    present in a ``simulate_grid`` output DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Output of ``simulate_grid``.
    var_names : list of str
        Variable columns to include.
    method : {'pearson', 'spearman', 'kendall'}

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns:
        sim_dist_set, sim_corr_set, var_x, var_y, correlation
    """
    rows = []
    for (d_idx, c_idx), grp in data.groupby(["sim_dist_set", "sim_corr_set"]):
        corr_mat = grp[var_names].corr(method=method)
        for i, vx in enumerate(var_names):
            for j, vy in enumerate(var_names):
                if i < j:
                    rows.append({
                        "sim_dist_set": d_idx,
                        "sim_corr_set": c_idx,
                        "var_x": vx,
                        "var_y": vy,
                        "correlation": corr_mat.loc[vx, vy],
                    })
    return pd.DataFrame(rows)


def get_scenario_metadata(
    dist_sets,
    corr_sets,
    var_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a human-readable metadata table describing every simulated scenario.

    Parameters
    ----------
    dist_sets : list of list
        Distribution sets as passed to ``simulate_grid``.
    corr_sets : list of list
        Correlation vectors as passed to ``simulate_grid``.
    var_names : list of str, optional

    Returns
    -------
    pd.DataFrame
        One row per (dist_set, corr_set) combination with columns:
        sim_dist_set, sim_corr_set, dist_set_label, corr_set_label
    """
    def _label_dist(spec):
        if isinstance(spec, str):
            return spec
        if isinstance(spec, dict):
            params = {k: v for k, v in spec.items() if k != "name"}
            pstr = ",".join(f"{k}={v}" for k, v in params.items())
            return f"{spec['name']}({pstr})" if pstr else spec["name"]
        if callable(spec):
            return getattr(spec, "__name__", "custom")
        return str(spec)

    def _is_single_spec(x):
        return isinstance(x, str) or callable(x) or (isinstance(x, dict) and "name" in x)

    if all(_is_single_spec(d) for d in dist_sets):
        dist_sets = [dist_sets]

    n_vars = len(dist_sets[0])
    if var_names is None:
        var_names = [f"var_{i+1}" for i in range(n_vars)]

    from itertools import product as iproduct
    rows = []
    for d_idx, dist_set in enumerate(dist_sets):
        dist_label = " | ".join(
            f"{v}:{_label_dist(s)}" for v, s in zip(var_names, dist_set)
        )
        for c_idx, corr_vec in enumerate(corr_sets):
            rows.append({
                "sim_dist_set": d_idx,
                "sim_corr_set": c_idx,
                "dist_set_label": dist_label,
                "corr_set_label": str(list(corr_vec)),
            })
    return pd.DataFrame(rows)
