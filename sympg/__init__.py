"""
SymPG - Synthetic Multivariate Data Generator using Gaussian Copulas.

Quick start
-----------
>>> import sympg
>>>
>>> result = sympg.simulate_grid(
...     distributions=["normal", "beta", "gamma"],
...     correlations=[0.1, 0.5, 0.9],
...     n_obs=500,
...     random_state=42,
... )
>>> df   = result["data"]          # (500, 9) wide DataFrame
>>> cm   = result["corr_matrix"]   # 9×9 Spearman matrix
>>> meta = result["metadata"]      # column descriptions
>>>
>>> # Plotting
>>> sympg.plot_corr_heatmap(result).show()
>>> sympg.plot_distributions(result).show()
>>> sympg.plot_scatter_block(result, block_corr=0.9).show()
>>> sympg.plot_corr_accuracy(result).show()
"""

from .simulator import simulate, simulate_grid
from .distributions import DISTRIBUTIONS
from .diagnostics import summarise_correlations, get_scenario_metadata
from .plotting import (
    plot_corr_heatmap,
    plot_distributions,
    plot_scatter_block,
    plot_corr_accuracy,
    plot_all,
)

__version__ = "0.1.0"
__all__ = [
    "simulate",
    "simulate_grid",
    "summarise_correlations",
    "get_scenario_metadata",
    "DISTRIBUTIONS",
    "plot_corr_heatmap",
    "plot_distributions",
    "plot_scatter_block",
    "plot_corr_accuracy",
    "plot_all",
]
