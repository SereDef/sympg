"""
SymPG plotting module
Visualisation helpers for simulate_grid() output.

All functions accept the dict returned by simulate_grid() and return
Plotly figures. Call fig.show() to display or fig.write_image() to save.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Optional, Dict


def _get_blocks(metadata: pd.DataFrame) -> List[float]:
    return sorted(metadata["block_corr"].unique())


def _get_dists(metadata: pd.DataFrame) -> List[str]:
    return list(metadata["dist_label"].unique())


# ---------------------------------------------------------------------------
# 1. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_corr_heatmap(
    result: Dict,
    method: str = "spearman",
    title: Optional[str] = None,
    colorscale: str = "RdBu",
):
    """
    Heatmap of the full observed correlation matrix, with block boundaries
    annotated to show which correlation level each group belongs to.

    Parameters
    ----------
    result : dict
        Output of simulate_grid().
    method : {'spearman', 'pearson'}
    title : str, optional
    colorscale : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    data = result["data"]
    metadata = result["metadata"]

    cm = data.corr(method=method).round(3)
    cols = cm.columns.tolist()
    n = len(cols)

    # Build short tick labels
    tick_labels = [c.replace("_r", "<br>r=") for c in cols]

    # Block boundary lines
    blocks = _get_blocks(metadata)
    n_dists = len(_get_dists(metadata))
    shapes = []
    for b_idx in range(1, len(blocks)):
        boundary = b_idx * n_dists - 0.5
        for orientation in ["h", "v"]:
            shapes.append(dict(
                type="line",
                x0=boundary if orientation == "v" else -0.5,
                x1=boundary if orientation == "v" else n - 0.5,
                y0=boundary if orientation == "h" else -0.5,
                y1=boundary if orientation == "h" else n - 0.5,
                line=dict(color="black", width=1.5, dash="dot"),
            ))

    fig = go.Figure(go.Heatmap(
        z=cm.values,
        x=tick_labels,
        y=tick_labels,
        zmin=-1, zmax=1,
        colorscale=colorscale,
        colorbar=dict(title=f"{method.capitalize()} r"),
        text=cm.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig.update_layout(
        title=title or f"Observed {method.capitalize()} Correlation Matrix",
        shapes=shapes,
        height=max(500, 60 * n),
        width=max(550, 65 * n),
        xaxis=dict(tickangle=-45),
        margin=dict(l=120, b=120),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Distribution histograms (one panel per column, grouped by block)
# ---------------------------------------------------------------------------

def plot_distributions(
    result: Dict,
    nbins: int = 40,
    title: Optional[str] = None,
):
    """
    Grid of histograms: one subplot per simulated column, arranged so that
    each row = one distribution type and each column = one correlation level.

    Parameters
    ----------
    result : dict
    nbins : int
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metadata = result["metadata"]
    data = result["data"]
    dists = _get_dists(metadata)
    blocks = _get_blocks(metadata)

    n_rows = len(dists)
    n_cols = len(blocks)

    subplot_titles = []
    for r in blocks:
        for d in dists:
            subplot_titles.append(f"{d} | r={r}")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    ]

    for r_idx, r in enumerate(blocks):
        for d_idx, d in enumerate(dists):
            row_meta = metadata[
                (metadata["block_corr"] == r) & (metadata["dist_label"] == d)
            ]
            if row_meta.empty:
                continue
            col_name = row_meta.iloc[0]["column_name"]
            values = data[col_name].dropna()
            color = colors[d_idx % len(colors)]

            fig.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=nbins,
                    marker_color=color,
                    opacity=0.75,
                    showlegend=False,
                    name=col_name,
                ),
                row=d_idx + 1,
                col=r_idx + 1,
            )

    fig.update_layout(
        title=title or "Simulated Distributions by Type and Correlation Level",
        height=260 * n_rows,
        width=280 * n_cols,
        bargap=0.05,
        margin=dict(t=80),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Scatter matrix within a single correlation block
# ---------------------------------------------------------------------------

def plot_scatter_block(
    result: Dict,
    block_corr: float,
    title: Optional[str] = None,
    max_points: int = 500,
):
    """
    Scatter-plot matrix (SPLOM) for all variables in one correlation block,
    with a linear trendline and marginal histograms on the diagonal.

    Parameters
    ----------
    result : dict
    block_corr : float
        The correlation level of the block to plot.
    title : str, optional
    max_points : int
        Downsample to this many points for readability.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    metadata = result["metadata"]
    data = result["data"]

    block_cols = metadata[metadata["block_corr"] == block_corr]["column_name"].tolist()
    if not block_cols:
        raise ValueError(f"No block found with block_corr={block_corr}. "
                         f"Available: {_get_blocks(metadata)}")

    subset = data[block_cols]
    if len(subset) > max_points:
        subset = subset.sample(max_points, random_state=0)

    # Short axis labels
    labels = {c: c.replace(f"_r{block_corr}", "") for c in block_cols}
    cm = data[block_cols].corr(method="spearman").round(2)

    dims = []
    for c in block_cols:
        dims.append(dict(label=labels[c], values=subset[c]))

    fig = go.Figure(go.Splom(
        dimensions=dims,
        showupperhalf=False,
        diagonal_visible=True,
        marker=dict(size=3, opacity=0.5, color="#636EFA"),
    ))

    # Annotate each off-diagonal cell with observed Spearman r
    annotations = []
    n = len(block_cols)
    cell_size = 1 / n
    for i in range(n):
        for j in range(i):
            r_val = cm.iloc[i, j]
            annotations.append(dict(
                x=(j + 0.5) * cell_size,
                y=1 - (i + 0.2) * cell_size,
                xref="paper", yref="paper",
                text=f"r={r_val:.2f}",
                showarrow=False,
                font=dict(size=10, color="crimson"),
            ))

    fig.update_layout(
        title=title or f"Scatter Matrix — block correlation r={block_corr}",
        annotations=annotations,
        height=220 * n,
        width=220 * n,
        margin=dict(t=60),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Observed vs requested correlation bar chart
# ---------------------------------------------------------------------------

def plot_corr_accuracy(
    result: Dict,
    method: str = "spearman",
    title: Optional[str] = None,
):
    """
    Bar chart comparing **requested** vs **observed** within-block pairwise
    correlations for every distribution pair at every correlation level.

    Parameters
    ----------
    result : dict
    method : {'spearman', 'pearson'}
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    metadata = result["metadata"]
    data = result["data"]
    blocks = _get_blocks(metadata)

    rows = []
    for r in blocks:
        block_cols = metadata[metadata["block_corr"] == r]["column_name"].tolist()
        if len(block_cols) < 2:
            continue
        cm = data[block_cols].corr(method=method)
        for c1, c2 in combinations(block_cols, 2):
            l1 = c1.replace(f"_r{r}", "")
            l2 = c2.replace(f"_r{r}", "")
            rows.append({
                "pair": f"{l1} × {l2}",
                "block_corr": r,
                "observed": cm.loc[c1, c2],
            })

    df_acc = pd.DataFrame(rows)

    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    for i, (pair, grp) in enumerate(df_acc.groupby("pair")):
        grp = grp.sort_values("block_corr")
        color = colors[i % len(colors)]
        fig.add_trace(go.Bar(
            name=pair,
            x=[str(r) for r in grp["block_corr"]],
            y=grp["observed"],
            marker_color=color,
            opacity=0.8,
        ))

    # Reference lines for requested values
    for r in blocks:
        fig.add_hline(y=r, line_dash="dot", line_color="black",
                      annotation_text=f"target {r}", annotation_position="right")

    fig.update_layout(
        barmode="group",
        title=title or f"Requested vs Observed {method.capitalize()} Correlations",
        xaxis_title="Requested correlation (block_corr)",
        yaxis_title=f"Observed {method} r",
        yaxis=dict(range=[-1.05, 1.05]),
        legend_title="Distribution pair",
        height=500,
        width=800,
        margin=dict(r=120),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Convenience: plot all
# ---------------------------------------------------------------------------

def plot_all(result: Dict, show: bool = True):
    """
    Generate and optionally display all four standard plots.

    Returns
    -------
    dict with keys: 'heatmap', 'distributions', 'scatter_blocks', 'accuracy'
        scatter_blocks is itself a dict keyed by block_corr value.
    """
    figs = {}
    figs["heatmap"] = plot_corr_heatmap(result)
    figs["distributions"] = plot_distributions(result)
    figs["accuracy"] = plot_corr_accuracy(result)

    blocks = _get_blocks(result["metadata"])
    figs["scatter_blocks"] = {
        r: plot_scatter_block(result, block_corr=r) for r in blocks
    }

    if show:
        figs["heatmap"].show()
        figs["distributions"].show()
        figs["accuracy"].show()
        for fig in figs["scatter_blocks"].values():
            fig.show()

    return figs
