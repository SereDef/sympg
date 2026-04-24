import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sympg

RESULT = sympg.simulate_grid(
    distributions=["normal", "beta", "gamma"],
    correlations=[0.1, 0.5, 0.9],
    n_obs=200,
    random_state=0,
)

def test_plot_corr_heatmap_returns_figure():
    import plotly.graph_objects as go
    fig = sympg.plot_corr_heatmap(RESULT)
    assert isinstance(fig, go.Figure)

def test_plot_distributions_returns_figure():
    import plotly.graph_objects as go
    fig = sympg.plot_distributions(RESULT)
    assert isinstance(fig, go.Figure)

def test_plot_scatter_block_valid():
    import plotly.graph_objects as go
    fig = sympg.plot_scatter_block(RESULT, block_corr=0.5)
    assert isinstance(fig, go.Figure)

def test_plot_scatter_block_invalid():
    with pytest.raises(ValueError, match="No block found"):
        sympg.plot_scatter_block(RESULT, block_corr=0.99)

def test_plot_corr_accuracy_returns_figure():
    import plotly.graph_objects as go
    fig = sympg.plot_corr_accuracy(RESULT)
    assert isinstance(fig, go.Figure)

def test_plot_corr_accuracy_pearson():
    import plotly.graph_objects as go
    fig = sympg.plot_corr_accuracy(RESULT, method="pearson")
    assert isinstance(fig, go.Figure)

def test_plot_all_returns_dict():
    figs = sympg.plot_all(RESULT, show=False)
    assert set(figs.keys()) == {"heatmap", "distributions", "scatter_blocks", "accuracy"}
    assert set(figs["scatter_blocks"].keys()) == {0.1, 0.5, 0.9}
