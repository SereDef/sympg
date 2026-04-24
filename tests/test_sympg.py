import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sympg


# --- _make_var_names ---

def test_var_names_unique_dists():
    from sympg.simulator import _make_var_names
    assert _make_var_names(["normal", "beta", "gamma"]) == ["normal1", "beta1", "gamma1"]

def test_var_names_repeated_dists():
    from sympg.simulator import _make_var_names
    assert _make_var_names(["normal", "beta", "normal"]) == ["normal1", "beta1", "normal2"]

def test_var_names_all_same():
    from sympg.simulator import _make_var_names
    assert _make_var_names(["beta", "beta", "beta"]) == ["beta1", "beta2", "beta3"]

def test_var_names_dict_spec():
    from sympg.simulator import _make_var_names
    specs = [{"name": "beta", "a": 2, "b": 5}, "normal", {"name": "beta", "a": 1}]
    assert _make_var_names(specs) == ["beta1", "normal1", "beta2"]


# --- simulate() ---

def test_simulate_default_var_names():
    df = sympg.simulate(["normal", "beta", "normal"], [0.3, 0.1, 0.2], n_obs=10)
    assert list(df.columns) == ["normal1", "beta1", "normal2"]

def test_simulate_dict_dist():
    df = sympg.simulate(
        [{"name": "beta", "a": 2, "b": 5}, "normal"],
        correlations=[0.5], n_obs=100, random_state=0
    )
    assert df.shape == (100, 2)
    # beta is [0,1] by definition
    assert df.iloc[:, 0].between(0, 1).all()

def test_simulate_dict_does_not_mutate():
    spec = {"name": "beta", "a": 2, "b": 5}
    spec_copy = dict(spec)
    sympg.simulate([spec, "normal"], [0.3], n_obs=50)
    assert spec == spec_copy, "simulate() must not mutate the caller's dict"

def test_simulate_shape():
    df = sympg.simulate(["normal", "beta"], [0.5], n_obs=100, var_names=["x","y"])
    assert df.shape == (100, 2)

def test_simulate_reproducible():
    kw = dict(distributions=["normal","beta"], correlations=[0.3], n_obs=50, random_state=7)
    pd.testing.assert_frame_equal(sympg.simulate(**kw), sympg.simulate(**kw))

def test_simulate_corr_approx():
    df = sympg.simulate(["normal","normal"], [0.7], n_obs=5000, random_state=1)
    assert abs(df.corr(method="spearman").iloc[0,1] - 0.7) < 0.05

def test_simulate_wrong_corr_length():
    with pytest.raises(ValueError, match="Expected 3"):
        sympg.simulate(["normal","normal","beta"], [0.5], n_obs=10)

def test_simulate_unknown_dist():
    with pytest.raises(ValueError, match="Unknown distribution"):
        sympg.simulate(["foobar","normal"], [0.0], n_obs=10)


# --- simulate_grid() ---

def test_grid_default_var_names():
    """Columns should be <dist><n>_r<corr>, including same-dist vars."""
    result = sympg.simulate_grid(
        distributions=["normal", "beta", "normal"],
        correlations=[0.5],
        n_obs=50,
    )
    cols = result["data"].columns.tolist()
    assert "normal1_r0.5" in cols
    assert "beta1_r0.5" in cols
    assert "normal2_r0.5" in cols

def test_grid_shape():
    result = sympg.simulate_grid(
        distributions=["normal", "beta", "gamma"],
        correlations=[0.1, 0.5, 0.9],
        n_obs=500, random_state=42,
    )
    assert result["data"].shape == (500, 9)   # 3 dists × 3 corrs

def test_grid_same_dist_intrablock_corr():
    """normal1 × normal2 within the same block must have corr ≈ r."""
    result = sympg.simulate_grid(
        distributions=["normal", "normal"],
        correlations=[0.7],
        n_obs=4000, random_state=0,
    )
    df = result["data"]
    r_obs = df.corr(method="spearman").loc["normal1_r0.7", "normal2_r0.7"]
    assert abs(r_obs - 0.7) < 0.08, f"Got r={r_obs:.3f}"

def test_grid_cross_dist_intrablock_corr():
    """normal × beta within the same block must have corr ≈ r."""
    result = sympg.simulate_grid(
        ["normal", "beta"], [0.6], n_obs=4000, random_state=1
    )
    df = result["data"]
    r_obs = df.corr(method="spearman").loc["normal1_r0.6", "beta1_r0.6"]
    assert abs(r_obs - 0.6) < 0.08, f"Got r={r_obs:.3f}"

def test_grid_interblock_corr_near_zero():
    """Variables from different blocks should be uncorrelated."""
    result = sympg.simulate_grid(
        ["normal"], [0.0, 0.9], n_obs=3000, random_state=1
    )
    df = result["data"]
    r = df.corr(method="spearman").loc["normal1_r0.0", "normal1_r0.9"]
    assert abs(r) < 0.08, f"Expected ~0, got {r:.3f}"

def test_grid_metadata_rows():
    result = sympg.simulate_grid(["normal","beta","gamma"], [0.1, 0.5], n_obs=50)
    assert len(result["metadata"]) == 6   # 3 dists × 2 corrs

def test_grid_metadata_columns():
    result = sympg.simulate_grid(["normal"], [0.5], n_obs=10)
    expected = {"column_name","dist_label","var_name","var_index","block_corr"}
    assert expected.issubset(result["metadata"].columns)

def test_grid_dict_dist():
    result = sympg.simulate_grid(
        [{"name": "beta", "a": 2, "b": 5}, "normal"],
        [0.4], n_obs=100, random_state=3,
    )
    df = result["data"]
    assert "beta1_r0.4" in df.columns
    assert "normal1_r0.4" in df.columns

def test_grid_custom_var_names():
    result = sympg.simulate_grid(
        ["normal", "beta"], [0.5],
        n_obs=50, var_names=["pheno", "CpG1"]
    )
    cols = result["data"].columns.tolist()
    assert "pheno_r0.5" in cols and "CpG1_r0.5" in cols

def test_grid_negative_corr():
    result = sympg.simulate_grid(["normal","beta"], [-0.6], n_obs=3000, random_state=3)
    df = result["data"]
    r = df.corr(method="spearman").loc["normal1_r-0.6", "beta1_r-0.6"]
    assert r < -0.45, f"Expected negative corr, got {r:.3f}"

def test_grid_returns_keys():
    result = sympg.simulate_grid(["normal"], [0.3], n_obs=50)
    assert set(result.keys()) == {"data", "corr_matrix", "metadata"}

def test_grid_corr_matrix_shape():
    result = sympg.simulate_grid(["normal","beta"], [0.3, 0.7], n_obs=100)
    assert result["corr_matrix"].shape == (4, 4)
