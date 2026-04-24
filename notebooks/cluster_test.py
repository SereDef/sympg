
import sympg

# distributions=["normal", "beta",  'gamma', 'mixture']

distributions = [
    # --- Extreme Low Methylation (Sharp left peaks) ---
    dict(name="cluster2.",  type="beta", a=9.8,   b=430.2), # Tallest yellow peak (~0.02)
    dict(name="cluster1.",  type="beta", a=7.4,   b=207.6), # Sharp purple peak (~0.03)
    # dict(name="cluster7.",  type="beta", a=7.1,   b=147.9), # Light green peak (~0.04)
    dict(name="cluster10.", type="beta", a=6.4,   b=103.6), # Green peak (~0.05)
    
    # --- Low-to-Mid Methylation (Wider left peaks) ---
    # dict(name="cluster8.",  type="beta", a=6.0,   b=59.0),  # Teal peak (~0.08)
    # dict(name="cluster9.",  type="beta", a=8.2,   b=41.8),  # Grey peak (~0.15)
    # dict(name="cluster13.", type="beta", a=13.5,  b=27.5),  # Orange-yellow peak (~0.32)
    # dict(name="cluster6.",  type="beta", a=11.3,  b=17.7),  # Light blue peak (~0.38)
    
    # --- Mid-to-High Methylation (Wider right peaks) ---
    # dict(name="cluster11.", type="beta", a=19.4,  b=9.6),   # Pink peak (~0.68)
    dict(name="cluster5.",  type="beta", a=32.2,  b=8.8),   # Red-pink peak (~0.80)
    # dict(name="cluster4.",  type="beta", a=41.8,  b=8.2),   # Purple peak (~0.85)
    # dict(name="cluster12.", type="beta", a=35.3,  b=5.7),   # Light blue peak (~0.88)
    
    # --- Extreme High Methylation (Sharp right peaks) ---
    # dict(name="cluster3.",  type="beta", a=52.3,  b=6.7),   # Green peak (~0.90)
    # dict(name="cluster14.", type="beta", a=59.0,  b=6.0),   # Light purple peak (~0.92)
    dict(name="cluster15.", type="beta", a=117.9, b=7.2),   # Sharp yellow right peak (~0.95)

    # Controls
    'normal',
    'mixture'
]

correlations = [0.03, 0.1, 0.5, 0.9]

dnam = sympg.simulate_grid(
    distributions=distributions,
    correlations=correlations,
    n_obs=500,
    random_state=42,
)

data = dnam['data']

meta = dnam['meta']


import pandas as pd
from sklearn.linear_model import LassoCV

# EWAS -- normal linear - robust linear - 

# cluster finalize 
# - data: birth genr epic1 450 alspac 450 genr next! epic1
# - can i regress cohort ? -- 
# - separability 
# - characterizen(prenatal facors)
# is this statistical? 
# simulation - EWAS - regression and robust (most common methods for EWAS) eBayes - frankenstein full baysian with cluster as priors
# with cpg as outcomes
# for 


# Identify the columns from the r=0.5 block (suffix '_2')
block_cols = [c for c in data.columns if c.endswith('2')]

# We test three distinctly shaped targets to see how shapes affect estimation
targets_to_test = ['cluster2.2', 'normal2', 'mixture2']

results = []

for target in targets_to_test:
    # Predictors are all OTHER variables in the same block
    predictors = [c for c in block_cols if c != target]
    
    # We must standardize the inputs so the Lasso coefficients are on the same scale
    X = data[predictors]
    y = data[target]
    
    # Fit the Lasso model with cross-validation
    lasso = LassoCV(cv=5, random_state=42).fit(X, y)
    
    for pred_col, coef in zip(predictors, lasso.coef_):
        results.append({
            "Target": target.replace(".2", ""),
            "Predictor": pred_col.replace(".2", ""),
            "Intended_Copula_Corr": 0.5,
            "Observed_Spearman": data[target].corr(data[pred_col], method='spearman').round(3),
            "Observed_Pearson": data[target].corr(data[pred_col], method='pearson').round(3),
            "Lasso_Coef": round(coef, 3)
        })

# Display the demonstration table
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))