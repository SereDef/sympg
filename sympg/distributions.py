from scipy.stats import (
    beta, gamma, norm, lognorm, truncnorm, uniform as sp_uniform, rv_continuous)
import numpy as np

class MixtureModel(rv_continuous):
    
    def __init__(self, submodels, *args, weights = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise(ValueError(f'There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal.'))
        self.weights = [w / sum(weights) for w in weights]
        
    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x)  * weight
        return pdf
            
    def _sf(self, x):
        sf = self.submodels[0].sf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            sf += submodel.sf(x)  * weight
        return sf

    def _cdf(self, x):
        cdf = self.submodels[0].cdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            cdf += submodel.cdf(x)  * weight
        return cdf

    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.submodels), size=size, p = self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs


def _build_distribution_registry():

    def _normal(u, loc=0.0, scale=1.0):
        return norm.ppf(u, loc=loc, scale=scale)

    def _beta(u, a=0.5, b=0.5):
        return beta.ppf(u, a=a, b=b)

    def _gamma_norm(u, a=2.0, scale=1.0):
        raw = gamma.ppf(u, a=a, scale=scale)
        rmin, rmax = np.nanmin(raw), np.nanmax(raw)
        return (raw - rmin) / (rmax - rmin) if rmax > rmin else raw

    def _gamma(u, a=2.0, scale=1.0):
        return gamma.ppf(u, a=a, scale=scale)

    def _lognormal(u, s=1.0, scale=1.0):
        return lognorm.ppf(u, s=s, scale=scale)

    def _uniform(u, low=0.0, high=1.0):
        return sp_uniform.ppf(u, loc=low, scale=high - low)

    def _truncnorm(u, a=-2.0, b=2.0, loc=0.0, scale=1.0):
        return truncnorm.ppf(u, a=a, b=b, loc=loc, scale=scale)

    def _mixture(u, submodels=None, weights=None):
        # Provide default submodels if none are passed
        if submodels is None:
            # Example: A mixture of two normal distributions
            submodels = [norm(-1, .5), norm(1, .5), norm(3, .8)]
        if weights is None: 
            weights = [0.5, 0.3, 0.2]
        
        # Instantiate your custom MixtureModel
        mix = MixtureModel(submodels=submodels, weights=weights)
        
        # Use the ppf method inherited from rv_continuous
        return mix.ppf(u)

    return {
        "normal":    _normal,
        "beta":      _beta,
        "gamma":     _gamma,
        "gamma_norm":_gamma_norm,
        "lognormal": _lognormal,
        "uniform":   _uniform,
        "truncnorm": _truncnorm,
        "mixture":   _mixture
    }
