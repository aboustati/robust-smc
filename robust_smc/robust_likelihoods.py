from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm


class BetaRobustLikelihood(ABC):
    def __init__(self, base_distribution, beta):
        self.base_distribution = base_distribution
        self.beta = beta

    def annealed_term(self, y, *args, **kwargs):
        annealed = self.beta * self.base_distribution.logpdf(y, *args, **kwargs)
        annealed = np.sum(annealed, axis=1)
        return np.exp(annealed)

    @abstractmethod
    def integral_term(self, y, *args, **kwargs):
        pass

    def log_likelihood(self, y, *args, **kwargs):
        lik = self.annealed_term(y, *args, **kwargs) / self.beta
        integral_term = self.integral_term(y, *args, **kwargs) / (1 / (1 + self.beta))
        integral_term = np.broadcast_to(integral_term, lik.shape)
        lik -= integral_term
        return lik


class BetaRobustGaussian(BetaRobustLikelihood):
    def __init__(self, beta, compute_integral_term=False):
        super().__init__(norm, beta)
        self.compute_integral_term = compute_integral_term

    def integral_term(self, y, *args, **kwargs):
        integral = 0.0
        if self.compute_integral_term:
            sigma_squared = kwargs['scale'] ** 2
            integral += ((self.beta + 1) * (2 * np.pi * sigma_squared)) ** (-self.beta / 2)  # D
            integral = np.prod(integral)
        return integral


class BetaRobustAsymmetricGaussian(BetaRobustGaussian):
    def annealed_term(self, y, loc, scale_1, scale_2, **kwargs):
        idx_2 = y > loc
        annealed = self.beta * self.base_distribution.logpdf(y, loc=loc, scale=scale_1, **kwargs)
        annealed_2 = self.beta * self.base_distribution.logpdf(y, loc=loc, scale=scale_2, **kwargs)

        annealed[idx_2] = annealed_2[idx_2]
        annealed = np.sum(annealed, axis=1)

        return np.exp(annealed)
