from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm


class RobustLikelihood(ABC):
    def __init__(self, base_distribution, beta):
        self.base_distribution = base_distribution
        self.beta = beta

    def annealed_term(self, y, *args, **kwargs):
        annealed = self.base_distribution.pdf(y, *args, **kwargs) ** self.beta
        annealed = np.prod(annealed, axis=1)
        return annealed

    @abstractmethod
    def integral_term(self, y, *args, **kwargs):
        pass

    def log_likelihood(self, y, *args, **kwargs):
        lik = self.annealed_term(y, *args, **kwargs) / self.beta
        integral_term = self.integral_term(y, *args, **kwargs) / (1 / (1 + self.beta))
        integral_term = np.broadcast_to(integral_term, lik.shape)
        lik -= integral_term
        return lik


class RobustGaussian(RobustLikelihood):
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
