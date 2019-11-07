from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm


class RobustLikelihood(ABC):
    def __init__(self, base_distribution, beta):
        self.base_distribution = base_distribution
        self.beta = beta

    def annealed_term(self, y, *args, **kwargs):
        return self.base_distribution.pdf(y, *args, **kwargs) ** self.beta

    @abstractmethod
    def integral_term(self, y, *args, **kwargs):
        pass

    def log_likelihood(self, y, *args, **kwargs):
        lik = self.annealed_term(y, *args, **kwargs) / self.beta
        lik -= self.integral_term(y, *args, **kwargs) / (1 / (1 + self.beta))
        return lik


class RobustGaussian(RobustLikelihood):
    def __init__(self, beta):
        super().__init__(norm, beta)

    def integral_term(self, y, *args, **kwargs):
        simga_squared = kwargs['scale'] ** 2
        return ((self.beta + 1) * (2 * np.pi * simga_squared)) ** (-self.beta / 2)
