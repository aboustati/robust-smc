from abc import ABC, abstractmethod

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from .robust_likelihoods import BetaRobustGaussian


class SMCSampler(ABC):
    def __init__(self, data, num_samples=100, X_init=None, seed=None):
        self.data = data
        self.time_steps = self.data.shape[0]
        self.num_samples = num_samples
        self.X_init = X_init

        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def multinomial_resampling(self, w, x):
        """
        Standard multinomial resampling for SMCSampler
        :param w: weights Nx1 numpy array
        :param x: sample trajectories NxD numpy array
        :return: NxD numpy array
        """
        u = self.rng.rand(*w.shape)
        bins = np.cumsum(w)
        return x[np.digitize(u, bins).squeeze()]

    @abstractmethod
    def proposal_sample(self, X_prev):
        """
        Samples from the proposal distribution
        :param X_prev: conditioning set NxD numpy array
        :return: NxD numpy array
        """
        pass

    @abstractmethod
    def compute_logw(self, t):
        """
        logarithms of importance weights
        :return: Nx1 numpy array
        """
        pass

    def sample(self):
        self._reset()
        self.X_samples.append(self.proposal_sample(self.X_init))
        self.logw.append(self.compute_logw(0))
        for t in range(self.time_steps - 1):
            # Resample
            w = self.normalised_weights(self.logw[-1])
            self.X_trajectories.append(self.multinomial_resampling(w, self.X_samples[-1]))
            # Sample
            self.X_samples.append(self.proposal_sample(self.X_trajectories[-1]))
            # Weight
            self.logw.append(self.compute_logw(t + 1))

        w = self.normalised_weights(self.logw[-1])
        self.X_trajectories.append(self.multinomial_resampling(w, self.X_samples[-1]))

    def _reset(self):
        self.X_samples = []
        self.X_trajectories = []
        self.logw = []

    @staticmethod
    def normalised_weights(logw):
        """
        Compute normalised weights from log-weights
        :param logw: log-weights Nx1 numpy array
        :return: Nx1 numpy array
        """
        return np.exp(logw - logsumexp(logw))

    @staticmethod
    def effective_sample_size(logw):
        """
        Compute the effective sample size from importance weights
        :param logw: log-weights Nx1 numpy array
        """
        return 1 / np.sum(SMCSampler.normalised_weights(logw) ** 2)


class LinearGaussianBPF(SMCSampler):
    def __init__(self, data, transition_matrix, transition_cov, observation_model,
                 observation_cov, X_init, num_samples=100, seed=None):
        """
        BPF for Linear (in state transition) Gaussian state-space models.
        :param data: observation data TxD_out array. To perform filter predictions replace value with np.nan
        :param transition_matrix: state transition matrix DxD.
        :param transition_cov: state transition covariance: vector of size D or matrix of size DxD
        :param observation_model: function that return the observation model
        :param observation_cov: likelihood standard deviation
        :param X_init: initial state NxD with N==num_samples, typically samples from the prior
        :param num_samples: number of samples
        :param seed: random seed
        """
        super().__init__(data, num_samples=num_samples, X_init=X_init, seed=seed)
        self.transition_matrix = transition_matrix
        assert transition_cov.ndim <= 2
        self.transition_cov = transition_cov
        observation_cov = np.atleast_1d(observation_cov)
        assert observation_cov.ndim <= 1, "LinearGaussianBFP does not support correlated noise in this implementation"
        self.observation_cov = observation_cov
        self.observation_model = observation_model

    def proposal_sample(self, X_prev):
        """
        Samples from the proposal distribution
        :param X_prev: conditioning set NxD numpy array
        :return: NxD numpy array
        """
        D = X_prev.shape[1]
        noiseless = (self.transition_matrix @ X_prev[:, :, None]).squeeze(axis=-1)  # NxD

        # If full prior covariance is given
        if self.transition_cov.ndim == 2:
            L = np.linalg.cholesky(self.transition_cov)
            noise = L[None, :, :] @ self.rng.randn(self.num_samples, D, 1)
            noise = np.squeeze(noise, axis=-1)
        # If only diagonal covariance is given
        else:
            std = np.sqrt(self.transition_cov)
            noise = std[None, :] * self.rng.randn(self.num_samples, D)

        samples = noiseless + noise
        return samples

    def compute_logw(self, t):
        """
        logarithms of importance weights
        :return: Nx1 numpy array
        """
        if np.isnan(self.data[t]).any():
            logw = self.logw[-1]
        else:
            observed = np.tile(self.data[t], (self.num_samples, 1))  # NxD_out
            predicted = self.observation_model(self.X_samples[-1])  # NxD_out
            logw = np.sum(norm.logpdf(observed, loc=predicted, scale=np.sqrt(self.observation_cov)), axis=1)[:, None]  # Nx1
        return logw


class RobustifiedLinearGaussianBPF(LinearGaussianBPF):
    def __init__(self, data, beta, transition_matrix, transition_cov, observation_model,
                 observation_cov, X_init, num_samples=100, seed=None):
        """
        Robustified BPF for Linear (in state transition) Gaussian state-space models.
        :param data: observation data TxD_out array. To perform filter predictions replace value with np.nan
        :param beta: tempering parameter for beta divergence
        :param transition_matrix: state transition matrix DxD.
        :param transition_cov: state transition covariance: vector of size D or matrix of size DxD
        :param observation_model: function that return the observation model
        :param observation_cov: likelihood standard deviation
        :param X_init: initial state NxD with N==num_samples, typically samples from the prior
        :param num_samples: number of samples
        :param seed: random seed
        """
        super().__init__(data, transition_matrix=transition_matrix, transition_cov=transition_cov, X_init=X_init,
                         observation_cov=observation_cov, observation_model=observation_model, num_samples=num_samples, seed=seed)
        self.robust_likelihood = BetaRobustGaussian(beta)

    def compute_logw(self, t):
        """
        logarithms of importance weights
        :return: Nx1 numpy array
        """
        if np.isnan(self.data[t]).any():
            logw = self.logw[-1]
        else:
            observed = np.tile(self.data[t], (self.num_samples, 1))  # NxD_out
            predicted = self.observation_model(self.X_samples[-1])
            logw = self.robust_likelihood.log_likelihood(observed, loc=predicted,
                                                         scale=np.sqrt(self.observation_cov))  # NxD_out
            logw = logw[:, None]  # Nx1
        return logw
