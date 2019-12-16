from abc import ABC, abstractmethod

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from .robust_likelihoods import RobustGaussian


class SMCSampler(ABC):
    def __init__(self, data, num_samples=100, x_init=None, seed=None):
        self.data = data
        self.time_steps = self.data.shape[0]
        self.num_samples = num_samples
        self.x_init = x_init

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
    def proposal_sample(self, x_prev):
        """
        Samples from the proposal distribution
        :param x_prev: conditioning set NxD numpy array
        :return: NxD numpy array
        """
        if not self.initialized:
            # return prior on x_init
            self.initialized = True
            pass
        else:
            # return prior from proposal
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
        self.x_samples.append(self.proposal_sample(self.x_init))
        self.logw.append(self.compute_logw(0))
        for t in range(self.time_steps - 1):
            # Resample
            w = self.normalised_weights(self.logw[-1])
            self.x_trajectories.append(self.multinomial_resampling(w, self.x_samples[-1]))
            # Sample
            self.x_samples.append(self.proposal_sample(self.x_trajectories[-1]))
            # Weight
            self.logw.append(self.compute_logw(t + 1))

        w = self.normalised_weights(self.logw[-1])
        self.x_trajectories.append(self.multinomial_resampling(w, self.x_samples[-1]))

    def _reset(self):
        self.x_samples = []
        self.x_trajectories = []
        self.logw = []
        self.initialized = False

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
    def __init__(self, data, transition_matrix, prior_cov, x_init, observation_model,
                 noise_cov=1.0, num_samples=100, seed=None):
        """
        BPF for Linear (in state transition) Gaussian state-space models.
        :param data: observation data TxD_out array. To perform filter predictions replace value with np.nan
        :param transition_matrix: state transition matrix DxD.
        :param prior_cov: state trasition covariance: vector of size D or matrix of size DxD
        :param x_init: initial state NxD with N==num_samples
        :param observation_model: function that return the observation model
        :param noise_cov: likelihood standard deviation
        :param num_samples: number of samples
        :param seed: random seed
        """
        super().__init__(data, num_samples=num_samples, x_init=x_init, seed=seed)
        self.transition_matrix = transition_matrix
        assert prior_cov.ndim <= 2
        self.prior_cov = prior_cov
        noise_cov = np.atleast_1d(noise_cov)
        assert noise_cov.ndim <= 1, "LinearGaussianBFP does not support correlated noise"
        self.noise_cov = noise_cov
        self.observation_model = observation_model

    def proposal_sample(self, x_prev):
        """
        Samples from the proposal distribution
        :param x_prev: conditioning set NxD numpy array
        :return: NxD numpy array
        """
        self.initialized = True
        D = x_prev.shape[1]
        noiseless = np.matmul(self.transition_matrix, x_prev[:, :, None])[:, :, 0]

        # If full prior covariance is given
        if self.prior_cov.ndim == 2:
            L = np.linalg.cholesky(self.prior_cov)
            noise = L[None, :, :] @ self.rng.randn(self.num_samples, D, 1)
            noise = np.squeeze(noise, axis=-1)
        # If only diagonal covariance is given
        else:
            std = np.sqrt(self.prior_cov)
            noise = std[None, :] * self.rng.randn(self.num_samples, D)

        samples = noiseless + noise
        return samples

    def compute_logw(self, t):
        """
        logarithms of importance weights
        :return: Nx1 numpy array
        """
        if np.isnan(self.data[t]):
            logw = self.logw[-1]
        else:
            observed = np.tile(self.data[t], (self.num_samples, 1))
            predicted = self.observation_model(self.x_samples[-1])
            logw = np.sum(norm.logpdf(observed, loc=predicted, scale=self.noise_cov), axis=1)[:, None]
        return logw


class RobustifiedLinearGaussianBPF(LinearGaussianBPF):
    def __init__(self, data, beta, transition_matrix, prior_cov, x_init, observation_model,
                 noise_cov=1.0, num_samples=100, seed=None):
        """
        Robustified BPF for Linear (in state transition) Gaussian state-space models.
        :param data: observation data TxD_out array. To perform filter predictions replace value with np.nan
        :param beta: tempering parameter for beta divergence
        :param transition_matrix: state transition matrix DxD.
        :param prior_cov: state trasition covariance: vector of size D or matrix of size DxD
        :param x_init: initial state NxD with N==num_samples
        :param observation_model: function that return the observation model
        :param noise_cov: likelihood standard deviation
        :param num_samples: number of samples
        :param seed: random seed
        """
        super().__init__(data, transition_matrix=transition_matrix, prior_cov=prior_cov, x_init=x_init,
                         noise_cov=noise_cov, observation_model=observation_model, num_samples=num_samples, seed=seed)
        self.robust_likelihood = RobustGaussian(beta)

    def compute_logw(self, t):
        """
        logarithms of importance weights
        :return: Nx1 numpy array
        """
        if np.isnan(self.data[t]):
            logw = self.logw[-1]
        else:
            observed = np.tile(self.data[t], self.num_samples)[:, None]
            predicted = self.observation_model(self.x_samples[-1])
            logw = self.robust_likelihood.log_likelihood(observed, loc=predicted, scale=np.sqrt(self.noise_cov))
        return logw
