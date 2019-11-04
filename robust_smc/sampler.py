from abc import ABC, abstractmethod

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


class SMCSampler(ABC):
    def __init__(self, data, num_samples=100, x_init=None, seed=None):
        self.data = data
        self.time_steps = self.data.shape[0]
        self.num_samples = num_samples
        self.x_init = x_init

        self.x_samples = []
        self.x_trajectories = []
        self.logw = []
        self.initialized = False
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
            # return prior on on x_init
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
        self.x_samples.append(self.proposal_sample(self.x_init))
        self.logw.append(self.compute_logw(0))
        for t in range(self.time_steps - 1):
            # Resample
            w = self.get_normalised_weights(self.logw[-1])
            self.x_trajectories.append(self.multinomial_resampling(w, self.x_samples[-1]))
            # Sample
            self.x_samples.append(self.proposal_sample(self.x_trajectories[-1]))
            # Weight
            self.logw.append(self.compute_logw(t + 1))

    def reset(self):
        self.x_samples = []
        self.x_trajectories = []
        self.logw = []
        self.initialized = False

    @staticmethod
    def get_normalised_weights(logw):
        """
        Compute normalised weights from log-weights
        :param logw: log-weights Nx1 numpy array
        :return: Nx1 numpy array
        """
        return np.exp(logw - logsumexp(logw))


class LinearDiagonalGaussianBPF(SMCSampler):
    def __init__(self, data, transition_matrix, prior_std, x_init, observation_model, num_samples=100, seed=None):
        super().__init__(data, num_samples=num_samples, x_init=x_init, seed=seed)
        self.transition_matrix = transition_matrix
        self.prior_std = prior_std
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
        noise = self.prior_std[None, :] * self.rng.randn(self.num_samples, D)
        samples = noiseless + noise
        return samples

    def compute_logw(self, t):
        observed = np.tile(self.data[t], self.num_samples)[:, None]
        predicted = self.observation_model(self.x_samples[-1])
        return norm.logpdf(observed, predicted)
