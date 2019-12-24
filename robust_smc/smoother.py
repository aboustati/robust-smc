from abc import ABC, abstractmethod

import numpy as np
from scipy.special import logsumexp

from tqdm import trange


class Smoother(ABC):
    def __init__(self, sampler, num_samples, seed=None):
        """
        :param sampler: And SMC sampler object that has been run
        :param num_samples: number of samples for smoother
        :param seed: random seed
        """
        self.sampler = sampler
        self.num_samples = num_samples
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def log_backward_kernel(self, X_next, X):
        """
        Computes the logarithm of the backward kernel
        :param X_next: state at the future time step MxD array
        :param X: state at the current time step NxD array
        """
        pass

    def categorical_sampling(self, p):
        """
        Vectorizes sampling from the categorical distribution
        :param p: Probability matrix  NxD, with N # of samples & D # of indices
        """
        u = self.rng.rand(p.shape[0])[:, None]
        cdf = np.cumsum(p, axis=1)
        idx = np.argmax(cdf > u, axis=1)
        return idx

    @staticmethod
    def weight_normaliser(logw):
        """
        Vectorizes weight normalisation
        :param logw: weight vector per sample MxN array with M # of smoother samples, N # of filter samples
        """
        return np.exp(logw - logsumexp(logw, axis=-1)[:, None])

    def smoothing_step(self, t, X_next):
        """
        Returns smoothed samples for time t
        :param t: time index
        :param X_next: State samples from the filtering distribution for time t+1, MxD
        """
        p = self.log_backward_kernel(X_next, self.sampler.X_samples[t])
        smooth_logw = self.sampler.logw[t][:, 0][None, :] + p  # MxN
        w = self.weight_normaliser(smooth_logw)  # MxN
        sample_idx = self.categorical_sampling(p=w)
        smoother_sample = self.sampler.X_samples[t][sample_idx].copy()
        return smoother_sample

    def sample_smooth_trajectories(self):
        """
        Run smoother to compute smooth trajectories
        """
        w = self.sampler.normalised_weights(self.sampler.logw[-1])
        x = self.sampler.multinomial_resampling(w, self.sampler.X_trajectories[-1])[:self.num_samples]
        self.smoother_samples = [x]
        for t in trange(len(self.sampler.X_samples) - 1, 0, -1):
            self.smoother_samples.append(self.smoothing_step(t - 1, self.smoother_samples[-1]))
        self.smoother_samples.reverse()
        self.smoother_samples = np.stack(self.smoother_samples)


class LinearGaussianSmoother(Smoother):
    def log_backward_kernel(self, X_next, X):
        """
        Computes the logarithm of the backward kernel
        :param X_next: state at the future time step MxD array
        :param X: state at the current time step NxD array
        """
        A = self.sampler.transition_matrix
        cov = self.sampler.transition_cov

        cov = np.atleast_1d(cov)
        if cov.ndim < 2:
            cov = np.diag(cov)

        M, N, D = X_next.shape[0], X.shape[0], X.shape[1]
        mean = np.matmul(A[None, :, :], X[:, :, None])[:, :, 0]  # NxD
        assert mean.shape == (N, D)
        L = np.linalg.cholesky(cov)[None, None, :, :]  # DxD
        dev = X_next[:, None, :] - mean[None, :, :]  # MxNxD
        y = np.linalg.solve(L, dev)  # MxNxD
        mahalanobis = np.sum(y ** 2, axis=-1)  # MxN
        assert mahalanobis.shape == (M, N)
        return -0.5 * mahalanobis
