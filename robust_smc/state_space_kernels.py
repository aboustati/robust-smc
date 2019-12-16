import numpy as np
from scipy.sparse.linalg import expm


class StateSpaceMatern52:
    def __init__(self, lengthscale=1.0, variance=10.0):
        self.lengthscale = lengthscale
        self.variance = variance
        self._initialize()

    def _initialize(self):
        self.lambda_constant = np.sqrt(5.0) / self.lengthscale
        self.transition_matrix = np.array([
            [0, 1., 0.],
            [0., 0., 1.],
            [- self.lambda_constant ** 3, - 3 * self.lambda_constant ** 2, - 3 * self.lambda_constant]
        ])
        self.process_noise = np.array([0., 0., 1.])[:, None]  # 3x1
        self.observation_matrix = np.array([1., 0., 0.])[None, :]  # 1x3

        self.spectral_density = self.variance * (400 * np.sqrt(5)) / (3 * (self.lengthscale ** 5))

        kappa = self.variance * (self.lambda_constant ** 2) / 3

        self.Pinf = np.array([
            [self.variance, 0., -kappa],
            [0., kappa, 0.],
            [-kappa, 0., self.variance * (self.lambda_constant ** 4)]
        ])

    def discretize(self, delta=None):
        delta = delta or 1.0
        A = expm(self.transition_matrix * delta)
        Q = self.Pinf - A @ self.Pinf @ A.T
        H = self.observation_matrix
        m_0 = np.zeros((A.shape[0], 1))
        P_0 = self.Pinf
        return A, Q, H, m_0, P_0