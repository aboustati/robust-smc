from scipy.stats import levy_stable

import numpy as np

from numpy.linalg import cholesky


def peaks(x, y):
    """
    Re-implementation of MATLAB's peaks function
    :param x: Nx1 Numpy array
    :param y: Nx1 Numpy array
    :return: Nx1 Numpy array
    """
    first_term = 3 * ((1 - x) ** 2) * np.exp(- (x ** 2) - ((y + 1) ** 2))
    second_term = - 10 * ((x / 5) - (x ** 3) - (y ** 5)) * np.exp(-(x ** 2 + y ** 2))
    third_term = - (1 / 3) * np.exp(-(x + 1) ** 2 + y ** 2)
    return 200 * (first_term + second_term + third_term)


def dem(x, y, num_frequencies=4):
    """
    Synthetic Digital Elevation Model (DEM) map
    :param x: x coordinates Nx1 numpy array
    :param y: y coordinates Nx1 numpy array
    :return: z coordinates Nx1 numpy array
    """
    a = np.array([300, 80, 60, 40, 20, 10])[None, :num_frequencies]  # 1x6
    omega = np.array([5, 10, 20, 30, 80, 150])[None, :num_frequencies]  # 1x6
    omega_bar = np.array([4, 10, 20, 40, 90, 150])[None, :num_frequencies]  # 1x6
    q = 3 / (2.96 * 1e4)
    # q = 0.5
    peak = peaks(q * x, q * y)
    fourier = np.sum(a * np.sin(omega * q * x) * np.cos(omega_bar * q * x), axis=1)[:, None]
    return peak + fourier
    # return fourier


class TANSimulator:
    """
    Terrain Aided Navigation (TAN) simulator. Taken from Merlinge et. al. 2019
    """
    def __init__(self, final_time, time_step=0.1, observation_std=5, num_frequencies=6,
                 X0=None, transition_matrix=None, process_std=None, seed=None):
        """
        :param final_time: final time period
        :param time_step: time step
        :param observation_std: observation noise standard deviation
        :param X0: initial state
        :param transition_matrix: transition matrix
        :param process_std: process standard deviation
        :param seed: random seed
        """
        self.final_time = final_time
        self.time_step = time_step
        if X0 is None:
            X0 = np.array([-7.5 * 1e3, 5.0 * 1e3, 1.1 * 1e3, 88.15, -60.53, 0.0])
        self.X0 = X0
        self.transition_matrix = transition_matrix or np.vstack(
            [np.hstack([np.eye(3), time_step * np.eye(3)]), np.hstack([0 * np.eye(3), np.eye(3)])]
        )
        self.simulation_steps = int(final_time / time_step)
        self.observation_std = observation_std
        self.num_frequencies = num_frequencies
        if process_std is None:
            process_std = np.array([0.1, 0.1, 0.3, 1.45 * 1e-2, 2.28 * 1e-2, 11.5 * 1e-2]) * 20
        self.process_std = process_std
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self._simulate_system()

    def observation_model(self, X):
        """
        TANSimulator observation model

        m = z - DEM(x, y)

        :param X: Nx6 numpy array
        :return: z-coordinates
        """
        return X[:, 2][:, None] - dem(X[:, 0][:, None], X[:, 1][:, None], self.num_frequencies)

    def noise_model(self, Y):
        return self.observation_std * self.rng.randn(*Y.shape)

    def _simulate_system(self):
        """
        Simulates the TAN system
        """
        X = np.zeros((self.simulation_steps + 1, 6))
        X[0, :] = self.X0
        for t in range(self.simulation_steps):
            state_evolution = np.matmul(self.transition_matrix, X[t, :][:, None])
            process_noise = self.process_std[:, None] * self.rng.randn(6, 1)
            X[t + 1, :] = (state_evolution + process_noise)[:, 0]

        Y = self.observation_model(X)
        Y += self.noise_model(Y)
        self.X, self.Y = X, Y

    def renoise(self):
        Y = self.observation_model(self.X)
        Y += self.noise_model(Y)
        return Y
    
    
class SpecifiedTanSimulator(TANSimulator):
    def observation_model(self, X):
        """
        TANSimulator observation model

        m = z - DEM(x, y)

        :param X: Nx6 numpy array
        :return: z-coordinates
        """
        height = X[:, 2][:, None] - dem(X[:, 0][:, None], X[:, 1][:, None], self.num_frequencies)
        distance = np.sqrt(np.sum((X[:, :2] - self.X0[:2][None, :]) ** 2, axis=1))[:, None]
        return np.concatenate([height, distance], axis=-1)


class LinearTANSimulator(TANSimulator):
    """
    Terrain Aided Navigation (TAN) simulator with linear observation model.
    Adapted from Merlinge et. al. 2019
    """
    @staticmethod
    def observation_model(X):
        """
        TANSimulator observation model

        m = (x, y, z)
        """
        return X[:, :3].copy()


class ExplosiveTANSimulator(SpecifiedTanSimulator):
    """
    Terrain Aided Navigation (TAN) simulator with explosive noise. Taken from Merlinge et. al. 2019
    """
    def __init__(self, final_time, time_step=0.1, observation_std=5, contamination_probability=0.05,
                 degrees_of_freedom=1, num_frequencies=6, X0=None,
                 transition_matrix=None, process_std=None, seed=None):

        self.contamination_probability = contamination_probability
        self.degrees_of_freedom = degrees_of_freedom

        super().__init__(final_time=final_time, time_step=time_step, observation_std=observation_std,
                         num_frequencies=num_frequencies, X0=X0, transition_matrix=transition_matrix,
                         process_std=process_std, seed=seed)

    def noise_model(self, Y):
        u = self.rng.rand(Y.shape[0])
        noise = np.zeros_like(Y)
        norm_loc = (u > self.contamination_probability)
        t_loc = (u <= self.contamination_probability)
        self.contamination_locations = np.argwhere(t_loc)
        noise[norm_loc] = self.rng.randn(norm_loc.sum(), Y.shape[1])
        noise[t_loc] = self.rng.standard_t(df=self.degrees_of_freedom, size=(t_loc.sum(), 1))
        return self.observation_std * noise


class ConstantVelocityModel:
    """
    Constant Velosity model with Gaussian Explosion
    """
    def __init__(self, final_time, time_step=0.1, observation_cov=None,
                 explosion_scale=10.0, contamination_probability=0.05, seed=None):
        self.final_time = final_time
        self.time_step = time_step
        self.simulation_steps = int(final_time / time_step)
        self.observation_cov = np.eye(2) if observation_cov is None else observation_cov
        self.explosion_scale = explosion_scale
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.contamination_probability = contamination_probability
        self._build_system()
        self._simulate_system()

    def _build_system(self):
        self.dim_X = 4
        self.dim_Y = 2

        transition_matrix = np.eye(self.dim_X)
        transition_matrix[0, 2] = transition_matrix[1, 3] = self.time_step
        self.transition_matrix = transition_matrix
        self.observation_matrix = np.eye(2, 4)

        off_diagonal = (self.time_step ** 2 / 2) * np.eye(2)
        upper_diagonal = (self.time_step ** 3 / 3) * np.eye(2)
        lower_diagonal = self.time_step * np.eye(2)
        self.process_cov = np.vstack([np.hstack([upper_diagonal, off_diagonal]),
                                      np.hstack([off_diagonal, lower_diagonal])])

        self.initial_cov = np.eye(self.dim_X)
        self.initial_state = np.array([140., 140., 50., 0.])[:, None]

    def _simulate_system(self):
        L = cholesky(self.process_cov)
        R = cholesky(self.observation_cov)

        X = [self.initial_state + cholesky(self.initial_cov) @ self.rng.randn(self.dim_X, 1)]
        Y = [self.observation_matrix @ X[-1] + R @ self.rng.randn(self.dim_Y, 1)]

        for _ in range(self.simulation_steps - 1):
            X_new = self.transition_matrix @ X[-1] + L @ self.rng.randn(self.dim_X, 1)
            Y_new = self.observation_matrix @ X_new + R @ self.rng.randn(self.dim_Y, 1)

            u = self.rng.rand()
            if u < self.contamination_probability:
                Y_new += self.explosion_scale * self.rng.randn(self.dim_Y, 1)

            X.append(X_new)
            Y.append(Y_new)

        self.X = np.stack(X).squeeze(axis=-1)
        self.Y = np.stack(Y).squeeze(axis=-1)

    def renoise(self):
        R = cholesky(self.observation_cov)

        Y = []
        for X in self.X:
            Y_new = self.observation_matrix @ X[:, None] + R @ self.rng.randn(self.dim_Y, 1)

            u = self.rng.rand()
            if u < self.contamination_probability:
                Y_new += self.explosion_scale * self.rng.randn(self.dim_Y, 1)

            Y.append(Y_new)

        return np.stack(Y).squeeze(axis=-1)


class AsymmetricConstantVelocity(ConstantVelocityModel):
    def _simulate_system(self):
        L = cholesky(self.process_cov)
        R = cholesky(self.observation_cov)

        X = [self.initial_state + cholesky(self.initial_cov) @ self.rng.randn(self.dim_X, 1)]
        Y = [self.observation_matrix @ X[-1] + R @ self.rng.randn(self.dim_Y, 1)]

        for _ in range(self.simulation_steps - 1):
            X_new = self.transition_matrix @ X[-1] + L @ self.rng.randn(self.dim_X, 1)
            Y_new = self.observation_matrix @ X_new

            p = self.rng.rand()
            if p < 0.5:
                noise = -1 * np.abs(R @ self.rng.randn(self.dim_Y, 1))
            else:
                noise = 1 * self.explosion_scale * np.abs(R @ self.rng.randn(self.dim_Y, 1))

            u = self.rng.rand()
            if u < self.contamination_probability:
                Y_new += self.rng.exponential(1e3) * noise
            else:
                Y_new += noise

            X.append(X_new)
            Y.append(Y_new)

        self.X = np.stack(X).squeeze(axis=-1)
        self.Y = np.stack(Y).squeeze(axis=-1)

    def renoise(self):
        R = cholesky(self.observation_cov)

        Y = []
        for X in self.X:
            Y_new = self.observation_matrix @ X[:, None]
            p = self.rng.rand()
            if p < 0.5:
                noise = -1 * np.abs(R @ self.rng.randn(self.dim_Y, 1))
            else:
                noise = 1 * self.explosion_scale * np.abs(R @ self.rng.randn(self.dim_Y, 1))

            u = self.rng.rand()
            if u < self.contamination_probability:
                Y_new += self.rng.exponential(1e2) * noise
            else:
                Y_new += noise

            Y.append(Y_new)

        return np.stack(Y).squeeze(axis=-1)


class SensorLocalisation:
    def __init__(self, final_time, time_step=0.1, observation_cov=None, explosion_scale=20.0, seed=None):
        self.final_time = final_time
        self.time_step = time_step
        self.simulation_steps = int(final_time / time_step)
        self.observation_cov = np.eye(4) if observation_cov is None else observation_cov
        self.explosion_scale = explosion_scale
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._build_system()
        self._simulate_system()

    def _build_system(self):
        self.dim_X = 4
        self.dim_Y = 4

        transition_matrix = np.eye(self.dim_X)
        transition_matrix[2, 2] = transition_matrix[3, 3] = 0.99
        transition_matrix[0, 2] = transition_matrix[1, 3] = self.time_step
        self.transition_matrix = transition_matrix

        policy = np.zeros((self.dim_X, self.dim_X))
        policy[2, 0] = policy[3, 1] = -0.0134 / 2
        policy[2, 2] = policy[3, 3] = -0.0381 / 2
        self.policy = policy

        off_diagonal = (self.time_step ** 2 / 2) * np.eye(2)
        upper_diagonal = (self.time_step ** 3 / 3) * np.eye(2)
        lower_diagonal = self.time_step * np.eye(2)
        self.process_cov = np.vstack([np.hstack([upper_diagonal, off_diagonal]),
                                      np.hstack([off_diagonal, lower_diagonal])])

        self.initial_cov = np.eye(self.dim_X)
        self.initial_state = np.array([200, -50.0, 10000., 15.])[:, None]
        self.target_state = np.array([0., 0., 0., 0.])[:, None]
        self.sensor_locations = np.array(
            [[0. , -100],
             [0. ,  100],
             [2000, -100],
             [2000,  100]]
        )

    def observation_model(self, X):
        norm = np.sum((X[None, :2] - self.sensor_locations) ** 2, axis=-1)
        return 10 * np.log10((1 / norm) + 1e-9)

    def _simulate_system(self):
        L = cholesky(self.process_cov)
        R = cholesky(self.observation_cov)

        X0 = self.initial_state + self.policy @ (self.initial_state - self.target_state)
        X0 = X0 + cholesky(self.initial_cov) @ self.rng.randn(self.dim_X, 1)
        X = [X0]
        Y = [self.observation_model(X[-1].squeeze())[:, None] + R @ self.rng.randn(self.dim_Y, 1)]

        for _ in range(self.simulation_steps - 1):
            X_new = self.transition_matrix @ X[-1] + self.policy @ (X[-1] - self.target_state)
            X_new = X_new + L @ self.rng.randn(self.dim_X, 1)
            Y_new = self.observation_model(X_new.squeeze())[:, None] + R @ self.rng.randn(self.dim_Y, 1)

            u = self.rng.rand()
            if u < 0.01:
                Y_new += levy_stable.rvs(alpha=0.9, beta=0)

            X.append(X_new)
            Y.append(Y_new)

        self.X = np.stack(X).squeeze(axis=-1)
        self.Y = np.stack(Y).squeeze(axis=-1)

    def renoise(self):
        R = cholesky(self.observation_cov)

        Y = []
        for X in self.X:
            Y_new = self.observation_model(X.squeeze())[:, None] + R @ self.rng.randn(self.dim_Y, 1)
            Y_new[-1, :] = Y_new[-1, :] + levy_stable.rvs(alpha=0.5, beta=0)

            Y.append(Y_new)

        return np.stack(Y).squeeze(axis=-1)


