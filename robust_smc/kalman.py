import numpy as np


class Kalman:
    def __init__(self, data, transition_matrix, transition_cov, observation_matrix, observation_cov, m_0, P_0):
        """
        Kalman filter and RTS smoother
        :param data: TxD_out array, if np.nan perform filter prediction
        :param transition_matrix: state transition matrix DxD
        :param transition_cov: state evolution covariance DxD
        :param observation_matrix: observation matrix DxD_out
        :param observation_cov: observation covariance scalar, D, D_outxD_out
        """
        self.data = data
        self.transition_matrix = transition_matrix
        self.transition_cov = transition_cov
        self.observation_matrix = observation_matrix

        observation_cov = np.atleast_1d(observation_cov)
        if observation_cov.ndim < 2:
            observation_cov = np.diag(observation_cov)

        self.observation_cov = observation_cov
        self.m_0 = m_0
        self.P_0 = P_0

    def one_step_prediction(self, filter_mean, filter_cov):
        """
        One step kalman filter prediction
        :param filter_mean: filter distribution mean for previous time-step
        :param filter_cov:  filter distribution covariance for previous time-step
        :return:
        """
        m_bar = self.transition_matrix @ filter_mean  # Dx1
        P_bar = self.transition_matrix @ filter_cov @ self.transition_matrix.T + self.transition_cov  # DxD
        return m_bar, P_bar

    def filter(self):
        """
        Run the Kalman filter
        """
        self.filter_means = [self.m_0]
        self.filter_covs = [self.P_0]
        self.marginal_covs = []
        for t in range(self.data.shape[0]):
            m_bar, P_bar = self.one_step_prediction(self.filter_means[-1], self.filter_covs[-1])

            # Update step
            y = self.data[t]
            if not np.isnan(y).any():
                v = y[:, None] - self.observation_matrix @ m_bar
                S = self.observation_matrix @ P_bar @ self.observation_matrix.T + self.observation_cov
                K = P_bar @ self.observation_matrix.T @ np.linalg.inv(S)

                m_bar = m_bar + K @ v
                P_bar = P_bar - K @ S @ K.T

                self.marginal_covs.append(S)

            self.filter_means.append(m_bar)
            self.filter_covs.append(P_bar)
        self.filter_means = self.filter_means[1:]
        self.filter_covs = self.filter_covs[1:]

    def smooth(self):
        """
        Run Rauch–Tung–Striebel
        :return:
        """
        self.smoother_means = [self.filter_means[-1].copy()]
        self.smoother_covs = [self.filter_covs[-1].copy()]

        for t in range(len(self.filter_means) - 1):
            m_f = self.filter_means[- (2 + t)]
            P_f = self.filter_covs[- (2 + t)]
            m_bar = self.transition_matrix @ m_f
            P_bar = self.transition_matrix @ P_f @ self.transition_matrix.T + self.transition_cov
            G = P_f @ self.transition_matrix.T @ np.linalg.inv(P_bar)
            m = m_f + G @ (self.smoother_means[-1] - m_bar)
            P = P_f + G @ (self.smoother_covs[-1] - P_bar) @ G.T
            self.smoother_means.append(m)
            self.smoother_covs.append(P)

        self.smoother_means.reverse()
        self.smoother_covs.reverse()


