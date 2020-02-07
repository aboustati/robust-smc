import numpy as np
import pandas as pd

from robust_smc.kalman import Kalman
from robust_smc.sampler import LinearGaussianBPF, RobustifiedLinearGaussianBPF
from robust_smc.smoother import LinearGaussianSmoother
from robust_smc.state_space_kernels import StateSpaceMatern52

from tqdm import trange

from sklearn.metrics import mean_squared_error, median_absolute_error
from experiment_utilities import pickle_save

# Experiment Settings
RNG_SEED = 2235
NUM_RUNS = 100
# BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
BETA = [0.005, 0.01, 0.05, 0.1, 0.2]

# Sampler Settings
NUM_LATENT = 3
NUM_SAMPLES = 1000

# RNG
RNG = np.random.RandomState(RNG_SEED)

# GP Settings
LENGTHSCALE = 0.03
SIGNAL_VARIANCE = 32.
NOISE_VARIANCE = 1.0


def get_data():
    path = '~/Documents/PhD/Data/Air Quality/data_mat_pm25.csv'
    air_data = pd.read_csv(path, header=None)
    sample_series = air_data.loc[59, 10201:10400]  # Series with heavy contamination
    test_series = np.nanmedian(air_data, axis=0)[10201:10401]  # Baseline series, median of all series
    X, Y = np.linspace(0, 1, num=sample_series.shape[0])[:, None], sample_series.values[:, None]
    return X, Y, test_series


def experiment_step(X, Y):
    sampler_seed = RNG.randint(0, 1000000)
    smoother_seed = RNG.randint(0, 1000000)

    ss_kernel = StateSpaceMatern52(lengthscale=LENGTHSCALE, variance=SIGNAL_VARIANCE)

    delta = X[1]
    A, Q, H, m_0, P_0 = ss_kernel.discretize(delta=delta)

    L_0 = np.linalg.cholesky(P_0)
    x_init = m_0 + L_0[None, :, :] @ RNG.randn(NUM_SAMPLES, NUM_LATENT, 1)
    x_init = x_init[:, :, 0]

    vanilla_bpf = LinearGaussianBPF(
        data=Y[:, None],
        transition_matrix=A,
        transition_cov=Q,
        X_init=x_init,
        observation_model=lambda x: (H[None, :, :] @ x[:, :, None])[:, :, 0],
        observation_cov=NOISE_VARIANCE,
        num_samples=NUM_SAMPLES,
        seed=sampler_seed
    )
    vanilla_bpf.sample()
    vanilla_smoother = LinearGaussianSmoother(vanilla_bpf, NUM_SAMPLES, seed=smoother_seed)
    vanilla_smoother.sample_smooth_trajectories()

    # Robust Sampler
    robust_bpfs = []
    robust_smoothers = []
    for b in BETA:
        robust_bpf = RobustifiedLinearGaussianBPF(
            data=Y[:, None],
            beta=b,
            transition_matrix=A,
            observation_model=lambda x: (H[None, :, :] @ x[:, :, None])[:, :, 0],
            transition_cov=Q,
            observation_cov=NOISE_VARIANCE,
            X_init=x_init,
            num_samples=NUM_SAMPLES,
            seed=smoother_seed
        )
        robust_bpf.sample()
        robust_bpfs.append(robust_bpf)
        robust_smoother = LinearGaussianSmoother(robust_bpf, NUM_SAMPLES, seed=smoother_seed)
        robust_smoother.sample_smooth_trajectories()
        robust_smoothers.append(robust_smoother)

    return vanilla_smoother, robust_smoothers, vanilla_bpf, robust_bpfs


def kalman_results(X, Y, test_series):
    ss_kernel = StateSpaceMatern52(lengthscale=LENGTHSCALE, variance=SIGNAL_VARIANCE)
    delta = X[1]
    A, Q, H, m_0, P_0 = ss_kernel.discretize(delta=delta)

    kalman = Kalman(data=Y, transition_matrix=A, transition_cov=Q,
                    observation_matrix=H, observation_cov=NOISE_VARIANCE, m_0=m_0, P_0=P_0)

    kalman.filter()
    kalman.smooth()

    traj = np.stack(kalman.smoother_means)
    uncertainty = np.stack([np.diagonal(P) for P in kalman.smoother_covs])

    mean = traj[:, 0, 0]
    std = np.sqrt(uncertainty[:, 0] + NOISE_VARIANCE)
    quantiles = np.stack([mean - 1.64 * std, mean + 1.64 * std])
    NMSE = (mean_squared_error(mean, test_series) / (test_series ** 2).mean())

    upper = test_series <= quantiles[1, :]
    lower = test_series >= quantiles[0, :]
    coverage = np.sum(upper * lower) / len(test_series)

    score = [NMSE, coverage]
    return score


def compute_mse_and_coverage(test_series, smoother):
    mc_pred = smoother.smoother_samples[:, :, 0] + np.sqrt(NOISE_VARIANCE) * RNG.randn(len(test_series), NUM_SAMPLES)
    mean = mc_pred.mean(axis=1)
    quantiles = np.quantile(mc_pred, q=[0.05, 0.95], axis=1)
    mse = mean_squared_error(test_series, mean) / np.mean(test_series ** 2)
    upper = test_series <= quantiles[1, :]
    lower = test_series >= quantiles[0, :]
    coverage = np.sum(upper * lower) / len(test_series)
    scores = [mse, coverage]
    return scores


def compute_predictive_score(Y, sampler):
    Y_pred = np.stack([sampler.observation_model(X) for X in sampler.X_samples]).mean(axis=1)
    score = median_absolute_error(Y, Y_pred)
    return score


def run(runs):
    X, Y, test_series = get_data()

    kalman_data = kalman_results(X, Y, test_series)

    vanilla_bpf_data, robust_bpf_data = [], []
    robust_predictive_data = []
    for _ in trange(runs):
        vanilla_smoother, robust_smoothers, vanilla_bpf, robust_bpfs = experiment_step(X, Y)
        vanilla_bpf_data.append(compute_mse_and_coverage(test_series, vanilla_smoother))
        robust_bpf_data.append([compute_mse_and_coverage(test_series, robust_smoother) for robust_smoother in robust_smoothers])

        robust_predictive_data.append([compute_predictive_score(Y, robust_bpf) for robust_bpf in robust_bpfs])

    return (np.array(kalman_data), np.array(vanilla_bpf_data), np.array(robust_bpf_data)), np.array(robust_predictive_data)


if __name__ == '__main__':
    results, predictive_results = run(NUM_RUNS)
    pickle_save(f'./results/air_quality/beta-sweep.pk', results)
    pickle_save(
        f'./results/air_quality/beta-predictive-sweep.pk',
        predictive_results
    )
