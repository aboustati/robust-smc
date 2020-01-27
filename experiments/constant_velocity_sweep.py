import numpy as np

from robust_smc.data import ConstantVelocityModel
from robust_smc.kalman import Kalman
from robust_smc.sampler import LinearGaussianBPF, RobustifiedLinearGaussianBPF

from tqdm import trange

from numpy.linalg import cholesky
from sklearn.metrics import mean_squared_error

from experiment_utilities import pickle_save

# Experiment Settings
SIMULATOR_SEED = 1400
RNG_SEED = 1218
NUM_RUNS = 100
BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
CONTAMINATION = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# Sampler Settings
NUM_LATENT = 4
NUM_SAMPLES = 1000
NOISE_VAR = 1.0
FINAL_TIME = 100
TIME_STEP = 0.1
EXPLOSION_SCALE = 100.0

# RNG
rng = np.random.RandomState(RNG_SEED)


def experiment_step(simulator):
    Y = simulator.renoise()

    # Kalman
    kalman = Kalman(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        observation_matrix=simulator.observation_matrix,
        transition_cov=simulator.process_cov,
        observation_cov=simulator.observation_cov,
        m_0=np.zeros((NUM_LATENT, 1)),
        P_0=simulator.initial_cov
    )
    kalman.filter()

    X_init = simulator.initial_state[None, ...] + cholesky(simulator.initial_cov) @ rng.randn(NUM_SAMPLES, NUM_LATENT, 1)
    X_init = X_init.squeeze()

    # BPF Sampler
    vanilla_bpf = LinearGaussianBPF(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        observation_model=lambda x: (simulator.observation_matrix @ x[:, :, None]).squeeze(),
        transition_cov=simulator.process_cov,
        observation_cov=np.diag(simulator.observation_cov),
        X_init=X_init,
        num_samples=NUM_SAMPLES
    )
    vanilla_bpf.sample()

    # Robust Sampler
    robust_bpfs = []
    for b in BETA:
        robust_bpf = RobustifiedLinearGaussianBPF(
            data=Y,
            beta=b,
            transition_matrix=simulator.transition_matrix,
            observation_model=lambda x: (simulator.observation_matrix @ x[:, :, None]).squeeze(),
            transition_cov=simulator.process_cov,
            observation_cov=np.diag(simulator.observation_cov),
            X_init=X_init,
            num_samples=NUM_SAMPLES
        )
        robust_bpf.sample()
        robust_bpfs.append(robust_bpf)

    return simulator, kalman, vanilla_bpf, robust_bpfs


def compute_mse_and_coverage(simulator, sampler):
    if isinstance(sampler, Kalman):
        filter_means = np.stack(sampler.filter_means)[:, :, 0]
        filter_vars = np.diagonal(np.stack(sampler.filter_covs), axis1=1, axis2=2)

        scores = []
        for var in range(NUM_LATENT):
            mean = filter_means[:, var]
            std = np.sqrt(filter_vars[:, var])

            mse = mean_squared_error(simulator.X[:, var], mean)
            upper = simulator.X[:, var] <= mean + 1.64 * std
            lower = simulator.X[:, var] >= mean - 1.64 * std
            coverage = np.sum(upper * lower) / simulator.X.shape[0]
            scores.append([mse, coverage])
    else:
        trajectories = np.stack(sampler.X_trajectories)
        mean = trajectories.mean(axis=1)
        quantiles = np.quantile(trajectories, q=[0.05, 0.95], axis=1)
        scores = []
        for var in range(NUM_LATENT):
            mse = mean_squared_error(simulator.X[:, var], mean[:, var])
            upper = simulator.X[:, var] <= quantiles[1, :, var]
            lower = simulator.X[:, var] >= quantiles[0, :, var]
            coverage = np.sum(upper * lower) / simulator.X.shape[0]
            scores.append([mse, coverage])
    return scores


def run(runs, contamination):
    observation_cov = NOISE_VAR * np.eye(2)
    simulator = ConstantVelocityModel(
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        observation_cov=observation_cov,
        explosion_scale=EXPLOSION_SCALE,
        contamination_probability=contamination,
        seed=SIMULATOR_SEED
    )
    vanilla_bpf_data, robust_bpf_data, kalman_data = [], [], []
    for _ in trange(runs):
        simulator, kalman, vanilla_bpf, robust_bpfs = experiment_step(simulator)
        kalman_data.append(compute_mse_and_coverage(simulator, kalman))
        vanilla_bpf_data.append(compute_mse_and_coverage(simulator, vanilla_bpf))
        robust_bpf_data.append([compute_mse_and_coverage(simulator, robust_bpf) for robust_bpf in robust_bpfs])

    return np.array(kalman_data), np.array(vanilla_bpf_data), np.array(robust_bpf_data)


if __name__ == '__main__':
    for contamination in CONTAMINATION:
        results = run(NUM_RUNS, contamination)
        pickle_save(f'./results/constant-velocity/impulsive_noise/beta-sweep-contamination-{contamination}.pk', results)

