import numpy as np

from robust_smc.data import TANSimulator, dem
from robust_smc.sampler import LinearGaussianBPF, RobustifiedLinearGaussianBPF

from tqdm import trange

from sklearn.metrics import mean_squared_error
from experiment_utilities import pickle_save

# Experiment Settings
NUM_RUNS = 100
BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
FREQUENCIES = [0, 1, 2]

# Sampler Settings
NUM_LATENT = 6
NUM_SAMPLES = 1000
NOISE_STD = 20.0
FINAL_TIME = 100
TIME_STEP = 0.1


def observation_model(X, num_frequencies):
    return X[:, 2][:, None].copy() - dem(X[:, 0][:, None].copy(), X[:, 1][:, None].copy(), num_frequencies)


def experiment_step(simulator, num_frequencies, seed=None):
    rng = np.random.RandomState(seed)
    Y = simulator.renoise()

    transition_matrix = simulator.transition_matrix
    transition_cov = np.diag(simulator.process_std ** 2)
    observation_cov = simulator.observation_std ** 2

    prior_std = np.array([1e-1, 1e-1, 1.0, 1e-2, 1e-2, 1e-1])
    X_init = simulator.X0[None, :] + prior_std[None, :] * rng.randn(NUM_SAMPLES, NUM_LATENT)
    X_init = X_init.squeeze()

    # BPF Sampler
    vanilla_bpf = LinearGaussianBPF(
        data=Y,
        transition_matrix=transition_matrix,
        observation_model=lambda X: observation_model(X, num_frequencies),
        transition_cov=transition_cov,
        observation_cov=observation_cov,
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
            observation_model=lambda X: observation_model(X, num_frequencies),
            transition_cov=transition_cov,
            observation_cov=observation_cov,
            X_init=X_init,
            num_samples=NUM_SAMPLES
        )
        robust_bpf.sample()
        robust_bpfs.append(robust_bpf)

    return simulator, vanilla_bpf, robust_bpfs


def compute_mse(simulator, sampler):
    trajectories = np.stack(sampler.X_trajectories)
    scores = []
    for var in range(NUM_LATENT):
        mean = trajectories[:, :, var].mean(axis=1)
        scores.append(mean_squared_error(simulator.X[:, var], mean))
    return scores


def run(runs, num_frequencies):
    process_std = None

    simulator = TANSimulator(
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        observation_std=NOISE_STD,
        process_std=process_std
    )
    vanilla_bpf_data, robust_bpf_data = [], []
    for _ in trange(runs):
        simulator, vanilla_bpf, robust_bpfs = experiment_step(simulator, num_frequencies)
        vanilla_bpf_data.append(compute_mse(simulator, vanilla_bpf))
        robust_bpf_data.append([compute_mse(simulator, robust_bpf) for robust_bpf in robust_bpfs])

    return np.array(vanilla_bpf_data), np.array(robust_bpf_data)


if __name__ == '__main__':
    for num_frequencies in FREQUENCIES:
        results = run(NUM_RUNS, num_frequencies)
        pickle_save(f'./results/tan/beta-sweep-frequencies-{num_frequencies}.pk', results)
