import numpy as np

from robust_smc.data import ExplosiveTANSimulator
from robust_smc.sampler import LinearGaussianBPF, RobustifiedLinearGaussianBPF

from tqdm import trange

from sklearn.metrics import mean_squared_error
from experiment_utilities import pickle_save

# Experiment Settings
SIMULATOR_SEED = 1992
RNG_SEED = 24
NUM_RUNS = 100
BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
CONTAMINATION = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# Sampler Settings
NUM_LATENT = 6
NUM_SAMPLES = 1000
NOISE_STD = 20.0
FINAL_TIME = 200
TIME_STEP = 0.1

# RNG
RNG = np.random.RandomState(RNG_SEED)


def experiment_step(simulator):
    Y = simulator.renoise()

    seed = RNG.randint(0, 1000000)

    transition_matrix = simulator.transition_matrix
    transition_cov = np.diag(simulator.process_std ** 2)
    observation_cov = simulator.observation_std ** 2

    prior_std = np.array([1e-1, 1e-1, 1.0, 1e-2, 1e-2, 1e-1])
    X_init = simulator.X0[None, :] + prior_std[None, :] * RNG.randn(NUM_SAMPLES, NUM_LATENT)
    X_init = X_init.squeeze()

    # BPF Sampler
    vanilla_bpf = LinearGaussianBPF(
        data=Y,
        transition_matrix=transition_matrix,
        observation_model=simulator.observation_model,
        transition_cov=transition_cov,
        observation_cov=observation_cov,
        X_init=X_init,
        num_samples=NUM_SAMPLES,
        seed=seed
    )
    vanilla_bpf.sample()

    # Robust Sampler
    robust_bpfs = []
    for b in BETA:
        robust_bpf = RobustifiedLinearGaussianBPF(
            data=Y,
            beta=b,
            transition_matrix=simulator.transition_matrix,
            observation_model=simulator.observation_model,
            transition_cov=transition_cov,
            observation_cov=observation_cov,
            X_init=X_init,
            num_samples=NUM_SAMPLES,
            seed=seed
        )
        robust_bpf.sample()
        robust_bpfs.append(robust_bpf)

    return simulator, vanilla_bpf, robust_bpfs


def compute_mse_and_coverage(simulator, sampler):
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
    process_std = None

    simulator = ExplosiveTANSimulator(
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        observation_std=NOISE_STD,
        process_std=process_std,
        contamination_probability=contamination,
        seed=SIMULATOR_SEED
    )
    vanilla_bpf_data, robust_bpf_data = [], []
    for _ in trange(runs):
        simulator, vanilla_bpf, robust_bpfs = experiment_step(simulator)
        vanilla_bpf_data.append(compute_mse_and_coverage(simulator, vanilla_bpf))
        robust_bpf_data.append([compute_mse_and_coverage(simulator, robust_bpf) for robust_bpf in robust_bpfs])

    return np.array(vanilla_bpf_data), np.array(robust_bpf_data)


if __name__ == '__main__':
    for contamination in CONTAMINATION:
        results = run(NUM_RUNS, contamination)
        pickle_save(f'./results/tan/impulsive_noise_long_run/beta-sweep-contamination-{contamination}.pk', results)
