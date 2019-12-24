import numpy as np

from robust_smc.data import ConstantVelocityModel
from robust_smc.kalman import Kalman
from robust_smc.sampler import LinearGaussianBPF, RobustifiedLinearGaussianBPF

from tqdm import trange

from numpy.linalg import cholesky

from experiment_utilities import pickle_save, smse

# Experiment Settings
NUM_RUNS = 100
BETA = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
CONTAMINATION = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# Sampler Settings
NUM_LATENT = 4
NUM_SAMPLES = 1000
NOISE_VAR = 1.0
FINAL_TIME = 100
TIME_STEP = 0.1
EXPLOSION_SCALE = 100.0


def experiment_step(simulator, seed=None):
    rng = np.random.RandomState(seed)
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


def compute_smse(simulator, sampler):
    if isinstance(sampler, Kalman):
        filter_means = np.stack(sampler.filter_means)
        scores = []
        for var in range(NUM_LATENT):
            mean = filter_means[:, var]
            scores.append(smse(simulator.X[:, var], mean))
    else:
        trajectories = np.stack(sampler.X_trajectories)
        scores = []
        for var in range(NUM_LATENT):
            mean = trajectories[:, :, var].mean(axis=1)
            scores.append(smse(simulator.X[:, var], mean))
    return scores


def run(runs, contamination):
    observation_cov = NOISE_VAR * np.eye(2)
    simulator = ConstantVelocityModel(
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        observation_cov=observation_cov,
        explosion_scale=EXPLOSION_SCALE,
        contamination_probability=contamination
    )
    vanilla_bpf_data, robust_bpf_data, kalman_data = [], [], []
    for _ in trange(runs):
        simulator, kalman, vanilla_bpf, robust_bpfs = experiment_step(simulator)
        kalman_data.append(compute_smse(simulator, kalman))
        vanilla_bpf_data.append(compute_smse(simulator, vanilla_bpf))
        robust_bpf_data.append([compute_smse(simulator, robust_bpf) for robust_bpf in robust_bpfs])

    return np.array(kalman_data), np.array(vanilla_bpf_data), np.array(robust_bpf_data)


if __name__ == '__main__':
    for contamination in CONTAMINATION:
        results = run(NUM_RUNS, contamination)
        pickle_save(f'./results/constant-velocity/beta-sweep-contamination-{contamination}.pk', results)



