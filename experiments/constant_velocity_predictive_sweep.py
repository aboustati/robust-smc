import numpy as np

from robust_smc.data import ConstantVelocityModel
from robust_smc.sampler import RobustifiedLinearGaussianBPF

from tqdm import trange

from numpy.linalg import cholesky
from sklearn.metrics import median_absolute_error

from experiment_utilities import pickle_save

# Experiment Settings
# SIMULATOR_SEED = 1400
SIMULATOR_SEED = 2000
RNG_SEED = 1218
NUM_RUNS = 100
BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
CONTAMINATION = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
# CONTAMINATION = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# Sampler Settings
NUM_LATENT = 4
NUM_SAMPLES = 1000
NOISE_VAR = 1.0
# FINAL_TIME = 100
FINAL_TIME = 10
TIME_STEP = 0.1
EXPLOSION_SCALE = 100.0

# RNG
RNG = np.random.RandomState(RNG_SEED)


def experiment_step(simulator):
    Y = simulator.renoise()

    X_init = simulator.initial_state[None, ...] + cholesky(simulator.initial_cov) @ RNG.randn(NUM_SAMPLES, NUM_LATENT, 1)
    X_init = X_init.squeeze()

    seed = RNG.randint(0, 1000000)

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
            num_samples=NUM_SAMPLES,
            seed=seed
        )
        robust_bpf.sample()
        robust_bpfs.append(robust_bpf)

    return simulator, robust_bpfs


def compute_predictive_score(simulator, sampler):
    Y_pred = np.stack([sampler.observation_model(X) for X in sampler.X_samples]).mean(axis=1)

    m = np.median(simulator.Y, axis=0)
    w = m / m.sum()

    score = median_absolute_error(simulator.Y, Y_pred, multioutput=1 / w)
    return score


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
    robust_bpf_data = []
    for _ in trange(runs):
        simulator, robust_bpfs = experiment_step(simulator)
        robust_bpf_data.append([compute_predictive_score(simulator, robust_bpf) for robust_bpf in robust_bpfs])

    return np.array(robust_bpf_data)


if __name__ == '__main__':
    # for contamination in CONTAMINATION:
    #     results = run(NUM_RUNS, contamination)
    #     pickle_save(
    #         f'./results/constant-velocity/impulsive_noise_predictive/beta-sweep-contamination-{contamination}.pk',
    #         results
    #     )

    for contamination in CONTAMINATION:
        results = run(NUM_RUNS, contamination)
        pickle_save(
            f'./results/constant-velocity/impulsive_noise_predictive_rebuttal_alternative_seed/beta-sweep-contamination-{contamination}.pk',
            results
        )
