import numpy as np

from robust_smc.data import ConstantVelocityModel
from robust_smc.sampler import LinearGaussianBPF
from robust_smc.kalman import Kalman

from tqdm import trange

from numpy.linalg import cholesky
from sklearn.metrics import median_absolute_error

from experiment_utilities import pickle_save

# Experiment Settings
SIMULATOR_SEED = 1400
RNG_SEED = 1218
NUM_RUNS = 100
CONTAMINATION = [0.1]

# Sampler Settings
NUM_LATENT = 4
NUM_SAMPLES = 1000
NOISE_VAR = 1.0
FINAL_TIME = 100
# FINAL_TIME = 10
TIME_STEP = 0.1
EXPLOSION_SCALE = 100.0

# RNG
RNG = np.random.RandomState(RNG_SEED)


def experiment_step(simulator):
    # Kalman
    Y = simulator.renoise()

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

    X_init = simulator.initial_state[None, ...] + cholesky(simulator.initial_cov) @ RNG.randn(NUM_SAMPLES, NUM_LATENT, 1)
    X_init = X_init.squeeze()

    seed = RNG.randint(0, 1000000)

    # Robust Sampler
    bpf = LinearGaussianBPF(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        observation_model=lambda x: (simulator.observation_matrix @ x[:, :, None]).squeeze(),
        transition_cov=simulator.process_cov,
        observation_cov=np.diag(simulator.observation_cov),
        X_init=X_init,
        num_samples=NUM_SAMPLES,
        seed=seed
    )
    bpf.sample()

    return simulator, kalman, bpf


def compute_predictive_score(simulator, sampler):
    if isinstance(sampler, Kalman):
        means, covs = sampler.filter_means, sampler.filter_covs
        preds = [sampler.one_step_prediction(m, P) for m, P in zip(means, covs)]
        Y_pred = np.stack([sampler.observation_matrix @ p for p, _ in preds])
        Y_pred = Y_pred.squeeze()
    else:
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
    data = []
    for _ in trange(runs):
        datum = []
        simulator, kalman, bpf = experiment_step(simulator)
        datum.append(compute_predictive_score(simulator, kalman))
        datum.append(compute_predictive_score(simulator, bpf))

        data.append(datum)

    return np.array(data)


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
            f'./results/constant-velocity/impulsive_noise_predictive_rebuttal_bpf/beta-sweep-contamination-{contamination}.pk',
            results
        )
