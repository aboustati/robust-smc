import numpy as np

from robust_smc.data import AsymmetricConstantVelocity
from robust_smc.kalman import Kalman
from robust_smc.sampler import AsymmetricLinearGaussianBPF, LinearStudentTBPF
from robust_smc.sampler import RobustifiedLinearGaussianBPF, AsymmetricRobustifiedLinearGaussianBPF

from tqdm import trange

from numpy.linalg import cholesky
from sklearn.metrics import mean_squared_error

from experiment_utilities import pickle_save

# Experiment Settings
SIMULATOR_SEED = 1400
RNG_SEED = 1218
NUM_RUNS = 100
BETA = 0.1
CONTAMINATION = 0.1

# Sampler Settings
NUM_LATENT = 4
NUM_SAMPLES = 1000
NOISE_VAR = 1.0
FINAL_TIME = 100
TIME_STEP = 0.1
EXPLOSION_SCALE = 10.0
DF = 1.01

# RNG
RNG = np.random.RandomState(RNG_SEED)


def experiment_step(simulator):
    Y = simulator.renoise()

    X_init = simulator.initial_state[None, ...] + cholesky(simulator.initial_cov) @ RNG.randn(NUM_SAMPLES, NUM_LATENT, 1)
    X_init = X_init.squeeze()

    seed = RNG.randint(0, 1000000)

    # BPF Sampler
    vanilla_bpf = AsymmetricLinearGaussianBPF(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        observation_model=lambda x: (simulator.observation_matrix @ x[:, :, None]).squeeze(),
        transition_cov=simulator.process_cov,
        observation_cov_1=np.diag(simulator.observation_cov),
        observation_cov_2=(EXPLOSION_SCALE ** 2) * np.diag(simulator.observation_cov),
        X_init=X_init,
        num_samples=NUM_SAMPLES,
        seed=seed
    )
    vanilla_bpf.sample()


    # t_BPF Sampler
    student_bpfs = []
    symmetric_robust_bpfs = []
    scales = [1.0, EXPLOSION_SCALE]
    for scale in scales:
        student_bpf = LinearStudentTBPF(
            data=Y,
            transition_matrix=simulator.transition_matrix,
            observation_model=lambda x: (simulator.observation_matrix @ x[:, :, None]).squeeze(),
            transition_cov=simulator.process_cov,
            observation_cov=(scale ** 2) * np.diag(simulator.observation_cov),
            X_init=X_init,
            num_samples=NUM_SAMPLES,
            seed=seed
        )
        student_bpf.sample()
        student_bpfs.append(student_bpf)

        robust_bpf = RobustifiedLinearGaussianBPF(
            data=Y,
            beta=BETA,
            transition_matrix=simulator.transition_matrix,
            observation_model=lambda x: (simulator.observation_matrix @ x[:, :, None]).squeeze(),
            transition_cov=simulator.process_cov,
            observation_cov=(scale ** 2) * np.diag(simulator.observation_cov),
            X_init=X_init,
            num_samples=NUM_SAMPLES,
            seed=seed
        )
        robust_bpf.sample()
        symmetric_robust_bpfs.append(robust_bpf)

    # Robust Sampler
    asymmetric_robust_bpf = AsymmetricRobustifiedLinearGaussianBPF(
        data=Y,
        beta=BETA,
        transition_matrix=simulator.transition_matrix,
        observation_model=lambda x: (simulator.observation_matrix @ x[:, :, None]).squeeze(),
        transition_cov=simulator.process_cov,
        observation_cov_1=np.diag(simulator.observation_cov),
        observation_cov_2=(EXPLOSION_SCALE ** 2) * np.diag(simulator.observation_cov),
        X_init=X_init,
        num_samples=NUM_SAMPLES,
        seed=seed
    )
    asymmetric_robust_bpf.sample()

    return simulator, vanilla_bpf, student_bpfs, symmetric_robust_bpfs, asymmetric_robust_bpf


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


def run(runs):
    observation_cov = NOISE_VAR * np.eye(2)
    simulator = AsymmetricConstantVelocity(
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        observation_cov=observation_cov,
        explosion_scale=EXPLOSION_SCALE,
        contamination_probability=CONTAMINATION,
        seed=SIMULATOR_SEED
    )
    vanilla_bpf_data, student_bpf_data, symmetric_robust_data, asymmetric_robust_data = [], [], [], []
    for _ in trange(runs):
        simulator, vanilla_bpf, student_bpfs, s_robust_bpfs, a_robust_bpf = experiment_step(simulator)
        vanilla_bpf_data.append(compute_mse_and_coverage(simulator, vanilla_bpf))
        student_bpf_data.append([compute_mse_and_coverage(simulator, t_bpf) for t_bpf in student_bpfs])
        symmetric_robust_data.append([compute_mse_and_coverage(simulator, robust_bpf) for robust_bpf in s_robust_bpfs])
        asymmetric_robust_data.append(compute_mse_and_coverage(simulator, a_robust_bpf))

    return np.array(vanilla_bpf_data), np.array(student_bpf_data),\
           np.array(symmetric_robust_data), np.array(asymmetric_robust_data)


if __name__ == '__main__':
    results = run(NUM_RUNS)
    pickle_save(f'./results/constant-velocity/asymmetric_noise/student_t_comparison.pk', results)
