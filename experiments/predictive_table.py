import numpy as np

from robust_smc.data import ConstantVelocityModel
from experiment_utilities import pickle_load

BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
# CONTAMINATION = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
CONTAMINATION = 0.1
LABELS = ['Kalman Filter', 'BPF'] + [r'$\beta$ = {}'.format(b) for b in BETA]
TITLES = [
    'Displacement in $x$ direction',
    'Displacement in $y$ direction',
    'Velocity in $x$ direction',
    "Velocity in $y$ direction"
]

SIMULATOR_SEED = 1400
NOISE_VAR = 1.0
FINAL_TIME = 100
TIME_STEP = 0.1
EXPLOSION_SCALE = 100.0
NUM_LATENT = 4

if __name__ == '__main__':
    other_predictive_scores = pickle_load(
        f'./results/constant-velocity/impulsive_noise_predictive_rebuttal_bpf/beta-sweep-contamination-{CONTAMINATION}.pk'
    )

    beta_predictive_scores = pickle_load(
        f'./results/constant-velocity/impulsive_noise_predictive/beta-sweep-contamination-{CONTAMINATION}.pk'
    )

    scores = np.hstack([other_predictive_scores, beta_predictive_scores])

    mean = np.mean(scores, axis=0)
    std_err = (np.std(scores, axis=0)) / np.sqrt(99)

    for i, l in enumerate(LABELS):
        print(l, ':', '{0:.2f} ({1:.2f})'.format(mean[i], std_err[i]))

    for i, l in enumerate(LABELS):
        print(l, '&', '{0:.2f} & {1:.2f}'.format(mean[i], std_err[i]))
