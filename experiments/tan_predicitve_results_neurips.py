import os
from cycler import cycler

import numpy as np
from scipy.stats import mode

import matplotlib.pyplot as plt

from matplotlib import rc, cm

from robust_smc.data import ExplosiveTANSimulator

from experiment_utilities import pickle_load


SIMULATOR_SEED = 1992
NOISE_STD = 20.0
FINAL_TIME = 200
TIME_STEP = 0.1

# BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2] #, 0.5, 0.8]
# # CONTAMINATION = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]#, 0.4]
# CONTAMINATION = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

BETA = [0.005, 0.01, 0.05, 0.1, 0.2]
CONTAMINATION = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

LABELS = np.array(['BPF', 't-BPF'] + [r'$\beta$-BPF = {}'.format(b) for b in BETA] + \
         ['APF'] + [r'$\beta$-APF = {}'.format(b) for b in BETA])
print(len(LABELS))
TITLES = [
    'Displacement in $x$ direction',
    'Displacement in $y$ direction',
    'Displacement in $z$ direction',
    'Velocity in $x$ direction',
    'Velocity in $y$ direction',
    'Velocity in $z$ direction'
]

NUM_LATENT = 6


def get_mse_and_coverage(results_dict):
    vanilla_bpf_data = np.stack([item['metrics'] for item in results_dict['vanilla_bpf']])
    student_bpf_data = np.stack([item['metrics'] for item in results_dict['student_bpf']])
    # vanilla_apf_data = np.stack([item['metrics'] for item in results_dict['student_bpf']])
    robust_bpf_data = np.stack([[item['metrics'] for item in beta] for beta in results_dict['robust_bpfs']])
    # robust_apf_data = np.stack([[item['metrics'] for item in beta] for beta in results_dict['robust_apfs']])
    concatenated_data = np.concatenate([
        vanilla_bpf_data[:, None, :, :],
        # vanilla_apf_data[:, None, :, :],
        student_bpf_data[:, None, :, :],
        robust_bpf_data[:, :, :, :],
        # robust_apf_data[:, :, :, :]
    ], axis=1)
    return concatenated_data


def get_apf_mse_and_coverage(results_dict):
    vanilla_apf_data = np.stack([item['metrics'] for item in results_dict['vanilla_apf']])
    robust_apf_data = np.stack([[item['metrics'] for item in beta] for beta in results_dict['robust_apfs']])
    concatenated_data = np.concatenate([
        vanilla_apf_data[:, None, :, :],
        robust_apf_data[:, :, :, :]
    ], axis=1)
    return concatenated_data


def get_preditive_score(results_dict, filter):
    stacked = np.stack(results_dict[filter])
    return np.mean(stacked, axis=0), np.sqrt(np.var(stacked, ddof=1, axis=0) / (stacked.shape[0] - 1))


def print_predictive_table():
    bpf_results_path = f'./results/tan/neurips_impulsive_noise_with_bpf_and_3000_samples/'
    apf_results_path = f'./results/tan/neurips_impulsive_noise_apf_and_3000_samples/'

    means_over_pc, stds_over_pc = [], []
    for pc in CONTAMINATION:
        bpf_results = pickle_load(os.path.join(bpf_results_path, f'beta-predictive-sweep-contamination-{pc}.pk'))
        apf_results = pickle_load(os.path.join(apf_results_path, f'beta-predictive-sweep-contamination-{pc}.pk'))

        means, stds = [], []
        for method in ['vanilla_bpf', 'robust_bpfs', 'student_bpf']:
            m, s = get_preditive_score(bpf_results, method)
            means.append(m.reshape((-1, 1)))
            stds.append(s.reshape((-1, 1)))
        for method in ['vanilla_apf', 'robust_apfs']:
            m, s = get_preditive_score(apf_results, method)
            means.append(m.reshape((-1, 1)))
            stds.append(s.reshape((-1, 1)))

        means = np.concatenate(means, axis=0)
        stds = np.concatenate(stds, axis=0)

        means_over_pc.append(means)
        stds_over_pc.append(stds)

    means_over_pc = np.stack(means_over_pc).squeeze().T
    stds_over_pc = np.stack(stds_over_pc).squeeze().T

    for i, label in enumerate(LABELS):
        line = label
        for j in range(means_over_pc.shape[-1]):
            line += r' & ' + '{:.2f}'.format(means_over_pc[i, j]) + '({:.2f})'.format(stds_over_pc[i, j])
        line += r' \\'
        print(line)


if __name__ == '__main__':

    bpf_results_path = f'./results/tan/neurips_impulsive_noise_with_bpf_and_3000_samples/'
    apf_results_path = f'./results/tan/neurips_impulsive_noise_apf_and_3000_samples/'
    pc = 0.1
    bpf_results = pickle_load(os.path.join(bpf_results_path, f'beta-sweep-contamination-{pc}.pk'))
    print(bpf_results['vanilla_bpf'][0]['statistics'][1])

