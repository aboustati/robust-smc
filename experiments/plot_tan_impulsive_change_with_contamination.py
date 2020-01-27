import os
from cycler import cycler

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rc, cm

from robust_smc.data import ExplosiveTANSimulator

from experiment_utilities import pickle_load


# matplotlib Global Settings
palette = cycler(color=cm.Set1.colors)
rc('axes', prop_cycle=palette)
rc('lines', lw=2)
rc('axes', lw=1.2, titlesize='x-large', labelsize='x-large')
rc('legend', fontsize='x-large')


SIMULATOR_SEED = 1992
NOISE_STD = 20.0
FINAL_TIME = 200
TIME_STEP = 0.1

BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
CONTAMINATION = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

LABELS = np.array(['BPF'] + [r'$\beta$ = {}'.format(b) for b in BETA])
TITLES = [
    'Displacement in $x$ direction',
    'Displacement in $y$ direction',
    'Displacement in $z$ direction',
    'Velocity in $x$ direction',
    'Velocity in $y$ direction',
    'Velocity in $z$ direction'
]

NUM_LATENT = 6


def plot_metric(results_path, nrows, ncols, figsize, metric='mse', save_path=None):
    if metric == 'mse':
        metric_idx = 0
        label = 'NMSE'
        scale = 'log'
    elif metric == 'coverage':
        metric_idx = 1
        label = '90% Empirical Coverage'
        scale = 'linear'
    else:
        raise NotImplementedError

    plot_data = []
    for contamination in CONTAMINATION:
        simulator = ExplosiveTANSimulator(
            final_time=FINAL_TIME,
            time_step=TIME_STEP,
            observation_std=NOISE_STD,
            process_std=None,
            contamination_probability=contamination,
            seed=SIMULATOR_SEED
        )

        if metric == 'mse':
            normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
        else:
            normaliser = 1

        results_file = os.path.join(results_path, f'beta-sweep-contamination-{contamination}.pk')
        vanilla_bpf_data, robust_bpf_data = pickle_load(results_file)
        concatenated_data = np.concatenate([vanilla_bpf_data[:, None, :, metric_idx], robust_bpf_data[:, :, :, metric_idx]], axis=1)
        concatenated_data = concatenated_data / normaliser  # contamination x N x models x num_latent
        plot_data.append(concatenated_data)

    plot_data = np.stack(plot_data)

    selected_models = [0, 3, 5, 7]
    colors = ['C1', 'C3', 'C4', 'C0']
    labels = LABELS[selected_models]
    positions = np.arange(1, len(selected_models) + 1)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, sharex=True)
    ax = ax.flatten()
    for var in range(NUM_LATENT):
        ax[var].set_yscale(scale)
        for i in range(len(CONTAMINATION)):
            bplot = ax[var].boxplot(plot_data[i, :, selected_models, var].T, positions=(i * 7) + positions,
                                    sym='', patch_artist=True, manage_ticks=False)
            for box, m, color, l in zip(bplot['boxes'], bplot['medians'], colors, labels):
                box.set_facecolor(color)
                box.set_label(l)
                m.set_color('k')

        ax[var].set_title(TITLES[var], fontsize='x-large')
        ax[var].set_ylabel(label)
        ax[var].legend(handles=bplot['boxes'], loc='center right', bbox_to_anchor=(1.12, 0.5), frameon=False)
        ax[var].set_xticks(np.arange(2.5, 55, 7))
        ax[var].grid(axis='y')
        xtickNames = plt.setp(ax[var], xticklabels=CONTAMINATION)
        plt.setp(xtickNames, fontsize=12)

    ax[var].set_xlabel('Contamination probability')

    # fig.suptitle('Terrain Aided Navigation Experiment', fontsize='xx-large')
    if save_path:
        save_file = os.path.join(save_path, f'{metric}.pdf')
        plt.savefig(save_file, bbox_inches='tight')


def plot_single_latent_metric(results_path, figsize, latent, metric='mse', save_path=None):
    if metric == 'mse':
        metric_idx = 0
        label = 'NMSE'
        scale = 'log'
    elif metric == 'coverage':
        metric_idx = 1
        label = '90% Empirical Coverage'
        scale = 'linear'
    else:
        raise NotImplementedError

    plot_data = []
    for contamination in CONTAMINATION:
        simulator = ExplosiveTANSimulator(
            final_time=FINAL_TIME,
            time_step=TIME_STEP,
            observation_std=NOISE_STD,
            process_std=None,
            contamination_probability=contamination,
            seed=SIMULATOR_SEED
        )

        if metric == 'mse':
            normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
        else:
            normaliser = 1

        results_file = os.path.join(results_path, f'beta-sweep-contamination-{contamination}.pk')
        vanilla_bpf_data, robust_bpf_data = pickle_load(results_file)
        concatenated_data = np.concatenate([vanilla_bpf_data[:, None, :, metric_idx], robust_bpf_data[:, :, :, metric_idx]], axis=1)
        concatenated_data = concatenated_data / normaliser  # contamination x N x models x num_latent
        plot_data.append(concatenated_data)

    plot_data = np.stack(plot_data)

    selected_models = [0, 3, 9, 7]
    colors = ['C1', 'C3', 'C4', 'C0']
    labels = LABELS[selected_models]
    positions = np.arange(1, len(selected_models) + 1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=150)
    ax.set_yscale(scale)
    for i in range(len(CONTAMINATION)):
        bplot = ax.boxplot(plot_data[i, :, selected_models, latent].T, positions=(i * 7) + positions,
                                sym='', patch_artist=True, manage_ticks=False)
        for box, m, color, l in zip(bplot['boxes'], bplot['medians'], colors, labels):
            box.set_facecolor(color)
            box.set_label(l)
            m.set_color('k')

    ax.set_title(TITLES[latent], fontsize='x-large')
    ax.set_ylabel(label)
    ax.legend(handles=bplot['boxes'], loc='center right', bbox_to_anchor=(1.12, 0.5), frameon=False)
    ax.set_xticks(np.arange(2.5, 55, 7))
    ax.grid(axis='y')
    xtickNames = plt.setp(ax, xticklabels=CONTAMINATION)
    plt.setp(xtickNames, fontsize=12)

    ax.set_xlabel('Contamination probability')

    # fig.suptitle('Terrain Aided Navigation Experiment', fontsize='xx-large')
    if save_path:
        save_file = os.path.join(save_path, f'plot_{latent}_{metric}.pdf')
        plt.savefig(save_file, bbox_inches='tight')


if __name__ == '__main__':
    for metric in ['mse', 'coverage']:
        plot_metric(
            f'./results/tan/impulsive_noise_long_run/',
            nrows=NUM_LATENT,
            ncols=1,
            figsize=(20, 24),
            metric=metric,
            save_path='./figures/tan/impulsive_noise_long_run/variation_with_contamination/'
        )

        plot_single_latent_metric(
            f'./results/tan/impulsive_noise_long_run/',
            latent=5,
            figsize=(10, 5),
            metric=metric,
            save_path='./figures/tan/impulsive_noise_long_run/variation_with_contamination/'
        )

