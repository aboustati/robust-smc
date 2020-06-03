import os
from cycler import cycler

import numpy as np
from scipy.stats import mode

import matplotlib.pyplot as plt

from matplotlib import rc, cm

from robust_smc.data import ExplosiveTANSimulator

from experiment_utilities import pickle_load


# matplotlib Global Settings
palette = cycler(color=cm.Set1.colors)
rc('axes', prop_cycle=palette)
rc('lines', lw=2)
rc('axes', lw=1.2, titlesize='large', labelsize='x-large')
rc('legend', fontsize='x-large')
rc('font', family='serif')


SIMULATOR_SEED = 1992
NOISE_STD = 20.0
FINAL_TIME = 200
TIME_STEP = 0.1

BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]#, 0.5, 0.8]
# CONTAMINATION = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]#, 0.4]
CONTAMINATION = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

LABELS = np.array(['BPF', 't-BPF'] + [r'$\beta$ = {}'.format(b) for b in BETA])
TITLES = [
    'Displacement in $x$ direction',
    'Displacement in $y$ direction',
    'Displacement in $z$ direction',
    'Velocity in $x$ direction',
    'Velocity in $y$ direction',
    'Velocity in $z$ direction'
]

NUM_LATENT = 6


def plot_metrics(results_path, figsize, save_path=None):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=150, sharex=True)
    ax = ax.flatten()

    plt.subplots_adjust(hspace=0.05)

    for metric in ['mse', 'coverage']:
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
            predictive_scores = pickle_load(
                os.path.join(results_path, f'beta-predictive-sweep-contamination-{contamination}.pk')
            )
            best_beta = np.argmin(predictive_scores, axis=1)
            majority_vote = mode(best_beta)

            print(majority_vote)

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
            vanilla_bpf_data, student_bpf_data, robust_bpf_data = pickle_load(results_file)
            concatenated_data = np.concatenate([
                vanilla_bpf_data[:, None, :, metric_idx],
                student_bpf_data[:, None, :, metric_idx],
                robust_bpf_data[:, :, :, metric_idx]
            ], axis=1)
            concatenated_data = concatenated_data / normaliser  # contamination x N x models x num_latent
            plot_data.append(concatenated_data)

        plot_data = np.stack(plot_data).mean(axis=-1)[:, :, :-2]

        print(plot_data.shape)

        colors = [f'C{i}' for i in range(len(BETA) + 2)] * 12 #, 'C3', 'C4', 'C0']
        colors = cm.tab10.colors
        labels = LABELS
        positions = np.arange(1, len(BETA) + 3)

        ax[metric_idx].set_yscale(scale)
        for i in range(len(CONTAMINATION)):
            bplot = ax[metric_idx].boxplot(plot_data[i, :, :], positions=(i * 15) + positions,
                                    sym='x', patch_artist=True, manage_ticks=False, whis='range')
            ax[metric_idx].axvline((i * 15) + 8, color='gold', ls='-.', zorder=0)
            for box, m, color, l in zip(bplot['boxes'], bplot['medians'], colors, labels):
                box.set_facecolor(color)
                box.set_label(l)
                m.set_color('k')
                
        if metric == 'coverage':
            ax[metric_idx].set_ylim([0, 1.])
            ax[metric_idx].axhline(0.9, c='k', ls='--', alpha=0.6, lw=1.2, zorder=-1)

        ax[metric_idx].set_ylabel(label)
        ax[metric_idx].set_xticks(np.arange(5.5, 125, 15))
        ax[metric_idx].grid(axis='y')
        xtickNames = plt.setp(ax[metric_idx], xticklabels=CONTAMINATION)
        plt.setp(xtickNames, fontsize=12)

    ax[0].legend(handles=bplot['boxes'], loc='center right', bbox_to_anchor=(1.15, 0.0), frameon=False)

    ax[0].set_title('TAN experiment: aggregate metrics', fontsize=14)
    ax[-1].set_xlabel(r'Contamination probability $p_c$')
    if save_path:
        save_file = os.path.join(save_path, f'full_plots.pdf')
        plt.savefig(save_file, bbox_inches='tight')


def plot_aggregate_latent(results_path, figsize, save_path=None):
    selected_models = [0, 1, 7, 8]
    colors = ['C1', 'C2', 'C6', 'C0']

    labels = LABELS[selected_models]
    positions = np.arange(1, len(selected_models) + 1)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=150, sharex='all')
    axes = axes.flatten()

    plt.subplots_adjust(hspace=0.05)

    for ax, metric in zip(axes, ['mse', 'coverage']):
        if metric == 'mse':
            metric_idx = 0
            label = 'NMSE'
            scale = 'log'
        elif metric == 'coverage':
            metric_idx = 1
            label = '90% EC'
            scale = 'linear'
        else:
            raise NotImplementedError

        plot_data = []
        for contamination in CONTAMINATION:
            predictive_scores = pickle_load(
                os.path.join(results_path, f'beta-predictive-sweep-contamination-{contamination}.pk')
            )
            best_beta = np.argmin(predictive_scores, axis=1)
            majority_vote = mode(best_beta)

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
            vanilla_bpf_data, student_bpf_data, robust_bpf_data = pickle_load(results_file)
            concatenated_data = np.concatenate([
                vanilla_bpf_data[:, None, :, metric_idx],
                student_bpf_data[:, None, :, metric_idx],
                robust_bpf_data[:, :, :, metric_idx]
            ], axis=1)
            concatenated_data = concatenated_data / normaliser  # contamination x N x models x num_latent
            plot_data.append(concatenated_data)

        plot_data = np.stack(plot_data)
        plot_data = np.median(plot_data, axis=-1)

        if metric == 'coverage':
            ax.axhline(0.9, c='k', ls='--', alpha=0.6, lw=1.2, zorder=-1)

        ax.set_yscale(scale)
        for i in range(len(CONTAMINATION)):
            ax.axvline((i * 7) + 3, color='gold', ls='-.', zorder=0)
            bplot = ax.boxplot(plot_data[i, :, selected_models].T, positions=(i * 7) + positions,
                               sym='x', patch_artist=True, manage_ticks=False, whis='range',
                               widths=0.6, flierprops={'markersize': 4})
            for box, m, color, l in zip(bplot['boxes'], bplot['medians'], colors, labels):
                box.set_facecolor(color)
                box.set_label(l)
                m.set_color('k')

        ax.set_xticks(np.arange(2.5, 58.5, 7))
        xtickNames = plt.setp(ax, xticklabels=CONTAMINATION)
        plt.setp(xtickNames, fontsize=12)
        ax.set_ylabel(label)
        ax.grid(axis='y')

    axes[0].set_title('TAN experiment: aggregate metrics', fontsize=14)
    axes[-1].set_xlabel(r'Contamination probability $p_c$')
    axes[-1].legend(handles=bplot['boxes'], loc='center', bbox_to_anchor=(0.5, -0.4), frameon=False, ncol=6)
    if save_path:
        save_file = os.path.join(save_path, f'aggregate_plot.pdf')
        plt.savefig(save_file, bbox_inches='tight')


if __name__ == '__main__':
    plot_metrics(
        f'./results/tan/impulsive_noise_with_student_t/',
        figsize=(20, 8),
        save_path='./figures/tan/impulsive_noise_with_student_t/variation_with_contamination/'
    )
    #
    # for latent in range(6):
    #     plot_single_latent(
    #         f'./results/tan/impulsive_noise_with_student_t/',
    #         latent=latent,
    #         figsize=(8, 5),
    #         save_path='./figures/tan/impulsive_noise_with_student_t/variation_with_contamination/'
    #     )

    plot_aggregate_latent(
        f'./results/tan/impulsive_noise_with_student_t/',
        figsize=(8, 5),
        save_path='./figures/tan/impulsive_noise_with_student_t/variation_with_contamination/'
    )

