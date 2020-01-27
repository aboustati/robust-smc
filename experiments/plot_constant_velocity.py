from cycler import cycler

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, rc, patches

from robust_smc.data import ConstantVelocityModel
from experiment_utilities import pickle_load

# matplotlib Global Settings
palette = cycler(color=cm.Set1.colors)
rc('axes', prop_cycle=palette)
rc('lines', lw=2)
rc('axes', lw=1.2, titlesize='x-large', labelsize='x-large')
rc('legend', fontsize='x-large')

BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
CONTAMINATION = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
LABELS = ['Kaman Filter', 'BPF'] + [r'$\beta$ = {}'.format(b) for b in BETA]
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


def plot(results_file, nrows, ncols, figsize, metric='mse', save_path=None):
    if metric == 'mse':
        metric_idx = 0
        ylabel = 'MSE'
        scale = 'log'
    elif metric == 'coverage':
        metric_idx = 1
        ylabel = '90% Empirical Coverage'
        scale = 'linear'
    else:
        raise NotImplementedError
    kalman_data, vanilla_bpf_data, robust_bpf_data = pickle_load(results_file)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, sharex=True)
    ax = ax.flatten()
    for var in range(NUM_LATENT):
        ax[var].set_yscale(scale)
        boxes = [kalman_data[:, var, metric_idx], vanilla_bpf_data[:, var, metric_idx]] \
                + [robust_bpf_data[:, i, var, metric_idx] for i in range(len(BETA))]
        ax[var].boxplot(boxes)
        ax[var].set_title(TITLES[var])
        ax[var].set_ylabel(ylabel)
        xtickNames = plt.setp(ax[var], xticklabels=LABELS)
        plt.setp(xtickNames, fontsize=12)

    if save_path:
        plt.savefig(save_path)


def violin_plot(contamination, results_file, nrows, ncols, figsize, metric='mse', save_path=None):
    if metric == 'mse':
        metric_idx = 0
        ylabel = 'NMSE'
        scale = 'log'
    elif metric == 'coverage':
        metric_idx = 1
        ylabel = '90% Empirical Coverage'
        scale = 'linear'
    else:
        raise NotImplementedError

    observation_cov = NOISE_VAR * np.eye(2)
    simulator = ConstantVelocityModel(
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        observation_cov=observation_cov,
        explosion_scale=EXPLOSION_SCALE,
        contamination_probability=contamination,
        seed=SIMULATOR_SEED
    )

    if metric == 'mse':
        normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
    else:
        normaliser = np.ones((1, NUM_LATENT))

    kalman_data, vanilla_bpf_data, robust_bpf_data = pickle_load(results_file)

    kalman_data = kalman_data[:, :, metric_idx] / normaliser
    vanilla_bpf_data = vanilla_bpf_data[:, :, metric_idx] / normaliser
    robust_bpf_data = robust_bpf_data[:, :, :, metric_idx] / normaliser[None, ...]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, sharex=True)
    ax = ax.flatten()
    points = 25
    for var in range(NUM_LATENT):
        ax[var].set_yscale(scale)
        kalman_plot = ax[var].violinplot(dataset=kalman_data[:, var], points=points,
                                         showmedians=True, positions=[1])
        bpf_plot = ax[var].violinplot(dataset=vanilla_bpf_data[:, var], points=points,
                                      showmedians=True, positions=[2])
        robust_bpf_plot = ax[var].violinplot(dataset=robust_bpf_data[:, :, var], points=points,
                                             showmedians=True, positions=range(3, len(BETA) + 3))

        kalman_plot['bodies'][0].set_facecolor('C2')
        kalman_plot['bodies'][0].set_edgecolor('black')
        kalman_plot['bodies'][0].set_alpha(1)

        bpf_plot['bodies'][0].set_facecolor('C1')
        bpf_plot['bodies'][0].set_edgecolor('black')
        bpf_plot['bodies'][0].set_alpha(1)

        for pc in robust_bpf_plot['bodies']:
            pc.set_facecolor('C0')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        for element in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
            kalman_plot[element].set_color('black')
            bpf_plot[element].set_color('black')
            robust_bpf_plot[element].set_color('black')

        ax[var].set_title(TITLES[var])
        ax[var].set_ylabel(ylabel)
        ax[var].set_xticks(range(1, len(BETA) + 3))
        xtickNames = plt.setp(ax[var], xticklabels=LABELS)
        plt.setp(xtickNames, fontsize=12)

        colors = ['C2', 'C1', 'C0']
        labels = ['Kalman Filter', 'BPF', r'$\beta$-BPF']
        plot_patches = [patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]

        ax[var].legend(handles=plot_patches, loc='center right',
                       bbox_to_anchor=(1.15, 0.5), frameon=False)
        ax[var].grid(axis='y')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


def individual_violin_plot(contamination, state, results_file, figsize, save_path=None):

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=150, sharex=True)
    plt.subplots_adjust(hspace=0.1)
    ax = ax.flatten()
    points = 25

    for metric in ['mse', 'coverage']:
        if metric == 'mse':
            metric_idx = 0
            ylabel = 'NMSE'
            scale = 'log'
        elif metric == 'coverage':
            metric_idx = 1
            ylabel = '90% Empirical Coverage'
            scale = 'linear'
        else:
            raise NotImplementedError

        observation_cov = NOISE_VAR * np.eye(2)
        simulator = ConstantVelocityModel(
            final_time=FINAL_TIME,
            time_step=TIME_STEP,
            observation_cov=observation_cov,
            explosion_scale=EXPLOSION_SCALE,
            contamination_probability=contamination,
            seed=SIMULATOR_SEED
        )

        if metric == 'mse':
            normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
        else:
            normaliser = np.ones((1, NUM_LATENT))

        kalman_data, vanilla_bpf_data, robust_bpf_data = pickle_load(results_file)

        kalman_data = kalman_data[:, :, metric_idx] / normaliser
        vanilla_bpf_data = vanilla_bpf_data[:, :, metric_idx] / normaliser
        robust_bpf_data = robust_bpf_data[:, :, :, metric_idx] / normaliser[None, ...]

        ax[metric_idx].set_yscale(scale)
        kalman_plot = ax[metric_idx].violinplot(dataset=kalman_data[:, state], points=points,
                                         showmedians=True, positions=[1])
        bpf_plot = ax[metric_idx].violinplot(dataset=vanilla_bpf_data[:, state], points=points,
                                      showmedians=True, positions=[2])
        robust_bpf_plot = ax[metric_idx].violinplot(dataset=robust_bpf_data[:, :, state], points=points,
                                             showmedians=True, positions=range(3, len(BETA) + 3))

        if metric == 'coverage':
            ax[metric_idx].axhline(0.9, c='k', ls='--', alpha=0.6, lw=1.2, zorder=1000)

        kalman_plot['bodies'][0].set_facecolor('C2')
        kalman_plot['bodies'][0].set_edgecolor('black')
        kalman_plot['bodies'][0].set_alpha(1)

        bpf_plot['bodies'][0].set_facecolor('C1')
        bpf_plot['bodies'][0].set_edgecolor('black')
        bpf_plot['bodies'][0].set_alpha(1)

        for pc in robust_bpf_plot['bodies']:
            pc.set_facecolor('C0')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        for element in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
            kalman_plot[element].set_color('black')
            bpf_plot[element].set_color('black')
            robust_bpf_plot[element].set_color('black')

        ax[metric_idx].set_ylabel(ylabel)
        ax[metric_idx].set_xticks(range(1, len(BETA) + 3))
        xtickNames = plt.setp(ax[metric_idx], xticklabels=['', ''] + BETA)
        plt.setp(xtickNames, fontsize=12, rotation=-45)

        ax[metric_idx].grid(axis='y')

    colors = ['C2', 'C1', 'C0']
    labels = ['Kalman Filter', 'BPF', r'$\beta$-BPF']
    plot_patches = [patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]

    ax[0].legend(handles=plot_patches, loc='upper center', frameon=False)
    ax[1].set_xlabel(r'$\beta$')
    ax[0].set_title(TITLES[state], fontsize=18)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    for metric in ['mse', 'coverage']:
        for contamination in CONTAMINATION:
            # plot(
            #     f'./results/constant-velocity/impulsive_noise/beta-sweep-contamination-{contamination}.pk',
            #     nrows=4,
            #     ncols=1,
            #     figsize=(20, 14),
            #     metric=metric,
            #     save_path=f'./figures/constant-velocity/impulsive_noise/{metric}/beta-sweep-contamination-{contamination}.pdf'
            # )

            violin_plot(
                contamination=contamination,
                results_file=f'./results/constant-velocity/impulsive_noise/beta-sweep-contamination-{contamination}.pk',
                nrows=4,
                ncols=1,
                figsize=(20, 14),
                metric=metric,
                save_path=f'./figures/constant-velocity/impulsive_noise/{metric}/violin_beta-sweep-contamination-{contamination}.pdf'
            )

    for state in range(NUM_LATENT):
        individual_violin_plot(
            contamination=0.1,
            state=state,
            results_file=f'./results/constant-velocity/impulsive_noise/beta-sweep-contamination-{contamination}.pk',
            figsize=(6, 8),
            save_path=f'./figures/constant-velocity/impulsive_noise/latents/violin_latent_{state}.pdf'
        )
