from cycler import cycler

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, rc, patches, lines

from robust_smc.data import AsymmetricConstantVelocity
from experiment_utilities import pickle_load

# matplotlib Global Settings
palette = cycler(color=cm.Set1.colors)
rc('axes', prop_cycle=palette)
rc('lines', lw=2)
rc('axes', lw=1.2, titlesize='large', labelsize='x-large')
rc('legend', fontsize='large')
rc('font', family='serif')

BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
CONTAMINATION = 0.1
# LABELS = ['BPF'] + [r'$\beta$ = {}'.format(b) for b in BETA]
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


def aggregate_box_plot(results_file, figsize, save_path=None):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=150, sharex=True)
    plt.subplots_adjust(wspace=0.4, top=0.8)
    ax = ax.flatten()

    for metric in ['mse', 'coverage']:
        if metric == 'mse':
            metric_idx = 0
            ylabel = 'NMSE'
            scale = 'log'
        elif metric == 'coverage':
            metric_idx = 1
            ylabel = '90% EC'
            scale = 'linear'
        else:
            raise NotImplementedError

        observation_cov = NOISE_VAR * np.eye(2)
        simulator = AsymmetricConstantVelocity(
            final_time=FINAL_TIME,
            time_step=TIME_STEP,
            observation_cov=observation_cov,
            explosion_scale=EXPLOSION_SCALE,
            contamination_probability=CONTAMINATION,
            seed=SIMULATOR_SEED
        )

        if metric == 'mse':
            normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
        else:
            normaliser = np.ones((1, NUM_LATENT))

        vanilla_bpf_data, student_bpf_data, s_robust_bpf_data, a_robust_bpf_data = pickle_load(results_file)

        vanilla_bpf_data = vanilla_bpf_data[:, :, metric_idx] / normaliser
        student_bpf_data = student_bpf_data[:, :, :, metric_idx] / normaliser[None, ...]
        s_robust_bpf_data = s_robust_bpf_data[:, :, :, metric_idx] / normaliser[None, ...]
        a_robust_bpf_data = a_robust_bpf_data[:, :, metric_idx] / normaliser

        vanilla_bpf_data = vanilla_bpf_data.mean(axis=-1)
        student_bpf_data = student_bpf_data.mean(axis=-1)
        s_robust_bpf_data = s_robust_bpf_data.mean(axis=-1)
        a_robust_bpf_data = a_robust_bpf_data.mean(axis=-1)

        ax[metric_idx].set_yscale(scale)

        bpf_plot = ax[metric_idx].boxplot(vanilla_bpf_data, positions=[1], sym='x',
                                             patch_artist=True, widths=0.5, whis='range')
        student_bpf_plot = ax[metric_idx].boxplot(student_bpf_data, positions=[2, 3], sym='x',
                                          patch_artist=True, widths=0.5, whis='range')
        # s_robust_bpf_plot = ax[metric_idx].boxplot(s_robust_bpf_data, positions=[4, 5],
        #                                          sym='x', patch_artist=True, widths=0.5, whis='range')
        a_robust_bpf_plot = ax[metric_idx].boxplot(a_robust_bpf_data, positions=[4],
                                                   sym='x', patch_artist=True, widths=0.5, whis='range')

        if metric == 'coverage':
            ax[metric_idx].axhline(0.9, c='k', ls='--', alpha=0.6, lw=1.2, zorder=-1)

        bpf_plot['boxes'][0].set_facecolor('C1')
        bpf_plot['boxes'][0].set_edgecolor('black')
        bpf_plot['boxes'][0].set_alpha(1)

        a_robust_bpf_plot['boxes'][0].set_facecolor('C0')
        a_robust_bpf_plot['boxes'][0].set_edgecolor('black')
        a_robust_bpf_plot['boxes'][0].set_alpha(1)

        for pc in student_bpf_plot['boxes']:
            pc.set_facecolor('C2')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        # for pc in s_robust_bpf_plot['boxes']:
        #     pc.set_facecolor('C3')
        #     pc.set_edgecolor('black')
        #     pc.set_alpha(1)

        for element in ['medians']:
            a_robust_bpf_plot[element][0].set_color('black')
            bpf_plot[element][0].set_color('black')
            [box.set_color('black') for box in student_bpf_plot[element]]
            # [box.set_color('black') for box in s_robust_bpf_plot[element]]

        ax[metric_idx].set_ylabel(ylabel)
        ax[metric_idx].set_xticks(range(2, 4))
        xtickNames = plt.setp(ax[metric_idx], xticklabels=['1', '10'])
        plt.setp(xtickNames, fontsize=12)

        ax[metric_idx].grid(axis='y')

    colors = ['C1', 'C2', 'C0']
    labels = ['BPF', 't-BPF', r'$\beta$-BPF']
    plot_patches = [patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]

    ax[1].legend(handles=plot_patches, loc='center right',
                 frameon=False, bbox_to_anchor=(1.7, 0.5), ncol=1)
    ax[0].set_xlabel(r'$\sigma$')
    ax[1].set_xlabel(r'$\sigma$')
    plt.suptitle(
        r'Asymmetric Wiener velocity: aggregate metrics for $p_c = {}$'.format(CONTAMINATION),
        x=0.56,
        fontsize=14
    )

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    aggregate_box_plot(
        results_file=f'./results/constant-velocity/asymmetric_noise/student_t_comparison.pk',
        figsize=(6, 2),
        save_path=f'./figures/constant-velocity/asymmetric_noise/asymmetric_boxplot_aggregate.pdf'
    )
