import matplotlib.pyplot as plt

from experiment_utilities import pickle_load

BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2] #, 0.5, 0.8]
CONTAMINATION = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

LABELS = ['BPF', 't-BPF'] + [f'Robustified BPF - beta = {b}' for b in BETA]
TITLES = [
    'Displacement in $x$ direction',
    'Displacement in $y$ direction',
    'Displacement in $z$ direction',
    'Velocity in $x$ direction',
    'Velocity in $y$ direction',
    'Velocity in $z$ direction'
]

NUM_LATENT = 6


def plot_mse(results_file, nrows, ncols, figsize, save_path=None):
    vanilla_bpf_data, student_bpf_data, robust_bpf_data = pickle_load(results_file)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, sharex=True)
    ax = ax.flatten()
    for var in range(NUM_LATENT):
        ax[var].set_yscale('log')
        boxes = [vanilla_bpf_data[:, var, 0], student_bpf_data[:, var, 0]] \
                + [robust_bpf_data[:, i, var, 0] for i in range(len(BETA))]
        ax[var].boxplot(boxes)
        ax[var].set_title(TITLES[var])
        ax[var].set_ylabel('MSE')
        xtickNames = plt.setp(ax[var], xticklabels=LABELS)
        plt.setp(xtickNames, fontsize=12, rotation=-45)

    if save_path:
        plt.savefig(save_path)
    # plt.show()


def plot_coverage(results_file, nrows, ncols, figsize, save_path=None):
    vanilla_bpf_data, student_bpf_data, robust_bpf_data = pickle_load(results_file)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, sharex=True)
    ax = ax.flatten()
    for var in range(NUM_LATENT):
        boxes = [vanilla_bpf_data[:, var, 1], student_bpf_data[:, var, 1]] \
                + [robust_bpf_data[:, i, var, 1] for i in range(len(BETA))]
        ax[var].boxplot(boxes)
        ax[var].axhline(y=0.9, ls='--', c='k', alpha=0.6)
        ax[var].set_title(TITLES[var])
        ax[var].set_ylabel('Coverage')
        ax[var].set_ylim((0.0, 1.0))
        xtickNames = plt.setp(ax[var], xticklabels=LABELS)
        plt.setp(xtickNames, fontsize=12, rotation=-45)

    if save_path:
        plt.savefig(save_path)
    # plt.show()


def plot_pairwise_mse(results_file, nrows, ncols, figsize, save_path=None):
    vanilla_bpf_data, student_bpf_data, robust_bpf_data = pickle_load(results_file)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, sharex=True)
    ax = ax.flatten()
    for var in range(NUM_LATENT):
        # ax[var].set_yscale('log')
        boxes = [vanilla_bpf_data[:, var, 0], student_bpf_data[:, var, 0]] \
                + [robust_bpf_data[:, i, var, 0] for i in range(len(BETA))]
        ax[var].boxplot(boxes)
        ax[var].axhline(ls='--', c='k', alpha=0.6)
        ax[var].set_title(TITLES[var])
        ax[var].set_ylabel('MSE Difference')
        xtickNames = plt.setp(ax[var], xticklabels=LABELS[1:])
        plt.setp(xtickNames, fontsize=12, rotation=-45)

    if save_path:
        plt.savefig(save_path)
    # plt.show()


if __name__ == '__main__':
    for contamination in CONTAMINATION:
        plot_mse(
            f'./results/tan/impulsive_noise_with_student_t/beta-sweep-contamination-{contamination}.pk',
            nrows=NUM_LATENT,
            ncols=1,
            figsize=(20, 24),
            save_path=f'./figures/tan/impulsive_noise_with_student_t/mse/beta-sweep-contamination-{contamination}.pdf'
        )

        plot_coverage(
            f'./results/tan/impulsive_noise_with_student_t/beta-sweep-contamination-{contamination}.pk',
            nrows=NUM_LATENT,
            ncols=1,
            figsize=(20, 24),
            save_path=f'./figures/tan/impulsive_noise_with_student_t/coverage/beta-sweep-contamination-{contamination}.pdf'
        )

        # plot_pairwise_mse(
        #     f'./results/tan/impulsive_noise_long_run/beta-sweep-contamination-{contamination}.pk',
        #     nrows=NUM_LATENT,
        #     ncols=1,
        #     figsize=(20, 24),
        #     save_path=f'./figures/tan/impulsive_noise_long_run/pairwise/beta-sweep-contamination-{contamination}.pdf'
        # )
