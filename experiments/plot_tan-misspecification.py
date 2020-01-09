import matplotlib.pyplot as plt

from experiment_utilities import pickle_load

BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
FREQUENCIES = [0, 1, 2, 3, 4, 5, 6]
LABELS = ['BPF'] + [f'Robustified BPF - beta = {b}' for b in BETA]
TITLES = [
    'Displacement in $x$ direction',
    'Displacement in $y$ direction',
    'Displacement in $z$ direction',
    'Velocity in $x$ direction',
    'Velocity in $y$ direction',
    'Velocity in $z$ direction'
]

NUM_LATENT = 6


def plot(results_file, nrows, ncols, figsize, save_path=None):
    vanilla_bpf_data, robust_bpf_data = pickle_load(results_file)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, sharex=True)
    ax = ax.flatten()
    for var in range(NUM_LATENT):
        ax[var].set_yscale('log')
        boxes = [vanilla_bpf_data[:, var]] + [robust_bpf_data[:, i, var] for i in range(len(BETA))]
        ax[var].boxplot(boxes)
        ax[var].set_title(TITLES[var])
        ax[var].set_ylabel('MSE')
        xtickNames = plt.setp(ax[var], xticklabels=LABELS)
        plt.setp(xtickNames, fontsize=12, rotation=-45)

    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    for num_frequencies in FREQUENCIES:
        plot(
            f'./results/tan/beta-sweep-frequencies-{num_frequencies}.pk',
            nrows=NUM_LATENT,
            ncols=1,
            figsize=(20, 24),
            save_path=f'./figures/tan/beta-sweep-frequencies-{num_frequencies}.pdf'
        )
