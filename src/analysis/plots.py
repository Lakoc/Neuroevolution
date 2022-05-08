import matplotlib.pyplot as plt
from src.datasets import MNIST, FashionMNIST
import numpy as np


def plot_labels(data, labels, path):
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {labels[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.savefig(path)


def conv_plot(outputs):
    plt.figure(figsize=(6, 4))

    # pro jedno nastaveni mutace vykreslete konvergencni krivku
    x = np.arange(0, 5)
    plt.xscale('log')
    plt.plot(x, np.median(outputs, axis=0), color='C0')
    plt.fill_between(x, np.min(outputs, axis=0), np.max(outputs, axis=0), color='C0', alpha=0.4)
    plt.axhline(100, color="black", linestyle="--")
    plt.xlabel('Pocet evaluaci')
    plt.ylabel('Fitness')
    plt.title('Konvergencni krivka')

    plt.show()
    plt.close()


def conv_plot_multiple_runs(results, x_label, y_label, title):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    joined_runs = np.transpose(results, axes=[1, 0, 2]).reshape((results.shape[1], -1))
    x = np.arange(0, joined_runs.shape[0])
    per_epoch_median = np.median(joined_runs, axis=1)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, per_epoch_median, rcond=None)[0]

    ax.plot(x, per_epoch_median, label='Average fitness median over runs')
    ax.plot(x, m * x + c, label=f'Median slope {m:.2e}')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def histogram(outputs):
    plt.figure(figsize=(6, 4))

    # pro jedno nastaveni mutace vykreslete histogram
    plt.hist(outputs, bins=50, range=(70, 85), density=True)
    plt.xlabel('Hodnota')
    plt.ylabel('Cetnost')
    plt.title('Histogram')

    plt.show()
    plt.close()


def boxplot(data, title, x_label, y_label, boxplot_args):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.boxplot(data, **boxplot_args)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def scatter_distribution(data, x_label, y_label, title, legend_labels, marker_types=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for cls in range(len(legend_labels)):
        class_indexes = data[:, 2] == cls
        color = f'C{cls}'
        if marker_types is not None:
            pareto = data[class_indexes & marker_types]
            others = data[class_indexes & ~marker_types]
            ax.scatter(pareto[:, 0], pareto[:, 1], color=color, marker="x",
                       label=f'Pareto optimal - {legend_labels[cls]}')
            ax.scatter(others[:, 0], others[:, 1], color=color, marker="o", label=legend_labels[cls])
        else:
            filtered = data[class_indexes]
            ax.scatter(filtered[:, 0], filtered[:, 1], label=legend_labels[cls])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    batch, labels = next(iter(MNIST(batch_size=12).train))
    plot_labels(batch, labels, 'mnist_sample.pdf')
    batch, labels = next(iter(FashionMNIST(batch_size=12).train))
    plot_labels(batch, labels, 'fashion_mnist_sample.pdf')
