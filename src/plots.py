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


def histogram(outputs):
    plt.figure(figsize=(6,4))

    # pro jedno nastaveni mutace vykreslete histogram
    plt.hist(outputs, bins=50, range=(70, 85), density=True)
    plt.xlabel('Hodnota')
    plt.ylabel('Cetnost')
    plt.title('Histogram')

    plt.show()
    plt.close()

def boxplot(outputs):
    plt.figure(figsize=(6,4))
    plt.boxplot(outputs, labels=['0.1',  '0.2'], notch=True)
    plt.xlabel('Mutation prob')
    plt.ylabel('Hodnoty')
    plt.title('Boxplot')

if __name__ == '__main__':
    data, labels = next(iter(MNIST().train))
    plot_labels(data, labels, 'mnist_sample.pdf')
    data, labels = next(iter(FashionMNIST().train))
    plot_labels(data, labels, 'fashion_mnist_sample.pdf')
