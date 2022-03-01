import matplotlib.pyplot as plt


def plot_labels_pred(data, labels, predictions):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {labels[i]}\nPrediction: {predictions[i]}")
        plt.xticks([])
        plt.yticks([])
    fig.show()
