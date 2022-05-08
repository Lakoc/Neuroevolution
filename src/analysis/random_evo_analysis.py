import os

import scipy.stats as st
import numpy as np

import src.analysis.plots as plots

experiment_dir = "experiments/random_search"


def load_experiment_data(experiments):
    fitness = np.zeros((len(experiments), 250))
    for i, experiment in enumerate(experiments):
        experiment_setting = os.listdir(os.path.join(experiment_dir, experiment, 'best'))[0]
        fitness_path = os.path.join(experiment_dir, experiment, 'fitness', f'{experiment_setting}.npy')

        if 'Random' in experiment_setting:
            fitness[i, :] = np.load(fitness_path)
        else:
            fitness[i, :] = np.load(fitness_path)[0:5, :].reshape(-1)

    return fitness


if __name__ == '__main__':
    experiments = sorted(os.listdir(experiment_dir))
    fitness = load_experiment_data(experiments)
    plots.boxplot(fitness.T, "Random search vs. Evolutionary search", "Run", "Accuracy",
                  {'labels': ['Evo1', 'Evo2', 'Evo3', 'Rand1', 'Rand2', 'Rand3'], 'notch': True,
                   'showfliers': False}).savefig(
        'doc/random_run.pdf')

    evo = fitness[0:3, :].reshape(-1)
    random = fitness[3:, :].reshape(-1)

    alpha = 0.05
    _, p1 = st.normaltest(evo * 100)
    _, p2 = st.normaltest(random * 100)
    if np.all([p1 > alpha, p2 > alpha]):
        print('Normally distributed, t-test.')
        _, p = st.ttest_ind(evo, random)
        if p > alpha:
            print('The difference is NOT significant, the distributions MAY have the same mean value, p = ', p)
        else:
            print('The difference is significant, the distributions does not have the same mean value, p = ', p)

    else:
        print('Mann whitney test.')
        _, p = st.mannwhitneyu(evo, random)
        if p > alpha:
            print('The difference is NOT significant, the distributions MAY have the same mean value, p = ', p)
        else:
            print('The difference is significant, the distributions does not have the same mean value, p = ', p)
