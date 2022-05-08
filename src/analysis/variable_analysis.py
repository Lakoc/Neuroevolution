import os
import pickle

import numpy as np

import src.analysis.plots as plots
import src.utils as utils

experiment_dir = "experiments/variable"


def load_experiment_data(experiments):
    best_individuals = np.zeros((len(experiments), 3))
    fitness = np.zeros((len(experiments), 50, 11))
    for i, experiment in enumerate(experiments):
        experiment_setting = os.listdir(os.path.join(experiment_dir, experiment, 'best'))[0]
        fitness_path = os.path.join(experiment_dir, experiment, 'fitness', f'{experiment_setting}.npy')

        fitness[i, ...] = np.load(fitness_path).T

        best_individual = pickle.load(
            open(os.path.join(experiment_dir, experiment, 'best', experiment_setting.split('.npy')[0]), 'rb'))
        best_individuals[i][0:2] = best_individual[0:2]

    return best_individuals, fitness


if __name__ == '__main__':
    experiments = os.listdir(experiment_dir)

    best_individuals, fitness = load_experiment_data(experiments)
    fitness = fitness.reshape((-1, fitness.shape[-1]))
    pareto_optimal = utils.is_pareto_efficient(best_individuals[:, :-1])
    plots.scatter_distribution(best_individuals, "Accuracy", "MACs",
                               "Best individuals per configuration",
                               np.array(
                                   ["population_size: 50, n_generations: 10"]), marker_types=pareto_optimal).savefig(
        'doc/variable_individuals.pdf')

    plots.boxplot(fitness, "Fitness per generation", "Generation", "Accuracy",
                  {'labels': np.arange(0, 11)}).savefig(
        'doc/variable_generations.pdf')
