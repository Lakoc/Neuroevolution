import os
import pickle

import numpy as np

import src.analysis.plots as plots
import src.utils as utils

experiment_dir = "experiments/hierarchical"


def load_fitness_from_txt(path):
    content = open(path, 'r').readlines()[1:]
    fitness_local = np.zeros((6, 100))

    for line in content:
        gen_ind, _, fit = line.split()
        if '_' not in gen_ind:
            fitness_local[0, int(gen_ind)] = float(fit) / 100
        else:
            gen, ind = gen_ind.split('_')
            fitness_local[int(gen) + 1, int(ind)] = float(fit) / 100
    return fitness_local.T


def load_experiment_data(experiments):
    best_individuals = np.zeros((len(experiments), 3))
    fitness = np.zeros((len(experiments), 100, 6))
    for i, experiment in enumerate(experiments):
        experiment_setting = os.listdir(os.path.join(experiment_dir, experiment, 'best'))[0]
        fitness_path = os.path.join(experiment_dir, experiment, 'fitness', f'{experiment_setting}.npy')

        fitness[i, ...] = np.load(fitness_path).T if os.path.exists(fitness_path) else load_fitness_from_txt(
            os.path.join(experiment_dir, experiment, 'results.txt'))

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
                                   ["population_size: 100, n_generations: 5"]), marker_types=pareto_optimal).savefig(
        'doc/hierarchical_individuals.pdf')

    plots.boxplot(fitness, "Fitness per generation", "Generation", "Accuracy",
                  {'labels': np.arange(0, 6)}).savefig(
        'doc/hierarchical_generations.pdf')
