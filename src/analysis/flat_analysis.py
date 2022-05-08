import os
import pickle

import matplotlib
import numpy as np

import src.analysis.plots as plots
import src.utils as utils

experiment_dir = "experiments/flat"


def load_experiment_data(experiments):
    best_individuals = np.zeros((len(experiments), 3))
    fitness_small_pop = np.zeros((len(experiments), 101, 5))
    fitness_big_pop = np.zeros((len(experiments), 6, 100))
    population_type = np.zeros(len(experiments), dtype=int)
    init_mutations = np.zeros(len(experiments), dtype=bool)
    bigger_kernel = np.zeros(len(experiments), dtype=bool)
    bigger_architectures = np.zeros(len(experiments), dtype=int)

    for i, experiment in enumerate(experiments):
        experiment_setting = os.listdir(os.path.join(experiment_dir, experiment, 'fitness'))[0]

        best_individuals[i][0:2] = pickle.load(
            open(os.path.join(experiment_dir, experiment, 'best', experiment_setting.split('.npy')[0]), 'rb'))[0:2]

        experiment_type = 1 if 'population_size:100' in experiment_setting else 2
        best_individuals[i][2] = experiment_type
        population_type[i] = experiment_type

        fitness_local = np.load(os.path.join(experiment_dir, experiment, 'fitness', experiment_setting))
        if experiment_type == 1:
            fitness_big_pop[i, :] = fitness_local
        else:
            fitness_small_pop[i, :] = fitness_local

        init_mutations[i] = True if 'init_mutations:1000' in experiment_setting else False
        bigger_kernel[i] = True if 'kernel_size:5' in experiment_setting else False
        more_nodes = 'n_nodes:8' in experiment_setting or 'n_nodes:4' in experiment_setting
        more_channels = 'out_channels:16' in experiment_setting or 'out_channels:8' in experiment_setting
        bigger_architectures[i] = 1 if more_nodes and not more_channels else (
            2 if more_channels and not more_nodes else 0)
    return best_individuals, (
        fitness_small_pop, fitness_big_pop), population_type, init_mutations, bigger_kernel, bigger_architectures


if __name__ == '__main__':
    experiments = os.listdir(experiment_dir)
    best_individuals, (
        fitness_small_pop,
        fitness_big_pop), population_type, init_mutations, bigger_kernel, bigger_architectures = load_experiment_data(
        experiments)

    architecture_difference = (best_individuals[bigger_architectures == 1][:, 0],
                               best_individuals[bigger_architectures == 2][:, 0])

    fitness_small_pop = fitness_small_pop[population_type == 2]
    fitness_big_pop = fitness_big_pop[population_type == 1]

    kernel_difference = (best_individuals[bigger_kernel][:, 0],
                         best_individuals[~bigger_kernel][:, 0])

    best_individuals[:, 2] -= 1
    pareto_optimal = utils.is_pareto_efficient(best_individuals[:, :-1])

    plots.scatter_distribution(best_individuals, "Accuracy", "MACs",
                               "Best individuals per configuration",
                               np.array(
                                   ["population_size:100, generations:5",
                                    "population_size:5, generations:100"]), marker_types=pareto_optimal).savefig('doc/flat_individuals.pdf')
    # plots.conv_plot_multiple_runs(fitness_small_pop, "Epoch", "Accuracy", "population_size:5,n_generations:100").show()
    # plots.conv_plot_multiple_runs(fitness_big_pop, "Epoch", "Accuracy", "population_size:100,n_generations:5").show()
    matplotlib.rcParams.update({'font.size': 12})

    plots.conv_plot_multiple_runs(fitness_small_pop[init_mutations[population_type == 2]], "Epoch", "Accuracy",
                                  "population_size:5,n_generations:100,init_mutations:1000").savefig(
        'doc/flat_pop5mut1000.pdf')
    plots.conv_plot_multiple_runs(fitness_small_pop[~init_mutations[population_type == 2]], "Epoch", "Accuracy",
                                  "population_size:5,n_generations:100,init_mutations:0").savefig(
        'doc/flat_pop5mut0.pdf')
    plots.conv_plot_multiple_runs(fitness_big_pop[init_mutations[population_type == 1]], "Epoch", "Accuracy",
                                  "population_size:100,n_generations:5,init_mutations:1000").savefig(
        'doc/flat_pop100mut10000.pdf')
    plots.conv_plot_multiple_runs(fitness_big_pop[~init_mutations[population_type == 1]], "Epoch", "Accuracy",
                                  "population_size:100,n_generations:5,init_mutations:0").savefig(
        'doc/flat_pop100mut0.pdf')

    plots.boxplot(kernel_difference, "Accuracy according to kernel size", "Kernel size", "Accuracy",
                  {'labels': [3, 5], 'notch': True}).savefig(
        'doc/flat_kernel.pdf')

    plots.boxplot(architecture_difference, "Accuracy of selected architecture types", "Architecture", "Accuracy",
                  {'labels': ['n_nodes >= 4', 'n_channels >= 8'], 'notch': True}).savefig(
        'doc/flat_architecture.pdf')
