import copy
import os
import pickle

import numpy as np
import torch
import src.analysis.plots as plots
import src.utils as utils
import torch.nn as nn
from ptflops import get_model_complexity_info
from thop import profile

experiment_dir = "experiments/variable"



class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x

def load_experiment_data(experiments):
    best_individuals = np.zeros((len(experiments), 3))
    fitness = np.zeros((len(experiments), 50, 11))
    for i, experiment in enumerate(experiments):
        experiment_setting = os.listdir(os.path.join(experiment_dir, experiment, 'best'))[0]
        fitness_path = os.path.join(experiment_dir, experiment, 'fitness', f'{experiment_setting}.npy')

        fitness[i, ...] = np.load(fitness_path).T

        best_individual = pickle.load(
            open(os.path.join(experiment_dir, experiment, 'best', experiment_setting.split('.npy')[0]), 'rb'))
        model = torch.load(os.path.join(experiment_dir, experiment, 'models', f'{experiment_setting}'))
        # model.layers = nn.ModuleList(model.layers)
        # x = [l._modules[item] for l in model.layers for item in l._modules]
        # x.append(model.linear)
        model.layers = nn.ModuleList(model.layers)

        macs, params = profile(model, inputs=(torch.zeros((1,1, 28, 28)).to('cuda'),), verbose=False)

        # macs, params = get_model_complexity_info(model, (1, 28, 28), as_strings=False,
        #                                          print_per_layer_stat=False, verbose=False)
        best_individuals[i][0] = best_individual[0]
        best_individuals[i][1] = macs

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
