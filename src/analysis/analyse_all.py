import os

import numpy as np

import src.analysis.plots as plots
import src.utils as utils
from src.analysis import flat_analysis, hierarchical_analysis, variable_analysis

if __name__ == '__main__':
    flat_experiments = flat_analysis.load_experiment_data(os.listdir("experiments/flat"))
    variable_experiments = variable_analysis.load_experiment_data(os.listdir("experiments/variable"))
    hierarchical_experiments = hierarchical_analysis.load_experiment_data(os.listdir("experiments/hierarchical"))

    best_individuals = np.concatenate([flat_experiments[0], variable_experiments[0], hierarchical_experiments[0]])
    pareto_optimal = utils.is_pareto_efficient(best_individuals[:, :-1])
    classes = np.concatenate([np.zeros(flat_experiments[0].shape[0]), np.ones(variable_experiments[0].shape[0]),
                              np.ones(hierarchical_experiments[0].shape[0]) * 2])
    best_individuals[:, -1] = classes
    plots.scatter_distribution(best_individuals, "Accuracy", "MACs",
                               "Best individuals per configuration",
                               np.array(["Flat", "Variable length", "Hierarchical"]),
                               marker_types=pareto_optimal).savefig(
        'doc/all_individuals.pdf')
