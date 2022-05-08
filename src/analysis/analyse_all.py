import os
import scipy.stats as st
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

    alpha = 0.05
    classes = [best_individuals[best_individuals[:, -1] == i][:, 0] for i in range(3)]
    p_s = np.array([st.normaltest(cls)[1] for cls in classes])
    if np.all(p_s > alpha):
        print('Normally distributed, t-test.')
        _, p0 = st.ttest_ind(classes[0], classes[1])
        _, p1 = st.ttest_ind(classes[0], classes[2])
        _, p2 = st.ttest_ind(classes[1], classes[2])
        for p in [p0, p1, p2]:
            if (p > alpha):
                print('The difference is NOT significant, the distributions MAY have the same mean value, p = ', p)
            else:
                print('The difference is significant, the distributions does not have the same mean value, p = ', p)

    else:
        print('Mann whitney test.')
        t0, p0 = st.mannwhitneyu(classes[0], classes[1])
        t1, p1 = st.mannwhitneyu(classes[0], classes[2])
        t2, p2 = st.mannwhitneyu(classes[1], classes[2])
        for p in [p0, p1, p2]:
            if (p > alpha):
                print('The difference is NOT significant, the distributions MAY have the same mean value, p = ', p)
            else:
                print('The difference is significant, the distributions does not have the same mean value, p = ', p)
