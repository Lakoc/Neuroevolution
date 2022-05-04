import numpy as np


def is_pareto_efficient(costs):
    costs = np.copy(costs)
    costs[:, 0] = 1 - costs[:, 0]  # 1 - accuracy to minimize all parameters
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True
    return is_efficient
