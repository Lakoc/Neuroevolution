import copy
from torch import nn
from tqdm import tqdm
import numpy as np


class Search:
    def __init__(self, population_size, architecture_model, primitive_operations,
                 conv_set, trainer, logger, architecture):
        self.population_size = population_size
        self.population = [architecture(str(i), architecture_model, primitive_operations, conv_set) for i in
                           range(self.population_size)]
        self.trainer = trainer
        self.logger = logger
        self.best_individual = (0, 0, 0, None, None)

    def init_population(self):
        tqdm.write('Initialization of population...')
        for individual in self.population:
            individual.init_genotype_random()

    def evaluate_init_pop(self, per_generation_fitness, per_generation_macs, per_generation_params):
        tqdm.write('Evaluation of initial population began.')
        init_fitness, init_macs, init_params = self.evaluate_individuals(self.population)
        tqdm.write(
            f'Best fitness: {np.max(init_fitness):.2f}, '
            f'median fitness: {np.median(init_fitness):.2f}')
        per_generation_fitness[0, :] = init_fitness
        per_generation_params[0, :] = init_params
        per_generation_macs[0, :] = init_macs

    def evaluate_individuals(self, population):
        fitness_all = np.empty(self.population_size)
        params_all = np.empty(self.population_size)
        macs_all = np.empty(self.population_size)
        for i, individual in enumerate(tqdm(population)):
            (acc_mean, acc_std, macs, params), net = individual.evaluate(self.trainer, self.logger)
            fitness_all[i] = acc_mean
            params_all[i] = params
            macs_all[i] = macs
            if acc_mean >= self.best_individual[0]:
                self.best_individual = (acc_mean, macs, params, individual.genotype, net)
        return fitness_all, macs_all, params_all


class RandomSearch(Search):
    def __init__(self, population_size, architecture_model, primitive_operations,
                 conv_set, trainer, logger, architecture, **_):
        super(RandomSearch, self).__init__(population_size, architecture_model, primitive_operations,
                                           conv_set, trainer, logger, architecture)

    def evolve(self):
        tqdm.write(f"Starting random search with {self.population_size} individuals...")
        tqdm.write('Evaluation of all individuals...')
        fitness, macs, params = self.evaluate_individuals(self.population)
        tqdm.write(f'Best fitness: {np.max(fitness):.2f}, '
                   f'median fitness: {np.median(fitness):.2f}')
        return fitness, macs, params, self.best_individual


class EvolutionarySearch(Search):
    def __init__(self, population_size, selection_pressure, n_generations, architecture_model, primitive_operations,
                 conv_set, init_mutations, trainer, logger, architecture):
        super(EvolutionarySearch, self).__init__(population_size, architecture_model, primitive_operations,
                                                 conv_set, trainer, logger, architecture)
        self.n_generations = n_generations
        self.init_mutations = init_mutations
        self.tournament_size = max(2, int(population_size * selection_pressure))

    def init_population(self):
        tqdm.write('Initialization of population...')
        for individual in self.population:
            individual.init_genotype_from_identities(self.init_mutations)

    def tournament_selection(self, fitness_all):
        individuals = np.random.randint(0, self.population_size, self.tournament_size)
        winner_ind = np.argmax(fitness_all[individuals])
        winner = individuals[winner_ind]
        return winner

    def evolve(self):
        tqdm.write(
            f"Starting evolutionary search of {self.n_generations} generations "
            f"with {self.population_size} individuals in population...")
        per_generation_fitness = np.empty((self.n_generations + 1, self.population_size))
        per_generation_macs = np.empty_like(per_generation_fitness)
        per_generation_params = np.empty_like(per_generation_fitness)
        self.evaluate_init_pop(per_generation_fitness, per_generation_macs, per_generation_params)

        tqdm.write('Starting evolution ...\n')
        for generation in range(self.n_generations):
            generation_curr = generation + 1
            tqdm.write(f'Generation: {generation_curr}/{self.n_generations}')

            offspring = [self.population[self.tournament_selection(per_generation_fitness[generation])].generate_copy()
                         for _ in range(self.population_size)]

            for i, individual in enumerate(offspring):
                individual.id = f'{generation}_{i}'
                individual.mutate()
            fitness, macs, params = self.evaluate_individuals(offspring)
            per_generation_fitness[generation_curr, :] = fitness
            per_generation_macs[generation_curr, :] = macs
            per_generation_params[generation_curr, :] = params

            tqdm.write(f'Best fitness overall: {self.best_individual[0]:.2f}, '
                       f'best fitness curr: {np.max(per_generation_fitness[generation_curr, :]):.2f}, '
                       f'median fitness: {np.median(per_generation_fitness[generation_curr, :]):.2f}\n')

            self.population = self.population + offspring  # Do not remove any genotypes from the population,
            # allowing it to grow with time, maintaining architecture diversity.

        return per_generation_fitness, per_generation_macs, per_generation_params, self.best_individual


class GeneticSearch(Search):
    def __init__(self, population_size, architecture_model, n_generations, primitive_operations,
                 conv_set, trainer, logger, architecture, alpha=0.03, beta=100, elitism=0.2, mating_pool=1,
                 crossover_prob=0.2, **_):
        super(GeneticSearch, self).__init__(population_size, architecture_model, primitive_operations,
                                            conv_set, trainer, logger, architecture)
        self.n_generations = n_generations
        self.alpha = alpha
        self.beta = beta
        self.crossover_prob = crossover_prob
        self.elitism_size = max(int(population_size * elitism), 1)
        self.mating_pool = []
        self.mating_pool_size = int(population_size * mating_pool)

    def environmental_selection(self, merged_pop):
        all_fitness = [individual.acc_mean for individual in merged_pop]
        elite_indexes = np.argsort(all_fitness)[-self.elitism_size:]
        selected = []
        for index in elite_indexes:
            selected.append(merged_pop[index])
        rest_mask = np.ones(len(merged_pop), dtype=int)
        rest_mask[elite_indexes] = 0
        rest_pop = [merged_pop[index] for index, mask in enumerate(rest_mask) if mask]
        for _ in range(self.population_size - self.elitism_size):
            tournament_individuals = np.random.randint(0, len(rest_pop), 2)
            winner = self.slack_binary_tournament(rest_pop[tournament_individuals[0]],
                                                  rest_pop[tournament_individuals[1]])
            selected.append(winner)
        return selected

    def generate_offspring(self):
        offspring = []
        for _ in range(len(self.mating_pool) // 2):
            p1 = self.mating_pool.pop(np.random.randint(len(self.mating_pool)))
            p2 = self.mating_pool.pop(np.random.randint(len(self.mating_pool)))
            offspring1, offspring2 = self.crossover(p1, p2)
            offspring1.mutate()
            offspring2.mutate()
            offspring.extend([offspring1, offspring2])
        return offspring

    def get_random_parents(self):
        p1 = self.population[np.random.randint(self.population_size)]
        p2 = self.population[np.random.randint(self.population_size)]
        return p1, p2

    def generate_mating_pool(self):
        self.mating_pool = [self.slack_binary_tournament(*self.get_random_parents()) for _ in
                            range(self.mating_pool_size)]

    def swap_units_crossover(self, layers, layer_type, item):
        if np.random.rand() <= self.crossover_prob:
            c1_layer = layers[layer_type][0][item]
            c2_layer = layers[layer_type][1][item]

            # get correct layer indexes
            tmp = c1_layer[0]
            c1_layer[0] = c2_layer[0]
            c2_layer[0] = tmp

            # swap items
            layers[layer_type][0][item] = c2_layer
            layers[layer_type][1][item] = c1_layer
        return layers

    @staticmethod
    def split_layers(c1, c2):
        layers = {'conv': ([], []), 'pool': ([], []), 'dense': ([], [])}
        for child_index, child in enumerate([c1, c2]):
            for layer_index, layer in enumerate(child.genotype):
                if isinstance(layer, tuple):
                    layer_type = 'dense'
                elif isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.MaxPool2d) or isinstance(layer,
                                                                                                      nn.Dropout2d):
                    layer_type = 'pool'
                else:
                    layer_type = 'conv'
                layers[layer_type][child_index].append([layer_index, layer])
        return layers

    @staticmethod
    def collect_layers(c1, c2, layers):
        for layer_type in layers.keys():
            for child_index, child in enumerate([c1, c2]):
                for layer_index, layer in layers[layer_type][child_index]:
                    child.genotype[layer_index] = layer
        return c1, c2

    def crossover(self, p1, p2):
        c1 = copy.deepcopy(p1)
        c2 = copy.deepcopy(p2)
        layers = self.split_layers(c1, c2)

        for layer_type in layers.keys():
            items_to_cross = min(len(layers[layer_type][0]), len(layers[layer_type][1]))
            for item in range(items_to_cross):
                self.swap_units_crossover(layers, layer_type, item)

        c1, c2 = self.collect_layers(c1, c2, layers)
        return c1, c2

    def slack_binary_tournament(self, ind1, ind2):
        better_index = np.argsort([ind1.acc_mean, ind2.acc_mean])[1]
        individuals = (ind1, ind2)
        s1 = individuals[better_index]
        s2 = individuals[1 - better_index]
        if s1.acc_mean - s2.acc_mean > self.alpha:
            return s1
        else:
            if s1.n_params - s2.n_params > self.beta:
                return s1
            elif s1.acc_std < s2.acc_std:
                return s1
            elif s1.acc_std > s2.acc_std:
                return s2
            else:
                random_one = np.random.randint(0, 2)
                return individuals[random_one]

    def collect_stats(self):
        fitness_all = np.empty(self.population_size)
        params_all = np.empty(self.population_size)
        macs_all = np.empty(self.population_size)
        for i, individual in enumerate(self.population):
            fitness_all[i] = individual.fitness
            params_all[i] = individual.n_params
            macs_all[i] = individual.macs
        return fitness_all, macs_all, params_all

    def evolve(self):
        tqdm.write(
            f"Starting genetic search for {self.n_generations} generations "
            f"with {self.population_size} individuals in population...")
        per_generation_fitness = np.empty((self.n_generations + 1, self.population_size))
        per_generation_params = np.empty_like(per_generation_fitness)
        per_generation_macs = np.empty_like(per_generation_fitness)

        self.evaluate_init_pop(per_generation_fitness, per_generation_macs, per_generation_params)

        tqdm.write('Starting evolution ...\n')
        for generation in range(self.n_generations):
            generation_curr = generation + 1
            tqdm.write(f'Generation: {generation_curr}/{self.n_generations}')

            self.generate_mating_pool()
            offspring = self.generate_offspring()
            self.evaluate_individuals(offspring)
            merged_pop = offspring + self.population
            new_pop = self.environmental_selection(merged_pop)
            self.population = new_pop
            fitness, macs, params = self.collect_stats()
            per_generation_fitness[generation_curr, :] = fitness
            per_generation_macs[generation_curr, :] = macs
            per_generation_params[generation_curr, :] = params

            tqdm.write(f'Best fitness overall: {self.best_individual[0]:.2f}, '
                       f'best fitness curr: {np.max(per_generation_fitness[generation_curr, :]):.2f}, '
                       f'median fitness: {np.median(per_generation_fitness[generation_curr, :]):.2f}\n')

        return per_generation_fitness, per_generation_macs, per_generation_params, self.best_individual
