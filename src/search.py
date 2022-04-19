from tqdm import tqdm
import numpy as np
import copy


class Search:
    def __init__(self, population_size, architecture_model, primitive_operations,
                 conv_set,  trainer, logger, architecture):
        self.population_size = population_size
        self.population = [architecture(str(i), architecture_model, primitive_operations, conv_set) for i in
                           range(self.population_size)]
        self.trainer = trainer
        self.logger = logger
        self.best_individual = (0, None, None)

    def evaluate_individuals(self, population):
        fitness_all = np.empty(self.population_size)
        for i, individual in enumerate(tqdm(population)):
            fitness, net = individual.evaluate(self.trainer, self.logger)
            fitness_all[i] = fitness
            if fitness >= self.best_individual[0]:
                self.best_individual = (fitness, individual.genotype, net)
        return fitness_all


class RandomSearch(Search):
    def __init__(self, population_size, architecture_model, primitive_operations,
                 conv_set, trainer, logger, architecture, **_):
        super(RandomSearch, self).__init__(population_size, architecture_model, primitive_operations,
                                           conv_set, trainer, logger, architecture)

    def init_population(self):
        print('Initialization of population...')
        for individual in self.population:
            individual.init_genotype_random()

    def evolve(self):
        print(f"Starting random search with {self.population_size} individuals...")
        print('Evaluation of all individuals...')
        fitness = self.evaluate_individuals(self.population)
        print(f'Best fitness: {np.max(fitness):.2f}, '
              f'median fitness: {np.median(fitness):.2f}')
        return fitness, self.best_individual


class EvolutionarySearch(Search):
    def __init__(self, population_size, selection_pressure, n_generations, architecture_model, primitive_operations,
                 conv_set, init_mutations, trainer, logger, architecture, **_):
        super(EvolutionarySearch, self).__init__(population_size, architecture_model, primitive_operations,
                                                 conv_set, trainer, logger, architecture)
        self.n_generations = n_generations
        self.init_mutations = init_mutations
        self.tournament_size = max(2, int(population_size * selection_pressure))

    def init_population(self):
        print('Initialization of population...')
        for individual in self.population:
            individual.init_genotype_from_identities(self.init_mutations)

    def tournament_selection(self, fitness_all):
        individuals = np.random.randint(0, self.population_size, self.tournament_size)
        winner_ind = np.argmax(fitness_all[individuals])
        winner = individuals[winner_ind]
        return winner

    def evolve(self):
        print(
            f"Starting evolutionary search of {self.n_generations} generations "
            f"with {self.population_size} individuals in population...")
        print('Evaluation of initial population began.')
        init_fitness = self.evaluate_individuals(self.population)
        print(
            f'Best fitness: {np.max(init_fitness):.2f}, '
            f'median fitness: {np.median(init_fitness):.2f}')
        per_generation_fitness = np.empty((self.n_generations + 1, self.population_size))
        per_generation_fitness[0, :] = init_fitness

        print('Starting evolution ...\n')
        for generation in range(self.n_generations):
            generation_curr = generation + 1
            print(f'Generation: {generation_curr}/{self.n_generations}')

            offspring = [copy.deepcopy(self.population[self.tournament_selection(per_generation_fitness[generation])])
                         for _ in range(self.population_size)]
            for individual in offspring:
                individual.mutate()
            per_generation_fitness[generation_curr, :] = self.evaluate_individuals(offspring)

            print(f'Best fitness overall: {self.best_individual[0]:.2f}, '
                  f'best fitness curr: {np.max(per_generation_fitness[generation_curr, :]):.2f}, '
                  f'median fitness: {np.median(per_generation_fitness[generation_curr, :]):.2f}\n')

            self.population = self.population + offspring  # We do not remove any genotypes from the population,
            # allowing it to grow with time, maintaining architecture diversity.

        return per_generation_fitness, self.best_individual
