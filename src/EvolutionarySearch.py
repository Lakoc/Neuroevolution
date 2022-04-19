from src.Individuals import Flat
from tqdm import tqdm


class FlatSearch:
    def __init__(self, population_size, n_generations, per_level_motifs, primitive_operations, conv_set,
                 init_mutations, trainer, logger):
        self.population_size = population_size
        self.population = [Flat(str(i), per_level_motifs, primitive_operations, conv_set) for i in
                           range(self.population_size)]
        self.init_mutations = init_mutations
        self.n_generations = n_generations
        self.trainer = trainer
        self.logger = logger

    def init_population(self):
        for individual in self.population:
            individual.init_genotype(self.init_mutations)

    def evolve(self):
        individual_fitness = [individual.evaluate(self.trainer, self.logger) for individual in tqdm(self.population)]
        print(individual_fitness)
