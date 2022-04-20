import torch
import numpy as np
from src.models import FlatModel, HierarchicalModel


class Individual:
    def __init__(self, genotype_id, primitive_operations, conv_set):
        self.operations = primitive_operations
        self.id = genotype_id
        self.fitness = 0
        self.conv_set = conv_set
        self.genotype = None

    def eval(self, model, trainer, logger):
        device = trainer.device
        net = model(self.genotype, self.operations, conv_set=self.conv_set, n_classes=trainer.dataset.n_classes,
                    batch_shape=trainer.dataset.batch_shape, device=device).to(device)
        trainer.train(net, logger.train_logs(self.id))
        self.fitness = trainer.eval(net, logger.eval_log(self.id))

        return self.fitness, net


class Flat(Individual):
    def __init__(self, genotype_id, architecture_model, primitive_operations, conv_set):
        super().__init__(genotype_id, primitive_operations, conv_set)
        self.motif_size = architecture_model[0][0]

    def init_genotype_from_identities(self, init_mutations):
        self.genotype = torch.triu(torch.ones((self.motif_size, self.motif_size)), diagonal=1)
        for i in range(init_mutations):
            self.mutate()

    def init_genotype_random(self):
        self.genotype = torch.triu(torch.randint(0, len(self.operations), (self.motif_size, self.motif_size)),
                                   diagonal=1)

    def mutate(self):
        predecessor = np.random.randint(self.motif_size - 1)
        successor = np.random.randint(predecessor + 1, self.motif_size)
        operation = np.random.randint(len(self.operations))
        self.genotype[predecessor, successor] = operation

    def evaluate(self, trainer, logger):
        return self.eval(FlatModel, trainer, logger)


class Hierarchical(Individual):
    def __init__(self, genotype_id, architecture, primitive_operations, conv_set):
        super().__init__(genotype_id, primitive_operations, conv_set)
        self.architecture = architecture
        self.level_motifs = [len(primitive_operations)] + [len(level) for level in architecture]

    def init_genotype_from_identities(self, init_mutations):
        self.genotype = [[torch.triu(torch.ones((motif, motif)), diagonal=1) for motif in level]
                         for level in self.architecture]
        for i in range(init_mutations):
            self.mutate()

    def init_genotype_random(self):
        self.genotype = [
            [torch.triu(torch.randint(0, self.level_motifs[index], (motif, motif)), diagonal=1) for motif in level] for
            index, level in enumerate(self.architecture)]

    def mutate(self):
        level = np.random.randint(len(self.genotype))
        motif = np.random.randint(len(self.genotype[level]))
        n_nodes = self.genotype[level][motif].shape[0]
        predecessor = np.random.randint(n_nodes - 1)
        successor = np.random.randint(predecessor + 1, n_nodes)
        operation = np.random.randint(self.level_motifs[level])
        self.genotype[level][motif][predecessor, successor] = operation

    def evaluate(self, trainer, logger):
        return self.eval(HierarchicalModel, trainer, logger)
