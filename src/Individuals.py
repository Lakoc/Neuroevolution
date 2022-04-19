import torch
import numpy as np
from src.models import FlatModel


class Flat:
    def __init__(self, genotype_id, per_level_motifs, primitive_operations, conv_set):
        self.genotype = [[torch.triu(torch.ones((motif['n_nodes'], motif['n_nodes'])), diagonal=1) for motif in level]
                         for level in per_level_motifs]
        self.operations = primitive_operations
        self.id = genotype_id
        self.fitness = 0
        self.conv_set = conv_set

    def init_genotype(self, init_mutations):
        for i in range(init_mutations):
            self.mutate()

    def mutate(self):
        level = np.random.randint(len(self.genotype))
        motif = np.random.randint(len(self.genotype[level]))
        n_nodes = self.genotype[level][motif].shape[0]
        predecessor = np.random.randint(n_nodes - 1)
        successor = np.random.randint(predecessor + 1, n_nodes)
        operation = np.random.randint(len(self.operations))
        self.genotype[level][motif][predecessor, successor] = operation

    def evaluate(self, trainer, logger):
        device = trainer.device
        input_sample = torch.zeros(trainer.dataset.batch_shape).to(device)
        net = FlatModel(self.genotype, self.operations, conv_set=self.conv_set, n_classes=trainer.dataset.n_classes,
                        input_sample=input_sample, device=device).to(device)
        trainer.train(net, logger.train_logs(self.id))
        self.fitness = trainer.eval(net, logger.eval_log(self.id))
        return self.fitness
