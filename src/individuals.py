import torch
import numpy as np
from src.models import FlatModel


class Flat:
    def __init__(self, genotype_id, architecture_model, primitive_operations, conv_set):
        self.motif_size = architecture_model[0][0]
        self.operations = primitive_operations
        self.id = genotype_id
        self.fitness = 0
        self.conv_set = conv_set
        self.genotype = None

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
        device = trainer.device
        input_sample = torch.zeros(trainer.dataset.batch_shape).to(device)
        net = FlatModel(self.genotype, self.operations, conv_set=self.conv_set, n_classes=trainer.dataset.n_classes,
                        input_sample=input_sample, device=device).to(device)
        trainer.train(net, logger.train_logs(self.id))
        self.fitness = trainer.eval(net, logger.eval_log(self.id))

        return self.fitness, net

# class Hierarchical:
#     def __init__(self, genotype_id, per_level_motifs, primitive_operations, conv_set):
#         self.genotype = [[torch.triu(torch.ones((motif, motif)), diagonal=1) for motif in level]
#                          for level in per_level_motifs]
#         self.operations = primitive_operations
#         self.id = genotype_id
#         self.fitness = 0
#         self.conv_set = conv_set
#
#     def init_genotype(self, init_mutations):
#         for i in range(init_mutations):
#             self.mutate()
#
#     def mutate(self):
#         level = np.random.randint(len(self.genotype))
#         motif = np.random.randint(len(self.genotype[level]))
#         n_nodes = self.genotype[level][motif].shape[0]
#         predecessor = np.random.randint(n_nodes - 1)
#         successor = np.random.randint(predecessor + 1, n_nodes)
#         operation = np.random.randint(len(self.operations))
#         self.genotype[level][motif][predecessor, successor] = operation
#
#     def evaluate(self, trainer, logger):
#         device = trainer.device
#         input_sample = torch.zeros(trainer.dataset.batch_shape).to(device)
#         net = FlatModel(self.genotype, self.operations, conv_set=self.conv_set, n_classes=trainer.dataset.n_classes,
#                         input_sample=input_sample, device=device).to(device)
#         trainer.train(net, logger.train_logs(self.id))
#         self.fitness = trainer.eval(net, logger.eval_log(self.id))
#
#         return self.fitness, net
