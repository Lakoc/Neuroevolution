import copy
import torch
import numpy as np
from src.models import FlatModel, HierarchicalModel, VariableLengthModel


class Individual:
    def __init__(self, genotype_id, primitive_operations, conv_set):
        self.operations = primitive_operations
        self.id = genotype_id
        self.fitness = 0
        self.conv_set = conv_set
        self.genotype = None

    def generate_copy(self):
        return copy.deepcopy(self)

    def init_genotype_from_identities(self, init_mutations):
        pass

    def init_genotype_random(self):
        pass

    def eval(self, model, trainer, logger):
        device = trainer.device
        net = model(self.genotype, self.operations, conv_set=self.conv_set, n_classes=trainer.dataset.n_classes,
                    batch_shape=trainer.dataset.batch_shape, device=device).to(device)
        trainer.train(net, logger.train_logs(self.id))
        results = trainer.eval(net, logger.eval_log(self.id))
        self.fitness = results[0]
        return results, net


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


class VariableLength(Individual):
    def __init__(self, genotype_id, architecture, primitive_operations, conv_set, pool_range=(7, 9), conv_range=(2, 7),
                 max_dense=1024, mutation_prob=0.2):
        Individual.__init__(self, genotype_id, primitive_operations, conv_set)
        self.conv_layers_max = architecture[0][0]
        self.full_layers_max = architecture[0][1]
        self.max_len = self.conv_layers_max + self.full_layers_max
        self.num_conv = np.random.randint(0, self.conv_layers_max + 1)
        self.num_full = np.random.randint(1, self.full_layers_max + 1)
        self.pool_range = pool_range
        self.conv_range = conv_range
        self.max_out_neurons_full = max_dense
        self.genotype = []
        self.init_genotype()
        self.m_prob = mutation_prob
        self.acc_mean = 0
        self.acc_std = 0
        self.n_params = 0
        self.macs = 0

    def init_genotype(self):
        for layer in range(self.num_conv):
            if np.random.rand() <= 0.5:
                self.genotype.append(self.get_random_conv_layer())
            else:
                self.genotype.append(self.get_random_pool_layer())
        for layer in range(self.num_full):
            self.genotype.append(self.get_random_full_layer())

    def get_random_conv_layer(self):
        index = np.random.randint(self.conv_range[0], self.conv_range[1])
        return self.operations[index]

    def get_random_pool_layer(self):
        index = np.random.randint(self.pool_range[0], self.pool_range[1])
        return self.operations[index]

    def get_random_full_layer(self):
        return torch.nn.Linear, np.random.randint(1, self.max_out_neurons_full)

    def add_layer_conv(self):
        if np.random.rand() <= 0.5:
            return self.get_random_conv_layer()
        else:
            return self.get_random_pool_layer()

    def mutate(self):
        ops = []
        num_conv_prev = self.num_conv
        self.num_conv = 0
        self.num_full = 0
        if np.random.rand() <= 1.0:
            new_genotype = []
            for index, layer in enumerate(self.genotype):
                if np.random.rand() <= 0.5:
                    p_op = np.random.randint(3)
                    ops.append(p_op)  # 0 add, 1 modify, 2 remove
                    if index < num_conv_prev:
                        if p_op == 0:
                            if self.num_conv < self.conv_layers_max:
                                new_genotype.append(self.get_random_conv_layer())
                                new_genotype.append(layer)
                                self.num_conv += 2
                        elif p_op == 1:
                            new_genotype.append(self.get_random_conv_layer())
                            self.num_conv += 1
                    else:
                        if p_op == 0:
                            if num_conv_prev == 0 and index == 0 and np.random.rand() <= 0.5:
                                new_genotype.append(self.get_random_conv_layer())
                                new_genotype.append(layer)
                                self.num_full += 1
                                self.num_conv += 1
                            if self.num_full < self.full_layers_max:
                                new_genotype.append(self.get_random_full_layer())
                                new_genotype.append(layer)
                                self.num_full += 2
                        elif p_op == 1:
                            new_genotype.append(self.get_random_full_layer())
                            self.num_full += 1
                else:
                    if index < num_conv_prev:
                        self.num_conv += 1
                    else:
                        self.num_full += 1
                    new_genotype.append(layer)
                    ops.append(-1)  # 0 add, 1 modify, 2 remove
            if self.num_full == 0:
                self.num_full += 1
                new_genotype.append(self.get_random_full_layer())
            self.genotype = new_genotype

    def evaluate(self, trainer, logger):
        device = trainer.device
        net = VariableLengthModel(self.genotype, self.operations, conv_set=self.conv_set,
                                  n_classes=trainer.dataset.n_classes,
                                  batch_shape=trainer.dataset.batch_shape, device=device).to(device)
        trainer.train(net, logger.train_logs(self.id))
        acc_mean, acc_std, macs, params = trainer.eval(net, logger.eval_log(self.id))
        self.fitness = acc_mean
        self.macs = macs
        self.acc_mean = acc_mean
        self.acc_std = acc_std
        self.n_params = params
        return (acc_mean, acc_std, macs, params), net
