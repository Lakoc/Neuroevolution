import torch
from src.datasets import FashionMNIST, MNIST
from src.Trainer import Trainer
from src.Logger import Logger
from torch import nn
from src.EvolutionarySearch import FlatSearch
from src.models import SimpleCNN

if __name__ == "__main__":
    out_channels = 3
    kernel_size = 3
    n_nodes = 5
    default_stride = 1
    default_padding = kernel_size // 2
    operations = [None,
                  nn.Identity(),
                  [(nn.Conv2d, {'out_channels': out_channels, 'kernel_size': 1}), nn.BatchNorm2d(out_channels),
                   nn.ReLU()],
                  [(nn.Conv2d, {'out_channels': out_channels, 'kernel_size': 1}), nn.BatchNorm2d(out_channels),
                   nn.ELU()],
                  [(nn.Conv2d, {'out_channels': out_channels, 'kernel_size': kernel_size, 'stride': default_stride,
                                'padding': 'same'}),
                   nn.BatchNorm2d(out_channels), nn.ReLU()],
                  [(nn.Conv2d, {'out_channels': out_channels, 'kernel_size': kernel_size, 'stride': default_stride,
                                'padding': 'same'}),
                   nn.BatchNorm2d(out_channels), nn.ELU()],
                  nn.Dropout2d(),
                  nn.MaxPool2d(kernel_size=kernel_size, padding=default_padding, stride=default_stride),
                  nn.AvgPool2d(kernel_size=kernel_size, padding=default_padding, stride=default_stride)]
    flat_representation = [[{'n_nodes': n_nodes}]]
    dataset = FashionMNIST()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(dataset, device, optimizer=torch.optim.Adam)
    logger = Logger('results', 'results.txt')
    flat_search = FlatSearch(per_level_motifs=flat_representation, primitive_operations=operations, init_mutations=1000,
                             n_generations=100, population_size=5, trainer=trainer, logger=logger,
                             conv_set={'out_channels': out_channels, 'kernel_size': kernel_size})
    flat_search.init_population()
    flat_search.evolve()
