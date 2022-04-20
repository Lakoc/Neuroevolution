import time
import torch
from src.Trainer import Trainer
from src.Logger import Logger
from torch import nn
from argparse import ArgumentParser
import numpy as np
import src.datasets as datasets
import src.search as search
import src.individuals as individuals

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Neuroevolution alg interface.')

    # Search alg arguments
    parser.add_argument('--population_size', type=int, default=5,
                        help='Population size')
    parser.add_argument('--n_generations', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--search_alg', type=str, default='EvolutionarySearch',
                        help='Search algorithm to be used')

    # Trainer arguments
    parser.add_argument('--n_epochs', type=int, default=3,
                        help='Number of train epochs of each individual')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset to processes evolution on')

    # Individual arguments
    parser.add_argument('--architecture', type=str, default='Hierarchical',
                        help='Architecture of population individual')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Default convolution kernel size')
    parser.add_argument('--out_channels', type=int,
                        default=3,
                        help='Number of output channels of each convolution block')

    # Flat architecture args
    parser.add_argument('--init_mutations', type=int,
                        default=1000,
                        help='Number of initial mutations')
    parser.add_argument('--n_nodes', type=str,
                        default='3,3,3,3,3,3;4,4;3',
                        help='Number of nodes for flat representation')

    # Evolutionary search args
    parser.add_argument('--selection_pressure', type=float, default=0.05,
                        help='Selection pressure in the tournament offspring selection')

    args = parser.parse_args()

    default_stride = 1
    default_padding = args.kernel_size // 2

    operations = [None,
                  nn.Identity(),
                  [(nn.Conv2d, {'out_channels': args.out_channels, 'kernel_size': 1}),
                   nn.BatchNorm2d(args.out_channels), nn.ReLU()],
                  [(nn.Conv2d, {'out_channels': args.out_channels, 'kernel_size': 1}),
                   nn.BatchNorm2d(args.out_channels), nn.ELU()],
                  [(nn.Conv2d,
                    {'out_channels': args.out_channels, 'kernel_size': args.kernel_size, 'stride': default_stride,
                     'padding': 'same'}), nn.BatchNorm2d(args.out_channels), nn.ReLU()],
                  [(nn.Conv2d,
                    {'out_channels': args.out_channels, 'kernel_size': args.kernel_size, 'stride': default_stride,
                     'padding': 'same'}), nn.BatchNorm2d(args.out_channels), nn.ELU()],
                  nn.Dropout2d(),
                  nn.MaxPool2d(kernel_size=args.kernel_size, padding=default_padding, stride=default_stride),
                  nn.AvgPool2d(kernel_size=args.kernel_size, padding=default_padding, stride=default_stride)]

    dataset = getattr(datasets, args.dataset)()
    search_alg = getattr(search, args.search_alg)
    architecture = getattr(individuals, args.architecture)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(dataset, device, optimizer=torch.optim.Adam, epochs=args.n_epochs)
    logger = Logger('results', 'results.txt')

    architecture_model = [[int(val) for val in level.split(',')] for level in args.n_nodes.split(';')]

    search = search_alg(architecture_model=architecture_model, primitive_operations=operations,
                        init_mutations=args.init_mutations, selection_pressure=args.selection_pressure,
                        n_generations=args.n_generations, population_size=args.population_size, trainer=trainer,
                        logger=logger, conv_set={'out_channels': args.out_channels, 'kernel_size': args.kernel_size,
                                                 'padding': default_padding}, architecture=architecture)

    search.init_population()
    fitness_all, params_count, best_individual = search.evolve()

    out_name = str(vars(args)).replace(' ', '').replace("'", "") + f"_{time.time():.2f}_{best_individual[0]:.2f}"
    torch.save(best_individual[2], f"models/{out_name}")
    np.save(f"fitness/{out_name}", fitness_all)
    np.save(f"params/{out_name}", params_count)
