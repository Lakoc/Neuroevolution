import numpy as np
from src.plots import conv_plot
filename = '{population_size:5,n_generations:5,search_alg:EvolutionarySearch,n_epochs:2,dataset:FashionMNIST,architecture:Flat,kernel_size:3,out_channels:3,init_mutations:1000,n_nodes:4,selection_pressure:0.05}_1650668304.22_0.86.npy'
if __name__ == '__main__':
    stat_files = ['fitness', 'macs', 'params']
    stats = {}
    for stat in stat_files:
        stats[stat] = np.load(f'{stat}/{filename}')
    per_generation_stats = [{stat: stats[stat][gen] for stat in stat_files} for gen in range(stats['fitness'].shape[0])]

    outputs= stats['fitness'] * 100
    conv_plot(outputs)

