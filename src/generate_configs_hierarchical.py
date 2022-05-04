options = [
    ('n_nodes', ['2,2,2;2', '2,2,2;2,2,2;2', '3,3,3;3,3,3;3', '4,4,4;4', '3,3,3;6']),
    ('out_channels', ['2', '8', '16', '32']),
    ('kernel_size', ['3', '5'])]

if __name__ == "__main__":
    fixed_args = "--search_alg EvolutionarySearch --architecture Hierarchical --population_size 100 --n_generations 5 --n_epochs 2 --batch_size 32 --init_mutations 1000 --dataset FashionMNIST"
    all_configs = ['']
    for key, values in options:
        prev_configs = all_configs.copy()
        all_configs = [f"{config} {f'--{key}' if key else ''} {value}" for value in values for config in prev_configs]
    for index, config in enumerate(all_configs):
        with open(f'configs/config{index + 1}.txt', 'w') as f:
            f.write(f"{fixed_args}{config}")
