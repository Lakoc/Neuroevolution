options = [('', ["--population_size 100 --n_generations 30", "--population_size 30 --n_generations 100"]),
           ('n_nodes', ['2', '4', '8']), ('out_channels', ['2', '8', '16', '32']),
           ('kernel_size', ['3', '5']), ('init_mutations', ['0', '1000'])]

if __name__ == "__main__":
    fixed_args = "--search_alg EvolutionarySearch --architecture Flat --n_epochs 2 --batch_size 64 --dataset FashionMNIST"
    all_configs = ['']
    for key, values in options:
        prev_configs = all_configs.copy()
        all_configs = [f"{config} {f'--{key}' if key else ''} {value}" for value in values for config in prev_configs]
    for index, config in enumerate(all_configs):
        with open(f'configs/config{index+1}.txt', 'w') as f:
            f.write(f"{fixed_args}{config}")
