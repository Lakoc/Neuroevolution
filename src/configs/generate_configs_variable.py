options = [
    ('n_nodes', ['2,1', '3,2', '5,2', '7,2', '5,3']),
    ('out_channels', ['2', '8', '16', '32']),
    ('kernel_size', ['3', '5'])]

if __name__ == "__main__":
    fixed_args = "--search_alg GeneticSearch --architecture VariableLength --population_size 50 --n_generations 10 --n_epochs 2 --batch_size 32 --dataset FashionMNIST"
    all_configs = ['']
    for key, values in options:
        prev_configs = all_configs.copy()
        all_configs = [f"{config} {f'--{key}' if key else ''} {value}" for value in values for config in prev_configs]
    for index, config in enumerate(all_configs):
        with open(f'configs_variable/config{index + 1}.txt', 'w') as f:
            f.write(f"{fixed_args}{config}")
