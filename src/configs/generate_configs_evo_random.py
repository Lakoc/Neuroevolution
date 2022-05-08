if __name__ == "__main__":
    fixed_args = "--kernel_size 3 --n_nodes 4 --out_channels 8 --population_size 50 --n_generations 5 --architecture Flat --n_epochs 2 --batch_size 128 --dataset FashionMNIST"
    for i, search_alg in enumerate(["--search_alg EvolutionarySearch", "--search_alg RandomSearch"]):
        for experiment in range(0, 3):
            with open(f'configs_random/config{(i * 3) + experiment + 1}.txt', 'w') as f:
                f.write(f"{search_alg} {fixed_args}")
