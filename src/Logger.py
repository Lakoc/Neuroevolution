import os
from functools import partial


class Logger:
    def __init__(self, folder, results_file='results.txt'):
        self.folder = folder
        self.results_file = os.path.join(self.folder, results_file)

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w') as fp:
                fp.write('Genotype\tLoss\tAccuracy\n')

    @staticmethod
    def __train_logs(file, logs):
        pass
        # with open(file, 'w') as fp:
        #     fp.write('Epoch\tIteration\tLoss\n')
        #     fp.write('\n'.join(f'{epoch}\t\t{iteration}\t\t\t{loss:.3f}' for (epoch, iteration, loss) in logs) + '\n')

    def train_logs(self, genotype):
        return partial(self.__train_logs, os.path.join(self.folder, genotype))

    def __eval_log(self, genotype, loss, accuracy):
        with open(self.results_file, 'a') as fp:
            fp.write(f'{genotype}\t\t\t{loss:.3f}\t{accuracy:.3f}\n')

    def eval_log(self, genotype):
        return partial(self.__eval_log, genotype)
