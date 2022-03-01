import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from src.MNIST import MNIST
from src.Worker import Worker
from src.Models import Flat
from src.Logger import Logger

if __name__ == '__main__':
    dataset = MNIST()
    logger = Logger('results', 'results.txt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    genotype = np.array([1, 0, 1])
    genotype_string = ''.join(str(x) for x in genotype)

    net = Flat(genotype)

    worker = Worker(dataset.train, dataset.eval, device, net)
    worker.train(logger.train_logs(genotype_string))
    worker.eval(logger.eval_log(genotype_string))
