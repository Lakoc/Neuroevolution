import torch
from src.datasets import MNIST,FashionMNIST
from src.Trainer import Trainer
from src.Logger import Logger
from src.models import SimpleCNN
import numpy as np

if __name__ == '__main__':
    dataset = FashionMNIST(batch_size=64)
    logger = Logger('results', 'results.txt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    genotype_string = 'simple_cnn'
    net = SimpleCNN()
    worker = Trainer(dataset, device, torch.optim.Adam)
    worker.train(net, logger.train_logs(genotype_string))
    acc = worker.eval(net, logger.eval_log(genotype_string))
    print(acc, sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, net.parameters())]))
