import torch
from src.datasets import FashionMNIST
from src.Trainer import Trainer
from src.Logger import Logger
from src.models import SimpleCNN

if __name__ == '__main__':
    dataset = FashionMNIST()
    logger = Logger('results', 'results.txt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    genotype_string = 'simple_cnn'
    net = SimpleCNN()
    worker = Trainer(dataset, device, torch.optim.Adam)
    worker.train(net, logger.train_logs(genotype_string))
    acc = worker.eval(net, logger.eval_log(genotype_string))
    print(acc)
