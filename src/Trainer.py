import numpy as np
import torch.optim
import torch.nn as nn
from ptflops import get_model_complexity_info


class Trainer:
    def __init__(self, dataset, device, optimizer, epochs=3, iterations_per_log=200):
        self.epochs = epochs
        self.iterations_per_log = iterations_per_log
        self.device = device
        self.dataset = dataset
        self.criterion = nn.NLLLoss()
        self.optimizer = optimizer

    def train(self, net, logger=None):
        net = net.to(self.device, non_blocking=True)
        optimizer = self.optimizer(net.parameters())
        losses = []
        net.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (data, labels) in enumerate(self.dataset.train, 0):
                data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                outputs = net(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % self.iterations_per_log == self.iterations_per_log - 1:
                    losses.append((epoch, i, running_loss / (self.iterations_per_log * self.dataset.train.batch_size)))
                    running_loss = 0.0
        if logger:
            logger(losses)

    def eval(self, net, logger):
        net.eval()
        test_loss = 0
        per_batch_accuracies = np.empty(len(self.dataset.eval))
        with torch.no_grad():
            for i, (data, labels) in enumerate(self.dataset.eval):
                data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = net(data)
                test_loss += self.criterion(outputs, labels)
                pred = outputs.data.max(1, keepdim=True)[1]
                per_batch_accuracies[i] = pred.eq(labels.data.view_as(pred)).sum() / data.shape[0]

        test_loss /= len(self.dataset.eval.dataset)
        acc_mean = np.mean(per_batch_accuracies)
        acc_std = np.std(per_batch_accuracies)
        accuracy = 100. * acc_mean
        macs, params = get_model_complexity_info(net, tuple(self.dataset.batch_shape)[1:], as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        if logger:
            logger(test_loss, accuracy)
        return acc_mean, acc_std, macs, params
