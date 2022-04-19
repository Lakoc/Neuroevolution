import torch.optim
import torch.nn as nn


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
        correct = 0
        with torch.no_grad():
            for data, labels in self.dataset.eval:
                data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = net(data)
                test_loss += self.criterion(outputs, labels)
                pred = outputs.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum()

        test_loss /= len(self.dataset.eval.dataset)
        accuracy = 100. * correct / len(self.dataset.eval.dataset)
        if logger:
            logger(test_loss, accuracy)
        return accuracy
