import torch.optim
import torch.nn as nn


class Worker:
    def __init__(self, dataset_train, dataset_eval, device, net, epochs=3, iterations_per_log=200):
        self.epochs = epochs
        self.iterations_per_log = iterations_per_log
        self.device = device
        self.dataset_train = dataset_train
        self.dataset_eval = dataset_eval
        self.net = net.to(self.device, non_blocking=True)
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def train(self, logger=None):
        losses = []
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (data, labels) in enumerate(self.dataset_train, 0):
                data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()

                outputs = self.net(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % self.iterations_per_log == self.iterations_per_log - 1:
                    losses.append((epoch, i, running_loss / (self.iterations_per_log * self.dataset_train.batch_size)))
                    running_loss = 0.0
        if logger:
            logger(losses)

    def eval(self, logger):
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, labels in self.dataset_eval:
                data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = self.net(data)
                test_loss += self.criterion(outputs, labels)
                pred = outputs.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum()

        if logger:
            test_loss /= len(self.dataset_eval.dataset)
            accuracy = 100. * correct / len(self.dataset_eval.dataset)
            logger(test_loss, accuracy)
