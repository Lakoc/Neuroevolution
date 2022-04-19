import torch.nn as nn
import torch
import copy
import numpy as np
import torch.nn.functional as F


class Motif(nn.Module):
    def __init__(self, genotype, operations, input_sample, device):
        super(Motif, self).__init__()
        adjacency_matrix = genotype.to(torch.int)
        predecessors = [np.argwhere(adjacency_matrix[:, node] > 0).squeeze(0) for node in
                        range(adjacency_matrix.shape[0])]
        self.device = device
        self.layers = [
            [(copy.deepcopy(operations[(adjacency_matrix[predecessor, node]).item()]), predecessor) for predecessor
             in node_predecessors]
            for node, node_predecessors in enumerate(predecessors)]
        self.nodes = [None for _ in range(adjacency_matrix.shape[0])]
        self.output = self.__pre_forward(input_sample)

    def __pre_forward(self, x):
        with torch.no_grad():
            self.nodes[0] = x
            for node in range(1, len(self.nodes)):
                res = []
                for index, (operation, input_node) in enumerate(self.layers[node]):
                    input_tensor = self.nodes[input_node]
                    if input_tensor is not None:
                        if isinstance(operation, list):
                            input_channels = input_tensor.shape[1]
                            operation = nn.Sequential(operation[0][0](in_channels=input_channels, **operation[0][1]),
                                                      operation[1], operation[2])
                            self.layers[node][index] = (operation, input_node)
                        operation = operation.to(self.device)
                        res.append(operation(input_tensor))
                if len(res) > 0:
                    self.nodes[node] = torch.concat(res, dim=1)

        return x if self.nodes[-1] is None else self.nodes[-1]

    def forward(self, x):
        self.nodes[0] = x
        for node in range(1, len(self.nodes)):
            res = []
            for index, (operation, input_node) in enumerate(self.layers[node]):
                input_tensor = self.nodes[input_node]
                if input_tensor is not None:
                    res.append(operation(input_tensor))
            if len(res) > 0:
                self.nodes[node] = torch.concat(res, dim=1)
        return x if self.nodes[-1] is None else self.nodes[-1]


class FlatModel(nn.Module):
    def __init__(self, adjacency_matrix, operations, n_classes, input_sample, device, conv_set):
        super(FlatModel, self).__init__()
        self.motif = Motif(adjacency_matrix, operations, input_sample, device)
        self.last_conv = nn.Sequential(
            nn.Conv2d(self.motif.output.shape[1], conv_set['out_channels'], kernel_size=conv_set['kernel_size'],
                      padding=1), nn.BatchNorm2d(conv_set['out_channels']),
            nn.ReLU())
        self.glob_pooling = nn.AvgPool2d(kernel_size=conv_set['kernel_size'])
        flattened_size = conv_set['out_channels'] * (input_sample.shape[-2] // conv_set['kernel_size']) * (
                    input_sample.shape[-1] // conv_set['kernel_size'])
        self.linear = nn.Linear(flattened_size, n_classes)

    def forward(self, x):
        x = self.motif(x)
        x = self.last_conv(x)
        x = self.glob_pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return F.log_softmax(x, dim=0)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)
