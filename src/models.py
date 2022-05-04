import torch.nn as nn
import torch
import copy
import numpy as np
import torch.nn.functional as F


class Motif(nn.Module):
    def __init__(self, genotype, operations, conv_set, input_sample, device, hierarchical):
        super(Motif, self).__init__()
        adjacency_matrix = genotype.to(torch.int)
        predecessors = [np.argwhere(adjacency_matrix[:, node] > 0).squeeze(0) for node in
                        range(adjacency_matrix.shape[0])]
        self.device = device
        self.layers = [
            [(copy.deepcopy(operations[(adjacency_matrix[predecessor, node]).item()]), predecessor) for predecessor
             in node_predecessors]
            for node, node_predecessors in enumerate(predecessors)]
        self.conv_set = conv_set
        self.hierarchical = hierarchical
        self.nodes = [None for _ in range(adjacency_matrix.shape[0])]
        output = self.pre_forward(input_sample)
        self.last_conv = nn.Sequential(
            nn.Conv2d(output.shape[1], conv_set['out_channels'], kernel_size=conv_set['kernel_size'],
                      padding=conv_set['padding']), nn.BatchNorm2d(conv_set['out_channels']), nn.ELU()).to(self.device)
        self.channel_safe = nn.Identity()

    def hierarchical_safe(self, x):
        if x.shape[-3] != self.conv_set['out_channels'] and isinstance(self.channel_safe, nn.Identity):
            self.channel_safe = nn.Conv2d(x.shape[-3], self.conv_set['out_channels'], kernel_size=1).to(self.device)

    def pre_forward(self, x):
        with torch.no_grad():
            self.nodes[0] = x
            for node in range(1, len(self.nodes)):
                res = []
                for index, (operation, input_node) in enumerate(self.layers[node]):
                    input_tensor = self.nodes[input_node]
                    if input_tensor is not None:
                        input_channels = input_tensor.shape[1]
                        if isinstance(operation, list):
                            operation = nn.Sequential(operation[0][0](in_channels=input_channels, **operation[0][1]),
                                                      operation[1], operation[2])
                            self.layers[node][index] = (operation, input_node)
                        if isinstance(operation, Motif):
                            operation = nn.Sequential(
                                nn.Conv2d(in_channels=input_channels, out_channels=self.conv_set['out_channels'],
                                          kernel_size=1), operation)
                        operation = operation.to(self.device)

                        res.append(operation(input_tensor))
                if len(res) > 0:
                    self.nodes[node] = torch.concat(res, dim=1)

        return x if self.nodes[-1] is None else self.nodes[-1]

    def forward(self, x):
        if self.hierarchical:
            self.hierarchical_safe(x)
            x = self.channel_safe(x)
        self.nodes[0] = x
        for node in range(1, len(self.nodes)):
            res = []
            for index, (operation, input_node) in enumerate(self.layers[node]):
                input_tensor = self.nodes[input_node]
                if input_tensor is not None:
                    res.append(operation(input_tensor))
            if len(res) > 0:
                self.nodes[node] = torch.concat(res, dim=1)
        y = x if self.nodes[-1] is None else self.nodes[-1]
        return self.last_conv(y)


class FlatModel(nn.Module):
    def __init__(self, adjacency_matrix, operations, n_classes, batch_shape, device, conv_set):
        super(FlatModel, self).__init__()
        input_sample = torch.zeros(batch_shape).to(device)
        self.motif = Motif(adjacency_matrix, operations, conv_set, input_sample, device, hierarchical=False)
        self.glob_pooling = nn.AvgPool2d(kernel_size=conv_set['kernel_size'])
        flattened_size = conv_set['out_channels'] * (batch_shape[-2] // conv_set['kernel_size']) * (
                batch_shape[-1] // conv_set['kernel_size'])
        self.linear = nn.Linear(flattened_size, n_classes)

    def forward(self, x):
        x = self.motif(x)
        x = self.glob_pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class HierarchicalModel(nn.Module):
    def __init__(self, adjacency_matrices, operations, n_classes, batch_shape, device, conv_set):
        super(HierarchicalModel, self).__init__()
        batch_next_shape = list(batch_shape)
        batch_next_shape[-3] = conv_set['out_channels']
        input_sample = torch.zeros(batch_next_shape).to(device)
        self.input_layer = nn.Sequential(
            nn.Conv2d(batch_shape[-3], batch_next_shape[-3], kernel_size=conv_set['kernel_size'],
                      padding=conv_set['padding']), nn.BatchNorm2d(batch_next_shape[-3]), nn.ELU())

        self.next_motifs = [Motif(motif, operations, conv_set, input_sample, device, hierarchical=True) for motif in
                            adjacency_matrices[0]]

        for i, level in enumerate(adjacency_matrices[1:]):
            self.prev_motifs = self.next_motifs
            self.next_motifs = []
            for motif in level:
                self.next_motifs.append(
                    Motif(motif, self.prev_motifs, conv_set, input_sample, device, hierarchical=True))
        self.last_level_motif = self.next_motifs[0]
        self.glob_pooling = nn.AvgPool2d(kernel_size=conv_set['kernel_size'])
        flattened_size = conv_set['out_channels'] * (batch_shape[-2] // conv_set['kernel_size']) * (
                batch_shape[-1] // conv_set['kernel_size'])
        self.linear = nn.Linear(flattened_size, n_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.last_level_motif(x)
        x = self.glob_pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class VariableLengthModel(nn.Module):
    def __init__(self, architecture, operations, n_classes, batch_shape, device, conv_set):
        super(VariableLengthModel, self).__init__()
        input_shape = torch.Tensor(list(batch_shape))
        flattened_size = None
        self.layers = []
        for layer_setting in architecture:
            if isinstance(layer_setting, list):
                layer = nn.Sequential(
                    layer_setting[0][0](in_channels=int(input_shape[-3].item()), **layer_setting[0][1]),
                    layer_setting[1], layer_setting[2])
                input_shape[-3] = conv_set['out_channels']
            elif isinstance(layer_setting, tuple):
                if flattened_size is None:
                    flattened_size = int(torch.prod(input_shape[-3:]).item())
                    layer = nn.Sequential(nn.Flatten(start_dim=1),
                                          layer_setting[0](in_features=flattened_size, out_features=layer_setting[1]))
                else:
                    layer = layer_setting[0](in_features=flattened_size, out_features=layer_setting[1])
                flattened_size = layer_setting[1]
            else:
                layer = layer_setting
            self.layers.append(layer.to(device))

        self.linear = nn.Linear(flattened_size, n_classes).to(device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


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
        return F.log_softmax(x, dim=1)
