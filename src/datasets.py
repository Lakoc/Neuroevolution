import torch
import torchvision


class FashionMNIST:
    def __init__(self, batch_size):
        self.mean = 0.2859
        self.std = 0.3530
        self.batch_size_train = batch_size
        self.batch_size_test = 1000
        self.n_workers = 6
        self.dataset_path = 'files/'

        self.train = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(self.dataset_path, train=True, download=True,
                                              transform=torchvision.transforms.Compose(
                                                  [torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((self.mean,), (self.std,))])),
            batch_size=self.batch_size_train, pin_memory=True, num_workers=self.n_workers, shuffle=True)

        self.eval = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(self.dataset_path, train=False, download=True,
                                              transform=torchvision.transforms.Compose(
                                                  [torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((self.mean,), (self.std,))])),
            batch_size=self.batch_size_test, pin_memory=True, num_workers=self.n_workers, shuffle=True)

        self.batch_shape = next(iter(self.train))[0].shape
        self.n_classes = 10


class MNIST:
    def __init__(self, batch_size):
        self.mnist_mean = 0.1307
        self.mnist_std = 0.3081
        self.batch_size_train = batch_size
        self.batch_size_test = 1000
        self.n_workers = 6
        self.dataset_path = 'files/'

        self.train = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(self.dataset_path, train=True, download=True,
                                       transform=torchvision.transforms.Compose(
                                           [torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((self.mnist_mean,), (self.mnist_std,))])),
            batch_size=self.batch_size_train, pin_memory=True, num_workers=self.n_workers, shuffle=True)

        self.eval = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(self.dataset_path, train=False, download=True,
                                       transform=torchvision.transforms.Compose(
                                           [torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((self.mnist_mean,), (self.mnist_std,))])),
            batch_size=self.batch_size_test, pin_memory=True, num_workers=self.n_workers, shuffle=True)

        self.batch_shape = next(iter(self.train))[0].shape
        self.n_classes = 10
