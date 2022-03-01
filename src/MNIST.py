import torch
import torchvision


class MNIST:
    def __init__(self):
        self.mnist_mean = 0.1307
        self.mnist_std = 0.3081
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.n_workers = 6
        self.dataset_path = 'files/'

        self.train = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('files/', train=True, download=True, transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((self.mnist_mean,), (self.mnist_std,))])),
            batch_size=self.batch_size_train, pin_memory=True, num_workers=self.n_workers, shuffle=True)

        self.eval = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('files/', train=False, download=True, transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((self.mnist_mean,), (self.mnist_std,))])),
            batch_size=self.batch_size_test, pin_memory=True, num_workers=self.n_workers, shuffle=True)
