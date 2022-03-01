import torch
import torchvision


class FashionMNIST:
    def __init__(self):
        self.mean = 0.2859
        self.std = 0.3530
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.n_workers = 6
        self.dataset_path = 'files/'

        self.train = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST('files/', train=True, download=True,
                                              transform=torchvision.transforms.Compose(
                                                  [torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((self.mean,), (self.std,))])),
            batch_size=self.batch_size_train, pin_memory=True, num_workers=self.n_workers, shuffle=True)

        self.eval = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST('files/', train=False, download=True,
                                              transform=torchvision.transforms.Compose(
                                                  [torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((self.mean,), (self.std,))])),
            batch_size=self.batch_size_test, pin_memory=True, num_workers=self.n_workers, shuffle=True)
