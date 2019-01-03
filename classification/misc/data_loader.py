from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.transforms as transforms
import torchvision
import torch


class Flatten(object):
    def __call__(self, tensor):
        return tensor.view(-1)

    def __repr__(self):
        return self.__class__.__name__


class Transpose(object):
    def __call__(self, tensor):
        return tensor.permute(1, 2, 0)

    def __repr__(self):
        return self.__class__.__name__


def load_pytorch(config):
    if config.dataset == 'cifar10':
        if config.data_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Transpose()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Transpose()
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Transpose()
        ])
        trainset = torchvision.datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=test_transform)
    elif config.dataset == 'cifar100':
        if config.data_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                Transpose()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                Transpose()
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            Transpose()
        ])
        trainset = torchvision.datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=test_transform)
    elif config.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            Flatten(),
        ])
        trainset = torchvision.datasets.MNIST(root=config.data_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=config.data_path, train=False, download=True, transform=transform)
    elif config.dataset == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            Flatten(),
        ])
        trainset = torchvision.datasets.FashionMNIST(root=config.data_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=config.data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset!")

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=config.test_batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers)
    return trainloader, testloader
