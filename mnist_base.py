"""
File: mnist_v0.py
Author: Suibin Sun
Created Date: 2023-12-24, 11:57:18 pm
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-24, 11:57:18 pm
-----
"""

from torch.utils.data import DataLoader
from torchvision import datasets

import configs
import mnist_common

trainset = datasets.MNIST(
    configs.DATA_ROOT, train=True, download=True, transform=mnist_common.transform
)
trainloader = DataLoader(trainset, batch_size=mnist_common.batch_size, shuffle=True)
testset = datasets.MNIST(
    configs.DATA_ROOT, train=False, transform=mnist_common.transform
)
testloader = DataLoader(testset, batch_size=mnist_common.batch_size, shuffle=False)

model = mnist_common.Net()
optimizer = mnist_common.get_optimizer(model)
trained_model = mnist_common.train_model(model, trainloader, optimizer)
print('Test result for clean model and clean testset')
mnist_common.test_model(model, testloader)
